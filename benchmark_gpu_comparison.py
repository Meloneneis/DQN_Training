"""
GPU Comparison Benchmark for DQN Training
Tests training speed with parallel agents (simulating wandb sweep agents)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import gymnasium as gym
from sdc_wrapper import SDC_Wrapper
from utils import get_state
from replay_buffer import ReplayBuffer
import multiprocessing as mp
from multiprocessing import Process, Queue


def collect_real_data(num_samples=10000, seed=42):
    """Collect real observations from CarRacing environment"""
    print(f"Collecting {num_samples} real samples from CarRacing environment...")

    env = SDC_Wrapper(
        gym.make('CarRacing-v2', render_mode='rgb_array'),
        remove_score=True,
        return_linear_velocity=False
    )

    replay_buffer = ReplayBuffer(num_samples)

    obs, _ = env.reset(seed=seed)
    obs = get_state(obs)
    episode_count = 0

    while len(replay_buffer) < num_samples:
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        next_obs = get_state(next_obs)

        # For continuous actions: [steering, gas, brake]
        continuous_action = np.array([action[0], action[1], action[2]], dtype=np.float32)
        replay_buffer.add(obs, continuous_action, reward, next_obs, float(done))

        obs = next_obs

        if done:
            obs, _ = env.reset()
            obs = get_state(obs)
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"  Collected {len(replay_buffer)}/{num_samples} samples ({episode_count} episodes)")

    env.close()
    print(f"Data collection complete: {len(replay_buffer)} samples from {episode_count} episodes\n")

    return replay_buffer


def training_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """Perform one training step for continuous actions (NAF)"""
    obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(batch_size)

    obs_batch = torch.FloatTensor(obs_batch).pin_memory().to(device, non_blocking=True)
    act_batch = torch.FloatTensor(act_batch).pin_memory().to(device, non_blocking=True)
    rew_batch = torch.FloatTensor(rew_batch).pin_memory().to(device, non_blocking=True)
    next_obs_batch = torch.FloatTensor(next_obs_batch).pin_memory().to(device, non_blocking=True)
    done_batch = torch.FloatTensor(done_batch).pin_memory().to(device, non_blocking=True)

    q_values, _ = policy_net(obs_batch, act_batch)

    with torch.no_grad():
        next_actions = target_net(next_obs_batch)
        next_q_values, _ = target_net(next_obs_batch, next_actions)
        next_q_values = next_q_values * (1 - done_batch)
        target_q_values = rew_batch + gamma * next_q_values

    loss = F.mse_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return loss.item()


def agent_worker(agent_id, batch_size, num_steps, result_queue, replay_buffer_data):
    """Worker function for a single agent (runs in separate process)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    # Reconstruct replay buffer from shared data
    replay_buffer = ReplayBuffer(replay_buffer_data['maxsize'])
    replay_buffer._storage = replay_buffer_data['storage']
    replay_buffer._next_idx = replay_buffer_data['next_idx']

    # Use continuous action model (NAF) matching wandb config
    from model import ContinuousActionDQN
    action_dim = 3  # steering, gas, brake

    policy_net = ContinuousActionDQN(
        action_dim=action_dim,
        device=device,
        hidden_sizes=[1024, 512],
        dropout_rate=0.0223099016747042,
        cnn_channels=[128, 256, 512, 512],
        cnn_kernels=[8, 4, 3, 3],
        cnn_strides=[4, 2, 1, 1],
        final_spatial_size=6,
        activation='relu',
        normalization='none'
    ).to(device)

    target_net = ContinuousActionDQN(
        action_dim=action_dim,
        device=device,
        hidden_sizes=[1024, 512],
        dropout_rate=0.0223099016747042,
        cnn_channels=[128, 256, 512, 512],
        cnn_kernels=[8, 4, 3, 3],
        cnn_strides=[4, 2, 1, 1],
        final_spatial_size=6,
        activation='relu',
        normalization='none'
    ).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    gamma = 0.99

    # Warmup
    for _ in range(10):
        _ = training_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    for step in range(num_steps):
        _ = training_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)

        # Print progress for agent 0 only
        if agent_id == 0 and (step + 1) % 100 == 0:
            print(f"  Agent 0: Step {step + 1}/{num_steps}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time

    result_queue.put({
        'agent_id': agent_id,
        'total_time': total_time,
        'time_per_step': total_time / num_steps * 1000
    })


def benchmark_parallel_agents(batch_size, num_steps, num_agents, replay_buffer_data):
    """Run multiple agents in parallel"""
    result_queue = Queue()
    processes = []

    print(f"  Starting {num_agents} parallel agent(s)...")
    start_time = time.time()

    for agent_id in range(num_agents):
        p = Process(target=agent_worker, args=(agent_id, batch_size, num_steps, result_queue, replay_buffer_data))
        p.start()
        processes.append(p)

    # Wait for all agents to complete
    for p in processes:
        p.join()

    wall_time = time.time() - start_time

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    results = sorted(results, key=lambda x: x['agent_id'])

    return results, wall_time


def main():
    """Run GPU comparison benchmark"""
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    print("="*100)
    print("GPU PARALLEL TRAINING BENCHMARK - Simulating Multiple Wandb Agents")
    print("="*100)
    print()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()

    num_steps = 1000
    batch_sizes = [32, 64, 128]
    num_agents_list = [1, 2, 4]

    # Collect data once and share across all agents
    print("Collecting training data (shared across all agents)...")
    replay_buffer = collect_real_data(num_samples=10000, seed=42)

    # Convert replay buffer to dictionary for multiprocessing
    replay_buffer_data = {
        'storage': replay_buffer._storage,
        'maxsize': replay_buffer._maxsize,
        'next_idx': replay_buffer._next_idx
    }

    all_results = {}

    for batch_size in batch_sizes:
        print("\n" + "="*100)
        print(f"BATCH SIZE: {batch_size}")
        print("="*100)

        all_results[batch_size] = {}

        for num_agents in num_agents_list:
            print(f"\n{num_agents} Parallel Agent(s):")
            print("-" * 100)

            agent_results, wall_time = benchmark_parallel_agents(batch_size, num_steps, num_agents, replay_buffer_data)

            all_results[batch_size][num_agents] = {
                'agent_results': agent_results,
                'wall_time': wall_time
            }

            print(f"\n  Wall clock time (all {num_agents} agent(s) complete): {wall_time:.2f}s")
            print(f"\n  Individual agent times:")
            for res in agent_results:
                print(f"    Agent {res['agent_id']}: {res['total_time']:.2f}s ({res['time_per_step']:.2f}ms/step)")

            avg_agent_time = np.mean([r['total_time'] for r in agent_results])
            avg_time_per_step = np.mean([r['time_per_step'] for r in agent_results])

            print(f"\n  Average per-agent time: {avg_agent_time:.2f}s")
            print(f"  Average time/step per agent: {avg_time_per_step:.2f}ms")

            if num_agents > 1:
                slowdown = avg_agent_time / all_results[batch_size][1]['agent_results'][0]['total_time']
                print(f"  Slowdown vs 1 agent: {slowdown:.2f}x")

    # Summary table
    print("\n\n" + "="*120)
    print("SUMMARY - Parallel GPU Performance")
    print("="*120)
    print(f"{'Batch':<10} {'# Agents':<12} {'Wall Time (s)':<18} {'Avg Agent Time (s)':<22} {'Avg Time/Step (ms)':<22} {'Slowdown':<15}")
    print("-" * 120)

    for batch_size in batch_sizes:
        for num_agents in num_agents_list:
            data = all_results[batch_size][num_agents]
            wall_time = data['wall_time']
            agent_results = data['agent_results']

            avg_agent_time = np.mean([r['total_time'] for r in agent_results])
            avg_time_per_step = np.mean([r['time_per_step'] for r in agent_results])

            if num_agents == 1:
                slowdown_str = "1.00x (baseline)"
            else:
                slowdown = avg_agent_time / all_results[batch_size][1]['agent_results'][0]['total_time']
                slowdown_str = f"{slowdown:.2f}x"

            print(f"{batch_size:<10} {num_agents:<12} {wall_time:<18.2f} {avg_agent_time:<22.2f} {avg_time_per_step:<22.2f} {slowdown_str:<15}")

        if batch_size != batch_sizes[-1]:
            print("-" * 120)

    print("="*120)

    # Quick reference
    print("\n\n" + "="*100)
    print("QUICK REFERENCE - GPU Parallel Performance")
    print("="*100)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Configuration: Pinned Memory + TF32 enabled")
    print()
    print(f"{'Batch':<10} {'1 Agent (ms/step)':<20} {'2 Agents (ms/step)':<20} {'4 Agents (ms/step)':<20}")
    print("-" * 70)
    for batch_size in batch_sizes:
        print(f"{batch_size:<10} ", end="")
        for num_agents in num_agents_list:
            avg_time_per_step = np.mean([r['time_per_step'] for r in all_results[batch_size][num_agents]['agent_results']])
            print(f"{avg_time_per_step:<20.2f} ", end="")
        print()
    print("="*100)
    print("\nNote: Wall time shows total time until ALL agents complete (parallel execution)")
    print("      Avg time/step shows average training speed per individual agent")
    print("      Slowdown shows performance degradation when running multiple agents")
    print("="*100)


if __name__ == "__main__":
    main()
