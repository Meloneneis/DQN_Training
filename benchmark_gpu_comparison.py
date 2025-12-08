"""
GPU Comparison Benchmark for DQN Training
Tests training speed across different batch sizes and number of runs
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import gymnasium as gym
from model import DQN
from sdc_wrapper import SDC_Wrapper
from utils import get_state
from replay_buffer import ReplayBuffer


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

        action_id = np.random.randint(0, 9)
        replay_buffer.add(obs, action_id, reward, next_obs, float(done))

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
    """Perform one training step"""
    obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(batch_size)

    obs_batch = torch.FloatTensor(obs_batch).pin_memory().to(device, non_blocking=True)
    act_batch = torch.LongTensor(act_batch).pin_memory().to(device, non_blocking=True)
    rew_batch = torch.FloatTensor(rew_batch).pin_memory().to(device, non_blocking=True)
    next_obs_batch = torch.FloatTensor(next_obs_batch).pin_memory().to(device, non_blocking=True)
    done_batch = torch.FloatTensor(done_batch).pin_memory().to(device, non_blocking=True)

    q_values = policy_net(obs_batch)
    q_values = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_obs_batch)
        next_q_values = next_q_values.max(1)[0]
        next_q_values = next_q_values * (1 - done_batch)
        target_q_values = rew_batch + gamma * next_q_values

    loss = F.mse_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return loss.item()


def benchmark_training(replay_buffer, num_steps=500, batch_size=256):
    """Benchmark training with current optimizations"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU")

    action_size = 9
    policy_net = DQN(
        action_size=action_size,
        device=device,
        hidden_sizes=[1024, 512],
        dropout_rate=0.2,
        cnn_channels=[32, 64, 128, 128],
        cnn_kernels=[8, 4, 3, 3],
        cnn_strides=[4, 2, 1, 1],
        final_spatial_size=6,
        activation='relu',
        normalization='layer',
        use_dueling=False
    ).to(device)

    target_net = DQN(
        action_size=action_size,
        device=device,
        hidden_sizes=[1024, 512],
        dropout_rate=0.2,
        cnn_channels=[32, 64, 128, 128],
        cnn_kernels=[8, 4, 3, 3],
        cnn_strides=[4, 2, 1, 1],
        final_spatial_size=6,
        activation='relu',
        normalization='layer',
        use_dueling=False
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
    losses = []

    for step in range(num_steps):
        loss = training_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)
        losses.append(loss)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time

    return total_time, losses


def main():
    """Run GPU comparison benchmark"""
    print("="*80)
    print("GPU COMPARISON BENCHMARK - DQN Training Speed")
    print("="*80)
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

    num_steps = 500
    batch_sizes = [32, 64, 128]
    num_runs_list = [1, 2, 4]

    # Collect data once
    replay_buffer = collect_real_data(num_samples=10000)

    results = {}

    for batch_size in batch_sizes:
        print("\n" + "="*80)
        print(f"BATCH SIZE: {batch_size}")
        print("="*80)

        results[batch_size] = {}

        for num_runs in num_runs_list:
            print(f"\n{num_runs} Run(s):")
            print("-" * 80)

            run_times = []

            for run_idx in range(num_runs):
                print(f"  Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)
                total_time, losses = benchmark_training(replay_buffer, num_steps, batch_size)
                run_times.append(total_time)
                print(f"Done ({total_time:.2f}s)")

                time.sleep(0.5)

            avg_time = np.mean(run_times)
            std_time = np.std(run_times)
            time_per_step = avg_time / num_steps * 1000

            results[batch_size][num_runs] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'time_per_step': time_per_step,
                'run_times': run_times
            }

            print(f"\n  Average total time: {avg_time:.2f}s ± {std_time:.2f}s")
            print(f"  Average time per step: {time_per_step:.2f}ms")
            if num_runs > 1:
                print(f"  Individual runs: {[f'{t:.2f}s' for t in run_times]}")

    # Summary table
    print("\n\n" + "="*120)
    print("SUMMARY - GPU Comparison Results")
    print("="*120)
    print(f"{'Batch Size':<15} {'Metric':<25} {'1 Run':<20} {'2 Runs':<20} {'4 Runs':<20}")
    print("-" * 120)

    for batch_size in batch_sizes:
        # Total time
        print(f"{batch_size:<15} {'Total Time (s)':<25} ", end="")
        for num_runs in num_runs_list:
            avg = results[batch_size][num_runs]['avg_time']
            std = results[batch_size][num_runs]['std_time']
            if num_runs == 1:
                print(f"{avg:<20.2f} ", end="")
            else:
                print(f"{avg:.2f} ± {std:.2f}{' '*(20-len(f'{avg:.2f} ± {std:.2f}'))}", end="")
        print()

        # Time per step
        print(f"{'':<15} {'Time/Step (ms)':<25} ", end="")
        for num_runs in num_runs_list:
            time_per_step = results[batch_size][num_runs]['time_per_step']
            print(f"{time_per_step:<20.2f} ", end="")
        print()

        if batch_size != batch_sizes[-1]:
            print("-" * 120)

    print("="*120)

    # Export results for comparison
    print("\n\n" + "="*80)
    print("QUICK REFERENCE - Copy this for GPU comparison")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Configuration: Pinned Memory + TF32 enabled")
    print()
    print(f"{'Batch':<10} {'Time/Step (ms)':<20}")
    print("-" * 30)
    for batch_size in batch_sizes:
        time_per_step = results[batch_size][1]['time_per_step']
        print(f"{batch_size:<10} {time_per_step:<20.2f}")
    print("="*80)


if __name__ == "__main__":
    main()
