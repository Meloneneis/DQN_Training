"""
Benchmark script to compare training speed with and without mixed precision.
Uses actual gym CarRacing-v2 data and tests multiple batch sizes.
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


def collect_real_data(num_samples=5000, seed=42):
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
    step_count = 0

    while len(replay_buffer) < num_samples:
        # Take random actions
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        next_obs = get_state(next_obs)

        # Store action index (for discrete actions)
        action_id = np.random.randint(0, 9)  # Random action index
        replay_buffer.add(obs, action_id, reward, next_obs, float(done))

        obs = next_obs
        step_count += 1

        if done:
            obs, _ = env.reset()
            obs = get_state(obs)
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"  Collected {len(replay_buffer)}/{num_samples} samples ({episode_count} episodes)")

    env.close()
    print(f"Data collection complete: {len(replay_buffer)} samples from {episode_count} episodes\n")

    return replay_buffer


def training_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma,
                 device, use_mixed_precision, scaler=None):
    """Perform one training step with or without mixed precision"""

    # Sample batch from replay buffer
    obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(batch_size)

    obs_batch = torch.FloatTensor(obs_batch).to(device)
    act_batch = torch.LongTensor(act_batch).to(device)
    rew_batch = torch.FloatTensor(rew_batch).to(device)
    next_obs_batch = torch.FloatTensor(next_obs_batch).to(device)
    done_batch = torch.FloatTensor(done_batch).to(device)

    if use_mixed_precision and scaler is not None:
        # Mixed precision training
        with torch.cuda.amp.autocast():
            # Compute Q(s, a)
            q_values = policy_net(obs_batch)
            q_values = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)

            # Compute target Q values
            with torch.no_grad():
                next_q_values = target_net(next_obs_batch)
                next_q_values = next_q_values.max(1)[0]
                next_q_values = next_q_values * (1 - done_batch)
                target_q_values = rew_batch + gamma * next_q_values

            # Compute loss
            loss = F.mse_loss(q_values, target_q_values)

        # Backward pass with scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard FP32 training
        # Compute Q(s, a)
        q_values = policy_net(obs_batch)
        q_values = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = target_net(next_obs_batch)
            next_q_values = next_q_values.max(1)[0]
            next_q_values = next_q_values * (1 - done_batch)
            target_q_values = rew_batch + gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        optimizer.step()

    return loss.item()


def benchmark_training(replay_buffer, num_steps=500, batch_size=256, use_mixed_precision=True):
    """Benchmark training with or without mixed precision"""

    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU - mixed precision requires CUDA for speedup")

    # Set up model (use actual DQN from your codebase)
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

    # Set up optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    # Set up mixed precision scaler if needed
    scaler = None
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    # Hyperparameters
    gamma = 0.99

    # Warmup (not timed)
    print("  Warming up...")
    for _ in range(10):
        _ = training_step(policy_net, target_net, optimizer, replay_buffer, batch_size,
                         gamma, device, use_mixed_precision, scaler)

    # Synchronize CUDA before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print(f"  Running {num_steps} training steps...")

    start_time = time.time()
    losses = []

    for step in range(num_steps):
        loss = training_step(policy_net, target_net, optimizer, replay_buffer, batch_size,
                           gamma, device, use_mixed_precision, scaler)
        losses.append(loss)

        if (step + 1) % 100 == 0:
            print(f"    Step {step + 1}/{num_steps} - Avg loss: {np.mean(losses[-100:]):.4f}")

    # Synchronize CUDA after timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time

    return total_time, losses


def main():
    """Run the benchmark comparison"""
    print("="*70)
    print("Mixed Precision Training Benchmark with Real CarRacing Data")
    print("="*70)
    print()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Mixed precision training requires CUDA.")
        print("Benchmarks will run on CPU but won't show mixed precision speedup.\n")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}\n")

    # Configuration
    num_steps = 500  # Number of training steps per benchmark
    batch_sizes = [32, 64, 128]  # Different batch sizes to test

    # Collect real data once (reused for all benchmarks)
    replay_buffer = collect_real_data(num_samples=10000)

    # Store results
    results = {}

    # Test each batch size
    for batch_size in batch_sizes:
        print("\n" + "="*70)
        print(f"BATCH SIZE: {batch_size}")
        print("="*70)

        results[batch_size] = {}

        # Run FP32 benchmark
        print("\nFP32 (Standard Precision):")
        print("-" * 70)
        time_fp32, losses_fp32 = benchmark_training(
            replay_buffer, num_steps, batch_size, use_mixed_precision=False
        )
        results[batch_size]['fp32'] = time_fp32
        print(f"  Total time: {time_fp32:.2f}s | Avg time/step: {time_fp32/num_steps*1000:.2f}ms")

        # Small break
        time.sleep(1)

        # Run mixed precision benchmark
        print("\nMixed Precision:")
        print("-" * 70)
        time_mixed, losses_mixed = benchmark_training(
            replay_buffer, num_steps, batch_size, use_mixed_precision=True
        )
        results[batch_size]['mixed'] = time_mixed
        print(f"  Total time: {time_mixed:.2f}s | Avg time/step: {time_mixed/num_steps*1000:.2f}ms")

        # Comparison for this batch size
        if torch.cuda.is_available():
            speedup = time_fp32 / time_mixed
            time_saved = time_fp32 - time_mixed
            print(f"\n  Speedup: {speedup:.2f}x | Time saved: {time_saved:.2f}s ({time_saved/time_fp32*100:.1f}%)")

        print()

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY - Time per Training Step (milliseconds)")
    print("="*70)
    print(f"{'Batch Size':<15} {'FP32 (ms)':<15} {'Mixed (ms)':<15} {'Speedup':<15}")
    print("-" * 70)

    for batch_size in batch_sizes:
        fp32_time = results[batch_size]['fp32'] / num_steps * 1000
        mixed_time = results[batch_size]['mixed'] / num_steps * 1000
        speedup = results[batch_size]['fp32'] / results[batch_size]['mixed']

        print(f"{batch_size:<15} {fp32_time:<15.2f} {mixed_time:<15.2f} {speedup:<15.2f}x")

    print("="*70)

    # Best configuration
    if torch.cuda.is_available():
        best_batch = max(batch_sizes, key=lambda bs: results[bs]['fp32'] / results[bs]['mixed'])
        best_speedup = results[best_batch]['fp32'] / results[best_batch]['mixed']
        print(f"\nBest speedup: {best_speedup:.2f}x with batch size {best_batch}")
        print("="*70)


if __name__ == "__main__":
    main()
