"""
Minimal training test to catch any bugs before actual sweep
"""
import gymnasium as gym
import deepq
from sdc_wrapper import SDC_Wrapper

print("Starting minimal training test...")

# Create environment
env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='rgb_array'),
                  remove_score=True, return_linear_velocity=False)

# Minimal discrete action set
actions = [
    [0.0, 1.0, 0.0],    # straight
    [-0.5, 0.5, 0.0],   # left
    [0.5, 0.5, 0.0],    # right
]

print("Testing discrete DQN with minimal config...")
try:
    deepq.learn(
        env,
        lr=0.0001,
        total_timesteps=500,
        action_repeat=2,
        gamma=0.99,
        batch_size=32,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        target_network_update_freq=100,
        buffer_size=1000,
        learning_starts=100,
        model_identifier='test_discrete',
        outdir='test_output',
        new_actions=actions,
        use_doubleqlearning=True,
        validation_freq=10000,
        num_validation_seeds=1,
        early_stopping_patience=5,
        no_display=True,
        use_wandb=False,
        optimizer_type='adam',
        hidden_sizes=[128, 64],
        dropout_rate=0.1,
        cnn_channels=[16, 32],
        cnn_kernels=[8, 4],
        cnn_strides=[4, 2],
        final_spatial_size=10,
        activation='relu',
        use_continuous_actions=False,
        normalization='layer',
        lr_scheduler='none',
        weight_decay=0.0,
        use_dueling=False
    )
    print("✓ Discrete DQN training successful!")
except Exception as e:
    print(f"✗ ERROR in discrete DQN: {e}")
    import traceback
    traceback.print_exc()

env.close()

# Test continuous actions (NAF)
print("\nTesting continuous DQN (NAF) with minimal config...")
env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='rgb_array'),
                  remove_score=True, return_linear_velocity=False)

try:
    deepq.learn(
        env,
        lr=0.0001,
        total_timesteps=500,
        action_repeat=2,
        gamma=0.99,
        batch_size=32,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        target_network_update_freq=100,
        buffer_size=1000,
        learning_starts=100,
        model_identifier='test_continuous',
        outdir='test_output',
        new_actions=None,
        use_doubleqlearning=True,
        validation_freq=10000,
        num_validation_seeds=1,
        early_stopping_patience=5,
        no_display=True,
        use_wandb=False,
        optimizer_type='adam',
        hidden_sizes=[128, 64],
        dropout_rate=0.1,
        cnn_channels=[16, 32],
        cnn_kernels=[8, 4],
        cnn_strides=[4, 2],
        final_spatial_size=10,
        activation='relu',
        use_continuous_actions=True,
        normalization='layer',
        lr_scheduler='none',
        weight_decay=0.0,
        use_dueling=False
    )
    print("✓ Continuous DQN (NAF) training successful!")
except Exception as e:
    print(f"✗ ERROR in continuous DQN: {e}")
    import traceback
    traceback.print_exc()

env.close()

print("\n" + "="*70)
print("MINIMAL TRAINING TEST COMPLETE")
print("="*70)
