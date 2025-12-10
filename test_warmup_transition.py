"""
Test script to verify warmup to post-warmup transition works correctly.
This runs a very short training session to catch bugs like the epsilon crash.
"""

import gymnasium as gym
import deepq
import platform
import os

from sdc_wrapper import SDC_Wrapper


def test_warmup_transition():
    """Test warmup transition with minimal steps"""

    print("="*70)
    print("WARMUP TRANSITION TEST")
    print("="*70)
    print("\nTesting the transition from warmup to normal training...")
    print("This will run a SHORT training session to verify no crashes occur.\n")

    # Minimal hyperparameters for quick testing
    action_config = 'xlarge_aggressive_drifter'
    action_repeat = 4
    activation = 'gelu'
    batch_size = 32
    buffer_size = 5000  # Small buffer for testing
    cnn_config = 'xlarge_4layer'
    dropout_rate = 0
    early_stopping_patience = 100  # High to avoid early stopping during test
    exploration_final_eps = 0.05
    exploration_fraction = 0.3
    gamma = 0.96
    hidden_layer_config = 'medium_2layer'
    learning_starts = 50  # Start learning quickly
    lr = 0.0001
    lr_scheduler = 'none'
    normalization = 'none'
    num_validation_seeds = 5  # Few seeds for quick validation
    optimizer_type = 'adam'
    target_network_update_freq = 100
    total_timesteps = 500  # VERY SHORT - just to test transition
    use_doubleqlearning = True
    use_dueling = True
    validation_freq = 200  # Validate once during test
    weight_decay = 0.00002

    # Test warmup configuration
    warmup_teacher_path = 'lunar-sweep-7_best.pth'
    warmup_steps = 150  # Transition happens at step 150
    warmup_teacher_arch = {
        'cnn_channels': [64, 128, 256, 256],
        'cnn_kernels': [8, 4, 3, 3],
        'cnn_strides': [4, 2, 1, 1],
        'final_spatial_size': 6,
        'hidden_sizes': [512, 256],
        'activation': 'silu',
        'normalization': 'none',
        'use_dueling': True
    }

    # Validation seeds
    validation_seeds = [12345, 67890, 11111, 22222, 33333]

    # Fixed parameters
    agent_name = 'test_warmup'
    outdir = 'test_models'
    no_display = True
    use_continuous_actions = False

    # Action set
    def get_action_set(config_name):
        action_sets = {
            'xlarge_aggressive_drifter': [
                [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-0.2, 1.0, 0.0],
                [0.2, 1.0, 0.0], [-0.5, 0.8, 0.0], [0.5, 0.8, 0.0],
                [-1.0, 0.9, 0.0], [1.0, 0.9, 0.0], [-1.0, 0.0, 0.8],
                [1.0, 0.0, 0.8], [-0.5, 0.0, 0.5], [0.5, 0.0, 0.5],
            ],
        }
        return action_sets[config_name]

    def get_cnn_config(config_name):
        config_map = {
            'xlarge_4layer': ([128, 256, 512, 512], [8, 4, 3, 3], [4, 2, 1, 1], 6),
        }
        return config_map[config_name]

    def get_hidden_sizes(config_name):
        config_map = {
            'medium_2layer': [512, 256],
        }
        return config_map[config_name]

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Get action set and architecture configs
    actions = get_action_set(action_config)
    cnn_channels, cnn_kernels, cnn_strides, final_spatial_size = get_cnn_config(cnn_config)
    hidden_sizes = get_hidden_sizes(hidden_layer_config)

    print(f"\n{'='*70}")
    print("TEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Transition point: Step {warmup_steps}")
    print(f"Post-warmup steps: {total_timesteps - warmup_steps}")
    print(f"\nCRITICAL TESTS:")
    print(f"  1. Steps 0-{warmup_steps-1}: Warmup phase (teacher/student/random)")
    print(f"  2. Step {warmup_steps}: TRANSITION (switch to student + constant epsilon)")
    print(f"  3. Steps {warmup_steps+1}-{total_timesteps}: Post-warmup (student with epsilon={exploration_final_eps})")
    print(f"{'='*70}\n")

    # Create environment
    render_mode = 'rgb_array' if no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode),
                     remove_score=True, return_linear_velocity=False)

    # Start training
    print("Starting warmup transition test...\n")

    try:
        deepq.learn(
            env,
            lr=lr,
            total_timesteps=total_timesteps,
            action_repeat=action_repeat,
            gamma=gamma,
            batch_size=batch_size,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            target_network_update_freq=target_network_update_freq,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            model_identifier=agent_name,
            outdir=outdir,
            new_actions=actions,
            use_doubleqlearning=use_doubleqlearning,
            validation_freq=validation_freq,
            num_validation_seeds=num_validation_seeds,
            early_stopping_patience=early_stopping_patience,
            no_display=no_display,
            use_wandb=False,  # No wandb for testing
            optimizer_type=optimizer_type,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            cnn_channels=cnn_channels,
            cnn_kernels=cnn_kernels,
            cnn_strides=cnn_strides,
            final_spatial_size=final_spatial_size,
            activation=activation,
            use_continuous_actions=use_continuous_actions,
            normalization=normalization,
            lr_scheduler=lr_scheduler,
            weight_decay=weight_decay,
            use_dueling=use_dueling,
            warmup_teacher_path=warmup_teacher_path,
            warmup_teacher_arch=warmup_teacher_arch,
            warmup_steps=warmup_steps,
            validation_seeds_list=validation_seeds
        )

        print("\n" + "="*70)
        print("✓ TEST PASSED!")
        print("="*70)
        print("\nWarmup transition completed successfully:")
        print(f"  ✓ Warmup phase (0-{warmup_steps}): OK")
        print(f"  ✓ Transition at step {warmup_steps}: OK")
        print(f"  ✓ Post-warmup phase ({warmup_steps+1}-{total_timesteps}): OK")
        print("\nNo crashes detected. The epsilon bug is fixed!")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED!")
        print("="*70)
        print(f"\nError occurred: {e}")
        print("\nThe warmup transition has a bug that needs to be fixed.")
        print("="*70)
        raise

    finally:
        # Close environment
        env.close()

        # Clean up test models
        print("\nCleaning up test files...")
        import shutil
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        print("Test files removed.")


if __name__ == '__main__':
    test_warmup_transition()
