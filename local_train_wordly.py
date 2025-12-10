import gymnasium as gym
import deepq
import platform
import os
import wandb

from sdc_wrapper import SDC_Wrapper


def get_action_set(config_name):
    # Map config name to predefined action sets
    action_sets = {
        'xlarge_hybrid_drift': [
            # Speed & Cruise
            [0.0, 1.0, 0.0],    # straight, full gas
            [0.0, 0.5, 0.0],    # straight, cruise
            [0.0, 0.0, 0.6],    # brake
            # Micro Adjustments
            [-0.2, 1.0, 0.0],   # slight left, full gas
            [0.2, 1.0, 0.0],    # slight right, full gas
            # Racing Turns
            [-0.6, 0.7, 0.0],   # medium left, fast
            [0.6, 0.7, 0.0],    # medium right, fast
            # Tight Cornering
            [-1.0, 0.3, 0.0],   # hard left, slow
            [1.0, 0.3, 0.0],    # hard right, slow
            # Drifting
            [-1.0, 0.0, 0.5],   # hard left + brake
            [1.0, 0.0, 0.5],    # hard right + brake
            [-0.5, 0.0, 0.5],   # medium left + brake
            [0.5, 0.0, 0.5],    # medium right + brake
        ],
    }
    return action_sets[config_name]


def get_cnn_config(config_name):
    # Map config name to full CNN architecture specs
    config_map = {
        'xlarge_4layer': ([128, 256, 512, 512], [8, 4, 3, 3], [4, 2, 1, 1], 6),
    }
    return config_map[config_name]


def get_hidden_sizes(config_name):
    # Map config name to list of hidden layer sizes
    config_map = {
        'large_2layer': [1024, 512],
    }
    return config_map[config_name]


def main():
    """Train with wordly as teacher"""

    print("python version:\t{0}".format(platform.python_version()))
    print("gym version:\t{0}".format(gym.__version__))

    # Hyperparameters (same as other students)
    action_config = 'xlarge_hybrid_drift'  # 13 actions (wordly's action set)
    action_repeat = 2
    activation = 'gelu'
    batch_size = 32
    buffer_size = 200000
    cnn_config = 'xlarge_4layer'
    dropout_rate = 0
    early_stopping_patience = 5
    exploration_final_eps = 0.035
    exploration_fraction = 0.10500105770997512
    gamma = 0.999
    hidden_layer_config = 'large_2layer'
    learning_starts = 500
    lr = 0.0001
    lr_scheduler = 'cosine'
    normalization = 'layer'
    num_validation_seeds = 50
    optimizer_type = 'adamw'
    target_network_update_freq = 1500
    total_timesteps = 400000
    use_doubleqlearning = True
    use_dueling = True
    validation_freq = 10000
    weight_decay = 3.2265646597375465e-05

    # Validation seeds - same 50 seeds used in evaluate_racing.py
    validation_seeds = [
        22597174, 68545857, 75568192, 91140053, 86018367,
        49636746, 66759182, 91294619, 84274995, 31531469,
        10000019, 20000003, 30000001, 40000003, 50000017,
        60000011, 70000027, 80000023, 90000049, 10000079,
        12345678, 87654321, 23456789, 98765432, 34567890,
        11111111, 99999999, 55555555, 77777777, 13579246,
        64827193, 19283746, 73829164, 48291637, 92837465,
        38291647, 73829156, 18273645, 92837461, 47382916
    ]

    # Fixed parameters
    agent_name = 'wordly_student'
    outdir = 'local_models'
    no_display = True
    use_continuous_actions = False

    # Warm-up configuration - Wordly teacher
    warmup_teacher_path = 'wordly.pth'
    warmup_steps = 60000
    warmup_teacher_arch = {
        'cnn_channels': [32, 64, 128],
        'cnn_kernels': [8, 4, 3],
        'cnn_strides': [4, 2, 1],
        'final_spatial_size': 8,
        'hidden_sizes': [1024, 512, 256, 128],
        'activation': 'silu',
        'normalization': 'none',
        'use_dueling': True
    }

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Initialize wandb with custom run name
    wandb_project = "THE_FINAL_COUNTDOWN"
    wandb_run_name = f"wordly-student-wordly-teacher-large5-large2-gelu-60k-warmup"

    print(f"\n🔗 Initializing wandb...")
    print(f"   Project: {wandb_project}")
    print(f"   Run name: {wandb_run_name}")

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            'action_config': action_config,
            'action_repeat': action_repeat,
            'activation': activation,
            'batch_size': batch_size,
            'buffer_size': buffer_size,
            'cnn_config': cnn_config,
            'dropout_rate': dropout_rate,
            'early_stopping_patience': early_stopping_patience,
            'exploration_final_eps': exploration_final_eps,
            'exploration_fraction': exploration_fraction,
            'gamma': gamma,
            'hidden_layer_config': hidden_layer_config,
            'learning_starts': learning_starts,
            'lr': lr,
            'lr_scheduler': lr_scheduler,
            'normalization': normalization,
            'num_validation_seeds': num_validation_seeds,
            'optimizer_type': optimizer_type,
            'target_network_update_freq': target_network_update_freq,
            'total_timesteps': total_timesteps,
            'use_doubleqlearning': use_doubleqlearning,
            'use_dueling': use_dueling,
            'validation_freq': validation_freq,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'warmup_teacher': warmup_teacher_path,
            'training_type': 'wordly_student_with_wordly_warmup',
        },
        tags=['local-training', 'wordly-teacher', 'xlarge_hybrid_drift', 'large_5layer', 'large_2layer', 'gelu', '60k-warmup', '50-validation-seeds']
    )

    print(f"✓ Wandb initialized\n")

    # Get action set and architecture configs
    actions = get_action_set(action_config)
    cnn_channels, cnn_kernels, cnn_strides, final_spatial_size = get_cnn_config(cnn_config)
    hidden_sizes = get_hidden_sizes(hidden_layer_config)

    # Print configuration
    print("\n=== Wordly Student Training Configuration ===")
    print(f"Action config: {action_config} -> {len(actions)} actions")
    print(f"Action repeat: {action_repeat}")
    print(f"Activation: {activation}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size}")
    print(f"CNN config: {cnn_config} -> channels: {cnn_channels}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Exploration final eps: {exploration_final_eps}")
    print(f"Exploration fraction: {exploration_fraction}")
    print(f"Gamma: {gamma}")
    print(f"Hidden layer config: {hidden_layer_config} -> {hidden_sizes}")
    print(f"Learning rate: {lr}")
    print(f"LR scheduler: {lr_scheduler}")
    print(f"Normalization: {normalization}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Target network update freq: {target_network_update_freq}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Use double Q-learning: {use_doubleqlearning}")
    print(f"Use dueling: {use_dueling}")
    print(f"Validation starts: 100k steps")
    print(f"Validation freq: {validation_freq}")
    print(f"Num validation seeds: {num_validation_seeds}")
    print(f"Weight decay: {weight_decay}")
    print(f"Agent name: {agent_name}")
    print(f"Output directory: {outdir}")
    print()
    print(f"🔥 WARM-UP MODE: Using wordly teacher for first {warmup_steps} steps")
    print(f"   Teacher: {warmup_teacher_path}")
    print(f"📊 VALIDATION: Starts at 100k, using 50 evaluation seeds for best model selection")
    print("=" * 70)

    # Create environment
    render_mode = 'rgb_array' if no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode),
                     remove_score=True, return_linear_velocity=False)

    # Start training
    print("\nStarting wordly student training with warm-up...\n")
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
        use_wandb=True,
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

    # Close environment
    env.close()

    # Finish wandb run
    wandb.finish()

    print(f"\n✓ Wordly student training complete! Model saved in {outdir}/")


if __name__ == '__main__':
    main()
