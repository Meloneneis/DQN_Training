import gymnasium as gym
import deepq
import argparse
import platform
import wandb
import os

from sdc_wrapper import SDC_Wrapper


def get_action_set(config_name):
    # Map config name to predefined action sets
    # Format: [steering, gas, brake] where steering: -1 to 1, gas: 0 to 1, brake: 0 to 1

    action_sets = {
        # Minimal sets (5 actions) - very discrete control
        'minimal_conservative': [
            [0.0, 0.5, 0.0],    # straight, half gas
            [-0.5, 0.3, 0.0],   # left, slow
            [0.5, 0.3, 0.0],    # right, slow
            [0.0, 1.0, 0.0],    # straight, full gas
            [0.0, 0.0, 0.5],    # brake
        ],

        'minimal_aggressive': [
            [0.0, 1.0, 0.0],    # straight, full gas
            [-0.8, 0.8, 0.0],   # hard left, fast
            [0.8, 0.8, 0.0],    # hard right, fast
            [-0.4, 0.6, 0.0],   # soft left, medium
            [0.4, 0.6, 0.0],    # soft right, medium
        ],

        # Small sets (7 actions)
        'small_balanced': [
            [0.0, 1.0, 0.0],    # straight, full gas
            [0.0, 0.5, 0.0],    # straight, half gas
            [-0.4, 0.7, 0.0],   # left, fast
            [0.4, 0.7, 0.0],    # right, fast
            [-0.7, 0.4, 0.0],   # hard left, slow
            [0.7, 0.4, 0.0],    # hard right, slow
            [0.0, 0.0, 0.6],    # brake
        ],

        'small_precision': [
            [0.0, 0.8, 0.0],    # straight, fast
            [-0.3, 0.8, 0.0],   # slight left
            [0.3, 0.8, 0.0],    # slight right
            [-0.6, 0.5, 0.0],   # medium left
            [0.6, 0.5, 0.0],    # medium right
            [0.0, 0.0, 0.4],    # light brake
            [0.0, 0.0, 0.8],    # hard brake
        ],

        # Medium sets (9-10 actions)
        'medium_balanced': [
            [0.0, 1.0, 0.0],    # straight, full gas
            [0.0, 0.5, 0.0],    # straight, cruise
            [-0.3, 0.8, 0.0],   # slight left, fast
            [0.3, 0.8, 0.0],    # slight right, fast
            [-0.6, 0.6, 0.0],   # medium left
            [0.6, 0.6, 0.0],    # medium right
            [-0.9, 0.3, 0.0],   # hard left, slow
            [0.9, 0.3, 0.0],    # hard right, slow
            [0.0, 0.0, 0.5],    # brake
            [0.0, 0.0, 0.9],    # hard brake
        ],

        'medium_original': [  # Similar to improved_actions.txt
            [-1.0, 0.4, 0.0],
            [-0.6, 0.6, 0.0],
            [-0.3, 0.8, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.3, 0.8, 0.0],
            [0.6, 0.6, 0.0],
            [1.0, 0.4, 0.0],
            [0.0, 0.0, 0.8],
        ],

        # Large sets (12-15 actions) - fine-grained control
        'large_precision': [
            [0.0, 1.0, 0.0],    # straight, full gas
            [0.0, 0.7, 0.0],    # straight, fast
            [0.0, 0.4, 0.0],    # straight, cruise
            [-0.2, 0.9, 0.0],   # tiny left, very fast
            [0.2, 0.9, 0.0],    # tiny right, very fast
            [-0.5, 0.7, 0.0],   # medium left, fast
            [0.5, 0.7, 0.0],    # medium right, fast
            [-0.8, 0.5, 0.0],   # hard left, medium
            [0.8, 0.5, 0.0],    # hard right, medium
            [-1.0, 0.3, 0.0],   # max left, slow
            [1.0, 0.3, 0.0],    # max right, slow
            [0.0, 0.0, 0.6],    # brake
        ],

        'large_extended': [
            [-1.0, 0.4, 0.0],
            [-0.6, 0.6, 0.0],
            [-0.3, 0.8, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.3, 0.8, 0.0],
            [0.6, 0.6, 0.0],
            [1.0, 0.4, 0.0],
            [0.0, 0.0, 0.8],
            [-0.6, 0.0, 0.5],   # left + brake
            [0.6, 0.0, 0.5],    # right + brake
            [0.0, 0.0, 0.3],    # light brake
            [-0.4, 0.5, 0.0],   # medium left
            [0.4, 0.5, 0.0],    # medium right
            [0.0, 0.7, 0.0],    # straight, medium-fast
        ],

        'xlarge_full': [
            # Straight variations
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.3, 0.0],
            # Left turns (4 levels)
            [-0.3, 0.8, 0.0],
            [-0.5, 0.6, 0.0],
            [-0.8, 0.4, 0.0],
            [-1.0, 0.3, 0.0],
            # Right turns (4 levels)
            [0.3, 0.8, 0.0],
            [0.5, 0.6, 0.0],
            [0.8, 0.4, 0.0],
            [1.0, 0.3, 0.0],
            # Braking
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.8],
            # Turn + brake
            [-0.5, 0.0, 0.6],
            [0.5, 0.0, 0.6],
        ],
    }

    if config_name not in action_sets:
        raise ValueError(f"Unknown action config: {config_name}")

    return action_sets[config_name]


def get_cnn_config(config_name):
    # Map config name to full CNN architecture specs
    # Returns: (channels, kernels, strides, final_spatial_size)
    # Input image: 96x96

    config_map = {
        # 3 layers: 96 -> 23 -> 10 -> 8
        'small_3layer': ([16, 32, 64], [8, 4, 3], [4, 2, 1], 8),
        'medium_3layer': ([32, 64, 128], [8, 4, 3], [4, 2, 1], 8),
        'large_3layer': ([64, 128, 256], [8, 4, 3], [4, 2, 1], 8),

        # 4 layers: 96 -> 23 -> 10 -> 8 -> 6
        'small_4layer': ([16, 32, 64, 64], [8, 4, 3, 3], [4, 2, 1, 1], 6),
        'medium_4layer': ([32, 64, 128, 128], [8, 4, 3, 3], [4, 2, 1, 1], 6),
        'large_4layer': ([64, 128, 256, 256], [8, 4, 3, 3], [4, 2, 1, 1], 6),
        'xlarge_4layer': ([128, 256, 512, 512], [8, 4, 3, 3], [4, 2, 1, 1], 6),

        # 5 layers: 96 -> 23 -> 10 -> 8 -> 6 -> 4
        'medium_5layer': ([32, 64, 128, 256, 256], [8, 4, 3, 3, 3], [4, 2, 1, 1, 1], 4),
        'large_5layer': ([64, 128, 256, 512, 512], [8, 4, 3, 3, 3], [4, 2, 1, 1, 1], 4),
    }

    if config_name not in config_map:
        raise ValueError(f"Unknown CNN config: {config_name}")

    return config_map[config_name]


def get_hidden_sizes(config_name):
    # Map config name to list of hidden layer sizes
    config_map = {
        'small_2layer': [256, 128],
        'medium_2layer': [512, 256],
        'large_2layer': [1024, 512],
        'xlarge_2layer': [2048, 1024],
        'medium_3layer': [512, 256, 128],
        'large_3layer': [1024, 512, 256],
        'large_4layer': [1024, 512, 256, 128]
    }

    if config_name not in config_map:
        raise ValueError(f"Unknown hidden layer config: {config_name}")

    return config_map[config_name]


def load_actions(action_filename):
    actions = []
    with open(action_filename) as f:
        lines = f.readlines()
        for line in lines:
            action = []
            for tok in line.split():
                action.append(float(tok))
            actions.append(action)
    return actions


def train():
    # Train with hyperparameters from wandb sweep
    # Initialize wandb
    wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    print("python version:\t{0}".format(platform.python_version()))
    print("gym version:\t{0}".format(gym.__version__))

    # Fixed parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_filename', type=str, default='improved_actions.txt',
                       help='a list of actions')
    parser.add_argument('--use_doubleqlearning', default=True, action="store_true",
                       help='a flag that indicates the use of double q learning')
    parser.add_argument('--no_display', default=True, action="store_true",
                       help='a flag indicating whether training runs on the cluster')
    parser.add_argument('--agent_name', type=str, default=None,
                       help='an agent name')
    parser.add_argument('--outdir', type=str, default='sweep_models',
                       help='a directory for output')
    parser.add_argument('--validation_freq', type=int, default=10000,
                       help='how often to run validation (in timesteps)')
    parser.add_argument('--num_validation_seeds', type=int, default=20,
                       help='number of random seeds to use for validation')
    parser.add_argument('--early_stopping_patience', type=int, default=4,
                       help='stop training after N validations without improvement')

    # Use parse_known_args to ignore sweep parameters passed by wandb
    args, unknown = parser.parse_known_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Use wandb run name as agent identifier
    agent_name = args.agent_name if args.agent_name else wandb.run.name

    # Check if using continuous actions (for NAF)
    use_continuous_actions = getattr(config, 'use_continuous_actions', False)

    # Get action set from config (only for discrete actions)
    if use_continuous_actions:
        actions = None  # NAF doesn't use discrete action set
    else:
        actions = get_action_set(config.action_config)

    # Get architecture configs
    cnn_channels, cnn_kernels, cnn_strides, final_spatial_size = get_cnn_config(config.cnn_config)
    hidden_sizes = get_hidden_sizes(config.hidden_layer_config)

    # Log action space info to wandb
    if use_continuous_actions:
        wandb.config.update({
            'action_type': 'continuous',
            'action_dim': 3  # steering, gas, brake
        })
    else:
        wandb.config.update({
            'action_type': 'discrete',
            'num_actions': len(actions),
            'action_space_size': len(actions)
        })

    # Start training
    print("\nStart training with sweep configuration...")
    print(f"use_continuous_actions: {use_continuous_actions}")
    print(f"total_timesteps: {config.total_timesteps}")
    print(f"action_repeat: {config.action_repeat}")
    print(f"lr: {config.lr}")
    print(f"batch_size: {config.batch_size}")
    print(f"gamma: {config.gamma}")
    print(f"exploration_fraction: {config.exploration_fraction}")
    print(f"exploration_final_eps: {config.exploration_final_eps}")
    print(f"target_network_update_freq: {config.target_network_update_freq}")
    print(f"buffer_size: {config.buffer_size}")
    print(f"learning_starts: {config.learning_starts}")
    print(f"optimizer_type: {config.optimizer_type}")
    print(f"activation: {config.activation}")
    if use_continuous_actions:
        print(f"Action type: CONTINUOUS (NAF) - 3D action space")
    else:
        print(f"action_config: {config.action_config} -> {len(actions)} actions")
    print(f"cnn_config: {config.cnn_config} -> {len(cnn_channels)} layers: {cnn_channels}")
    print(f"  CNN output: {final_spatial_size}x{final_spatial_size}")
    print(f"hidden_layer_config: {config.hidden_layer_config} -> {hidden_sizes}")
    print(f"dropout_rate: {config.dropout_rate}")
    print(f"normalization: {config.normalization}")
    print(f"use_dueling: {config.use_dueling}")
    print(f"lr_scheduler: {config.lr_scheduler}")
    print(f"weight_decay: {config.weight_decay}")

    render_mode = 'rgb_array' if args.no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode),
                     remove_score=True, return_linear_velocity=False)

    deepq.learn(
        env,
        lr=config.lr,
        total_timesteps=config.total_timesteps,
        action_repeat=config.action_repeat,
        gamma=config.gamma,
        batch_size=config.batch_size,
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        target_network_update_freq=config.target_network_update_freq,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        model_identifier=agent_name,
        outdir=args.outdir,
        new_actions=actions,
        use_doubleqlearning=args.use_doubleqlearning,
        validation_freq=args.validation_freq,
        num_validation_seeds=args.num_validation_seeds,
        early_stopping_patience=args.early_stopping_patience,
        no_display=args.no_display,
        use_wandb=True,
        optimizer_type=config.optimizer_type,
        hidden_sizes=hidden_sizes,
        dropout_rate=config.dropout_rate,
        cnn_channels=cnn_channels,
        cnn_kernels=cnn_kernels,
        cnn_strides=cnn_strides,
        final_spatial_size=final_spatial_size,
        activation=config.activation,
        use_continuous_actions=use_continuous_actions,
        normalization=config.normalization,
        lr_scheduler=config.lr_scheduler,
        weight_decay=config.weight_decay,
        use_dueling=config.use_dueling
    )

    # Close environment
    env.close()

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    train()
