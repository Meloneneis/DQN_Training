import gymnasium as gym
import platform
import argparse
import torch
import numpy as np
import yaml

from action import ActionSet, get_action, get_continuous_action
from sdc_wrapper import SDC_Wrapper
from utils import get_state
from model import DQN, ContinuousActionDQN

def evaluate(env, new_actions=None, load_path='agent.pth', use_continuous=False,
             hidden_sizes=None, cnn_channels=None, cnn_kernels=None,
             cnn_strides=None, final_spatial_size=6, activation='relu',
             normalization='layer', use_dueling=False):
    """ Evaluate a trained model and compute your leaderboard scores

	NO CHANGES SHOULD BE MADE TO THIS FUNCTION

    Parameters
    -------
    env: gym.Env
        environment to evaluate on
    load_path: str
        path to load the model (.pth) from
    hidden_sizes: list
        hidden layer sizes for the model
    cnn_channels: list
        CNN channel sizes
    cnn_kernels: list
        CNN kernel sizes
    cnn_strides: list
        CNN strides
    final_spatial_size: int
        final spatial size after CNN
    activation: str
        activation function
    normalization: str
        normalization type
    use_dueling: bool
        whether to use dueling architecture
    """
    episode_rewards = []
    action_manager = ActionSet()

    if new_actions is not None:
        action_manager.set_actions(new_actions)

    actions = action_manager.get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # These are not the final evaluation seeds, do not overfit on these tracks!
    #seeds = [10000019, 20000003, 30000001, 40000003, 50000017,
    #           60000011, 70000027, 80000023, 90000049, 10000079]
    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
             49636746, 66759182, 91294619, 84274995, 31531469,
             22597174, 68545857, 75568192, 91140053, 86018367,
             49636746, 66759182, 91294619, 84274995, 31531469,
             10000019, 20000003, 30000001, 40000003, 50000017,
             60000011, 70000027, 80000023, 90000049, 10000079,
             12345678, 87654321, 23456789, 98765432, 34567890,
             11111111, 99999999, 55555555, 77777777, 13579246,
             64827193, 19283746, 73829164, 48291637, 92837465,
             38291647, 73829156, 18273645, 92837461, 47382916]

    # Set default architecture if not provided
    if hidden_sizes is None:
        hidden_sizes = [1024, 512]
    if cnn_channels is None:
        cnn_channels = [32, 64, 128, 128]
    if cnn_kernels is None:
        cnn_kernels = [8, 4, 3, 3]
    if cnn_strides is None:
        cnn_strides = [4, 2, 1, 1]

    # Build & load network (dropout_rate=0.0 for inference)
    # Try different normalization/activation combinations if loading fails
    checkpoint = torch.load(load_path, map_location=device)

    combinations_to_try = [
        (activation, normalization),
        (activation, 'layer' if normalization == 'none' else 'none'),
        ('relu', normalization),
        ('silu', normalization),
    ]

    loaded = False
    for act, norm in combinations_to_try:
        try:
            if use_continuous:
                policy_net = ContinuousActionDQN(3, device, hidden_sizes=hidden_sizes,
                                                dropout_rate=0.0,
                                                cnn_channels=cnn_channels, cnn_kernels=cnn_kernels,
                                                cnn_strides=cnn_strides, final_spatial_size=final_spatial_size,
                                                activation=act, normalization=norm).to(device)
            else:
                policy_net = DQN(action_size, device, hidden_sizes=hidden_sizes, dropout_rate=0.0,
                                 cnn_channels=cnn_channels, cnn_kernels=cnn_kernels,
                                 cnn_strides=cnn_strides, final_spatial_size=final_spatial_size,
                                 activation=act, normalization=norm,
                                 use_dueling=use_dueling).to(device)

            policy_net.load_state_dict(checkpoint)
            print(f"Successfully loaded model with activation='{act}', normalization='{norm}'")
            loaded = True
            break
        except RuntimeError as e:
            if act == combinations_to_try[-1][0] and norm == combinations_to_try[-1][1]:
                # Last attempt failed, raise the error
                raise e
            continue

    if not loaded:
        raise RuntimeError("Failed to load model with any activation/normalization combination")

    policy_net.eval()

    # Iterate over a number of evaluation episodes
    for episode, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        obs = get_state(obs)
        t = 0

        # Run each episode until episode has terminated or 600 time steps have been reached
        episode_rewards.append(0.0)
        while True:
            if use_continuous:
                action = get_continuous_action(obs, policy_net, exploration_noise=0.0, t=t)
            else:
                action, _ = get_action(obs, policy_net, action_size, actions, is_greedy=True, t=t)

            obs, rew, term, trunc, _ = env.step(action)
            done = term or trunc
            obs = get_state(obs)

            episode_rewards[-1] += rew
            t += 1

            if done or t >= 600:
                break
        print('episode %d \t reward %f' % (episode, episode_rewards[-1]))
    print('---------------------------')
    print(' mean score: %f' % np.mean(np.array(episode_rewards)))
    print(' median score: %f' % np.median(np.array(episode_rewards)))
    print(' std score: %f' % np.std(np.array(episode_rewards)))
    print('---------------------------')

def get_action_set(config_name):
    # Map config name to predefined action sets (same as in train_racing_sweep.py)
    action_sets = {
        'minimal_conservative': [
            [0.0, 0.5, 0.0], [-0.5, 0.3, 0.0], [0.5, 0.3, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 0.5],
        ],
        'minimal_aggressive': [
            [0.0, 1.0, 0.0], [-0.8, 0.8, 0.0], [0.8, 0.8, 0.0],
            [-0.4, 0.6, 0.0], [0.4, 0.6, 0.0],
        ],
        'small_balanced': [
            [0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [-0.4, 0.7, 0.0],
            [0.4, 0.7, 0.0], [-0.7, 0.4, 0.0], [0.7, 0.4, 0.0],
            [0.0, 0.0, 0.6],
        ],
        'small_precision': [
            [0.0, 0.8, 0.0], [-0.3, 0.8, 0.0], [0.3, 0.8, 0.0],
            [-0.6, 0.5, 0.0], [0.6, 0.5, 0.0], [0.0, 0.0, 0.4],
            [0.0, 0.0, 0.8],
        ],
        'medium_balanced': [
            [0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [-0.3, 0.8, 0.0],
            [0.3, 0.8, 0.0], [-0.6, 0.6, 0.0], [0.6, 0.6, 0.0],
            [-0.9, 0.3, 0.0], [0.9, 0.3, 0.0], [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.9],
        ],
        'medium_original': [
            [-1.0, 0.4, 0.0], [-0.6, 0.6, 0.0], [-0.3, 0.8, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.4, 0.0], [0.3, 0.8, 0.0],
            [0.6, 0.6, 0.0], [1.0, 0.4, 0.0], [0.0, 0.0, 0.8],
        ],
        'large_precision': [
            [0.0, 1.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.4, 0.0],
            [-0.2, 0.9, 0.0], [0.2, 0.9, 0.0], [-0.5, 0.7, 0.0],
            [0.5, 0.7, 0.0], [-0.8, 0.5, 0.0], [0.8, 0.5, 0.0],
            [-1.0, 0.3, 0.0], [1.0, 0.3, 0.0], [0.0, 0.0, 0.6],
        ],
        'large_extended': [
            [-1.0, 0.4, 0.0], [-0.6, 0.6, 0.0], [-0.3, 0.8, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.4, 0.0], [0.3, 0.8, 0.0],
            [0.6, 0.6, 0.0], [1.0, 0.4, 0.0], [0.0, 0.0, 0.8],
            [-0.6, 0.0, 0.5], [0.6, 0.0, 0.5], [0.0, 0.0, 0.3],
            [-0.4, 0.5, 0.0], [0.4, 0.5, 0.0], [0.0, 0.7, 0.0],
        ],
        'xlarge_full': [
            [0.0, 1.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.5, 0.0],
            [0.0, 0.3, 0.0], [-0.3, 0.8, 0.0], [-0.5, 0.6, 0.0],
            [-0.8, 0.4, 0.0], [-1.0, 0.3, 0.0], [0.3, 0.8, 0.0],
            [0.5, 0.6, 0.0], [0.8, 0.4, 0.0], [1.0, 0.3, 0.0],
            [0.0, 0.0, 0.5], [0.0, 0.0, 0.8], [-0.5, 0.0, 0.6],
            [0.5, 0.0, 0.6],
        ],
        'xlarge_hybrid_drift': [
            [0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.6],
            [-0.2, 1.0, 0.0], [0.2, 1.0, 0.0], [-0.6, 0.7, 0.0],
            [0.6, 0.7, 0.0], [-1.0, 0.3, 0.0], [1.0, 0.3, 0.0],
            [-1.0, 0.0, 0.5], [1.0, 0.0, 0.5], [-0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5],
        ],
        'xlarge_aggressive_drifter': [
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-0.2, 1.0, 0.0],
            [0.2, 1.0, 0.0], [-0.5, 0.8, 0.0], [0.5, 0.8, 0.0],
            [-1.0, 0.9, 0.0], [1.0, 0.9, 0.0], [-1.0, 0.0, 0.8],
            [1.0, 0.0, 0.8], [-0.5, 0.0, 0.5], [0.5, 0.0, 0.5],
        ],
    }
    if config_name not in action_sets:
        raise ValueError(f"Unknown action config: {config_name}")
    return action_sets[config_name]


def get_cnn_config(config_name):
    # Map config name to full CNN architecture specs
    config_map = {
        'small_3layer': ([16, 32, 64], [8, 4, 3], [4, 2, 1], 8),
        'medium_3layer': ([32, 64, 128], [8, 4, 3], [4, 2, 1], 8),
        'large_3layer': ([64, 128, 256], [8, 4, 3], [4, 2, 1], 8),
        'small_4layer': ([16, 32, 64, 64], [8, 4, 3, 3], [4, 2, 1, 1], 6),
        'medium_4layer': ([32, 64, 128, 128], [8, 4, 3, 3], [4, 2, 1, 1], 6),
        'large_4layer': ([64, 128, 256, 256], [8, 4, 3, 3], [4, 2, 1, 1], 6),
        'xlarge_4layer': ([128, 256, 512, 512], [8, 4, 3, 3], [4, 2, 1, 1], 6),
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


def load_config_from_yaml(config_path):
    """Load model configuration from wandb config.yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def infer_architecture_from_checkpoint(checkpoint_path):
    """Automatically infer model architecture from checkpoint file

    Returns
    -------
    dict
        Dictionary containing architecture parameters:
        - cnn_channels: list of CNN channel sizes
        - cnn_kernels: list of kernel sizes
        - cnn_strides: list of strides
        - final_spatial_size: final spatial dimension
        - hidden_sizes: list of FC layer sizes
        - action_size: number of actions
        - use_dueling: whether model uses dueling architecture
        - activation: activation function (default 'silu')
        - normalization: normalization type ('layer' or 'none')
    """
    device = torch.device("cpu")  # Load to CPU for inspection
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer CNN architecture
    cnn_channels = []
    num_cnn_layers = 0
    for key in checkpoint.keys():
        if key.startswith('conv_layers.') and key.endswith('.weight'):
            layer_num = int(key.split('.')[1])
            num_cnn_layers = max(num_cnn_layers, layer_num + 1)

    for i in range(num_cnn_layers):
        weight_shape = checkpoint[f'conv_layers.{i}.weight'].shape
        cnn_channels.append(weight_shape[0])  # output channels

    # Standard kernels and strides based on number of layers
    if num_cnn_layers == 3:
        cnn_kernels = [8, 4, 3]
        cnn_strides = [4, 2, 1]
        final_spatial_size = 8
    elif num_cnn_layers == 4:
        cnn_kernels = [8, 4, 3, 3]
        cnn_strides = [4, 2, 1, 1]
        final_spatial_size = 6
    elif num_cnn_layers == 5:
        cnn_kernels = [8, 4, 3, 3, 3]
        cnn_strides = [4, 2, 1, 1, 1]
        final_spatial_size = 4
    else:
        raise ValueError(f"Unsupported number of CNN layers: {num_cnn_layers}")

    # Check if using dueling architecture
    use_dueling = 'value_output.weight' in checkpoint

    # Infer action size
    if use_dueling:
        action_size = checkpoint['advantage_output.weight'].shape[0]
    else:
        action_size = checkpoint['fc_output.weight'].shape[0]

    # Infer hidden layer sizes
    hidden_sizes = []
    if use_dueling:
        # Shared layer
        if 'fc_shared.0.weight' in checkpoint:
            hidden_sizes.append(checkpoint['fc_shared.0.weight'].shape[0])

        # Value/Advantage streams (they should be the same size)
        stream_idx = 0
        while f'value_stream.{stream_idx}.weight' in checkpoint:
            hidden_sizes.append(checkpoint[f'value_stream.{stream_idx}.weight'].shape[0])
            stream_idx += 1
    else:
        # Standard DQN
        fc_idx = 0
        while f'fc_layers.{fc_idx}.weight' in checkpoint:
            hidden_sizes.append(checkpoint[f'fc_layers.{fc_idx}.weight'].shape[0])
            fc_idx += 1

    # Try to infer normalization by checking if it's layer norm or none
    # This is tricky - we'll try both and see which one works
    normalization = 'none'  # Default guess

    print(f"\n=== Auto-Inferred Architecture ===")
    print(f"CNN Layers: {num_cnn_layers}")
    print(f"CNN Channels: {cnn_channels}")
    print(f"CNN Kernels: {cnn_kernels}")
    print(f"CNN Strides: {cnn_strides}")
    print(f"Final Spatial Size: {final_spatial_size}")
    print(f"Hidden Sizes: {hidden_sizes}")
    print(f"Action Size: {action_size}")
    print(f"Use Dueling: {use_dueling}")
    print(f"Normalization: trying 'none' first, will try 'layer' if that fails")
    print(f"Activation: trying 'silu' first, will try others if that fails")
    print(f"===================================\n")

    return {
        'cnn_channels': cnn_channels,
        'cnn_kernels': cnn_kernels,
        'cnn_strides': cnn_strides,
        'final_spatial_size': final_spatial_size,
        'hidden_sizes': hidden_sizes,
        'action_size': action_size,
        'use_dueling': use_dueling,
        'activation': 'silu',  # Default, can't infer from checkpoint
        'normalization': normalization  # Default guess
    }


def main():
    """
    Evaluate a trained Deep Q-Learning agent
    """

    print("python version:\t{0}".format(platform.python_version()))
    print("gym version:\t{0}".format(gym.__version__))

    # it doesn't help that much.
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent_name', type=str, default="lunar-sweep-7_best(2)",
                       help='model name without .pth extension (default: lunar-sweep-7_best)')
    parser.add_argument('--config', type=str, default='lunar_config.yaml',
                       help='Path to config.yaml file (default: lunar_config.yaml). Use "auto" to auto-infer from checkpoint.')
    parser.add_argument('--no_display', default=False, action="store_true",
                       help='a flag indicating whether training/evaluation runs on the cluster')
    parser.add_argument('--use_continuous', default=False, action="store_true",
                       help='Use continuous actions (NAF) instead of discrete')

    args = parser.parse_args()

    filename = args.agent_name + '.pth'

    # Determine whether to use config file or auto-inference
    if args.config == 'auto':
        # Auto-infer architecture from checkpoint
        print(f"Auto-inferring architecture from checkpoint: {filename}")
        arch = infer_architecture_from_checkpoint(filename)

        # Build action set based on inferred action size
        action_size_to_config = {
            5: 'minimal_conservative',
            7: 'small_balanced',
            9: 'medium_original',
            10: 'medium_balanced',
            12: 'xlarge_aggressive_drifter',
            13: 'xlarge_hybrid_drift',
            15: 'large_extended',
            16: 'xlarge_full',
        }

        action_config = action_size_to_config.get(arch['action_size'])
        if action_config is None:
            raise ValueError(f"Cannot infer action config for {arch['action_size']} actions.")

        actions = get_action_set(action_config)
        print(f"Inferred action config: {action_config} -> {len(actions)} actions")

        cnn_channels = arch['cnn_channels']
        cnn_kernels = arch['cnn_kernels']
        cnn_strides = arch['cnn_strides']
        final_spatial_size = arch['final_spatial_size']
        hidden_sizes = arch['hidden_sizes']
        use_dueling = arch['use_dueling']
        activation = arch['activation']
        normalization = arch['normalization']
    else:
        # Load from config file
        print(f"\nLoading configuration from: {args.config}")
        config_data = load_config_from_yaml(args.config)

        # Extract action configuration
        action_config = config_data.get('action_config', {}).get('value', 'xlarge_aggressive_drifter')
        actions = get_action_set(action_config)
        print(f"Action config: {action_config} -> {len(actions)} actions")

        # Extract CNN configuration
        cnn_config = config_data.get('cnn_config', {}).get('value', 'large_4layer')
        cnn_channels, cnn_kernels, cnn_strides, final_spatial_size = get_cnn_config(cnn_config)
        print(f"CNN config: {cnn_config} -> channels: {cnn_channels}")

        # Extract hidden layer configuration
        hidden_config = config_data.get('hidden_layer_config', {}).get('value', 'medium_2layer')
        hidden_sizes = get_hidden_sizes(hidden_config)
        print(f"Hidden layer config: {hidden_config} -> sizes: {hidden_sizes}")

        # Extract other hyperparameters
        activation = config_data.get('activation', {}).get('value', 'silu')
        normalization = config_data.get('normalization', {}).get('value', 'none')
        use_dueling = config_data.get('use_dueling', {}).get('value', True)

        print(f"Activation: {activation}")
        print(f"Normalization: {normalization}")
        print(f"Use dueling: {use_dueling}")
        print()

    print(f"Loading model from: {filename}\n")

    render_mode = 'rgb_array' if args.no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode),
                     remove_score=True, return_linear_velocity=False)

    evaluate(env, new_actions=actions, load_path=filename,
            use_continuous=args.use_continuous,
            hidden_sizes=hidden_sizes,
            cnn_channels=cnn_channels, cnn_kernels=cnn_kernels,
            cnn_strides=cnn_strides, final_spatial_size=final_spatial_size,
            activation=activation, normalization=normalization,
            use_dueling=use_dueling)

    env.close()


if __name__ == '__main__':
    main()
