import numpy as np
import torch
import torch.optim as optim
from action import ActionSet, get_action, select_exploratory_action, select_greedy_action, get_continuous_action
from learning import perform_qlearning_step, update_target_net, perform_qlearning_step_continuous
from model import DQN, ContinuousActionDQN
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from utils import get_state, visualize_training
import os
import matplotlib
import time
import gymnasium as gym
from sdc_wrapper import SDC_Wrapper
import wandb

def validate_agent(policy_net, actions, action_size, validation_seeds, no_display=True, use_continuous_actions=False):
    """Validate the agent on a fixed set of validation seeds

    Parameters
    ----------
    policy_net: torch.nn.Module
        policy Q-network to evaluate
    actions: list
        list of available actions
    action_size: int
        number of actions
    validation_seeds: list
        list of seeds to use for validation
    no_display: bool
        whether to run without display
    use_continuous_actions: bool
        whether using continuous actions

    Returns
    -------
    float
        average reward across validation episodes
    """
    render_mode = 'rgb_array' if no_display else 'human'
    val_env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode),
                          remove_score=True, return_linear_velocity=False)

    policy_net.eval()
    episode_rewards = []

    for seed in validation_seeds:
        obs, _ = val_env.reset(seed=seed)
        obs = get_state(obs)
        episode_reward = 0.0
        t = 0

        while True:
            with torch.no_grad():
                if use_continuous_actions:
                    action = get_continuous_action(obs, policy_net, exploration_noise=0.0, t=t)
                else:
                    action, _ = get_action(obs, policy_net, action_size, actions, is_greedy=True, t=t)

            obs, rew, term, trunc, _ = val_env.step(action)
            done = term or trunc
            obs = get_state(obs)
            episode_reward += rew
            t += 1

            if done or t >= 600:
                break

        episode_rewards.append(episode_reward)

    val_env.close()
    policy_net.train()

    avg_reward = np.mean(episode_rewards)
    return avg_reward

def learn(env,
          lr=1e-4,
          total_timesteps = 300000,
          buffer_size = 300000,
          exploration_fraction=0.20,
          exploration_final_eps=0.02,
          train_freq=1,
          action_repeat=4,
          batch_size=256,
          learning_starts=100,
          gamma=0.99,
          target_network_update_freq=1500,
          validation_freq=5000,
          num_validation_seeds=10,
          early_stopping_patience=5,
          new_actions = None,
          model_identifier='agent',
          outdir = "",
          use_doubleqlearning = True,
          no_display = True,
          use_wandb = False,
          optimizer_type='adam',
          hidden_sizes=[1024, 512],
          dropout_rate=0.2,
          cnn_channels=[32, 64, 128, 128],
          cnn_kernels=[8, 4, 3, 3],
          cnn_strides=[4, 2, 1, 1],
          final_spatial_size=6,
          activation='relu',
          use_continuous_actions=False,
          normalization='layer',
          lr_scheduler='none',
          weight_decay=0.0,
          use_dueling=False,
          warmup_teacher_path=None,
          warmup_teacher_arch=None,
          warmup_steps=0,
          validation_seeds_list=None):
    """ Train a deep q-learning model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to take
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    action_repeat: int
        selection action on every n-th frame and repeat action for intermediate frames
    batch_size: int
        size of a batched sampled from replay buffer for training
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    model_identifier: string
        identifier of the agent
    """
    print(f"""
        learn(
            env,
            lr={lr},
            total_timesteps={total_timesteps},
            buffer_size={buffer_size},
            exploration_fraction={exploration_fraction},
            exploration_final_eps={exploration_final_eps},
            train_freq={train_freq},
            action_repeat={action_repeat},
            batch_size={batch_size},
            learning_starts={learning_starts},
            gamma={gamma},
            target_network_update_freq={target_network_update_freq},
            validation_freq={validation_freq},
            num_validation_seeds={num_validation_seeds},
            early_stopping_patience={early_stopping_patience},
            new_actions={new_actions},
            model_identifier={model_identifier!r},
            outdir={outdir!r},
            use_doubleqlearning={use_doubleqlearning},
            no_display={no_display},
        )
        """)

    if use_wandb:
        wandb.config.update({
            'lr': lr,
            'total_timesteps': total_timesteps,
            'buffer_size': buffer_size,
            'exploration_fraction': exploration_fraction,
            'exploration_final_eps': exploration_final_eps,
            'train_freq': train_freq,
            'action_repeat': action_repeat,
            'batch_size': batch_size,
            'learning_starts': learning_starts,
            'gamma': gamma,
            'target_network_update_freq': target_network_update_freq,
            'validation_freq': validation_freq,
            'num_validation_seeds': num_validation_seeds,
            'early_stopping_patience': early_stopping_patience,
            'use_doubleqlearning': use_doubleqlearning,
            'optimizer_type': optimizer_type,
            'hidden_sizes': hidden_sizes,
            'dropout_rate': dropout_rate,
            'cnn_channels': cnn_channels,
            'cnn_num_layers': len(cnn_channels),
            'activation': activation,
            'normalization': normalization,
            'lr_scheduler': lr_scheduler,
            'weight_decay': weight_decay,
            'use_dueling': use_dueling
        })

    # set float as default
    torch.set_default_dtype (torch.float32)

    # Enable performance optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        print("\nUsing CUDA.")
        print (torch.version.cuda,"\n")
    else:
        print ("\nNot using CUDA.\n")

    episode_rewards = [0.0]
    training_losses = []
    action_manager = ActionSet()

    if new_actions is not None:
        print ( "Set new actions")
        action_manager.set_actions(new_actions)

    actions = action_manager.get_action_set()

    action_size = len(actions)
    print ( action_size )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders) - GPU acceleration enabled!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Build networks
    if use_continuous_actions:
        action_dim = 3  # steering, gas, brake
        policy_net = ContinuousActionDQN(action_dim, device, hidden_sizes=hidden_sizes,
                                        dropout_rate=dropout_rate,
                                        cnn_channels=cnn_channels, cnn_kernels=cnn_kernels,
                                        cnn_strides=cnn_strides, final_spatial_size=final_spatial_size,
                                        activation=activation, normalization=normalization,
                                        ).to(device)
        target_net = ContinuousActionDQN(action_dim, device, hidden_sizes=hidden_sizes,
                                        dropout_rate=dropout_rate,
                                        cnn_channels=cnn_channels, cnn_kernels=cnn_kernels,
                                        cnn_strides=cnn_strides, final_spatial_size=final_spatial_size,
                                        activation=activation, normalization=normalization,
                                        ).to(device)
        print("\nUsing Continuous Action DQN (NAF)\n")
    else:
        policy_net = DQN(action_size, device, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate,
                         cnn_channels=cnn_channels, cnn_kernels=cnn_kernels,
                         cnn_strides=cnn_strides, final_spatial_size=final_spatial_size,
                         activation=activation, normalization=normalization,
                         use_dueling=use_dueling).to(device)
        target_net = DQN(action_size, device, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate,
                         cnn_channels=cnn_channels, cnn_kernels=cnn_kernels,
                         cnn_strides=cnn_strides, final_spatial_size=final_spatial_size,
                         activation=activation, normalization=normalization,
                         use_dueling=use_dueling).to(device)
        print(f"\nUsing Discrete Action DQN (Dueling: {use_dueling})\n")

    # Warm-up with teacher network
    teacher_net = None
    using_warmup = warmup_teacher_path is not None and warmup_steps > 0

    if using_warmup:
        print(f"\n{'='*60}")
        print(f"WARM-UP MODE ENABLED")
        print(f"{'='*60}")
        print(f"Teacher model: {warmup_teacher_path}")
        print(f"Warm-up steps: {warmup_steps}")
        print(f"During warm-up, the teacher network will be used as the target")
        print(f"After {warmup_steps} steps, normal target network updates resume")
        print(f"{'='*60}\n")

        # Load teacher network with teacher architecture
        if warmup_teacher_arch is None:
            raise ValueError("warmup_teacher_arch must be provided when using warmup")

        teacher_net = DQN(action_size, device,
                         hidden_sizes=warmup_teacher_arch['hidden_sizes'],
                         dropout_rate=0.0,
                         cnn_channels=warmup_teacher_arch['cnn_channels'],
                         cnn_kernels=warmup_teacher_arch['cnn_kernels'],
                         cnn_strides=warmup_teacher_arch['cnn_strides'],
                         final_spatial_size=warmup_teacher_arch['final_spatial_size'],
                         activation=warmup_teacher_arch['activation'],
                         normalization=warmup_teacher_arch['normalization'],
                         use_dueling=warmup_teacher_arch['use_dueling']).to(device)

        # Load teacher weights
        teacher_checkpoint = torch.load(warmup_teacher_path, map_location=device)
        teacher_net.load_state_dict(teacher_checkpoint)
        teacher_net.eval()

        print(f"✓ Teacher network loaded successfully")
        print(f"  Teacher architecture:")
        print(f"    CNN: {warmup_teacher_arch['cnn_channels']}")
        print(f"    Hidden: {warmup_teacher_arch['hidden_sizes']}")
        print(f"    Activation: {warmup_teacher_arch['activation']}")
        print(f"    Normalization: {warmup_teacher_arch['normalization']}")
        print(f"    Dueling: {warmup_teacher_arch['use_dueling']}\n")

    # Always initialize target_net from policy_net (student architecture)
    # During warmup, we'll use teacher_net for Q-targets instead of target_net
    target_net.load_state_dict(policy_net.state_dict())

    target_net.eval()

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Create optimizer based on type
    optimizer_type_lower = optimizer_type.lower()
    if optimizer_type_lower == 'adam':
        optimizer = optim.Adam(policy_net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type_lower == 'adamw':
        optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose from 'adam', 'adamw'")

    # Create learning rate scheduler
    scheduler = None
    if lr_scheduler.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_timesteps)
    elif lr_scheduler.lower() == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999994)
    elif lr_scheduler.lower() != 'none':
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}. Choose from 'none', 'cosine', 'exponential'")

    print(f"\nUsing optimizer: {optimizer_type}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"LR Scheduler: {lr_scheduler}")
    print(f"Activation function: {activation}")
    print(f"CNN architecture ({len(cnn_channels)} layers): {cnn_channels}")
    print(f"CNN output size: {final_spatial_size}x{final_spatial_size} -> {cnn_channels[-1] * final_spatial_size * final_spatial_size} features")
    print(f"FC layer architecture: {hidden_sizes}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Normalization type: {normalization}")
    print()

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize environment and get first state
    obs, _ = env.reset()
    obs = get_state(obs)
    start = time.time()

    # Use provided validation seeds or generate random ones
    if validation_seeds_list is not None:
        validation_seeds = validation_seeds_list
        print(f"\nUsing provided validation seeds ({len(validation_seeds)} seeds)")
        print(f"Validation seeds: {validation_seeds[:10]}... (showing first 10)\n")
    else:
        # Generate random validation seeds (different from test seeds)
        # Test seeds: [10000019, 20000003, 30000001, 40000003, 50000017,
        #              60000011, 70000027, 80000023, 90000049, 10000079]
        np.random.seed(42)  # For reproducibility of validation seeds
        validation_seeds = []
        test_seeds = {10000019, 20000003, 30000001, 40000003, 50000017,
                      60000011, 70000027, 80000023, 90000049, 10000079}
        while len(validation_seeds) < num_validation_seeds:
            seed = np.random.randint(1000000, 99999999)
            if seed not in test_seeds and seed not in validation_seeds:
                validation_seeds.append(seed)
        print(f"\nValidation seeds: {validation_seeds}\n")

    # Track best validation performance and early stopping
    best_val_reward = -float('inf')
    validation_rewards = []
    validations_without_improvement = 0

    # Iterate over the total number of time steps
    for t in range(total_timesteps):

        # Select action
        if use_continuous_actions:
            progress = min(t / (exploration_fraction * total_timesteps), 1.0)
            current_noise = 0.3 - (0.3 - 0.05) * progress

            policy_net.eval()
            env_action = get_continuous_action(obs, policy_net, exploration_noise=current_noise, t=t)
            policy_net.train()
            action_id = env_action  # For continuous, store the actual action
        else:
            env_action, action_id = get_action(obs, policy_net, action_size, actions, exploration, t, is_greedy=False)

        # TODO: if you want to implement the network associated with the continuous action set or the prioritized replay buffer, you need to reimplement the replay buffer.

        # Perform action fram_skip-times
        for f in range(action_repeat):
            new_obs, rew, term, trunc, _ = env.step(env_action)
            done = term or trunc
            episode_rewards[-1] += rew
            if done:
                break

        # Store transition in the replay buffer.
        new_obs = get_state(new_obs)
        replay_buffer.add(obs, action_id, rew, new_obs, float(done))
        obs = new_obs

        if done:
            # Start new episode after previous episode has terminated
            print("timestep: " + str(t) + " \t reward: " + str(episode_rewards[-1]))
            if use_wandb:
                wandb.log({
                    'episode_reward': episode_rewards[-1],
                    'episode_number': len(episode_rewards) - 1,
                    'timestep': t
                })
            obs, _ = env.reset()
            obs = get_state(obs)
            episode_rewards.append(0.0)

        if t > learning_starts and t % train_freq == 0:
            # Use teacher network during warmup, target network after
            active_target_net = teacher_net if (using_warmup and t < warmup_steps) else target_net

            if use_continuous_actions:
                loss = perform_qlearning_step_continuous(policy_net, active_target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning, grad_clip=10.0)
            else:
                loss = perform_qlearning_step(policy_net, active_target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning, grad_clip=10.0)
            training_losses.append(loss)

            if scheduler is not None:
                scheduler.step()

            if use_wandb:
                wandb.log({
                    'training_loss': loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'timestep': t
                })

        # Update target network
        # Handle warm-up transition
        if using_warmup and t == warmup_steps:
            print(f"\n{'='*60}")
            print(f"WARM-UP COMPLETE at step {t}")
            print(f"{'='*60}")
            print(f"Switching from teacher network to own target network")
            print(f"Copying policy network to target network...")
            target_net.load_state_dict(policy_net.state_dict())
            print(f"✓ Target network updated - resuming normal training")
            print(f"{'='*60}\n")

        # Regular target network updates (skip during warmup)
        if t > learning_starts and t % target_network_update_freq == 0:
            if not using_warmup or t >= warmup_steps:
                update_target_net(policy_net, target_net)

        if t % 1000 == 0:
            end = time.time()
            print(f"\n** {t} th timestep - {end - start:.5f} sec passed**\n")

            # Log exploration epsilon
            if use_wandb:
                current_eps = exploration.value(t)
                wandb.log({
                    'exploration_epsilon': current_eps,
                    'timestep': t
                })

        #if t % 10000 == 0:
        #    end = time.time()
        #    print(f"\n** {t} th timestep - Checkpoint saved! **\n")

        #    # Save timestamped checkpoint (keeps all versions)
        #    checkpoint_name = f"{model_identifier}_{t}.pth"
        #    torch.save(policy_net.state_dict(), os.path.join(outdir, checkpoint_name))

        # Validation check
        if t > learning_starts and t % validation_freq == 0:
            print(f"\n{'='*60}")
            print(f"Running validation at timestep {t}...")
            print(f"{'='*60}")

            val_reward = validate_agent(policy_net, actions, action_size,
                                       validation_seeds, no_display, use_continuous_actions)
            validation_rewards.append((t, val_reward))

            print(f"Validation average reward: {val_reward:.2f}")
            print(f"Previous best: {best_val_reward:.2f}")

            # Log validation metrics to wandb
            if use_wandb:
                wandb.log({
                    'val_reward': val_reward,
                    'best_val_reward': best_val_reward,
                    'timestep': t
                })

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                validations_without_improvement = 0  # Reset counter
                best_checkpoint_name = f"{model_identifier}_best.pth"
                best_checkpoint_path = os.path.join(outdir, best_checkpoint_name)
                torch.save(policy_net.state_dict(), best_checkpoint_path)
                print(f"*** NEW BEST! Saved {best_checkpoint_name} ***")
                if use_wandb:
                    wandb.log({
                        'best_val_reward': best_val_reward,
                        'timestep': t
                    })
                    # Save best model to wandb for cloud backup
                    wandb.save(best_checkpoint_path)
            else:
                validations_without_improvement += 1
                print(f"No improvement over best ({best_val_reward:.2f})")
                print(f"Validations without improvement: {validations_without_improvement}/{early_stopping_patience}")

            # Early stopping check
            if validations_without_improvement >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING: No improvement for {early_stopping_patience} validation checks")
                print(f"Best validation reward: {best_val_reward:.2f}")
                print(f"Stopping training at timestep {t}")
                print(f"{'='*60}\n")
                break

            print(f"{'='*60}\n")


    end = time.time()
    print(f"\n** Total {end - start:.5f} sec passed**\n")

    # Best model already saved during validation
    print(f"Best model saved as: {model_identifier}_best.pth")
    print(f"Best validation reward: {best_val_reward:.2f}")

    # Visualize the training loss and cumulative reward curves
    visualize_training(episode_rewards, training_losses, model_identifier, outdir )
 
