import gymnasium as gym
import platform
import argparse
import torch
import numpy as np

from action import ActionSet, get_action
from sdc_wrapper import SDC_Wrapper
from utils import get_state
from model import DQN

def evaluate(env, new_actions = None, load_path='agent.pth'):
    """ Evaluate a trained model and compute your leaderboard scores

	NO CHANGES SHOULD BE MADE TO THIS FUNCTION

    Parameters
    -------
    env: gym.Env
        environment to evaluate on
    load_path: str
        path to load the model (.pth) from
    """
    episode_rewards = []
    action_manager = ActionSet()

    if new_actions is not None:
        action_manager.set_actions ( new_actions )

    actions = action_manager.get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # These are not the final evaluation seeds, do not overfit on these tracks!
    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
             49636746, 66759182, 91294619, 84274995, 31531469]

    # Build & load network
    policy_net = DQN(action_size, device).to(device)
    checkpoint = torch.load(load_path, map_location=device)
    policy_net.load_state_dict(checkpoint)
    policy_net.eval()

    # Iterate over a number of evaluation episodes
    for episode, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        obs = get_state(obs)
        t = 0

        # Run each episode until episode has terminated or 600 time steps have been reached
        episode_rewards.append(0.0)
        while True:
            action, _ = get_action ( obs, policy_net, action_size, actions, is_greedy= True, t=t)

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

def load_actions ( action_filename ):

    actions = []

    with open ( action_filename ) as f:

        lines = f.readlines()

        for line in lines:
            action = []
            for tok in line.split():
                action.append ( float ( tok ))
            actions.append (action)

    return actions

def main():

    """ 
    Evaluate a trained Deep Q-Learning agent 
    """ 

    print ("python version:\t{0}".format (platform.python_version()))
    print ("gym version:\t{0}".format(gym.__version__))

    # it doesn't help that much.
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument ( '--action_filename', type=str, default = 'improved_actions.txt', help='a list of actions' )
    parser.add_argument ( '--agent_name', type=str, default='agent_best')
    parser.add_argument('--no_display', default=True, action="store_true", help='a flag indicating whether training/evaluation runs on the cluster')

    args = parser.parse_args()

    # load actions
    actions = load_actions ( args.action_filename ) 
    print ( "actions:\t\t", actions )

    filename = args.agent_name +'.pth'

    print("loading {0}".format( filename ) )
    render_mode = 'rgb_array' if args.no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode), remove_score=True, return_linear_velocity=False)
    evaluate(env, new_actions = actions, load_path = filename )

    env.close()


if __name__ == '__main__':
    main()
