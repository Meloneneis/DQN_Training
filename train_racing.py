import gymnasium as gym
import deepq
import argparse
import platform
import time

from sdc_wrapper import SDC_Wrapper

PROJECT_NAME = "car_racing_RL"
MODEL_NOTES = "Double Q-Learning"
BATCH_SIZE = 32
DROPOUT = 0.2
EPOCHS = 25
L1_SIZE = 16
L2_SIZE = 32
HIDDEN_LAYER_SIZE = 128
LEARNING_RATE = 0.01
DECAY = 1e-6
MOMENTUM = 0.9


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

def main ():

    """ 
    Train a Deep Q-Learning agent in headless mode on the cluster
    """ 

    print ("python version:\t{0}".format (platform.python_version()))
    print ("gym version:\t{0}".format(gym.__version__))

    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument ( '--total_timesteps', type=int, default=300000, help = 'The number of env steps to take' )
    parser.add_argument ( '--action_repeats', type=int, default=4, help='Update the model every action_repeatss steps' )
    parser.add_argument ( '--gamma', type=float, default=0.99, help='selection action on every n-th frame and repeat action for intermediate frames' )
    parser.add_argument ( '--action_filename', type=str, default = 'improved_actions.txt', help='a list of actions' )
    parser.add_argument ( '--use_doubleqlearning', default=True, action="store_true", help='a flag that indicates the use of double q learning' )
    parser.add_argument ( '--no_display', default=True, action="store_true", help='a flag indicating whether training runs on the cluster' )
    parser.add_argument ( '--agent_name', type=str, default='agent256', help='an agent name' )
    parser.add_argument ( '--outdir', type=str, default='', help='a directory for output' )
    parser.add_argument ( '--validation_freq', type=int, default=10000, help='how often to run validation (in timesteps)' )
    parser.add_argument ( '--num_validation_seeds', type=int, default=20, help='number of random seeds to use for validation' )
    parser.add_argument ( '--early_stopping_patience', type=int, default=5, help='stop training after N validations without improvement' )
    parser.add_argument ( '--batch_size', type=int, default=256, help='batch size for training' )
    parser.add_argument ( '--use_continuous_actions', default=False, action="store_true", help='Use continuous actions (NAF) instead of discrete' )

    args = parser.parse_args()


    # load actions
    actions = load_actions ( args.action_filename )

    # start training
    print ( "\nStart training..." )
    render_mode = 'rgb_array' if args.no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode), remove_score=True, return_linear_velocity=False)


    deepq.learn (
                    env, total_timesteps = args.total_timesteps,
                    action_repeat = args.action_repeats,
                    gamma = args.gamma,
                    model_identifier = args.agent_name,
                    outdir= args.outdir,
                    new_actions = actions,
                    use_doubleqlearning = args.use_doubleqlearning,
                    validation_freq = args.validation_freq,
                    num_validation_seeds = args.num_validation_seeds,
                    early_stopping_patience = args.early_stopping_patience,
                    no_display = args.no_display,
                    batch_size=args.batch_size,
                    use_continuous_actions = args.use_continuous_actions,
                )
    
    # wrap up
    env.close ()

if __name__ == '__main__':
    main()
