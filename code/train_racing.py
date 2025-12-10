import gymnasium as gym
import deepq
import argparse
import platform
import time

from sdc_wrapper import SDC_Wrapper

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

    parser.add_argument ( '--total_timesteps', type=int, default=100000, help = 'The number of env steps to take' )
    parser.add_argument ( '--action_repeats', type=int, default=4, help='Update the model every action_repeatss steps' )
    parser.add_argument ( '--gamma', type=float, default=0.99, help='selection action on every n-th frame and repeat action for intermediate frames' )
    parser.add_argument ( '--action_filename', type=str, default = 'improved_actions.txt', help='a list of actions' )
    parser.add_argument ( '--lr', type=float, default=5e-5, help='learning rate' )
    parser.add_argument ( '--batch_size', type=int, default=32, help='batch size for training' )
    parser.add_argument ( '--buffer_size', type=int, default=50000, help='replay buffer size' )
    parser.add_argument ( '--exploration_fraction', type=float, default=0.2, help='fraction of training for exploration' )
    parser.add_argument ( '--exploration_final_eps', type=float, default=0.02, help='final exploration epsilon' )
    parser.add_argument ( '--learning_starts', type=int, default=100, help='steps before learning starts' )
    parser.add_argument ( '--target_network_update_freq', type=int, default=1000, help='target network update frequency' )
    parser.add_argument ( '--weight_decay', type=float, default=0.0, help='weight decay for optimizer' )
    parser.add_argument ( '--optimizer_type', type=str, default='adam', help='optimizer type (adam or adamw)' )
    parser.add_argument ( '--activation', type=str, default='gelu', help='activation function (relu or gelu)' )
    parser.add_argument ( '--use_doubleqlearning', default=False, action="store_true", help='a flag that indicates the use of double q learning' )
    parser.add_argument ( '--use_dueling', default=False, action="store_true", help='a flag that indicates the use of dueling network' )
    parser.add_argument ( '--no_display', default=False, action="store_true", help='a flag indicating whether training runs on the cluster' )
    parser.add_argument ( '--agent_name', type=str, default='agent', help='an agent name' )
    parser.add_argument ( '--outdir', type=str, default='', help='a directory for output' )

    args = parser.parse_args()

    # check args
    print ( "\nArgs information")
    print ( "total_timesteps:     {0}".format ( args.total_timesteps ) )
    print ( "action_repeats:      {0}".format ( args.action_repeats ) )
    print ( "gamma:               {0}".format ( args.gamma ) )
    print ( "action_filename:     {0}".format ( args.action_filename ) )
    print ( "use_doubleqlearning: {0}".format ( "doubleq" if args.use_doubleqlearning else "no doubleq" ) )
    print ( "use_dueling:         {0}".format ( "dueling" if args.use_dueling else "no dueling" ) )
    print ( "no display:             {0}".format ( "true" if args.no_display else "false" ) )
    print ( "agent_name:          {0}".format ( args.agent_name ) )
    print ( "outdir:              {0}".format ( args.outdir ) )

    # load actions
    actions = load_actions ( args.action_filename ) 
    print ( "actions:\t\t", actions )

    # start training
    print ( "\nStart training..." )
    render_mode = 'rgb_array' if args.no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode), remove_score=True, return_linear_velocity=False)

    deepq.learn (
                    env,
                    lr = args.lr,
                    total_timesteps = args.total_timesteps,
                    action_repeat = args.action_repeats,
                    gamma = args.gamma,
                    batch_size = args.batch_size,
                    buffer_size = args.buffer_size,
                    exploration_fraction = args.exploration_fraction,
                    exploration_final_eps = args.exploration_final_eps,
                    learning_starts = args.learning_starts,
                    target_network_update_freq = args.target_network_update_freq,
                    model_identifier = args.agent_name,
                    outdir= args.outdir,
                    new_actions = actions,
                    use_doubleqlearning = args.use_doubleqlearning,
                    use_dueling = args.use_dueling,
                    weight_decay = args.weight_decay,
                    optimizer_type = args.optimizer_type,
                    activation = args.activation
                )
    
    # wrap up
    env.close ()

if __name__ == '__main__':
    main()
