import random
import torch

def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select greedy action

    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(policy_net.device)
    else:
        state = state.to(policy_net.device)

    policy_net.eval()

    with torch.no_grad():
        q_values = policy_net(state)

    action = q_values.argmax().item()

    policy_net.train()

    return action

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select exploratory action

    epsilon = exploration.value(t)

    if random.random() < epsilon:
        action = random.randrange(action_size)
    else:
        action = select_greedy_action(state, policy_net, action_size)

    return action


def get_action ( state, policy_net, action_size, actions = None, exploration = None, t = None, is_greedy = False):

    """
    Get an action regarding the mode of the policy.
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    actions: list
        list of actions
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    is_greedy: bool
        the mode of the policy
    Returns
    -------
    command (steering, throuput, brake)
        action
    """

    """
    This code is implemented for a discrete action set.

    If you want to develop networks with continuous action, you need to modify this.
    """

    # TODO: if you want to implement the network associated with the continuous action set, you need to reimplement this.
    if is_greedy:

        id = select_greedy_action(state, policy_net, action_size)
        return actions[id], id

    else:

        id = select_exploratory_action(state, policy_net, action_size, exploration, t)
        return actions[id], id


class ActionSet:

    def __init__(self):
        """ Initialize actions
        """
        self.actions = [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]

    def set_actions(self, new_actions):
        """ Set the list of available actions
        Parameters
        ------
        list
            list of available actions
        """
        self.actions = new_actions

    def get_action_set(self):
        """ Get the list of available actions
        Returns
        -------
        list
            list of available actions
        """
        return self.actions
