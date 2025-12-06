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

def get_continuous_action(state, policy_net, exploration_noise=0.1, t=0):
    """
    Select continuous action with exploration noise

    Parameters:
    -----------
    state: observation
    policy_net: ContinuousActionDQN network
    exploration_noise: std dev of Gaussian noise
    t: timestep
    """
    import numpy as np

    with torch.no_grad():
        action = policy_net(state).cpu().numpy().flatten()

    if exploration_noise > 0:
        noise = np.random.normal(0, exploration_noise, size=action.shape)
        action = action + noise

    action[0] = np.clip(action[0], -1.0, 1.0)  # steering
    action[1] = np.clip(action[1], 0.0, 1.0)   # gas
    action[2] = np.clip(action[2], 0.0, 1.0)   # brake

    return action


class OrnsteinUhlenbeckNoise:
    """Better exploration noise for continuous actions"""
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        import numpy as np
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        import numpy as np
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        import numpy as np
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

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
