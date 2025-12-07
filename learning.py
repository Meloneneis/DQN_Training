import numpy as np
import torch
import torch.nn.functional as F

def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning=False, grad_clip=None):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    use_doubleqlearning: bool
        whether to use double Q-learning
    grad_clip: float or None
        gradient clipping value (None = no clipping)
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """

    # Tip: You can use use_doubleqlearning to switch the learning modality.

    # Step 1
    obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(batch_size)

    obs_batch = torch.FloatTensor(obs_batch).pin_memory().to(device, non_blocking=True)
    act_batch = torch.LongTensor(act_batch).pin_memory().to(device, non_blocking=True)
    rew_batch = torch.FloatTensor(rew_batch).pin_memory().to(device, non_blocking=True)
    next_obs_batch = torch.FloatTensor(next_obs_batch).pin_memory().to(device, non_blocking=True)
    done_batch = torch.FloatTensor(done_batch).pin_memory().to(device, non_blocking=True)

    q_values = policy_net(obs_batch)
    q_values = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if use_doubleqlearning:
            next_q_values_policy = policy_net(next_obs_batch)
            next_actions = next_q_values_policy.max(1)[1]

            next_q_values_target = target_net(next_obs_batch)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q_values_target = target_net(next_obs_batch)
            next_q_values = next_q_values_target.max(1)[0]

        next_q_values = next_q_values * (1 - done_batch)
        target_q_values = rew_batch + gamma * next_q_values

    loss = F.mse_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
    optimizer.step()
    
    return loss.item()

def update_target_net(policy_net, target_net, tau=None):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    tau: float or None
        soft update coefficient (if None, performs hard update)
        target = tau * policy + (1-tau) * target
    """

    if tau is None:
        # Hard update: copy weights directly
        target_net.load_state_dict(policy_net.state_dict())
    else:
        # Soft update (Polyak averaging)
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


def perform_qlearning_step_continuous(policy_net, target_net, optimizer, replay_buffer,
                                     batch_size, gamma, device, use_doubleqlearning=False, grad_clip=None):
    """Q-learning step for continuous actions using NAF

    Parameters
    ----------
    grad_clip: float or None
        gradient clipping value (None = no clipping)
    """

    obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = \
        replay_buffer.sample(batch_size)

    obs_batch = torch.FloatTensor(obs_batch).pin_memory().to(device, non_blocking=True)
    act_batch = torch.FloatTensor(act_batch).pin_memory().to(device, non_blocking=True)
    rew_batch = torch.FloatTensor(rew_batch).pin_memory().to(device, non_blocking=True)
    next_obs_batch = torch.FloatTensor(next_obs_batch).pin_memory().to(device, non_blocking=True)
    done_batch = torch.FloatTensor(done_batch).pin_memory().to(device, non_blocking=True)

    q_values, _ = policy_net(obs_batch, act_batch)

    with torch.no_grad():
        next_actions = target_net(next_obs_batch)

        if use_doubleqlearning:
            next_actions_policy = policy_net(next_obs_batch)
            next_q_values, _ = target_net(next_obs_batch, next_actions_policy)
        else:
            next_q_values, _ = target_net(next_obs_batch, next_actions)

        next_q_values = next_q_values * (1 - done_batch)
        target_q_values = rew_batch + gamma * next_q_values

    loss = F.mse_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
    optimizer.step()

    return loss.item()
