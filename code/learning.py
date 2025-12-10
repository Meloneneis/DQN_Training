import numpy as np
import torch
import torch.nn.functional as F

def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning=False):
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
    
    obs_batch = torch.FloatTensor(obs_batch).to(device)
    act_batch = torch.LongTensor(act_batch).to(device)
    rew_batch = torch.FloatTensor(rew_batch).to(device)
    next_obs_batch = torch.FloatTensor(next_obs_batch).to(device)
    done_batch = torch.FloatTensor(done_batch).to(device)
    
    # Step 2
    q_values = policy_net(obs_batch)
    q_values = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)
    
    # Step 3
    with torch.no_grad():
        if use_doubleqlearning:
            # Double Q-Learning
            next_q_values_policy = policy_net(next_obs_batch)
            next_actions = next_q_values_policy.max(1)[1]
            
            next_q_values_target = target_net(next_obs_batch)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            # Standard DQN
            next_q_values_target = target_net(next_obs_batch)
            next_q_values = next_q_values_target.max(1)[0]
        
        # Step 4
        next_q_values = next_q_values * (1 - done_batch)
        
        # Step 5
        target_q_values = rew_batch + gamma * next_q_values
    
    # Step 6
    loss = F.mse_loss(q_values, target_q_values)
    
    # Step 7 & 8
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    
    # Step 9
    optimizer.step()
    
    return loss.item()

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    target_net.load_state_dict(policy_net.state_dict())

    # TODO: Update target network
