import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """ Add a transition to replay memory. 
        Parameters
        ----------
        obs_t: 
            State s_t
        action: 
            Action a_t taken in s_t
        reward: 
            Received reward r_t
        obs_tp1: 
            Follow-up state s_{t+1}
        done: bool
            Whether episode has terminated at s_{t+1}
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        batch_size = len(idxes)
        first_obs = self._storage[idxes[0]][0]
        first_action = self._storage[idxes[0]][1]

        obs_shape = first_obs.shape if hasattr(first_obs, 'shape') else (len(first_obs),)
        action_shape = first_action.shape if hasattr(first_action, 'shape') else (len(first_action) if hasattr(first_action, '__len__') else (),)

        # Pre-allocate arrays
        obses_t = np.empty((batch_size,) + obs_shape, dtype=np.float32)
        obses_tp1 = np.empty((batch_size,) + obs_shape, dtype=np.float32)
        actions = np.empty((batch_size,) + action_shape, dtype=np.float32 if action_shape else np.int64)
        rewards = np.empty(batch_size, dtype=np.float32)
        dones = np.empty(batch_size, dtype=np.float32)

        for idx, i in enumerate(idxes):
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t[idx] = obs_t
            actions[idx] = action
            rewards[idx] = reward
            obses_tp1[idx] = obs_tp1
            dones[idx] = done

        return np.squeeze(obses_t, axis=1), actions, rewards, np.squeeze(obses_tp1, axis=1), dones

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)