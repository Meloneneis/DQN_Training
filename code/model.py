import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_size, device, use_dueling=True, activation='gelu'):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        use_dueling: bool
            whether to use dueling network architecture
        activation: str
            activation function type
        """
        super().__init__()

        self.device = device
        self.action_size = action_size
        self.use_dueling = use_dueling

        # Set activation
        if activation == 'gelu':
            self.act = F.gelu
        else:
            self.act = F.relu

        # CNN layers
        self.conv1 = nn.Conv2d(3, 128, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1)

        self.conv_output_size = 512 * 6 * 6

        self.sensor_input_size = 7

        # First FC layer (shared)
        self.fc1 = nn.Linear(self.conv_output_size + self.sensor_input_size, 512)

        if use_dueling:
            # Dueling DQN: separate streams after fc1
            self.value_fc = nn.Linear(512, 256)
            self.value_stream = nn.Linear(256, 1)

            self.advantage_fc = nn.Linear(512, 256)
            self.advantage_stream = nn.Linear(256, self.action_size)
        else:
            # Standard DQN
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, self.action_size)

        # TODO: Create network

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        if not isinstance(observation, torch.Tensor):
            observation = torch.FloatTensor(observation).to(self.device)
        else:
            observation = observation.to(self.device)

        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
        
        batch_size = observation.shape[0]
        
        if observation.shape[-1] != 3:
            # If shape is (batch, channels, height, width), permute to (batch, height, width, channels)
            if observation.shape[1] == 3:
                observation = observation.permute(0, 2, 3, 1)
        
        # Permute from (batch, height, width, channels) to (batch, channels, height, width)
        observation = observation.permute(0, 3, 1, 2)

        # CNN forward
        x = self.act(self.conv1(observation))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        x = x.reshape(batch_size, -1)

        observation_original = observation.permute(0, 2, 3, 1)  # Back to (batch, H, W, C)
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation_original, batch_size)
        sensors = torch.cat([speed, abs_sensors, steering, gyroscope], dim=1)
        x = torch.cat([x, sensors], dim=1)

        # Shared FC layer
        x = self.act(self.fc1(x))

        if self.use_dueling:
            # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            value = self.act(self.value_fc(x))
            value = self.value_stream(value)

            advantage = self.act(self.advantage_fc(x))
            advantage = self.advantage_stream(advantage)

            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN
            x = self.act(self.fc2(x))
            q_values = self.fc3(x)

        return q_values

        # TODO: Forward pass through the network

    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels. The values are
        only approx. normalized, however, that suffices.
        Parameters
        ----------
        observation: list
            torch.Tensors of size (batch_size, 96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """

        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255 / 5

        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
        steer_crop[:, :10] *= -1
        steering = steer_crop.sum(dim=1, keepdim=True)

        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
        gyro_crop[:, :14] *= -1
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)

        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
