import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_size, device, hidden_sizes=[1024, 512], dropout_rate=0.2,
                 cnn_channels=[32, 64, 128, 128], cnn_kernels=[8, 4, 3, 3],
                 cnn_strides=[4, 2, 1, 1], final_spatial_size=6, activation='relu',
                 normalization='layer', use_dueling=False):
        """
        Deep Q-Network with configurable architecture.

        Parameters
        ----------
        action_size : int
            Number of discrete actions
        device : torch.device
            Device to run the model on
        hidden_sizes : list of int
            List of FC layer widths, e.g., [1024, 512] for 2 hidden layers
        dropout_rate : float
            Dropout probability for regularization
        cnn_channels : list of int
            List of CNN output channels, e.g., [32, 64, 128, 128] for 4 conv layers
        cnn_kernels : list of int
            Kernel sizes for each conv layer
        cnn_strides : list of int
            Stride for each conv layer
        final_spatial_size : int
            Spatial dimension after all conv layers (e.g., 6 for 6x6)
        activation : str
            Activation function name ('relu', 'gelu', 'silu')
        normalization : str
            Normalization type ('layer', 'none')
        use_dueling : bool
            Whether to use dueling network architecture (separates V(s) and A(s,a))
        """
        super().__init__()

        self.device = device
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.cnn_channels = cnn_channels
        self.num_cnn_layers = len(cnn_channels)
        self.activation_name = activation
        self.normalization = normalization
        self.use_dueling = use_dueling

        # Select activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build CNN layers dynamically based on configuration
        # Input: 96x96x3
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_channels = 3
        for i, (out_channels, kernel, stride) in enumerate(zip(cnn_channels, cnn_kernels, cnn_strides)):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride))

            # Add normalization layer based on type
            if normalization == 'layer':
                # LayerNorm for 2D expects [C, H, W] format, we'll compute H, W dynamically
                self.norm_layers.append(None)  # Will use F.layer_norm in forward
            elif normalization == 'none':
                self.norm_layers.append(None)
            else:
                raise ValueError(f"Unknown normalization: {normalization}")

            in_channels = out_channels

        # Conv output size: channels * spatial_size * spatial_size
        self.conv_output_size = cnn_channels[-1] * final_spatial_size * final_spatial_size

        self.sensor_input_size = 7

        if use_dueling:
            # Dueling DQN architecture: separate streams for V(s) and A(s,a)
            # Shared feature layers
            self.fc_shared = nn.ModuleList()
            input_size = self.conv_output_size + self.sensor_input_size

            # Use first hidden layer as shared
            if len(hidden_sizes) > 0:
                self.fc_shared.append(nn.Linear(input_size, hidden_sizes[0]))
                shared_output_size = hidden_sizes[0]
            else:
                shared_output_size = input_size

            # Value stream V(s)
            self.value_stream = nn.ModuleList()
            input_size = shared_output_size
            for i in range(1, len(hidden_sizes)):
                self.value_stream.append(nn.Linear(input_size, hidden_sizes[i]))
                input_size = hidden_sizes[i]
            self.value_output = nn.Linear(input_size, 1)

            # Advantage stream A(s,a)
            self.advantage_stream = nn.ModuleList()
            input_size = shared_output_size
            for i in range(1, len(hidden_sizes)):
                self.advantage_stream.append(nn.Linear(input_size, hidden_sizes[i]))
                input_size = hidden_sizes[i]
            self.advantage_output = nn.Linear(input_size, self.action_size)

        else:
            # Standard DQN architecture
            self.fc_layers = nn.ModuleList()
            input_size = self.conv_output_size + self.sensor_input_size

            for hidden_size in hidden_sizes:
                self.fc_layers.append(nn.Linear(input_size, hidden_size))
                input_size = hidden_size

            # Output layer
            self.fc_output = nn.Linear(input_size, self.action_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

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

        # Pass through dynamic CNN layers
        x = observation
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            x = conv_layer(x)
            if norm_layer is not None:
                x = norm_layer(x)
            elif self.normalization == 'layer':
                # Apply LayerNorm: normalize over [C, H, W] dimensions
                normalized_shape = x.shape[1:]  # [C, H, W]
                x = F.layer_norm(x, normalized_shape)
            x = self.activation(x)

        x = x.reshape(batch_size, -1)

        observation_original = observation.permute(0, 2, 3, 1)  # Back to (batch, H, W, C)
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation_original, batch_size)
        sensors = torch.cat([speed, abs_sensors, steering, gyroscope], dim=1)
        x = torch.cat([x, sensors], dim=1)

        if self.use_dueling:
            # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            # Shared features
            for fc_layer in self.fc_shared:
                x = self.activation(fc_layer(x))
                x = self.dropout(x)

            # Value stream
            value = x
            for layer in self.value_stream:
                value = self.activation(layer(value))
                value = self.dropout(value)
            value = self.value_output(value)

            # Advantage stream
            advantage = x
            for layer in self.advantage_stream:
                advantage = self.activation(layer(advantage))
                advantage = self.dropout(advantage)
            advantage = self.advantage_output(advantage)

            # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        else:
            # Standard DQN
            for fc_layer in self.fc_layers:
                x = self.activation(fc_layer(x))
                x = self.dropout(x)

            q_values = self.fc_output(x)

        return q_values

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

class ContinuousActionDQN(nn.Module):
    def __init__(self, action_dim, device, hidden_sizes=[512, 256], dropout_rate=0.2,
                 cnn_channels=[32, 64, 64], cnn_kernels=[8, 4, 3],
                 cnn_strides=[4, 2, 1], final_spatial_size=8, activation='relu',
                 normalization='layer'):
        super().__init__()

        self.device = device
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.normalization = normalization

        # Select activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build CNN layers dynamically
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_channels = 3
        for i, (out_channels, kernel, stride) in enumerate(zip(cnn_channels, cnn_kernels, cnn_strides)):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride))

            # Add normalization layer based on type
            if normalization == 'layer':
                # LayerNorm for 2D expects [C, H, W] format, we'll compute H, W dynamically
                self.norm_layers.append(None)  # Will use F.layer_norm in forward
            elif normalization == 'none':
                self.norm_layers.append(None)
            else:
                raise ValueError(f"Unknown normalization: {normalization}")

            in_channels = out_channels

        self.conv_output_size = cnn_channels[-1] * final_spatial_size * final_spatial_size
        self.sensor_input_size = 7

        # Shared feature layer
        self.fc_shared = nn.Linear(self.conv_output_size + self.sensor_input_size, hidden_sizes[0])

        # Value stream (V(s))
        self.value_layers = nn.ModuleList()
        input_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            self.value_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.value = nn.Linear(input_size, 1)

        # Mean action stream (mu(s))
        self.mu_layers = nn.ModuleList()
        input_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            self.mu_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.mu = nn.Linear(input_size, action_dim)

        # Advantage scale stream (for A(s,a))
        self.advantage_scale_layers = nn.ModuleList()
        input_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            self.advantage_scale_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.advantage_scale = nn.Linear(input_size, action_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, observation, action=None):
        """Forward pass"""
        if not isinstance(observation, torch.Tensor):
            observation = torch.FloatTensor(observation).to(self.device)
        else:
            observation = observation.to(self.device)

        if observation.dim() == 3:
            observation = observation.unsqueeze(0)

        batch_size = observation.shape[0]

        if observation.shape[-1] != 3:
            if observation.shape[1] == 3:
                observation = observation.permute(0, 2, 3, 1)

        observation = observation.permute(0, 3, 1, 2)

        # Pass through dynamic CNN layers
        x = observation
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            x = conv_layer(x)
            if norm_layer is not None:
                x = norm_layer(x)
            elif self.normalization == 'layer':
                # Apply LayerNorm: normalize over [C, H, W] dimensions
                normalized_shape = x.shape[1:]  # [C, H, W]
                x = F.layer_norm(x, normalized_shape)
            x = self.activation(x)
        x = torch.flatten(x, start_dim=1)

        observation_original = observation.permute(0, 2, 3, 1)
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(
            observation_original, batch_size
        )
        sensors = torch.cat([speed, abs_sensors, steering, gyroscope], dim=1)
        x = torch.cat([x, sensors], dim=1)

        features = self.activation(self.fc_shared(x))
        features = self.dropout(features)

        # Value stream
        value = features
        for layer in self.value_layers:
            value = self.activation(layer(value))
            value = self.dropout(value)
        value = self.value(value)

        # Mean action stream
        mu = features
        for layer in self.mu_layers:
            mu = self.activation(layer(mu))
            mu = self.dropout(mu)
        mu = self.mu(mu)

        mu_bounded = torch.cat([
            torch.tanh(mu[:, 0:1]),
            torch.sigmoid(mu[:, 1:2]),
            torch.sigmoid(mu[:, 2:3])
        ], dim=1)

        if action is None:
            return mu_bounded

        # Advantage scale stream
        adv_scale = features
        for layer in self.advantage_scale_layers:
            adv_scale = self.activation(layer(adv_scale))
            adv_scale = self.dropout(adv_scale)
        scales = torch.exp(torch.clamp(self.advantage_scale(adv_scale), -10, 10))

        action_diff = action - mu_bounded
        advantage = -0.5 * (scales * action_diff.pow(2)).sum(dim=1)

        q_value = value.squeeze(-1) + advantage

        return q_value, mu_bounded

    def _build_lower_triangular(self, L_elements, batch_size):
        """Build lower triangular matrix from vector"""
        L = torch.zeros(batch_size, self.action_dim, self.action_dim).to(self.device)
        tril_indices = torch.tril_indices(self.action_dim, self.action_dim)

        for b in range(batch_size):
            L[b, tril_indices[0], tril_indices[1]] = L_elements[b]

        return L

    def extract_sensor_values(self, observation, batch_size):
        """Extract sensor values (same as DQN)"""
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
