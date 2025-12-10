import torch

# Load checkpoint
checkpoint = torch.load('agent_best.pth', map_location='cpu')

# Print original keys
print("Original keys:")
for key in checkpoint.keys():
    print(f"  {key}")

print("\nConverting...")

# Rename keys
new_checkpoint = {}
for key, value in checkpoint.items():
    new_key = key

    # Rename conv layers
    new_key = new_key.replace('conv_layers.0.', 'conv1.')
    new_key = new_key.replace('conv_layers.1.', 'conv2.')
    new_key = new_key.replace('conv_layers.2.', 'conv3.')
    new_key = new_key.replace('conv_layers.3.', 'conv4.')

    # Rename fc_shared.0 to fc1
    new_key = new_key.replace('fc_shared.0.', 'fc1.')

    # For dueling architecture:
    # value_stream.0 -> value_fc (intermediate value layer)
    # value_output -> value_stream (final value output)
    # advantage_stream.0 -> advantage_fc (intermediate advantage layer)
    # advantage_output -> advantage_stream (final advantage output)
    new_key = new_key.replace('value_stream.0.', 'value_fc.')
    new_key = new_key.replace('value_output.', 'value_stream.')
    new_key = new_key.replace('advantage_stream.0.', 'advantage_fc.')
    new_key = new_key.replace('advantage_output.', 'advantage_stream.')

    new_checkpoint[new_key] = value
    print(f"  {key} -> {new_key}")

# Save converted checkpoint
torch.save(new_checkpoint, 'agent_best.pth')
print("\nCheckpoint converted and saved!")
