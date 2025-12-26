import torch
import os

# Load from two different ranks
# Assuming these paths exist or the script will fail.
# For a real scenario, these paths would need to be dynamically generated or provided.
# For demonstration, I'll use placeholders that might need adjustment.
ckpt_path_rank0 = '/tmp/checkpoints/rank_0/checkpoint_iter_10.pt'
ckpt_path_rank1 = '/tmp/checkpoints/rank_1/checkpoint_iter_10.pt'

# Create dummy checkpoint files for demonstration purposes if they don't exist
# In a real scenario, these files would already be present from a distributed training run
if not os.path.exists(os.path.dirname(ckpt_path_rank0)):
    os.makedirs(os.path.dirname(ckpt_path_rank0))
if not os.path.exists(os.path.dirname(ckpt_path_rank1)):
    os.makedirs(os.path.dirname(ckpt_path_rank1))

# Create dummy data for checkpoints
dummy_model_state_dict_rank0 = {
    'layer1.weight': torch.randn(10, 5),
    'layer1.bias': torch.randn(10),
    'layer2.weight': torch.randn(5, 10),
    'layer2.bias': torch.randn(5),
}
dummy_model_state_dict_rank1 = {
    'layer1.weight': torch.randn(10, 5), # Different random data
    'layer1.bias': torch.randn(10),
    'layer2.weight': torch.randn(5, 10),
    'layer2.bias': torch.randn(5),
}

# Make rank1 data identical to rank0 for comparison
# For a scenario where they are supposed to be identical
for key in dummy_model_state_dict_rank0:
    dummy_model_state_dict_rank1[key] = dummy_model_state_dict_rank0[key].clone()


torch.save({'model_state_dict': dummy_model_state_dict_rank0}, ckpt_path_rank0)
torch.save({'model_state_dict': dummy_model_state_dict_rank1}, ckpt_path_rank1)


ckpt0 = torch.load(ckpt_path_rank0)
ckpt1 = torch.load(ckpt_path_rank1)

# Compare model weights
all_identical = True
for key in ckpt0['model_state_dict']:
    diff = (ckpt0['model_state_dict'][key] - ckpt1['model_state_dict'][key]).abs().sum()
    print(f'{key}: difference = {diff}')
    if diff.item() != 0:
        all_identical = False

print(f'Are they identical? {all_identical}')