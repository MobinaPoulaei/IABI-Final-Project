import torch
import numpy as np


def randmixup(inputs, labels):
    """
    Random mixup augmentation for MIL bags.
    
    Args:
        inputs: list of tensors, each of shape (num_instances, 1024)
        labels: tensor of shape (batch_size,)
    
    Returns:
        mix_inputs: list of mixed bag features
        labels_a: first labels for mixing
        labels_b: second labels for mixing  
        lmbdas: mixing coefficients
    """
    batch_size = len(inputs)
    
    mix_inputs = []
    labels_a = []
    labels_b = []
    lmbdas = []
    
    for i in range(batch_size):
        # Random mixing coefficient
        lam = np.random.beta(1.0, 1.0)
        
        # Random pair for mixing
        j = np.random.randint(0, batch_size)
        
        # Mix features
        # inputs[i] shape: (num_instances_i, 1024)
        # inputs[j] shape: (num_instances_j, 1024)
        
        # Get minimum number of instances
        min_instances = min(inputs[i].shape[0], inputs[j].shape[0])
        
        # Randomly sample instances if needed
        if inputs[i].shape[0] > min_instances:
            idx_i = np.random.choice(inputs[i].shape[0], min_instances, replace=False)
            feat_i = inputs[i][idx_i]
        else:
            feat_i = inputs[i]
        
        if inputs[j].shape[0] > min_instances:
            idx_j = np.random.choice(inputs[j].shape[0], min_instances, replace=False)
            feat_j = inputs[j][idx_j]
        else:
            feat_j = inputs[j]
        
        # Mix
        mixed_feat = lam * feat_i + (1 - lam) * feat_j
        
        mix_inputs.append(mixed_feat)
        labels_a.append(labels[i])
        labels_b.append(labels[j])
        lmbdas.append(lam)
    
    labels_a = torch.stack(labels_a)
    labels_b = torch.stack(labels_b)
    lmbdas = torch.tensor(lmbdas)
    
    return mix_inputs, labels_a, labels_b, lmbdas


if __name__ == "__main__":
    # Test mixup
    print("Testing randmixup...")
    
    # Create dummy data
    batch_size = 4
    inputs = [torch.randn(np.random.randint(3, 10), 1024) for _ in range(batch_size)]
    labels = torch.tensor([0, 1, 2, 3])
    
    print(f"\nOriginal batch:")
    for i, inp in enumerate(inputs):
        print(f"  Sample {i}: shape {inp.shape}, label {labels[i].item()}")
    
    # Apply mixup
    mix_inputs, labels_a, labels_b, lmbdas = randmixup(inputs, labels)
    
    print(f"\nMixed batch:")
    for i in range(len(mix_inputs)):
        print(f"  Sample {i}: shape {mix_inputs[i].shape}, "
              f"label_a {labels_a[i].item()}, label_b {labels_b[i].item()}, "
              f"lambda {lmbdas[i].item():.3f}")
