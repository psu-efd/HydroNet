
def check_data_normalization(train_dataset, val_dataset):
    """
    Check normalization consistency between training and validation datasets.
    
    Args:
        train_dataset: SWE_DeepONetDataset for training data
        val_dataset: SWE_DeepONetDataset for validation data
    
    Returns:
        dict: Dictionary containing statistics for both datasets
    """
    print("\n=== Data Normalization Check ===")
    
    # Function to compute statistics for a dataset
    def compute_dataset_stats(dataset, name):
        branch_inputs = []
        trunk_inputs = []
        outputs = []
        
        for branch, trunk, target in dataset:
            branch_inputs.append(branch.numpy())
            trunk_inputs.append(trunk.numpy())
            outputs.append(target.numpy())
        
        branch_inputs = np.concatenate(branch_inputs)
        trunk_inputs = np.concatenate(trunk_inputs)
        outputs = np.concatenate(outputs)
        
        print(f"\n{name} Dataset Statistics:")
        
        # Branch inputs statistics (for each feature)
        print(f"\nBranch Inputs (shape: {branch_inputs.shape}):")
        for i in range(branch_inputs.shape[1]):
            print(f"  Feature {i}:")
            print(f"    Mean: {np.mean(branch_inputs[:, i]):.6f}")
            print(f"    Std: {np.std(branch_inputs[:, i]):.6f}")
            print(f"    Min: {np.min(branch_inputs[:, i]):.6f}")
            print(f"    Max: {np.max(branch_inputs[:, i]):.6f}")
        
        # Trunk inputs statistics (for each spatial/temporal coordinate)
        print(f"\nTrunk Inputs (shape: {trunk_inputs.shape}):")
        for i in range(trunk_inputs.shape[1]):
            print(f"  Coordinate {i}:")
            print(f"    Mean: {np.mean(trunk_inputs[:, i]):.6f}")
            print(f"    Std: {np.std(trunk_inputs[:, i]):.6f}")
            print(f"    Min: {np.min(trunk_inputs[:, i]):.6f}")
            print(f"    Max: {np.max(trunk_inputs[:, i]):.6f}")
        
        # Outputs statistics (for each output variable)
        print(f"\nOutputs (shape: {outputs.shape}):")
        output_names = ['X-Velocity (u)', 'Y-Velocity (v)', 'Water Depth (h)']
        for i in range(outputs.shape[1]):
            print(f"  {output_names[i]}:")
            print(f"    Mean: {np.mean(outputs[:, i]):.6f}")
            print(f"    Std: {np.std(outputs[:, i]):.6f}")
            print(f"    Min: {np.min(outputs[:, i]):.6f}")
            print(f"    Max: {np.max(outputs[:, i]):.6f}")
        
        return {
            'branch': {
                'mean': np.mean(branch_inputs, axis=0),
                'std': np.std(branch_inputs, axis=0),
                'min': np.min(branch_inputs, axis=0),
                'max': np.max(branch_inputs, axis=0)
            },
            'trunk': {
                'mean': np.mean(trunk_inputs, axis=0),
                'std': np.std(trunk_inputs, axis=0),
                'min': np.min(trunk_inputs, axis=0),
                'max': np.max(trunk_inputs, axis=0)
            },
            'output': {
                'mean': np.mean(outputs, axis=0),
                'std': np.std(outputs, axis=0),
                'min': np.min(outputs, axis=0),
                'max': np.max(outputs, axis=0)
            }
        }
    
    # Compute statistics for both datasets
    train_stats = compute_dataset_stats(train_dataset, "Training")
    val_stats = compute_dataset_stats(val_dataset, "Validation")
    
    # Compare statistics between datasets
    print("\n=== Normalization Comparison ===")
    
    def compare_stats(train_stat, val_stat, name, dim_names=None):
        if dim_names is None:
            dim_names = [f"Dimension {i}" for i in range(len(train_stat['mean']))]
            
        for i, (train_mean, val_mean, train_std, val_std) in enumerate(zip(
            train_stat['mean'], val_stat['mean'],
            train_stat['std'], val_stat['std']
        )):
            mean_diff = abs(train_mean - val_mean)
            std_diff = abs(train_std - val_std)
            print(f"\n{name} - {dim_names[i]}:")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Std difference: {std_diff:.6f}")
            if mean_diff > 0.1 or std_diff > 0.1:
                print(f"  ⚠️ WARNING: Large difference detected in {name} - {dim_names[i]} statistics!")
    
    # Compare branch inputs (25 features)
    branch_dim_names = [f"Feature {i}" for i in range(25)]
    compare_stats(train_stats['branch'], val_stats['branch'], "Branch Inputs", branch_dim_names)
    
    # Compare trunk inputs (3 spatial coordinates)
    trunk_dim_names = [f"Coordinate {i}" for i in range(3)]
    compare_stats(train_stats['trunk'], val_stats['trunk'], "Trunk Inputs", trunk_dim_names)
    
    # Compare outputs (3 variables)
    output_dim_names = ['Water Depth (h)', 'X-Velocity (u)', 'Y-Velocity (v)']
    compare_stats(train_stats['output'], val_stats['output'], "Outputs", output_dim_names)
    
    return {
        'train_stats': train_stats,
        'val_stats': val_stats
    }

