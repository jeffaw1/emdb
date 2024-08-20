"""
Input format specification for model evaluation.
"""

def validate_input_format(sequence_roots):
    """
    Validate the input format for the given sequence roots.
    
    Args:
    sequence_roots (list): List of paths to sequence root directories.
    
    Raises:
    ValueError: If the input format is invalid.
    """
    for root in sequence_roots:
        # Check for required files and their formats
        # This is a placeholder and should be replaced with actual checks
        required_files = ['poses.pkl', 'camera.json', 'joints.npy']
        for file in required_files:
            if not os.path.exists(os.path.join(root, file)):
                raise ValueError(f"Missing required file {file} in {root}")
    
    print("Input format validation successful.")

# Add more specific format checking functions as needed
