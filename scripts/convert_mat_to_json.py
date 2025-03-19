import os
import json
import scipy.io as sio
import numpy as np
import argparse
from pathlib import Path

def mat_to_json(mat_file, output_dir=None):
    """
    Convert a MATLAB .mat file containing propellant data to JSON format
    
    Parameters:
    -----------
    mat_file : str
        Path to the .mat file to convert
    output_dir : str, optional
        Directory to save the JSON file (default: same directory as mat_file)
    """
    try:
        # Load the MATLAB file
        print(f"Loading {mat_file}...")
        mat_data = sio.loadmat(mat_file, simplify_cells=True)
        
        # Check if the file contains 's' structure with propellant data
        if 's' not in mat_data:
            print(f"Error: {mat_file} does not contain propellant data in 's' structure")
            return False
        
        # Extract data from the 's' structure
        s = mat_data['s']
        
        # Create a new dictionary for JSON data
        json_data = {}
        
        # Map MATLAB fields to JSON
        field_map = [
            'prop_nm', 'prop_Pc', 'prop_OF', 'prop_k', 'prop_M', 'prop_T', 
            'prop_Reg', 'prop_Rho', 'opt_OF'
        ]
        
        # Transfer data with proper type conversion
        for field in field_map:
            if field in s:
                # Convert numpy arrays to Python lists for JSON serialization
                if isinstance(s[field], np.ndarray):
                    json_data[field] = s[field].tolist()
                else:
                    json_data[field] = s[field]
            else:
                print(f"Warning: Field '{field}' not found in {mat_file}")
        
        # Determine output file path
        if output_dir is None:
            output_dir = os.path.dirname(mat_file)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the same filename but with .json extension
        base_name = os.path.basename(mat_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{name_without_ext}.json")
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Successfully converted {mat_file} to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error converting {mat_file}: {str(e)}")
        return False

def batch_convert(input_dir, output_dir=None):
    """
    Convert all .mat files in a directory to JSON format
    
    Parameters:
    -----------
    input_dir : str
        Directory containing .mat files
    output_dir : str, optional
        Directory to save JSON files (default: same as input_dir)
    """
    input_path = Path(input_dir)
    
    # List all .mat files in the directory
    mat_files = list(input_path.glob('*.mat'))
    
    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return
    
    print(f"Found {len(mat_files)} .mat files in {input_dir}")
    
    # Convert each file
    success_count = 0
    for mat_file in mat_files:
        if mat_to_json(str(mat_file), output_dir):
            success_count += 1
    
    print(f"Converted {success_count} out of {len(mat_files)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MATLAB .mat propellant files to JSON")
    parser.add_argument("input", help="Input .mat file or directory containing .mat files")
    parser.add_argument("--output", help="Output directory for JSON files")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Convert all files in directory
        batch_convert(args.input, args.output)
    elif input_path.is_file() and input_path.suffix.lower() == '.mat':
        # Convert single file
        mat_to_json(args.input, args.output)
    else:
        print(f"Error: Input {args.input} is not a .mat file or directory")
