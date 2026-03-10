import torch
import torch.serialization
import os
import sys
import numpy as np
from dotenv import load_dotenv
from numpy.dtypes import Float64DType

load_dotenv()
file_path = os.getenv('WINDOWS_CHECKPOINT_ROOT')
file_path += '/' + sys.argv[1]

# Check if the file exists (optional, but good practice)
with torch.serialization.safe_globals([np.dtype, np.core.multiarray._reconstruct, np.core.multiarray.scalar, Float64DType]):
    if os.path.exists(file_path):
        # Load the model state dictionary
        try:
            model_state_dict = torch.load(file_path, map_location=torch.device('cpu')) # Load to CPU
            print("Successfully loaded .pth file.")
            print("\nKeys in the state dictionary:")

            results_file = sys.argv[1]
            results_file = results_file[:len(results_file)-4] + ".txt"
            print(results_file)
            with open("results.txt", "w") as f:
                for key in model_state_dict.keys():
                    print(key, model_state_dict[key]) 
                    
            # Example of how to access a specific weight (uncomment and modify as needed)
            # print("\nExample weight '...' data:")
            # print(model_state_dict['your_specific_key_name']) 

        except Exception as e:
            print(f"Error loading the file: {e}")
    else:
        print(f"Error: The file '{file_path}' was not found.")