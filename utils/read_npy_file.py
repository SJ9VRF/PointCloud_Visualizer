import numpy as np

def read_npy_file(file_path):
    try:
        # Load the .npy file
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
file_path = 'your_file.npy'
data = read_npy_file(file_path)
if data is not None:
    print("Data loaded successfully:")
    print(data)
