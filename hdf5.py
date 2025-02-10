import h5py
import numpy as np

def copy_hdf5_data(input_file, output_file):
    """
    Recursively copies data and groups from an existing HDF5 file to a new one without compression.
    """
    def copy_items(parent_input, parent_output):
        for item_name, item in parent_input.items():
            if isinstance(item, h5py.Group):
                # Create a new group in the output file
                new_group = parent_output.create_group(item_name)
                # Recursively copy items within the group
                copy_items(item, new_group)
            elif isinstance(item, h5py.Dataset):
                # Read data from the input dataset
                data = item[()]
                # Create a new dataset in the output file
                parent_output.create_dataset(item_name, data=data)

    with h5py.File(input_file, "r") as input_f:
        # Create a new HDF5 file
        with h5py.File(output_file, "w") as output_f:
            # Start copying from the root group
            copy_items(input_f, output_f)

if __name__ == "__main__":
    input_file_name = "/data/shared/rich/snap_full_252.h5"
    output_file_name = "output_file.hdf5"

    # Call the function to copy data and groups from input to output
    copy_hdf5_data(input_file_name, output_file_name)
    print(f"Data and groups recursively copied from '{input_file_name}' to '{output_file_name}' without compression.")
