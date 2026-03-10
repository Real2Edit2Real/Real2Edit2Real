import h5py
import numpy as np

def extrinsic_to_json(extrinsic):
    """
    Convert 4x4 camera extrinsic matrix to a JSON-formatted dictionary
    Args:
        extrinsic: np.ndarray, shape (4, 4)
    Returns:
        dict: {
            "extrinsic": {
                "rotation_matrix": [[...], [...], [...]],
                "translation_vector": [...]
            }
        }
    """
    assert extrinsic.shape == (4, 4), "Extrinsic matrix must be 4x4"

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    extrinsic_dict = {
        "extrinsic": {
            "rotation_matrix": R.tolist(),
            "translation_vector": t.tolist()
        }
    }

    return extrinsic_dict

def h5_to_dict(obj):
    """
    Recursively read groups and datasets in .h5 file and convert to Python dict
    """
    result = {}
    if isinstance(obj, h5py.Group):
        for key, item in obj.items():
            result[key] = h5_to_dict(item)
    elif isinstance(obj, h5py.Dataset):
        data = obj[()]
        # Try to convert to JSON-compatible type
        if isinstance(data, bytes):
            result = data.decode('utf-8')
        elif np.isscalar(data):
            result = data.item()
        else:
            result = data.tolist()
    return result

def dict_to_h5(h5file, data):
    """
    Recursively save a Python dict to an open h5py file or group.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            # Create a new group and write recursively
            subgroup = h5file.create_group(key)
            dict_to_h5(subgroup, value)
        else:
            # Save dataset
            # Handle Python scalars, lists, strings, numpy arrays, etc.
            if isinstance(value, (list, tuple)):
                pass
            if isinstance(value, str):
                pass
            h5file.create_dataset(key, data=value)

def save_dict_to_h5(data_dict, h5_path):
    """
    Save a Python dict as an HDF5 file.
    """
    with h5py.File(h5_path, 'w') as h5file:
        dict_to_h5(h5file, data_dict)
