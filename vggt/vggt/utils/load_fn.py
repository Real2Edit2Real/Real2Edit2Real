# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np

def load_and_preprocess_images(image_path_list, mode="crop", intrinsics_list=None):
    """
    Load and preprocess images (crop or pad) and apply same geometric transform to intrinsics.

    Args:
        image_path_list (list): List of image paths or NumPy arrays.
        mode (str): "crop" or "pad"
        intrinsics_list (list or None): List of (3,3) intrinsics matrices corresponding to images.

    Returns:
        images (torch.Tensor): (N, 3, H, W)
        intrinsics (torch.Tensor or None): (N, 3, 3), transformed intrinsics if provided
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    new_intrinsics = []
    to_tensor = TF.ToTensor()
    target_size = 518

    for idx, image_path in enumerate(image_path_list):
        # Load image
        if isinstance(image_path, np.ndarray):
            img = Image.fromarray(image_path)
        else:
            img = Image.open(image_path)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size

        # Resize
        if mode == "pad":
            if width >= height:
                new_width = target_size
                scale = new_width / width
                new_height = round(height * scale / 14) * 14
            else:
                new_height = target_size
                scale = new_height / height
                new_width = round(width * scale / 14) * 14
        else:  # crop
            new_width = target_size
            scale = new_width / width
            new_height = round(height * scale / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        # Adjust intrinsics if provided
        if intrinsics_list is not None:
            K = intrinsics_list[idx].copy()
            K[0, 0] *= scale  # fx
            K[1, 1] *= scale  # fy
            K[0, 2] *= scale  # cx
            K[1, 2] *= scale  # cy

        # Crop (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y:start_y+target_size, :]
            if intrinsics_list is not None:
                K[1, 2] -= start_y  # cy -= crop offset

        # Pad (only in pad mode)
        if mode == "pad":
            h_pad = target_size - img.shape[1]
            w_pad = target_size - img.shape[2]

            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left

            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )

            if intrinsics_list is not None:
                K[0, 2] += pad_left
                K[1, 2] += pad_top

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        if intrinsics_list is not None:
            new_intrinsics.append(torch.from_numpy(K).float())

    # Normalize shape if needed
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(s[0] for s in shapes)
        max_width = max(s[1] for s in shapes)
        padded_images = []
        for img in images:
            h_pad = max_height - img.shape[1]
            w_pad = max_width - img.shape[2]
            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left
            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
            padded_images.append(img)
        images = padded_images
        # You can also modify intrinsics accordingly here if needed

    images = torch.stack(images)

    if intrinsics_list is not None:
        intrinsics_tensor = torch.stack(new_intrinsics)
        return images, intrinsics_tensor
    else:
        intrinsics_tensor = None
        return images

    

"""
def load_and_preprocess_images(image_path_list, mode="crop"):
    
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Check if the input is a NumPy array (frame) or a file path
        if isinstance(image_path, np.ndarray):
            img = Image.fromarray(image_path)  # Convert NumPy array to PIL Image
        elif isinstance(image_path, str):
            img = Image.open(image_path)  # Open image from file path

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
"""
