import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_depth(depth_map, max_depth=None):
    """
    Visualize depth map
    :param depth_map: Input depth map (single channel)
    :param max_depth: Optional parameter, specify the maximum depth value for normalization
    :return: Visualized color depth map
    """

    # Use maximum value of depth map if max_depth is not specified
    if max_depth is None:
        max_depth = np.max(depth_map)
    
    # Normalize depth values to 0-255 range and convert to 8-bit unsigned integer
    depth_vis = np.clip(depth_map, 0, max_depth)  # Limit max depth
    depth_vis = (depth_vis / max_depth * 255).astype(np.uint8)
    
    # Apply color mapping (JET colormap is used here)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_colormap

def depth_canny(depth, threshold1=0, threshold2=10):
    # 1. Remove invalid depth values (e.g., 0 or nan):
    depth_valid = np.where((depth > 0) & np.isfinite(depth), depth, 0)

    # 2. Normalize to [0, 255]
    depth_norm = cv2.normalize(depth_valid, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    blurred = cv2.GaussianBlur(depth_uint8, (5, 5), sigmaX=1.0)

    # ✅ 4. Apply Canny edge detection (thresholds can be adjusted lower)
    edges = cv2.Canny(blurred, threshold1=0, threshold2=threshold2)

    return edges        

def visualize_depth_video(
    depth_array: np.ndarray, 
    output_path: str, 
    fps: int = 30,
    cmap_name: str = 'jet',
    invalid_depth_value: float = 0.0 # Used to fill invalid/zero depth
):
    """
    Visualize an NxCxHxW depth image array as a normalized heatmap video.
    Images from different cameras are stitched horizontally in a row.

    Args:
        depth_array: np.ndarray, depth data with shape (N, C, H, W).
        output_path: str, path to the output video file (e.g., "output.mp4").
        fps: int, video frame rate.
        cmap_name: str, Matplotlib colormap name (e.g., 'viridis', 'jet').
        invalid_depth_value: float, value used to replace invalid/zero depth before normalization.
    """
    if depth_array.ndim != 4:
        raise ValueError("depth_array must be (N, C, H, W).")

    # 1. Ensure data is on CPU and converted to NumPy 
    # Move all tensors from GPU/CPU to CPU and convert to float32 (if not already)
    depth_np = depth_array
    
    N, C, H, W = depth_np.shape
    
    # 2. Video encoder settings
    # Total width of output video is W * C, height is H
    video_width = W * C
    video_height = H
    
    # Use mp4v encoder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    
    # 3. Color mapper (get from Matplotlib)
    cmap = plt.get_cmap(cmap_name)

    # 4. **Global Normalization** (Crucial step)
    # Find valid depth min/max for the entire dataset (all frames, all cameras)
    valid_depths = depth_np[depth_np > 0] # Exclude zero or negative depth
    
    if valid_depths.size == 0:
        print("No valid depth, return.")
        return

    min_depth = valid_depths.min()
    max_depth = valid_depths.max()
    
    if max_depth == min_depth:
        print("max_depth == min_depth, return.")
        return

    # --- 5. Frame-by-frame processing and writing ---
    for i in range(N):
        frame_imgs = [] # Store RGB images for C cameras in the current frame
        
        # Process each camera
        for c in range(C):
            depth_map = depth_np[i, c] # (H, W)

            # 5.1 Data cleaning and normalization (using global Min/Max)
            # Replace zero values with min_depth (or specified invalid value)
            depth_map_normalized = depth_map.copy()
            depth_map_normalized[depth_map_normalized <= 0] = invalid_depth_value

            # Linearly map depth map to [0, 1]
            depth_map_normalized = (depth_map_normalized - min_depth) / (max_depth - min_depth)
            
            # Clip to [0, 1] range to prevent floating-point errors
            depth_map_normalized = np.clip(depth_map_normalized, 0, 1)

            # 5.2 Convert to heatmap RGB (Matplotlib output is RGBA 0-1 float)
            # *Note*: Matplotlib's cmap(X) returns RGBA float array [0, 1]
            rgb_float_map = cmap(depth_map_normalized)[:, :, :3] # Take RGB

            # Convert to BGR uint8 format expected by OpenCV
            rgb_uint8_map = (rgb_float_map * 255).astype(np.uint8)
            bgr_uint8_map = cv2.cvtColor(rgb_uint8_map, cv2.COLOR_RGB2BGR)

            frame_imgs.append(bgr_uint8_map)

        # 5.3 Image stitching (Horizontal stack)
        # Resulting shape: (H, W*C, 3)
        stitched_frame = np.hstack(frame_imgs)

        # 5.4 Write to video
        out.write(stitched_frame)

    # 6. Release resources
    out.release()

def visualize_pre_computed_canny_video(
    canny_array: np.ndarray, 
    output_path: str, 
    fps: int = 30,
    edge_value: int = 255 # Intensity value for edge pixels (typically 255)
):
    """
    Visualize an NxCxHxW pre-computed Canny edge map array as a video.
    Edge maps from different cameras are stitched horizontally in a row.

    Args:
        canny_array: np.ndarray, binary Canny edge map with shape (N, C, H, W).
        output_path: str, path to the output video file (e.g., "final_canny_video.mp4").
        fps: int, video frame rate.
        edge_value: int, intensity value for edge pixels (0-255).
    """
    if canny_array.ndim != 4:
        raise ValueError(f"Input array shape must be (N, C, H, W), but got {canny_array.shape}.")
    
    # Ensure array is uint8, as OpenCV video encoders typically require this
    # Canny results are often 0/1 float/int, scale to 0/255 uint8
    if canny_array.max() <= 1:
        # Scale 0/1 float or integer to 0/255
        canny_np_all = (canny_array * edge_value).astype(np.uint8)
    else:
        # Otherwise assume it's already in 0-255 range
        canny_np_all = canny_array.astype(np.uint8)
        
    N, C, H, W = canny_np_all.shape
    
    # 1. Video encoder settings
    video_width = W * C
    video_height = H
    
    # Use mp4v encoder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # isColor=True, because we output 3-channel BGR images
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height), isColor=True)

    # --- 2. Frame-by-frame processing and writing ---
    for i in range(N):
        frame_imgs_bgr = [] # Store BGR images for C cameras in the current frame
        
        # Process each camera
        for c in range(C):
            edge_map = canny_np_all[i, c] # (H, W) uint8 edge map

            # Convert to BGR color image (for video writing)
            # Canny output is single-channel grayscale, needs to be converted to 3-channel BGR
            bgr_edge_map = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

            frame_imgs_bgr.append(bgr_edge_map)

        # 3. Image stitching (Horizontal stack)
        # Resulting shape: (H, W*C, 3)
        stitched_frame = np.hstack(frame_imgs_bgr) 

        # 4. Write to video
        out.write(stitched_frame)

    # 5. Release resources
    out.release()