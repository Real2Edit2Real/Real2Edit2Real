import numpy as np
import plotly.graph_objects as go

def create_bbox_mesh_data(bbox):
    """
    Generate 3D coordinates for drawing a wireframe based on the bounding box [[x_min, y_min, z_min], [x_max, y_max, z_max]].
    """
    x_min, y_min, z_min = bbox[0]
    x_max, y_max, z_max = bbox[1]

    # Define the 8 vertices of the bounding box
    verts = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], 
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], 
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])

    # Define indices connecting the 12 edges
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges
    ]

    # Flatten coordinates and insert None to create discontinuous segments
    x_coords, y_coords, z_coords = [], [], []
    for i, j in lines:
        x_coords.extend([verts[i, 0], verts[j, 0], None])
        y_coords.extend([verts[i, 1], verts[j, 1], None])
        z_coords.extend([verts[i, 2], verts[j, 2], None])

    return x_coords, y_coords, z_coords

def visualize_pcd_and_bbox(pcd, bbox_list, port=8848):
    """
    Visualize point cloud and bounding boxes using Plotly. Compatible with (n, 3) or (n, 6) point clouds.

    Args:
        pcd (np.array): Point cloud data, shape (n, 3) [x, y, z] or (n, 6) [x, y, z, r, g, b].
        bbox_list (list): List of bounding boxes, each with shape (2, 3).
    """
    if pcd.shape[1] < 3:
        raise ValueError("Point cloud data `pcd` must have at least 3 columns (x, y, z).")
    
    fig = go.Figure()
    n_cols = pcd.shape[1]
    
    # --- 1. Handle point cloud color ---
    if n_cols >= 6:
        # Has color (n, 6)
        color_data = pcd[:, 3:6].astype(int)
        colors_rgb = [f'rgb({r},{g},{b})' for r, g, b in color_data]
        point_cloud_name = 'Point Cloud (Color)'
    else:
        # No color (n, 3)
        colors_rgb = 'gray' # Use default color
        point_cloud_name = 'Point Cloud (No Color)'

    # --- 2. Plot point cloud ---
    pcd_trace = go.Scatter3d(
        x=pcd[:, 0],
        y=pcd[:, 1],
        z=pcd[:, 2],
        mode='markers',
        name=point_cloud_name,
        marker=dict(
            size=2,  # Point size
            color=colors_rgb,
        )
    )
    fig.add_trace(pcd_trace)

    # --- 3. Plot bounding boxes ---
    bbox_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan'] 
    
    for i, bbox in enumerate(bbox_list):
        # Wireframe data for the bounding box
        x_coords, y_coords, z_coords = create_bbox_mesh_data(bbox)
        
        bbox_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            name=f'BBox {i+1}',
            line=dict(
                color=bbox_colors[i % len(bbox_colors)], 
                width=5
            )
        )
        fig.add_trace(bbox_trace)
        
    
    # --- 4. Configure layout and display ---
    fig.update_layout(
        title='Point Cloud and Bounding Boxes Visualization',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            # Maintain aspect ratio to prevent distortion
            aspectmode='data', 
        )
    )
    
    fig.write_html("debug.html")