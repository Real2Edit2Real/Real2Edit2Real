import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import os

if __name__ == "__main__":
    os.makedirs("data/vis_joint", exist_ok=True)
    for data_batch_id in ["mug_to_box"]:
        parent_data_path = os.path.join("data", "generated_data", data_batch_id)
        for job_id in sorted(os.listdir(parent_data_path)):
            for machine_id in sorted(os.listdir(os.path.join(parent_data_path, job_id))):
                for data_id in sorted(os.listdir(os.path.join(parent_data_path, job_id, machine_id))):
                    for episode_id in tqdm(sorted(os.listdir(os.path.join(parent_data_path, job_id, machine_id, data_id)))): 
                        aligned_joints = h5py.File(os.path.join(parent_data_path, job_id, machine_id, data_id, episode_id, "aligned_joints.h5"))    
                        data_matrix = aligned_joints['state']['joint']['position']
                        
                        # Set chart size
                        plt.figure(figsize=(12, 7))

                        # Draw line chart
                        for i in range(data_matrix.shape[1]):
                            plt.plot(data_matrix[:, i], label=f'{"Left" if i // 7 == 0 else "Right"} Arm Joint {i % 7}')

                        # Set title and labels
                        plt.title(f'Joints Visualization')
                        plt.xlabel('Frame')
                        plt.ylabel('Angle')

                        # Show legend and place it outside the chart at the top right
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                        # Enable grid lines
                        plt.grid(True, linestyle='--', alpha=0.6)

                        # Adjust layout to ensure the legend and title are not cropped
                        plt.tight_layout()

                        # --- 3. Save the chart (critical step) ---
                        # Use plt.savefig() to save the file
                        # Filenames can include extensions like .png, .jpg, .pdf, .svg, etc.
                        # bbox_inches='tight' is used to remove extra whitespace around the chart
                        # dpi=300 sets a higher resolution (optional, but recommended for high-quality output)
                        # plt.savefig(os.path.join(parent_data_path, job_id, machine_id, data_id, episode_id, "vis_joints.jpg"),
                        #             bbox_inches='tight', 
                        #             dpi=300) 
                        plt.savefig(os.path.join("data/vis_joint", f"vis_joints_{episode_id}.jpg"),
                                    bbox_inches='tight', 
                                    dpi=300) 
                        plt.close()
