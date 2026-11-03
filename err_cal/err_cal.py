import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh
import torch
from pytorch3d.structures import Pointclouds
from glob import glob

def load_normalization_params(config_dir):
    """
    Load normalization parameters from saved norm.pt file
    """
    try:
        norm_dict = torch.load(os.path.join(config_dir, "norm.pt"))
        mean = norm_dict['mean']
        std = norm_dict.get('std', None)  # Handle mean_subtraction_only case
        return mean, std
    except Exception as e:
        print(f"Error loading normalization parameters: {str(e)}")
        return None, None

def normalize_vertices(vertices, mean, std=None):
    """
    Normalize vertices using the same approach as during training
    """
    vertices = vertices - mean
    if std is not None:
        vertices = vertices / std
    return vertices

def load_mesh(file_path):
    try:
        mesh = trimesh.load(file_path)
        print(f"Successfully loaded mesh from {file_path}")
        print(f"Mesh vertices shape: {mesh.vertices.shape}")
        return mesh.vertices
    except Exception as e:
        print(f"Error loading mesh from {file_path}: {str(e)}")
        return None

def calculate_errors(pred, gt, mean=None, std=None):
    """
    Calculate errors in normalized space if normalization parameters are provided
    """
    if mean is not None:
        # Convert to torch tensors for consistent normalization
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        
        # Normalize both predictions and ground truth
        pred = normalize_vertices(pred, mean, std)
        gt = normalize_vertices(gt, mean, std)
        
        # Convert back to numpy for error calculation
        pred = pred.numpy()
        gt = gt.numpy()
    
    error = np.linalg.norm(pred - gt, axis=1)
    mean_error = np.mean(error)
    std_error = np.std(error)
    median_error = np.median(error)
    return mean_error, std_error, median_error, error

def calculate_l1_error(pred, gt, mean=None, std=None):
    if mean is not None:
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        pred = normalize_vertices(pred, mean, std)
        gt = normalize_vertices(gt, mean, std)
        pred = pred.numpy()
        gt = gt.numpy()
    
    error = np.abs(pred - gt)
    return np.mean(error)

def calculate_l2_error(pred, gt, mean=None, std=None):
    if mean is not None:
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        pred = normalize_vertices(pred, mean, std)
        gt = normalize_vertices(gt, mean, std)
        pred = pred.numpy()
        gt = gt.numpy()
    
    error = np.square(pred - gt)
    return np.sqrt(np.mean(error))

# Define file paths
input_folder = "D:\COURSAI\3DGeoMeshNet\Results\COMA\Coma_26_Mar_Feast\input"
output_folder = "D:\COURSAI\3DGeoMeshNet\Results\COMA\Coma_26_Mar_Feast\output"
save_path = "D:\COURSAI\3DGeoMeshNet\Results\COMA\Coma_26_Mar_Feast\error_analysis"
config_dir = "D:\COURSAI\3DGeoMeshNet\Results\COMA\Coma_26_Mar_Feast\Coma_21_Mar_GAT2"  # Add path to directory containing norm.pt

# Load normalization parameters
mean, std = load_normalization_params(config_dir)
if mean is not None:
    print("Loaded normalization parameters")
    print(f"Using {'mean-only' if std is None else 'mean-std'} normalization")

# Create save directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Get all input and output files
input_files = glob(os.path.join(input_folder, '*.ply'))
output_files = glob(os.path.join(output_folder, 'completed_*.ply'))

# Initialize lists to store errors
mean_errors = []
std_errors = []
median_errors = []
all_point_errors = []
l1_errors = []
l2_errors = []
max_errors = []
min_errors = []

# Process each file
for input_file in input_files:
    base_name = os.path.basename(input_file)
    output_file = os.path.join(output_folder, f"completed_{base_name}")
    
    print(f"Processing {base_name}...")
    
    # Load meshes
    gt_vertices = load_mesh(input_file)
    pred_vertices = load_mesh(output_file)
    
    if gt_vertices is None or pred_vertices is None:
        print(f"Error: Failed to load meshes for {base_name}. Skipping.")
        continue
    
    # Convert to mm if necessary (assuming the original is in meters)
    gt_vertices *= 1000
    pred_vertices *= 1000
    
    # Scale mean and std to mm as well if they exist
    if mean is not None:
        mean_mm = mean.clone() * 1000
        std_mm = std.clone() * 1000 if std is not None else None
    else:
        mean_mm = None
        std_mm = None
    
    # Calculate errors in normalized space
    mean_error, std_error, median_error, point_errors = calculate_errors(
        pred_vertices, gt_vertices, mean_mm, std_mm
    )
    l1_error = calculate_l1_error(pred_vertices, gt_vertices, mean_mm, std_mm)
    l2_error = calculate_l2_error(pred_vertices, gt_vertices, mean_mm, std_mm)
    
    # Store errors
    mean_errors.append(mean_error)
    std_errors.append(std_error)
    median_errors.append(median_error)
    all_point_errors.extend(point_errors)
    l1_errors.append(l1_error)
    l2_errors.append(l2_error)
    max_errors.append(np.max(point_errors))
    min_errors.append(np.min(point_errors))

# Calculate and save statistics (rest of the code remains the same)
print(f"Processed {base_name}")

# Calculate average errors
avg_mean_error = np.mean(mean_errors)
avg_std_error = np.mean(std_errors)
avg_median_error = np.mean(median_errors)
#avg_chamfer_dist = np.mean(chamfer_dists)
avg_l1_error = np.mean(l1_errors)
avg_l2_error = np.mean(l2_errors)
avg_max_error = np.mean(max_errors)
avg_min_error = np.mean(min_errors)

# Print and save average error statistics
print(f'\nError Statistics:')
print(f'Mean Error: {avg_mean_error:.3f} ± {avg_std_error:.3f} mm')
print(f'Median Error: {avg_median_error:.3f} mm')
#print(f'Chamfer Distance: {avg_chamfer_dist:.6f}')
print(f'L1 Error: {avg_l1_error:.6f} mm')
print(f'L2 Error: {avg_l2_error:.6f} mm')
print(f'Average Max Error: {avg_max_error:.3f} mm')
print(f'Average Min Error: {avg_min_error:.3f} mm')

# Create error distribution histogram
plt.figure(figsize=(10, 6))
plt.hist(all_point_errors, bins=50, edgecolor='black')
plt.title('Point-wise Error Distribution')
plt.xlabel('Error (mm)')
plt.ylabel('Number of Points')
plt.savefig(os.path.join(save_path, 'error_distribution.png'))
plt.close()

# Save detailed error statistics
with open(os.path.join(save_path, 'error_stats.txt'), 'w') as f:
    f.write(f'Error Statistics:\n')
    f.write(f'Mean Error: {avg_mean_error:.3f} ± {avg_std_error:.3f} mm\n')
    f.write(f'Median Error: {avg_median_error:.3f} mm\n')
    #f.write(f'Chamfer Distance: {avg_chamfer_dist:.6f}\n')
    f.write(f'L1 Error: {avg_l1_error:.6f} mm\n')
    f.write(f'L2 Error: {avg_l2_error:.6f} mm\n')
    f.write(f'Average Max Error: {avg_max_error:.3f} mm\n')
    f.write(f'Average Min Error: {avg_min_error:.3f} mm\n')

print(f"\nProcessing completed. Results saved to {save_path}")




#################################################### without trainniai normalization ####################################################
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import trimesh
# import torch
# #from pytorch3d.loss import chamfer_distance
# from pytorch3d.structures import Pointclouds
# from glob import glob

# def load_mesh(file_path):
#     try:
#         mesh = trimesh.load(file_path)
#         print(f"Successfully loaded mesh from {file_path}")
#         print(f"Mesh vertices shape: {mesh.vertices.shape}")
#         return mesh.vertices
#     except Exception as e:
#         print(f"Error loading mesh from {file_path}: {str(e)}")
#         return None

# def calculate_errors(pred, gt):
#     error = np.linalg.norm(pred - gt, axis=1)
#     mean_error = np.mean(error)
#     std_error = np.std(error)
#     median_error = np.median(error)
#     return mean_error, std_error, median_error, error

# # def calculate_chamfer_distance(pred, gt):
# #     pred_tensor = torch.tensor(pred, dtype=torch.float32).unsqueeze(0)
# #     gt_tensor = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)
# #     pred_pcl = Pointclouds(points=pred_tensor)
# #     gt_pcl = Pointclouds(points=gt_tensor)
# #     chamfer_dist, _ = chamfer_distance(pred_pcl, gt_pcl)
# #     return chamfer_dist.item()

# def calculate_l1_error(pred, gt):
#     error = np.abs(pred - gt)
#     return np.mean(error)

# def calculate_l2_error(pred, gt):
#     error = np.square(pred - gt)
#     return np.sqrt(np.mean(error))

# # Define file paths
# input_folder = "/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Results/9.Dec19_normalizing/input"
# output_folder = "/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Results/9.Dec19_normalizing/output"
# save_path = "/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Results/9.Dec19_normalizing/error_analysis"

# # Create save directory if it doesn't exist
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# # Get all input and output files
# input_files = glob(os.path.join(input_folder, '*.ply'))
# output_files = glob(os.path.join(output_folder, 're_*.ply'))

# # Print the number of files found
# print(f"Number of input files found: {len(input_files)}")
# print(f"Number of output files found: {len(output_files)}")

# # Check if files were found
# if len(input_files) == 0:
#     print("Error: No input files found.")
#     exit()
# if len(output_files) == 0:
#     print("Error: No output files found.")
#     exit()

# # Initialize lists to store errors
# mean_errors = []
# std_errors = []
# median_errors = []
# all_point_errors = []
# chamfer_dists = []
# l1_errors = []
# l2_errors = []
# max_errors = []
# min_errors = []

# # Process each file
# for input_file in input_files:
#     base_name = os.path.basename(input_file)
#     output_file = os.path.join(output_folder, f"re_{base_name}")
    
#     print(f"Processing {base_name}...")
    
#     # Load meshes
#     gt_vertices = load_mesh(input_file)
#     pred_vertices = load_mesh(output_file)
    
#     if gt_vertices is None or pred_vertices is None:
#         print(f"Error: Failed to load meshes for {base_name}. Skipping.")
#         continue
    
#     # Ensure the meshes have the same shape
#     if gt_vertices.shape != pred_vertices.shape:
#         print(f"Error: Mesh shapes do not match for {base_name}. Skipping.")
#         continue
    
#     # Convert to mm if necessary (assuming the original is in meters)
#     gt_vertices *= 1000
#     pred_vertices *= 1000
    
#     # Calculate errors
#     mean_error, std_error, median_error, point_errors = calculate_errors(pred_vertices, gt_vertices)
#     #chamfer_dist = calculate_chamfer_distance(pred_vertices, gt_vertices)
#     l1_error = calculate_l1_error(pred_vertices, gt_vertices)
#     l2_error = calculate_l2_error(pred_vertices, gt_vertices)
    
#     # Store errors
#     mean_errors.append(mean_error)
#     std_errors.append(std_error)
#     median_errors.append(median_error)
#     all_point_errors.extend(point_errors)
#     #chamfer_dists.append(chamfer_dist)
#     l1_errors.append(l1_error)
#     l2_errors.append(l2_error)
#     max_errors.append(np.max(point_errors))
#     min_errors.append(np.min(point_errors))
    
#     print(f"Processed {base_name}")

# # Calculate average errors
# avg_mean_error = np.mean(mean_errors)
# avg_std_error = np.mean(std_errors)
# avg_median_error = np.mean(median_errors)
# #avg_chamfer_dist = np.mean(chamfer_dists)
# avg_l1_error = np.mean(l1_errors)
# avg_l2_error = np.mean(l2_errors)
# avg_max_error = np.mean(max_errors)
# avg_min_error = np.mean(min_errors)

# # Print and save average error statistics
# print(f'\nError Statistics:')
# print(f'Mean Error: {avg_mean_error:.3f} ± {avg_std_error:.3f} mm')
# print(f'Median Error: {avg_median_error:.3f} mm')
# #print(f'Chamfer Distance: {avg_chamfer_dist:.6f}')
# print(f'L1 Error: {avg_l1_error:.6f} mm')
# print(f'L2 Error: {avg_l2_error:.6f} mm')
# print(f'Average Max Error: {avg_max_error:.3f} mm')
# print(f'Average Min Error: {avg_min_error:.3f} mm')

# # Create error distribution histogram
# plt.figure(figsize=(10, 6))
# plt.hist(all_point_errors, bins=50, edgecolor='black')
# plt.title('Point-wise Error Distribution')
# plt.xlabel('Error (mm)')
# plt.ylabel('Number of Points')
# plt.savefig(os.path.join(save_path, 'error_distribution.png'))
# plt.close()

# # Save detailed error statistics
# with open(os.path.join(save_path, 'error_stats.txt'), 'w') as f:
#     f.write(f'Error Statistics:\n')
#     f.write(f'Mean Error: {avg_mean_error:.3f} ± {avg_std_error:.3f} mm\n')
#     f.write(f'Median Error: {avg_median_error:.3f} mm\n')
#     #f.write(f'Chamfer Distance: {avg_chamfer_dist:.6f}\n')
#     f.write(f'L1 Error: {avg_l1_error:.6f} mm\n')
#     f.write(f'L2 Error: {avg_l2_error:.6f} mm\n')
#     f.write(f'Average Max Error: {avg_max_error:.3f} mm\n')
#     f.write(f'Average Min Error: {avg_min_error:.3f} mm\n')

# print(f"\nProcessing completed. Results saved to {save_path}")


