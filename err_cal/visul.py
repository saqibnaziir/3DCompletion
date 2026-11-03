import numpy as np
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
import os
from glob import glob

def calculate_vertex_errors(gt_mesh, recon_mesh):
    """Calculate Euclidean distance between corresponding vertices."""
    gt_vertices = gt_mesh.vertices
    recon_vertices = recon_mesh.vertices
    
    # Ensure the meshes have the same number of vertices
    assert gt_vertices.shape == recon_vertices.shape, "Meshes must have the same number of vertices"
    
    # Calculate Euclidean distances
    errors = np.linalg.norm(gt_vertices - recon_vertices, axis=1)
    
    # Print error statistics
    print(f"Error statistics:")
    print(f"  Min: {errors.min():.6f}")
    print(f"  Max: {errors.max():.6f}")
    print(f"  Mean: {errors.mean():.6f}")
    print(f"  Median: {np.median(errors):.6f}")
    
    return errors

def visualize_error(mesh, errors, output_path, cmap="jet"):
    """Visualize the mesh with error colors and save to file without color bar."""
    # Convert trimesh to PyVista mesh
    vertices = mesh.vertices
    faces = mesh.faces
    pv_mesh = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)))
    
    # Add error data to the mesh
    pv_mesh.point_data["Errors"] = errors
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.background_color = 'white'
    
    # Add mesh to the plotter 0.001 0.482                0.047416
    plotter.add_mesh(pv_mesh, scalars="Errors", cmap=cmap, show_edges=False, clim=[0.000000,  0.008523])
    
    # Set camera position for a front view and adjust to show full shape
    #####for normal 
    plotter.camera_position = 'xy'
    ########for ear plotter.camera_position = 'zy'
    plotter.camera.zoom(1.2)  # Reduced zoom to show more of the shape
    
    # Center the mesh in the view
    plotter.reset_camera()
    
    # Save screenshot
    # Save screenshot without color bar
    plotter.remove_scalar_bar()
    plotter.screenshot(output_path, transparent_background=True)
    print(f"Error visualization saved to {output_path}")
    
def process_file_pair(gt_path, recon_path, save_path):
    # Load meshes
    gt_mesh = trimesh.load(gt_path)
    recon_mesh = trimesh.load(recon_path)
    
    # Print mesh information
    print(f"\nProcessing: {os.path.basename(gt_path)}")
    print("Ground Truth Mesh:")
    print(f"  Vertices: {len(gt_mesh.vertices)}")
    print(f"  Faces: {len(gt_mesh.faces)}")
    print("Reconstructed Mesh:")
    print(f"  Vertices: {len(recon_mesh.vertices)}")
    print(f"  Faces: {len(recon_mesh.faces)}")
    
    # Calculate errors
    errors = calculate_vertex_errors(gt_mesh, recon_mesh)
    
    # Visualize and save
    output_filename = f"error_visualization_{os.path.basename(gt_path).replace('.ply', '.png')}"
    output_path = os.path.join(save_path, output_filename)
    visualize_error(gt_mesh, errors, output_path)  # Removed clim parameter

def main():
    # Define paths
    input_folder = "/content/input"
    output_folder = "/content/output"
    save_path = "/content/error_analysis"

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get all input files
    input_files = glob(os.path.join(input_folder, '*.ply'))

    # Process each file pair
    for input_file in input_files:
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_folder, f"pre_processed_{base_name}")
        
        if os.path.exists(output_file):
            process_file_pair(input_file, output_file, save_path)
        else:
            print(f"Warning: No matching output file found for {base_name}")

    print(f"\nProcessing completed. Results saved to {save_path}")

if __name__ == "__main__":
    main()
