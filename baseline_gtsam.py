import gtsam
import open3d as o3d
import numpy as np


def vector6(x, y, z, a, b, c):
    """Create 6d double numpy array."""
    return np.array([x, y, z, a, b, c], dtype=float)


# Convert GTSAM Pose3 objects to list of translations for plotting
def extract_translations_from_gtsam_values(poses_values):
    return [(poses_values.atPose3(i).translation()) for i in range(poses_values.size())]


# Define the plot function (from your reference)
def plot_two_poses_with_edges_open3d(
    initial_poses_list, optimized_poses_list, edges, skip=1
):
    """
    Visualize two lists of Pose objects as point clouds using Open3D, with edges between specific poses
    in the initial_poses_list, and include XYZ axes.

    Parameters:
        initial_poses_list (list of Pose): First list of Pose objects (initial poses).
        optimized_poses_list (list of Pose): Second list of Pose objects (optimized poses).
        edges (list of dict): List of edges, where each edge contains "i", "j" for the indices and
                              a "measurement" for the pose transformation between them.
        skip (int, optional): Plot every 'skip' poses (default is 1, which plots all poses).
    """

    # Helper function to convert pose list to Open3D point cloud
    def poses_to_point_cloud(pose_list, color, skip=1):
        points = [pose for idx, pose in enumerate(pose_list) if idx % skip == 0]
        points_np = np.array(points)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        # Set point colors
        colors_np = np.tile(color, (points_np.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors_np)

        return pcd

    # Create two point clouds with different colors
    pcd_initial = poses_to_point_cloud(
        initial_poses_list, color=[1, 0, 0], skip=skip
    )  # Red for initial poses
    pcd_optimized = poses_to_point_cloud(
        optimized_poses_list, color=[0, 1, 0], skip=skip
    )  # Green for optimized poses

    # Create lines for edges between poses in the initial_poses_list
    lines = []
    line_colors = []

    for edge in edges:
        from_idx = edge["i"]
        to_idx = edge["j"]

        # Only plot edges if they exist in the range of the initial pose list
        if from_idx < len(initial_poses_list) and to_idx < len(initial_poses_list):
            # Line connects the from and to poses
            line = [from_idx, to_idx]
            lines.append(line)
            line_colors.append([0, 0, 1])  # Blue color for the lines

    # Create Open3D LineSet for the edges in the initial poses
    if lines:
        # Use the points from the initial poses as vertices
        line_set = o3d.geometry.LineSet()
        points = np.array(optimized_poses_list)  # Points from optimized poses
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
    else:
        line_set = None

    # Create XYZ axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # Visualize both point clouds, lines, and axes
    geometries = [pcd_initial, pcd_optimized, axes]
    if line_set:
        geometries.append(line_set)

    o3d.visualization.draw_geometries(
        geometries,
        zoom=0.8,
        front=[-0.4999, -0.1659, -0.8499],
        lookat=[0, 0, 0],
        up=[0.1204, -0.9852, 0.1215],
    )


# Helper function to read .g2o file
def read_g2o_file(file_path):
    poses = []
    edges = []

    with open(file_path, "r") as f:
        for line in f:
            data = line.strip().split()
            if data[0] == "VERTEX_SE3:QUAT":
                # Parse pose (node)
                node_id = int(data[1])
                x, y, z = float(data[2]), float(data[3]), float(data[4])
                qx, qy, qz, qw = (
                    float(data[5]),
                    float(data[6]),
                    float(data[7]),
                    float(data[8]),
                )
                pose = gtsam.Pose3(
                    gtsam.Rot3.Quaternion(qw, qx, qy, qz), gtsam.Point3(x, y, z)
                )
                poses.append((node_id, pose))
            elif data[0] == "EDGE_SE3:QUAT":
                # Parse edge (constraint)
                id_from = int(data[1])
                id_to = int(data[2])
                x, y, z = float(data[3]), float(data[4]), float(data[5])
                qx, qy, qz, qw = (
                    float(data[6]),
                    float(data[7]),
                    float(data[8]),
                    float(data[9]),
                )
                # Parse the upper triangular information matrix
                upper_triangular = np.array([float(i) for i in data[10:]])

                # Reshape to the upper triangular form (size 6x6)
                upper_triangular_matrix = np.zeros((6, 6))
                upper_triangular_matrix[np.triu_indices(6)] = upper_triangular

                # Make it symmetric
                information_matrix = (
                    upper_triangular_matrix
                    + upper_triangular_matrix.T
                    - np.diag(upper_triangular_matrix.diagonal())
                )

                edge_pose = gtsam.Pose3(
                    gtsam.Rot3.Quaternion(qw, qx, qy, qz), gtsam.Point3(x, y, z)
                )
                edges.append((id_from, id_to, edge_pose, information_matrix))

    return poses, edges


# Load the g2o file
dataset_names = [
    "grid3D.g2o",
    "torus3D.g2o",
    "sphere_bignoise_vertex3.g2o",
    "rim.g2o",
    "cubicle.g2o",
    "parking-garage.g2o",
]
dataset_name = dataset_names[-2]
initial_poses, edges = read_g2o_file(f"data/{dataset_name}")

# Create a NonlinearFactorGraph
graph = gtsam.NonlinearFactorGraph()

# Add the prior to the first node
prior_noise = gtsam.noiseModel.Diagonal.Variances(
    vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4)
)
first_pose = initial_poses[0][1]
graph.add(gtsam.PriorFactorPose3(0, first_pose, prior_noise))

# Add edges to graph with or without robust noise model
for id_from, id_to, edge_pose, _ in edges:
    odomModel = gtsam.noiseModel.Diagonal.Variances(
        vector6(1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2)
    )
    loopModel = gtsam.noiseModel.Diagonal.Variances(
        vector6(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    )

    # Add robust noise model for non-consecutive edges
    if abs(id_from - id_to) != 1:
        robust_noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), loopModel
        )
        factor = gtsam.BetweenFactorPose3(id_from, id_to, edge_pose, robust_noise_model)
    else:
        factor = gtsam.BetweenFactorPose3(id_from, id_to, edge_pose, odomModel)

    graph.add(factor)

# Initial estimates (optional: use poses from .g2o as initial guess)
initial_estimate = gtsam.Values()
for node_id, pose in initial_poses:
    initial_estimate.insert(node_id, pose)

# Create Levenberg-Marquardt optimizer parameters
params = gtsam.LevenbergMarquardtParams()
params.setVerbosity("TERMINATION")
params.setVerbosityLM("SUMMARY")
params.setMaxIterations(500)  # Set maximum number of iterations to 500

# Initialize optimizer with verbose output
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
result = optimizer.optimize()

# Visualize the initial and optimized poses
edges = [
    {"i": id_from, "j": id_to, "measurement": edge_pose}
    for id_from, id_to, edge_pose, _ in edges
]
initial_poses_xyz = [pose.translation() for _, pose in initial_poses]
optimized_poses_xyz = extract_translations_from_gtsam_values(result)
plot_two_poses_with_edges_open3d(initial_poses_xyz, optimized_poses_xyz, edges)
