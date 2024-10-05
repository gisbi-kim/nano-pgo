import time
import numpy as np

np.set_printoptions(linewidth=np.inf, suppress=True, precision=4)
from scipy.spatial.transform import Rotation

import scipy.sparse as sp
import sksparse.cholmod as cholmod

import symforce.symbolic as sf
from symforce.ops import LieGroupOps

import open3d as o3d
import matplotlib.pyplot as plt

import multiprocessing


def quaternion_to_rotation(qx, qy, qz, qw):
    rotation = Rotation.from_quat([qx, qy, qz, qw])
    return rotation.as_matrix()


def se2_to_se3(x, y, theta):
    rotation = Rotation.from_euler("z", theta)
    R = rotation.as_matrix()
    t = np.array([x, y, 0.0])
    return R, t


def rotmat_to_rotvec(R):
    rotation = Rotation.from_matrix(R)
    rotvec = rotation.as_rotvec()
    return rotvec


def rotvec_to_rotmat(rotvec):
    rotation = Rotation.from_rotvec(rotvec)
    R = rotation.as_matrix()
    return R


def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def plot_two_poses_with_edges_open3d(
    initial_poses_list, optimized_poses_list, edges, skip=1
):
    """
    Visualizes initial and optimized poses along with edges using Open3D.

    Parameters:
        initial_poses_list (list of np.ndarray): List of initial pose translations (3D positions).
        optimized_poses_list (list of np.ndarray): List of optimized pose translations (3D positions).
        edges (list of dict): List of edges, where each edge contains "i", "j" for the indices.
        skip (int, optional): Plot every 'skip' poses (default is 1, which plots all poses).
    """

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

    # Create lines for edges between poses
    lines = []
    line_colors = []

    for edge in edges:
        from_idx = edge["i"]
        to_idx = edge["j"]

        # Only plot edges if indices are valid
        if from_idx < len(optimized_poses_list) and to_idx < len(optimized_poses_list):
            # Line connects the from and to poses
            line = [from_idx, to_idx]
            lines.append(line)
            line_colors.append([0, 0, 1])  # Blue color for the lines

    # Create Open3D LineSet for the edges
    if lines:
        # Creates a LineSet object to represent edges between poses.
        # Use the points from the optimized poses as vertices
        line_set = o3d.geometry.LineSet()
        points = np.array(optimized_poses_list)
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
    else:
        line_set = None

    # Visualize both point clouds, lines, and axes
    geometries = [pcd_initial, pcd_optimized]
    if line_set:
        geometries.append(line_set)

    o3d.visualization.draw_geometries(
        geometries,
        zoom=0.8,
        front=[0, 0, 1],  # top view
        lookat=initial_poses_list[-1],
        up=[0, 1, 0],
    )


#
# TODO: make these non-global (but must be called "once")
#

epsilon = 1e-7

# Define rotation variables (rotation vectors for each axis)
sf_ri = sf.V3.symbolic("ri")  # Rotation of pose_i
sf_rj = sf.V3.symbolic("rj")  # Rotation of pose_j
sf_rij = sf.V3.symbolic("rij")  # Measured relative rotation

# Define translation variables
sf_ti = sf.V3.symbolic("ti")  # Translation of pose_i
sf_tj = sf.V3.symbolic("tj")  # Translation of pose_j
sf_tij = sf.V3.symbolic("tij")  # Measured relative translation

# Create rotation matrices using Lie Group operations
sf_Ri = LieGroupOps.from_tangent(sf.Rot3, sf_ri)
sf_Rj = LieGroupOps.from_tangent(sf.Rot3, sf_rj)
sf_Rij = LieGroupOps.from_tangent(sf.Rot3, sf_rij)

# Construct SE(3) containers 
sf_Ti = sf.Pose3(R=sf_Ri, t=sf_ti)
sf_Tj = sf.Pose3(R=sf_Rj, t=sf_tj)
sf_Tij = sf.Pose3(R=sf_Rij, t=sf_tij)

# SE3 error: T_err = T_ij^{-1} * T_i^{-1} * T_j
sf_T_err = sf_Tij.inverse() * (sf_Ti.inverse() * sf_Tj)

# Convert SE3 error to a tangent vector [r, t], 6-dim.
#  NOTE: symforce uses the [r, t] order, not [t, r].
sf_se3_err = sf.Matrix(sf_T_err.to_tangent())

# Define residual as the rotation and translation error
sf_residual = sf_se3_err  # 6D vector

# Compute the full Jacobian
sf_J_ti = sf_residual.jacobian([sf_ti])  # 6 x 3
sf_J_ri = sf_residual.jacobian([sf_ri])  # 6 x 3
sf_J_tj = sf_residual.jacobian([sf_tj])  # 6 x 3
sf_J_rj = sf_residual.jacobian([sf_rj])  # 6 x 3


def between_factor_jacobian_by_symforce(pose_i, pose_j, pose_ij_meas):
    """
    Computes the Jacobians for the between factor residual using Symforce symbolic computation.

    Parameters:
        pose_i (dict): Dictionary containing rotation vector 'r' and translation 't' for pose i.
        pose_j (dict): Dictionary containing rotation vector 'r' and translation 't' for pose j.
        pose_ij_meas (dict): Dictionary containing the measured relative rotation matrix 'R' and translation vector 't'.

    Returns:
        Ji (np.ndarray): 6x6 Jacobian matrix with respect to pose i.
        Jj (np.ndarray): 6x6 Jacobian matrix with respect to pose j.

    Note: the Ji and Jj should have shapes (6, 6) like:
                                        |  translation_variable (3-dim), rotation_variable (3-dim) |
            cost_func_translation_part  |               *                          *               |
            cost_func_rotation_part     |               *                          *               |
    """

    # Create the substitutions dictionary
    substitutions = {
        sf_ri: sf.V3(pose_i["r"] + epsilon),
        sf_ti: sf.V3(pose_i["t"] + epsilon),
        sf_rj: sf.V3(pose_j["r"] + epsilon),
        sf_tj: sf.V3(pose_j["t"] + epsilon),
        sf_rij: sf.V3(rotmat_to_rotvec(pose_ij_meas["R"]) + epsilon),
        sf_tij: sf.V3(pose_ij_meas["t"] + epsilon),
    }

    sf_J_ti_val = sf_J_ti.subs(substitutions).to_numpy()
    sf_J_ri_val = sf_J_ri.subs(substitutions).to_numpy()
    sf_J_tj_val = sf_J_tj.subs(substitutions).to_numpy()
    sf_J_rj_val = sf_J_rj.subs(substitutions).to_numpy()

    # ps. the reason why the index 3: mapped to :3
    # is because this example uses [t, r], but symforce uses the order of [r, t]
    sf_Ji = np.zeros((6, 6))
    sf_Ji[:3, :3] = sf_J_ti_val[3:, :]
    sf_Ji[3:, :3] = sf_J_ti_val[:3, :]
    sf_Ji[:3, 3:] = sf_J_ri_val[3:, :]
    sf_Ji[3:, 3:] = sf_J_ri_val[:3, :]

    sf_Jj = np.zeros((6, 6))
    sf_Jj[:3, :3] = sf_J_tj_val[3:, :]
    sf_Jj[3:, :3] = sf_J_tj_val[:3, :]
    sf_Jj[:3, 3:] = sf_J_rj_val[3:, :]
    sf_Jj[3:, 3:] = sf_J_rj_val[:3, :]

    return sf_Ji, sf_Jj


def compute_between_factor_residual_and_jacobian(
    pose_i, pose_j, pose_ij_meas, use_jacobian_approx_fast=True
):
    """
    Computes the residual and Jacobians for a pair of poses given a measurement.

    Parameters:
        pose_i (dict): Dictionary containing rotation vector 'r' and translation 't' for pose i.
        pose_j (dict): Dictionary containing rotation vector 'r' and translation 't' for pose j.
        pose_ij_meas (dict): Dictionary containing rotation matrix 'R' and translation 't' from the measurement.

    Returns:
        residual (np.ndarray): 6-element residual vector.
        Ji (np.ndarray): 6x6 Jacobian matrix with respect to pose i.
        Jj (np.ndarray): 6x6 Jacobian matrix with respect to pose j.
    """
    # Unpack poses
    ti, ri = pose_i["t"], pose_i["r"]
    tj, rj = pose_j["t"], pose_j["r"]

    # Convert rotation vectors to matrices
    Ri = rotvec_to_rotmat(ri)
    Rj = rotvec_to_rotmat(rj)

    # Measurement
    Rij_meas, tij_meas = pose_ij_meas["R"], pose_ij_meas["t"]

    # Predicted relative transformation
    Ri_inv = Ri.T
    Rij_pred = Ri_inv @ Rj
    tij_pred = Ri_inv @ (tj - ti)

    # Error in rotation and translation
    R_err = Rij_meas.T @ Rij_pred
    t_err = Rij_meas.T @ (tij_pred - tij_meas)

    # Map rotation error to rotation vector
    r_err = rotmat_to_rotvec(R_err)
    residual = np.hstack((t_err, r_err))

    def between_factor_jacobian_by_hand_approx():
        # Jacobian w.r. to pose i
        Ji = np.zeros((6, 6))
        Ji[:3, :3] = -Rij_meas.T @ Ri_inv
        Ji[:3, 3:] = Rij_meas.T @ Ri_inv @ skew_symmetric(tj - ti)
        Ji[3:, 3:] = -np.eye(3)  # approx

        # Jacobian w.r. to pose j
        Jj = np.zeros((6, 6))
        Jj[:3, :3] = Rij_meas.T @ Ri_inv
        Jj[3:, 3:] = np.eye(3)  # approx

        # ps. the above approximations are valid for small angle differecne
        #  (may differ at the early iterations wrt the symforce version)

        return Ji, Jj

    # Compute Jacobians analytically
    if use_jacobian_approx_fast:
        Ji, Jj = between_factor_jacobian_by_hand_approx()
    else:
        Ji, Jj = between_factor_jacobian_by_symforce(pose_i, pose_j, pose_ij_meas)

    debug_compare_jacobians = False
    if debug_compare_jacobians:
        by_hand_Ji, by_hand_Jj = between_factor_jacobian_by_hand_approx()

        start_time = time.perf_counter()
        by_symb_Ji, by_symb_Jj = between_factor_jacobian_by_symforce(
            pose_i, pose_j, pose_ij_meas
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("=" * 30, f"elapsed time: {elapsed_time:.8f} sec.")
        print(f"by_hand_Ji\n {by_hand_Ji}")
        print(f"by_symbolic_Ji\n {by_symb_Ji}\n")
        print(f"by_hand_Jj\n {by_hand_Jj}")
        print(f"by_symbolic_Jj\n {by_symb_Jj}\n\n")

    return residual, Ji, Jj


class PoseGraphOptimizer:
    def __init__(
        self,
        max_iterations=50,
        initial_cauchy_c=10.0,
        use_jacobian_approx_fast=False,
        num_processes=1,
    ):
        self.STATE_DIM = 6

        self.max_iterations = max_iterations
        self.num_processes = num_processes

        # jacobian mode
        self.use_jacobian_approx_fast = use_jacobian_approx_fast  # if False, using Symforce-based auto-generated Symbolic Jacobian
        if not self.use_jacobian_approx_fast:
            print(
                "\n Using symforce-based automatically derived symbolic jacobian... (may slow)\n"
            )

        # Robust loss
        self.cauchy_c = initial_cauchy_c  # cauchy kernel

        # LM iterative optimization
        self.lambda_ = 0.001  # Initial damping factor, for LM opt
        self.lambda_allowed_range = [1e-7, 1e5]

        # weight ratio
        self.loop_information_matrix = 1.0 * np.identity(6)
        self.odom_information_matrix = 1.0 * np.identity(6)

        # A single prior
        self.add_prior_to_prevent_gauge_freedom = True

        # misc
        self.H_fig_saved = False
        self.loud_verbose = True

    def read_g2o_file(self, file_path):
        """
        Reads a g2o file and parses the poses and edges.

        Parameters:
            file_path (str): Path to the g2o file.

        Returns:
            poses (dict): Dictionary of poses with pose ID as keys and dictionaries containing rotation matrix 'R' and translation vector 't' as values.
            edges (list): List of edges, where each edge is a dictionary containing 'from', 'to', rotation matrix 'R', translation vector 't', and 'information' matrix.
        """
        self.dataset_name = file_path.split("/")[-1]

        print(f"Reading (parse) {file_path} ...")

        poses = {}
        edges = []

        def parse_information_matrix(data, size):
            """
            Parses the upper triangular part of the information matrix and constructs the full symmetric matrix.

            Parameters:
                data (list of float): Upper triangular elements of the information matrix.
                size (int): Size of the square information matrix.

            Returns:
                information_matrix (np.ndarray): size x size information matrix.
            """
            information_upper = np.array(data)
            information_matrix = np.zeros((size, size))
            indices = np.triu_indices(size)
            information_matrix[indices] = information_upper
            information_matrix += information_matrix.T - np.diag(
                information_matrix.diagonal()
            )
            return information_matrix

        def information_matrix_wrt_edge_type(is_consecutive):
            # Using a constant info matrix is more stable
            if is_consecutive:
                # Odometry edge
                information_matrix = self.odom_information_matrix
            else:
                # Loop edge
                information_matrix = self.loop_information_matrix

            return information_matrix

        def SE3_edge_dict(id_from, id_to, R, t, information_matrix):
            return {
                "from": id_from,
                "to": id_to,
                "R": R,
                "t": t,
                "information": information_matrix,
            }

        self.using_predefined_const_information_matrix_wrt_type = True

        with open(file_path, "r") as f:
            for line in f:
                data = line.strip().split()
                if not data:
                    continue

                tag = data[0]

                if tag.startswith("VERTEX"):
                    if tag == "VERTEX_SE3:QUAT":
                        node_id = int(data[1])
                        x, y, z = map(float, data[2:5])
                        qx, qy, qz, qw = map(float, data[5:9])
                        R = quaternion_to_rotation(qx, qy, qz, qw)
                        t = np.array([x, y, z])
                        poses[node_id] = {"R": R, "t": t}

                    elif tag == "VERTEX_SE2":
                        node_id = int(data[1])
                        x, y, theta = map(float, data[2:5])
                        R, t = se2_to_se3(x, y, theta)
                        poses[node_id] = {"R": R, "t": t}

                elif tag.startswith("EDGE"):
                    if tag == "EDGE_SE3:QUAT":
                        id_from = int(data[1])
                        id_to = int(data[2])
                        x, y, z = map(float, data[3:6])
                        qx, qy, qz, qw = map(float, data[6:10])
                        R = quaternion_to_rotation(qx, qy, qz, qw)
                        t = np.array([x, y, z])

                        if self.using_predefined_const_information_matrix_wrt_type:
                            # Using a constant info matrix seems generally more stable
                            information_matrix = information_matrix_wrt_edge_type(
                                is_consecutive=(abs(id_from - id_to) == 1)
                            )
                        else:
                            # The information matrix parses the original data,
                            information_matrix = parse_information_matrix(data[10:], 6)

                        edge = SE3_edge_dict(id_from, id_to, R, t, information_matrix)
                        edges.append(edge)

                    elif tag == "EDGE_SE2":
                        id_from = int(data[1])
                        id_to = int(data[2])
                        dx, dy, dtheta = map(float, data[3:6])
                        R, t = se2_to_se3(dx, dy, dtheta)

                        if self.using_predefined_const_information_matrix_wrt_type:
                            # Using a constant info matrix seems generally more stable
                            information_matrix = information_matrix_wrt_edge_type(
                                is_consecutive=(abs(id_from - id_to) == 1)
                            )
                        else:
                            # Parse the SE2 information matrix and pad it to 6x6
                            information_matrix_se2 = parse_information_matrix(
                                data[6:12], 3
                            )
                            information_matrix = np.zeros((6, 6))
                            information_matrix[:3, :3] = information_matrix_se2
                            information_matrix += np.diag(np.ones(6))

                        edge = SE3_edge_dict(id_from, id_to, R, t, information_matrix)
                        edges.append(edge)

        # Convert rotations to rotation vectors
        for pose_id, pose in poses.items():
            pose["r"] = rotmat_to_rotvec(pose["R"])

        return poses, edges

    def cauchy_weight(self, s):
        epsilon = 1e-5
        return self.cauchy_c / (np.sqrt(self.cauchy_c**2 + s) + epsilon)

    def initialize_variables_container(self, index_map):
        x = np.zeros(6 * self.num_poses)
        for pose_id, pose in self.poses_initial.items():
            idx = index_map[pose_id]
            t = pose["t"]
            r = pose["r"]
            x[self.STATE_DIM * idx : self.STATE_DIM * idx + 3] = t
            x[self.STATE_DIM * idx + 3 : self.STATE_DIM * idx + 6] = r

        return x

    def get_state_block(self, states_vector, block_idx):
        start_location = self.STATE_DIM * block_idx
        end_location = start_location + self.STATE_DIM
        return states_vector[start_location:end_location]

    def add_initials(self, poses_initial):
        self.poses_initial = poses_initial
        self.num_poses = len(self.poses_initial)
        self.pose_indices = list(self.poses_initial.keys())
        self.index_map = self.generate_poses_index_map(self.pose_indices)

    def add_edges(self, edges):
        self.edges = edges

    def add_prior(self, idx):
        # Current option: only single prior to avoid gauge problem
        self.prior_pose_id = self.pose_indices[idx]
        self.setup_fixed_single_prior(self.prior_pose_id)

    def setup_fixed_single_prior(self, prior_pose_id):
        # Identify the prior pose index
        self.idx_prior = self.index_map[prior_pose_id]

        # Information matrix for the prior
        self.information_prior = 1e9 * np.identity(self.STATE_DIM)  # Adjust as needed

    def generate_poses_index_map(self, pose_indices):
        return {pose_id: idx for idx, pose_id in enumerate(pose_indices)}

    def nodes_are_consecutive(self, id_to, id_from):
        # assumption: odom edges have consecutive indices
        return abs(id_to - id_from) == 1

    def process_edge(self, edge_data):
        ii, edge, index_map, x, STATE_DIM, use_jacobian_approx_fast = edge_data

        if self.loud_verbose and (ii % 1000 == 0):
            print(f" [(parr) build_sparse_system] processing edge {ii}/{len(edges)}")

        idx_i = index_map[edge["from"]]
        idx_j = index_map[edge["to"]]

        # Extract poses
        xi = self.get_state_block(x, idx_i)
        xj = self.get_state_block(x, idx_j)

        pose_i = {"t": xi[:3], "r": xi[3:]}
        pose_j = {"t": xj[:3], "r": xj[3:]}

        pose_ij_meas = {"R": edge["R"], "t": edge["t"]}
        information_edge = edge["information"]

        # Compute residual and Jacobians
        residual, Ji, Jj = compute_between_factor_residual_and_jacobian(
            pose_i, pose_j, pose_ij_meas, use_jacobian_approx_fast
        )

        # Check if edge is non-consecutive (loop closure)
        if not self.nodes_are_consecutive(edge["from"], edge["to"]):
            # For loop closure edges, robust kernel is applied
            s = residual.T @ information_edge @ residual
            weight = self.cauchy_weight(s)
        else:
            # for odom edges, no robust loss
            weight = 1.0

        # Deweighting
        residual *= weight
        Ji *= weight
        Jj *= weight

        # Accumulate error
        total_error = residual.T @ information_edge @ residual

        # Assemble H and b components
        Hii = Ji.T @ information_edge @ Ji
        Hjj = Jj.T @ information_edge @ Jj
        Hij = Ji.T @ information_edge @ Jj

        bi = Ji.T @ information_edge @ residual
        bj = Jj.T @ information_edge @ residual

        return (idx_i, idx_j, Hii, Hjj, Hij, bi, bj, total_error)

    def build_sparse_system(self, edges):

        # First step: Calculate each element of H and b
        #  Prepare data for parallel processing
        edge_data_list = [
            (
                ii,
                edge,
                self.index_map,
                self.x,
                self.STATE_DIM,
                self.use_jacobian_approx_fast,
            )
            for ii, edge in enumerate(edges)
        ]
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = pool.map(self.process_edge, edge_data_list)

        # Second step: Assemble H and b with a for loop
        H_row = []
        H_col = []
        H_data = []
        b = np.zeros(self.STATE_DIM * len(self.index_map))
        total_error = 0.0

        for result in results:
            idx_i, idx_j, Hii, Hjj, Hij, bi, bj, edge_error = result

            # Accumulate total error
            total_error += edge_error

            # Hii
            for i in range(self.STATE_DIM):
                for j in range(self.STATE_DIM):
                    H_row.append((self.STATE_DIM * idx_i) + i)
                    H_col.append((self.STATE_DIM * idx_i) + j)
                    H_data.append(Hii[i, j])

            # Hjj
            for i in range(self.STATE_DIM):
                for j in range(self.STATE_DIM):
                    H_row.append(self.STATE_DIM * idx_j + i)
                    H_col.append(self.STATE_DIM * idx_j + j)
                    H_data.append(Hjj[i, j])

            # Hij and Hji
            for i in range(self.STATE_DIM):
                for j in range(self.STATE_DIM):
                    # Hij
                    H_row.append(self.STATE_DIM * idx_i + i)
                    H_col.append(self.STATE_DIM * idx_j + j)
                    H_data.append(Hij[i, j])

                    # Hji
                    H_row.append(self.STATE_DIM * idx_j + i)
                    H_col.append(self.STATE_DIM * idx_i + j)
                    H_data.append(Hij[j, i])  # Transpose

            # b_i and b_j
            b[
                (self.STATE_DIM * idx_i) : (self.STATE_DIM * idx_i) + self.STATE_DIM
            ] -= bi
            b[
                (self.STATE_DIM * idx_j) : (self.STATE_DIM * idx_j) + self.STATE_DIM
            ] -= bj

        # prior
        if self.add_prior_to_prevent_gauge_freedom:
            """
            Adds a prior to the first pose to fix the gauge freedom.
            """
            # Current estimate of the prior pose
            xi_prior = self.get_state_block(self.x, self.idx_prior)

            # Initial estimate (measurement) of the prior pose
            pose_prior_meas = self.poses_initial[self.prior_pose_id]
            xi_prior_meas = np.hstack((pose_prior_meas["t"], pose_prior_meas["r"]))

            # Compute the residual (error) between current and initial estimates
            residual_prior = xi_prior - xi_prior_meas

            # Jacobian of the prior (identity matrix since it's a direct difference)
            J_prior = np.identity(self.STATE_DIM)

            # Compute the prior's contribution to H and b
            H_prior = J_prior.T @ self.information_prior @ J_prior
            b_prior = J_prior.T @ self.information_prior @ residual_prior

            # Append prior contributions to H_data, H_row, and H_col
            for i in range(self.STATE_DIM):
                for j in range(self.STATE_DIM):
                    H_row.append(self.STATE_DIM * self.idx_prior + i)
                    H_col.append(self.STATE_DIM * self.idx_prior + j)
                    H_data.append(H_prior[i, j])

            # Update b with the prior contribution
            b[
                (self.STATE_DIM * self.idx_prior) : (self.STATE_DIM * self.idx_prior)
                + self.STATE_DIM
            ] -= b_prior

        # Convert H to sparse matrix
        H = sp.coo_matrix(
            (H_data, (H_row, H_col)),
            shape=(self.STATE_DIM * self.num_poses, self.STATE_DIM * self.num_poses),
        )

        return H, b, total_error

    def solve_sparse_system(self, H, b, e):
        # Apply damping (Levenberg-Marquardt)
        H = H + sp.diags(self.lambda_ * H.diagonal(), format="csr")

        # Perform Cholesky factorization
        H = H.tocsc()
        factor = cholmod.cholesky(H)

        # Solve the system
        delta_x = factor.solve_A(b)

        return delta_x

    def plot_H_matrix(self, H, name):
        """
        Plots the sparsity pattern of the Hessian matrix H.

        Parameters:
            H (scipy.sparse.coo_matrix): The Hessian matrix.
        """
        if not hasattr(self, "fig") or not hasattr(self, "ax"):
            # Initialize the plot on the first call
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()  # Enable interactive mode

        self.ax.clear()
        self.ax.set_title(f"Sparsity Pattern of H")
        self.ax.spy(H, markersize=1, color="white")  # White for non-zero
        self.ax.set_facecolor("black")  # Black for zero
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.fig.savefig(f"docs/H/H_sparsity_{name}.png")
        self.H_fig_saved = True

    # Recalculate error with updated poses
    def evaluate_error_changes(self, x_new):
        total_error_after_iter_opt = 0

        for edge in self.edges:
            idx_i = self.index_map[edge["from"]]
            idx_j = self.index_map[edge["to"]]

            # Extract updated poses
            xi = self.get_state_block(x_new, idx_i)
            xj = self.get_state_block(x_new, idx_j)

            pose_i = {"t": xi[:3], "r": xi[3:]}
            pose_j = {"t": xj[:3], "r": xj[3:]}

            pose_ij_meas = {"R": edge["R"], "t": edge["t"]}
            information = edge["information"]

            # Compute residual
            residual, _, _ = compute_between_factor_residual_and_jacobian(
                pose_i, pose_j, pose_ij_meas, use_jacobian_approx_fast=True
            )

            # Apply Cauchy robust kernel if consecutive
            if not self.nodes_are_consecutive(edge["from"], edge["to"]):
                s = residual.T @ information @ residual
                weight = self.cauchy_weight(s)
            else:
                weight = 1.0

            # deweight
            residual *= weight

            total_error_after_iter_opt += residual.T @ information @ residual

        # Also include the prior in the total error
        if self.add_prior_to_prevent_gauge_freedom:
            x_meas = self.get_state_block(self.x, self.idx_prior)
            x_pred = self.get_state_block(x_new, self.idx_prior)
            prior_residual = x_meas - x_pred

            total_error_after_iter_opt += (
                prior_residual.T @ self.information_prior @ prior_residual
            )

        return total_error_after_iter_opt

    def adjust_parameters(self, iteration, delta_x, error_before_opt, error_after_opt):
        # Check if error decreased
        if error_after_opt < error_before_opt:
            # tune params
            if self.lambda_allowed_range[0] < self.lambda_:
                self.lambda_ /= 5

            # verbose
            print(
                f"\nIteration {iteration}: The total cost",
                f"decreased from {error_before_opt:.3f} to {error_after_opt:.3f}",
                f" \n - current lambda is {self.lambda_:.7f} and cauchy kernel is {self.cauchy_c:.2f}",
                f" \n - |delta_x|: {np.linalg.norm(delta_x):.4f}\n",
            )
        else:
            # tune params
            if self.lambda_ < self.lambda_allowed_range[1]:
                self.lambda_ *= 5
            if self.cauchy_c > 1.0:
                self.cauchy_c /= 2

            # verbose
            print(
                f"\nIteration {iteration}: The total cost did not decrease (from",
                f"{error_before_opt:.3f} to {error_after_opt:.3f}).",
                f" \n - increase lambda to {self.lambda_:.7f} and cauchy kernel to {self.cauchy_c:.2f}",
                f" \n - |delta_x|: {np.linalg.norm(delta_x):.4f}\n",
            )

    def process_single_iteration(self, iteration):
        # Build and solve the system
        H, b, total_error = self.build_sparse_system(self.edges)
        if not self.H_fig_saved:
            self.plot_H_matrix(H, self.dataset_name)

        delta_x = self.solve_sparse_system(H, b, total_error)

        # Update poses
        x_new = self.x + delta_x

        # Evaluate the error direction
        total_error_after_iter_opt = self.evaluate_error_changes(x_new)

        # Conditionally Accept the update
        self.x = x_new if total_error_after_iter_opt < total_error else self.x

        # Run-time adjustment of LM parameter and Cauchy kernel size
        self.adjust_parameters(
            iteration, delta_x, total_error, total_error_after_iter_opt
        )

        # Visualize this iterations's result
        self.visualize_3d_poses(self.get_optimized_poses())

        # Check for convergence
        termination_flag = False
        convergence_error_diff_threshold = 1e-1
        if (total_error_after_iter_opt < total_error) and (
            np.abs(total_error - total_error_after_iter_opt)
            < convergence_error_diff_threshold
        ):
            print("Converged.")
            termination_flag = True

        return termination_flag

    def optimize(self, visual_debug=True):
        """
        Performs pose graph optimization using the Gauss-Newton method with robust kernels.

        Parameters:
            poses_initial (dict): Initial poses with pose ID as keys and dictionaries containing rotation 'R', translation 't', and rotation vector 'r'.
            edges (list): List of edges, where each edge is a dictionary containing 'from', 'to', rotation matrix 'R', translation vector 't', and 'information' matrix.
            self.max_iterations (int, optional): Maximum number of optimization iterations (default is 10).
            c (float, optional): Parameter for the Cauchy robust kernel (default is 1.0).

        Returns:
            optimized_poses (dict): Optimized poses with pose ID as keys and dictionaries containing rotation matrix 'R' and translation vector 't'.
        """

        # Initialize pose parameters
        self.x = self.initialize_variables_container(self.index_map)

        # optimize
        for iteration in range(self.max_iterations):
            termination_flag = self.process_single_iteration(iteration)
            if termination_flag:
                break

        # Extract optimized poses
        return self.get_optimized_poses()

    def get_optimized_poses(self):
        optimized_poses = {}
        for pose_id, idx in self.index_map.items():
            xi = self.x[(6 * idx) : (6 * idx) + 6]
            t = xi[:3]
            r = xi[3:]
            R = rotvec_to_rotmat(r)
            optimized_poses[pose_id] = {"R": R, "t": t}

        return optimized_poses

    def visualize_3d_poses(self, poses_optimized):
        # Prepare data for plotting
        # Sort the poses based on pose IDs to maintain order
        sorted_pose_ids = sorted(self.poses_initial.keys())

        # Extract initial and optimized translations as lists
        initial_positions_list = [
            self.poses_initial[pose_id]["t"] for pose_id in sorted_pose_ids
        ]
        optimized_positions_list = [
            poses_optimized[pose_id]["t"] for pose_id in sorted_pose_ids
        ]

        # Prepare edges for plotting
        edges_for_plotting = [{"i": edge["from"], "j": edge["to"]} for edge in edges]

        # Plot the results using Open3D
        plot_two_poses_with_edges_open3d(
            initial_positions_list, optimized_positions_list, edges_for_plotting
        )


if __name__ == "__main__":
    """
    Main execution block.
    Loads a g2o file, performs pose graph optimization, and visualizes the results.
    """

    """
      Dataset selection
    """
    # Successed datasets
    # dataset_name = "data/cubicle.g2o"
    # dataset_name = "data/parking-garage.g2o"
    dataset_name = "data/input_INTEL_g2o.g2o"
    # dataset_name = "data/input_M3500_g2o.g2o"
    # dataset_name = "data/sphere2500.g2o"

    # TODO: these datasets still fail
    # dataset_name = "data/input_MITb_g2o.g2o"
    # dataset_name = "data/rim.g2o"

    """
      Pose-graph optimization
    """

    cauchy_c = 10.0  # robust kernel size
    num_processes = 8  # if want to use max, multiprocessing.cpu_count()
    max_iterations = 50
    use_jacobian_approx_fast = (
        False  # if False, using Symforce-based auto-generated Symbolic Jacobian
    )

    pgo = PoseGraphOptimizer(
        max_iterations=max_iterations,
        initial_cauchy_c=cauchy_c,
        use_jacobian_approx_fast=use_jacobian_approx_fast,
        num_processes=num_processes,
    )

    # read data
    poses_initial, edges = pgo.read_g2o_file(dataset_name)

    # add initials
    pgo.add_initials(poses_initial)

    # add constraints
    pgo.add_edges(edges)

    prior_node_idx = -1
    pgo.add_prior(prior_node_idx)

    # Optimize poses
    poses_optimized = pgo.optimize(visual_debug=True)
