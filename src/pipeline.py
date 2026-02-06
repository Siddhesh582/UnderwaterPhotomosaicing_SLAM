"""
Underwater Photomosaicing with Pose Graph SLAM
Main class for image feature matching and SLAM pipeline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import gtsam


class ImageFeatureMatching:
    """
    Complete SLAM pipeline for underwater photomosaicing.
    
    Pipeline:
    1. SIFT feature detection and matching
    2. RANSAC-based homography estimation
    3. Uncertainty quantification via Jacobian covariance propagation
    4. Loop closure detection
    5. GTSAM pose graph optimization
    6. Photomosaic generation
    """
    
    def __init__(self):
        """Initialize SIFT detector and matcher."""
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.homographies = []

    # ==================== IMAGE I/O ====================
    
    def read_image(self, image_paths, titles=None, show=True):
        """
        Read images from file paths.
        
        Args:
            image_paths: List of image file paths
            titles: Optional list of titles for display
            show: Whether to display images
            
        Returns:
            List of loaded images
        """
        images = []
        for idx, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return None
            
            title = titles[idx] if titles and idx < len(titles) else f'Image{idx+1}'
            
            if show:
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(title)
                plt.axis('off')
                plt.show()
            
            images.append(image)
        
        return images

    def normalize_images(self, image):
        """Normalize image intensity to [0, 255]."""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized

    # ==================== FEATURE MATCHING ====================
    
    def feature_matching(self, image1, image2, ratio_threshold=0.8):
        """
        SIFT feature detection and matching with Lowe's ratio test.
        
        Args:
            image1, image2: Input images
            ratio_threshold: Lowe's ratio test threshold (default: 0.8)
            
        Returns:
            Tuple of (points1, points2, keypoints1, keypoints2, good_matches)
        """
        normalized_image1 = self.normalize_images(image1)
        normalized_image2 = self.normalize_images(image2)
        
        keypoints1, descriptors1 = self.sift.detectAndCompute(normalized_image1, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(normalized_image2, None)
        
        if descriptors1 is None or descriptors2 is None:
            print('No descriptors detected in images')
            return None, None, None, None, None
        
        print(f'Detected {len(keypoints1)} features in 1st image')
        print(f'Detected {len(keypoints2)} features in 2nd image')
        
        # KNN matching with k=2
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches after Lowe's ratio test")
        
        # Extract 2D coordinates
        image1_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        image2_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        return image1_pts, image2_pts, keypoints1, keypoints2, good_matches

    # ==================== RANSAC ====================
    
    def ransac_homography(self, image1_pts, image2_pts, image_pair=None, 
                         ransac_threshold=5.0, max_iters=2000, confidence=0.995):
        """
        RANSAC-based affine partial 2D estimation.
        
        Args:
            image1_pts, image2_pts: Matched point correspondences
            image_pair: Tuple identifying the image pair
            ransac_threshold: Inlier threshold in pixels
            max_iters: Maximum RANSAC iterations
            confidence: RANSAC confidence level
            
        Returns:
            Tuple of (homography_matrix, inlier_mask)
        """
        if len(image1_pts) < 3 or len(image2_pts) < 3:
            print("Need at least 3 point correspondences for affine partial 2D")
            return None, None
        
        # Estimate affine transformation using RANSAC
        H, mask = cv2.estimateAffinePartial2D(
            image1_pts, image2_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=ransac_threshold, 
            maxIters=max_iters, 
            confidence=confidence
        )
        
        if H is None:
            print('RANSAC failed, no affine transformation')
            return None, None
        
        # Convert to 3x3 homography
        H = np.vstack((H, [0, 0, 1]))
        
        inlier_count = np.sum(mask)
        total_matches = len(mask)
        
        print(f"RANSAC Results:")
        print(f"Total matches: {total_matches}")
        print(f"Inliers: {inlier_count}")
        print(f"Outliers rejected: {total_matches - inlier_count}")
        
        self.homographies.append({
            'pair': image_pair if image_pair else ('ImageX', 'ImageY'), 
            'H': H
        })
        print(f"\nHomography for this pair found in RANSAC:\n{H}")
        
        return H, mask

    # ==================== VISUALIZATION ====================
    
    def visualize_feature_matches(self, image1, image2, keypoints1, keypoints2, 
                                  matches, mask=None):
        """
        Visualize feature matches with inliers/outliers.
        
        Args:
            image1, image2: Input images
            keypoints1, keypoints2: Detected keypoints
            matches: List of matches
            mask: Inlier mask from RANSAC (optional)
            
        Returns:
            Image with drawn matches
        """
        if mask is not None:
            matches_mask = mask.ravel().tolist()
            draw_params = dict(
                matchColor=(0, 255, 0),      # Green for inliers
                singlePointColor=(255, 0, 0), # Blue for outliers
                matchesMask=matches_mask, 
                flags=cv2.DrawMatchesFlags_DEFAULT
            )
            title = f'Feature Matches - {np.sum(mask)}/{len(mask)} inliers (Green=Inliers, Blue=Outliers)'
        else:
            draw_params = dict(
                matchColor=(0, 0, 255),      # Red matches
                singlePointColor=(0, 255, 255), # Cyan points
                flags=cv2.DrawMatchesFlags_DEFAULT
            )
            title = 'Features Matched'
        
        matched_img = cv2.drawMatches(
            image1, keypoints1, image2, keypoints2, 
            matches, None, **draw_params
        )
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
        
        return matched_img

    # ==================== UNCERTAINTY QUANTIFICATION ====================
    
    def homography_residuals(self, params, image1_pts, image2_pts):
        """
        Compute reprojection residuals for homography parameters.
        
        Args:
            params: Flattened homography parameters (8 values)
            image1_pts, image2_pts: Point correspondences
            
        Returns:
            Flattened residual vector
        """
        H = np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0]
        ])
        
        # Convert to homogeneous coordinates
        image1_homogeneous_coord = np.column_stack([image1_pts, np.ones(len(image1_pts))])
        
        # Apply homography transformation
        transformed = (H @ image1_homogeneous_coord.T).T
        
        # Convert back to Euclidean coordinates
        z = transformed[:, 2]
        z[z == 0] = 1e-8  # Avoid division by zero
        transformed_pts = transformed[:, :2] / z.reshape(-1, 1)
        
        # Compute residuals
        residuals = (image2_pts - transformed_pts).flatten()
        
        return residuals

    def compute_jacobian(self, H_params, inlier_pts1):
        """
        Compute analytical Jacobian matrix for homography.
        
        For each point (x1, y1), computes derivatives of reprojection errors
        with respect to 8 homography parameters.
        
        Args:
            H_params: Homography parameters (8 values)
            inlier_pts1: Inlier points from image 1
            
        Returns:
            Jacobian matrix of shape (2*N, 8)
        """
        n_points = len(inlier_pts1)
        J = np.zeros((2 * n_points, 8))
        
        for i, (x1, y1) in enumerate(inlier_pts1):
            w = H_params[6]*x1 + H_params[7]*y1 + 1.0
            w = max(abs(w), 1e-8)  # Avoid division by zero
            
            x2_h = H_params[0]*x1 + H_params[1]*y1 + H_params[2]
            y2_h = H_params[3]*x1 + H_params[4]*y1 + H_params[5]
            
            # Derivatives for x-residual
            J[2*i, 0] = -x1 / w
            J[2*i, 1] = -y1 / w
            J[2*i, 2] = -1.0 / w
            J[2*i, 3] = 0.0
            J[2*i, 4] = 0.0
            J[2*i, 5] = 0.0
            J[2*i, 6] = x2_h * x1 / (w * w)
            J[2*i, 7] = x2_h * y1 / (w * w)
            
            # Derivatives for y-residual
            J[2*i+1, 0] = 0.0
            J[2*i+1, 1] = 0.0
            J[2*i+1, 2] = 0.0
            J[2*i+1, 3] = -x1 / w
            J[2*i+1, 4] = -y1 / w
            J[2*i+1, 5] = -1.0 / w
            J[2*i+1, 6] = y2_h * x1 / (w * w)
            J[2*i+1, 7] = y2_h * y1 / (w * w)
        
        return J

    def compute_covariance(self, H_ransac, inlier_image1, inlier_image2):
        """
        Compute homography covariance from residuals and Jacobian.
        
        Args:
            H_ransac: Estimated homography matrix
            inlier_image1, inlier_image2: Inlier point correspondences
            
        Returns:
            Covariance matrix (8x8)
        """
        if H_ransac is None or len(inlier_image1) < 4:
            print("Not enough inliers")
            return None
        
        H_flat = H_ransac.flatten()[:8]
        residuals = self.homography_residuals(H_flat, inlier_image1, inlier_image2)
        sigma2 = np.mean(residuals**2)
        print(f"Residual sigma2: {sigma2:.2f}, sqrt: {np.sqrt(sigma2):.2f} pixels")
        
        J = self.compute_jacobian(H_flat, inlier_image1)
        
        try:
            JtJ = J.T @ J
            return sigma2 * np.linalg.inv(JtJ)
        except np.linalg.LinAlgError:
            print("Adding regularization, Singular Jacobian Matrix")
            JtJ += 1e-8 * np.eye(8)
            return sigma2 * np.linalg.inv(JtJ)

    # ==================== COMPLETE REGISTRATION ====================
    
    def register_image_pair(self, image1, image2, visualize=True):
        """
        Complete image pair registration pipeline.
        
        Steps:
        1. Feature detection and matching
        2. RANSAC homography estimation
        3. Covariance computation
        
        Args:
            image1, image2: Input images
            visualize: Whether to visualize matches
            
        Returns:
            Dictionary with registration results
        """
        # SIFT feature matching
        result = self.feature_matching(image1, image2)
        if result[0] is None:
            return {'success': False, 'error': 'Feature matching failed'}
        
        image1_pts, image2_pts, keypoints1, keypoints2, good_matches = result
        
        # RANSAC homography estimation
        H_ransac, mask = self.ransac_homography(image1_pts, image2_pts)
        if H_ransac is None:
            return {'success': False, 'error': 'RANSAC failed'}
        
        inlier_mask = mask.ravel() == 1
        inlier_image1 = image1_pts[inlier_mask]
        inlier_image2 = image2_pts[inlier_mask]
        
        # Covariance computation
        covariance_matrix = self.compute_covariance(H_ransac, inlier_image1, inlier_image2)
        
        if visualize:
            self.visualize_feature_matches(image1, image2, keypoints1, keypoints2, 
                                          good_matches, mask)
        
        matching_result = {
            'success': True,
            'homography_ransac': H_ransac,
            'inlier_mask': mask,
            'inlier_points_image1': inlier_image1,
            'inlier_points_image2': inlier_image2,
            'total_matches': len(good_matches),
            'inlier_count': np.sum(mask),
            'inlier_ratio': np.sum(mask) / len(good_matches),
            'covariance_matrix': covariance_matrix,
            'keypoints1': keypoints1,
            'keypoints2': keypoints2,
            'good_matches': good_matches
        }
        
        print(f"\nImage Feature Matching done")
        print(f"Final inlier ratio: {matching_result['inlier_ratio']:.2%}")
        
        return matching_result

    # ==================== POSE EXTRACTION ====================
    
    def homography_to_pose(self, H_ransac):
        """
        Extract SE(2) pose [x, y, θ] from homography matrix.
        
        Args:
            H_ransac: Homography matrix
            
        Returns:
            Pose array [x, y, theta]
        """
        H_norm = H_ransac / H_ransac[2, 2]
        x, y = -H_norm[0, 2], H_norm[1, 2]
        
        R = H_norm[:2, :2]
        theta = np.arctan2(R[1, 0], R[0, 0])
        
        return np.array([x, y, theta])

    def pose_covariance(self, covariance_matrix, H_ransac):
        """
        Propagate homography covariance to SE(2) pose covariance.
        
        Args:
            covariance_matrix: Homography covariance (8x8)
            H_ransac: Homography matrix
            
        Returns:
            Pose covariance matrix (3x3)
        """
        if covariance_matrix is None:
            return np.eye(3) * 1.0
        
        H_norm = H_ransac / H_ransac[2, 2]
        
        # Jacobian of pose extraction w.r.t. homography parameters
        J_pose = np.zeros((3, 8))
        
        # Translation derivatives
        J_pose[0, 2] = -1.0  # ∂x/∂h02
        J_pose[1, 5] = 1.0   # ∂y/∂h12
        
        # Rotation derivatives (from atan2(h10, h00))
        h00, h10 = H_norm[0, 0], H_norm[1, 0]
        denominator = h00**2 + h10**2
        
        if denominator > 1e-8:
            J_pose[2, 0] = -h10 / denominator  # ∂θ/∂h00
            J_pose[2, 3] = h00 / denominator   # ∂θ/∂h10
        
        # First-order uncertainty propagation
        pose_cov = J_pose @ covariance_matrix @ J_pose.T
        pose_cov = pose_cov / 50  # Scaling factor
        
        return pose_cov

    # ==================== POSE COMPOSITION ====================
    
    def pose_compose(self, pose1, pose2):
        """
        Compose two SE(2) poses: pose1 ⊕ pose2.
        
        Args:
            pose1, pose2: Poses as [x, y, theta]
            
        Returns:
            Composed pose
        """
        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2
        
        # Rotation matrix from pose1
        R1 = np.array([
            [np.cos(theta1), -np.sin(theta1)],
            [np.sin(theta1),  np.cos(theta1)]
        ])
        
        # Rotate and translate
        xy = np.dot(R1, np.array([x2, y2])) + np.array([x1, y1])
        theta = theta1 + theta2
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
        
        return np.array([xy[0], xy[1], theta])

    def propagate_covariance(self, pose1, cov1, pose2, cov2):
        """
        Propagate covariance during pose composition.
        
        Args:
            pose1, cov1: First pose and its covariance
            pose2, cov2: Second pose and its covariance
            
        Returns:
            Covariance of composed pose
        """
        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2
        
        # Jacobian w.r.t. pose1
        J1 = np.array([
            [1, 0, -np.sin(theta1)*x2 - np.cos(theta1)*y2],
            [0, 1,  np.cos(theta1)*x2 - np.sin(theta1)*y2],
            [0, 0, 1]
        ])
        
        # Jacobian w.r.t. pose2
        J2 = np.array([
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1),  np.cos(theta1), 0],
            [0, 0, 1]
        ])
        
        cov_new = J1 @ cov1 @ J1.T + J2 @ cov2 @ J2.T
        return cov_new

    def compose_pose_with_covariance(self, pose1, cov1, pose2, cov2):
        """
        Compose poses and propagate covariances.
        
        Args:
            pose1, cov1: First pose and covariance
            pose2, cov2: Second pose and covariance
            
        Returns:
            Tuple of (composed_pose, composed_covariance)
        """
        new_pose = self.pose_compose(pose1, pose2)
        new_cov = self.propagate_covariance(pose1, cov1, pose2, cov2)
        return new_pose, new_cov

    def global_poses_and_covariances(self, relative_poses, relative_covariances):
        """
        Compute global poses from relative transformations.
        
        Args:
            relative_poses: List of relative poses
            relative_covariances: List of relative covariances
            
        Returns:
            Tuple of (global_poses, global_covariances)
        """
        global_poses = [np.array([0.0, 0.0, 0.0])]  # Anchor at origin
        
        # Initial uncertainty
        initial_cov = np.diag([1.5**2, 1.5**2, (1.0 * np.pi/180)**2])
        global_covariances = [initial_cov]
        
        # Compose poses sequentially
        for rel_pose, rel_cov in zip(relative_poses, relative_covariances):
            new_pose, new_cov = self.compose_pose_with_covariance(
                global_poses[-1], global_covariances[-1], rel_pose, rel_cov
            )
            global_poses.append(new_pose)
            global_covariances.append(new_cov)
        
        return global_poses, global_covariances

    # ==================== LOOP CLOSURE ====================
    
    def extract_loop_closures(self, non_sequential_results, min_inlier_ratio=0.4):
        """
        Extract valid loop closures from non-sequential matches.
        
        Args:
            non_sequential_results: List of non-sequential matching results
            min_inlier_ratio: Minimum inlier ratio threshold
            
        Returns:
            List of valid loop closure constraints
        """
        loop_closures = []
        rejected_count = 0
        
        for result_dict in non_sequential_results:
            if result_dict['success']:
                inlier_ratio = result_dict['result']['inlier_ratio']
                
                if inlier_ratio >= min_inlier_ratio:
                    loop_closures.append({
                        'pair': result_dict['pair'],
                        'measurement': self.homography_to_pose(
                            result_dict['result']['homography_ransac']
                        ),
                        'covariance': self.pose_covariance(
                            result_dict['result']['covariance_matrix'],
                            result_dict['result']['homography_ransac']
                        ),
                        'edge_type': 'loop_closure',
                        'inlier_ratio': inlier_ratio,
                        'inlier_count': result_dict['result']['inlier_count']
                    })
                else:
                    rejected_count += 1
                    print(f"Rejected Loop closure {result_dict['pair_name']}: "
                          f"{inlier_ratio:.1%} inliers")
        
        return loop_closures

    def build_edges(self, sequential_results, non_sequential_results=None, 
                   min_inlier_ratio=0.4):
        """
        Build factor graph edges from sequential and loop closure constraints.
        
        Args:
            sequential_results: List of sequential matching results
            non_sequential_results: List of non-sequential matching results
            min_inlier_ratio: Minimum inlier ratio for loop closures
            
        Returns:
            List of graph edges
        """
        edges = []
        
        # Sequential edges
        for i, result_dict in enumerate(sequential_results):
            if result_dict['success']:
                rel_pose = self.homography_to_pose(
                    result_dict['result']['homography_ransac']
                )
                pose_cov = self.pose_covariance(
                    result_dict['result']['covariance_matrix'],
                    result_dict['result']['homography_ransac']
                )
                regularized_cov = pose_cov + 1e-6 * np.eye(pose_cov.shape[0])
                info = np.linalg.inv(regularized_cov)
                
                edge = {
                    'from': i,
                    'to': i + 1,
                    'measurement': rel_pose,
                    'information': info,
                    'edge_type': 'sequential'
                }
                edges.append(edge)
        
        # Loop closure edges
        if non_sequential_results is not None:
            loop_closures = self.extract_loop_closures(
                non_sequential_results, min_inlier_ratio
            )
            
            for lc in loop_closures:
                i, j = lc['pair']
                regularized_cov = lc['covariance'] + 1e-6 * np.eye(lc['covariance'].shape[0])
                info = np.linalg.inv(regularized_cov)
                
                edge = {
                    'from': i,
                    'to': j,
                    'measurement': lc['measurement'],
                    'information': info,
                    'edge_type': 'loop_closure',
                    'inlier_ratio': lc['inlier_ratio']
                }
                edges.append(edge)
        
        return edges

    # ==================== GTSAM OPTIMIZATION ====================
    
    def solve_with_gtsam(self, initial_poses, edges):
        """
        Optimize pose graph using GTSAM.
        
        Args:
            initial_poses: Initial pose estimates
            edges: List of graph edges (odometry + loop closures)
            
        Returns:
            Tuple of (optimized_poses, optimized_covariances)
        """
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        
        # Insert initial poses
        for i, pose in enumerate(initial_poses):
            x, y, theta = pose
            pose2d = gtsam.Pose2(x, y, theta)
            initial_estimate.insert(i, pose2d)
        
        # Prior on first pose
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.01])
        graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), prior_noise))
        
        # Add edges
        edges_added = 0
        for edge in edges:
            i, j = edge['from'], edge['to']
            measurement = edge['measurement']
            
            try:
                if edge['edge_type'] == 'sequential':
                    noise_model = gtsam.noiseModel.Diagonal.Sigmas([3.0, 3.0, 0.05])
                else:
                    noise_model = gtsam.noiseModel.Diagonal.Sigmas([8.0, 8.0, 0.15])
                
                rel_pose = gtsam.Pose2(measurement[0], measurement[1], measurement[2])
                graph.add(gtsam.BetweenFactorPose2(i, j, rel_pose, noise_model))
                edges_added += 1
            
            except Exception as e:
                print(f"Could not add edge {i}-{j}: {e}")
        
        # Optimize
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SILENT")
        params.setMaxIterations(100)
        
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        
        # Extract optimized poses
        optimized_poses = []
        for i in range(len(initial_poses)):
            pose_gtsam = result.atPose2(i)
            optimized_pose = [pose_gtsam.x(), pose_gtsam.y(), pose_gtsam.theta()]
            optimized_poses.append(optimized_pose)
            
            x, y, theta = optimized_pose
            print(f"Node {i+1}: ({x:.1f}, {y:.1f}, {np.degrees(theta):.1f}°)")
        
        # Compute marginal covariances
        marginals = gtsam.Marginals(graph, result)
        optimized_covariances = []
        print("\nMarginal Covariances per node:")
        for i in range(len(initial_poses)):
            cov_gtsam = marginals.marginalCovariance(i)
            optimized_covariances.append(cov_gtsam)
            print(f"Node {i} covariance:\n{cov_gtsam}\n")
        
        return optimized_poses, optimized_covariances

    # ==================== VISUALIZATION ====================
    
    def visualize_factor_graph(self, global_poses, global_covariances, 
                              loop_closures, edges, save_path):
        """
        Visualize factor graph with uncertainty ellipses.
        
        Args:
            global_poses: List of poses
            global_covariances: List of covariances
            loop_closures: List of loop closure constraints
            edges: List of all edges
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Plot sequential edges
        sequential_edges = [e for e in edges if e['edge_type'] == 'sequential']
        for edge in sequential_edges:
            i, j = edge['from'], edge['to']
            x1, y1 = global_poses[i][:2]
            x2, y2 = global_poses[j][:2]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.8,
                   label='Sequential Edges' if edge == sequential_edges[0] else "")
        
        # Plot loop closure edges
        loop_edges = [e for e in edges if e['edge_type'] == 'loop_closure']
        for edge in loop_edges:
            i, j = edge['from'], edge['to']
            x1, y1 = global_poses[i][:2]
            x2, y2 = global_poses[j][:2]
            ax.plot([x1, x2], [y1, y2], 'r--', linewidth=1, alpha=0.9,
                   label='Loop Closures' if edge == loop_edges[0] else "")
        
        # Plot nodes with uncertainty ellipses
        for i, (pose, cov) in enumerate(zip(global_poses, global_covariances)):
            x, y, theta = pose
            
            # Uncertainty ellipse
            eigenvals, eigenvecs = np.linalg.eigh(cov[:2, :2])
            eigenvals = np.maximum(eigenvals, 0.1)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 4 * np.sqrt(eigenvals)
            
            ellipse = Ellipse((x, y), width, height, angle=angle,
                            facecolor='none', alpha=0.4, edgecolor='black',
                            linewidth=2, zorder=2)
            ax.add_patch(ellipse)
            
            # Node circle
            circle = plt.Circle((x, y), radius=1, color='green', alpha=0.9, zorder=5)
            ax.add_patch(circle)
            
            # Label
            ax.annotate(f'{i+1}', (x, y), fontsize=12, ha='center', va='center',
                      color='black', weight='bold', zorder=6)
        
        ax.set_xlabel('X Position (pixels)', fontsize=14, weight='bold')
        ax.set_ylabel('Y Position (pixels)', fontsize=14, weight='bold')
        ax.set_title('Factor Graph Before GTSAM Optimization', 
                    fontsize=16, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_complete_factor_graph(self, sequential_results, non_sequential_results, 
                                   save_path=None):
        """
        Build and plot complete factor graph before optimization.
        
        Args:
            sequential_results: Sequential matching results
            non_sequential_results: Non-sequential matching results
            save_path: Path to save figure
            
        Returns:
            Dictionary with graph data
        """
        print("Creating factor graph before optimization")
        
        # Extract sequential poses and covariances
        sequential_poses = []
        sequential_covariances = []
        
        for i, result_dict in enumerate(sequential_results):
            if result_dict['success']:
                pose = self.homography_to_pose(result_dict['result']['homography_ransac'])
                pose_cov = self.pose_covariance(
                    result_dict['result']['covariance_matrix'],
                    result_dict['result']['homography_ransac']
                )
                sequential_poses.append(pose)
                sequential_covariances.append(pose_cov)
            else:
                sequential_poses.append(np.array([0.0, 0.0, 0.0]))
                sequential_covariances.append(np.eye(3) * 10.0)
                print(f"Sequential {i}-{i+1}: FAILED")
        
        # Compute global poses and covariances
        global_poses, global_covariances = self.global_poses_and_covariances(
            sequential_poses, sequential_covariances
        )
        
        print("\nNode positions before GTSAM optimization:")
        for i, pose in enumerate(global_poses):
            x, y, theta = pose
            print(f"Node {i+1}: ({x:.1f}, {y:.1f}, {np.degrees(theta):.1f}°)")
        
        # Extract loop closures
        loop_closures = self.extract_loop_closures(non_sequential_results, min_inlier_ratio=0.4)
        
        # Build edges
        edges = self.build_edges(sequential_results, non_sequential_results)
        
        # Visualize
        self.visualize_factor_graph(global_poses, global_covariances, 
                                   loop_closures, edges, save_path)
        
        return {
            'vertices': global_poses,
            'vertex_covariances': global_covariances,
            'edges': edges,
            'loop_closures': loop_closures
        }

    def plot_after_gtsam_optimization(self, optimized_poses, optimized_covariances, 
                                     edges, save_path=None):
        """
        Plot factor graph after GTSAM optimization.
        
        Args:
            optimized_poses: Optimized poses
            optimized_covariances: Optimized covariances
            edges: Graph edges
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Plot sequential edges
        sequential_edges = [e for e in edges if e['edge_type'] == 'sequential']
        for edge in sequential_edges:
            i, j = edge['from'], edge['to']
            x1, y1 = optimized_poses[i][:2]
            x2, y2 = optimized_poses[j][:2]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=3, alpha=0.8,
                   label='Sequential Edges' if edge == sequential_edges[0] else "")
        
        # Plot loop closure edges
        loop_edges = [e for e in edges if e['edge_type'] == 'loop_closure']
        for edge in loop_edges:
            i, j = edge['from'], edge['to']
            x1, y1 = optimized_poses[i][:2]
            x2, y2 = optimized_poses[j][:2]
            ax.plot([x1, x2], [y1, y2], 'r--', linewidth=2, alpha=0.9,
                   label='Loop Closures' if edge == loop_edges[0] else "")
        
        # Plot nodes with uncertainty ellipses
        for i, (pose, cov) in enumerate(zip(optimized_poses, optimized_covariances)):
            x, y, theta = pose
            
            # Uncertainty ellipse
            eigenvals, eigenvecs = np.linalg.eigh(cov[:2, :2])
            eigenvals = np.maximum(eigenvals, 0.1)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 4 * np.sqrt(eigenvals)
            
            ellipse = Ellipse((x, y), width, height, angle=angle,
                            facecolor='none', edgecolor='black', linewidth=1, zorder=2)
            ax.add_patch(ellipse)
            
            # Node circle
            circle = plt.Circle((x, y), radius=1.5, color='black', zorder=5)
            ax.add_patch(circle)
            
            # Label
            ax.annotate(f'{i+1}', (x, y), fontsize=12, ha='center', va='center',
                      color='black', weight='bold', zorder=6)
        
        ax.set_xlabel('X Position (pixels)', fontsize=14, weight='bold')
        ax.set_ylabel('Y Position (pixels)', fontsize=14, weight='bold')
        ax.set_title('Factor Graph After GTSAM Optimization', 
                    fontsize=16, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    # ==================== MOSAIC GENERATION ====================
    
    def calculate_canvas_bounds(self, images, transformations, padding=50):
        """
        Calculate canvas size for mosaic.
        
        Args:
            images: List of images
            transformations: List of transformation matrices
            padding: Border padding in pixels
            
        Returns:
            Tuple of (canvas_width, canvas_height, translation_matrix)
        """
        all_corners = []
        
        for i, (image, T) in enumerate(zip(images, transformations)):
            h, w = image.shape[:2]
            corners = np.array([[0,0,1], [w,0,1], [w,h,1], [0,h,1]], dtype=np.float32)
            
            # Transform corners
            transformed_corners_h = (T @ corners.T).T
            
            # Convert to Euclidean coordinates
            z = transformed_corners_h[:, 2]
            z[z == 0] = 1e-8
            transformed_corners = transformed_corners_h[:, :2] / z.reshape(-1, 1)
            
            all_corners.extend(transformed_corners)
        
        # Calculate canvas size
        all_corners = np.array(all_corners)
        min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int) - padding
        max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int) + padding
        
        canvas_width = max_x - min_x
        canvas_height = max_y - min_y
        
        # Translation to positive coordinates
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], 
                              dtype=np.float64)
        
        print(f"Canvas size: {canvas_width} x {canvas_height}")
        
        return canvas_width, canvas_height, translation

    def warp_images_to_mosaic(self, images, transformations, canvas_width, 
                              canvas_height, translation):
        """
        Warp images to mosaic with soft-edge blending.
        
        Args:
            images: List of images
            transformations: List of transformation matrices
            canvas_width, canvas_height: Canvas dimensions
            translation: Translation matrix
            
        Returns:
            Mosaic image
        """
        mosaic = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        
        for i, (image, T) in enumerate(zip(images, transformations)):
            T_final = translation @ T
            warped = cv2.warpPerspective(image.astype(np.float32), T_final, 
                                        (canvas_width, canvas_height))
            mask = np.any(warped > 0, axis=2).astype(np.uint8)
            
            # Soft blend using distance transform
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            edge_width = 20
            alpha = np.ones_like(dist_transform, dtype=np.float32)
            edge_pixels = dist_transform < edge_width
            alpha[edge_pixels] = dist_transform[edge_pixels] / edge_width
            alpha = np.clip(alpha, 0.3, 1.0)
            
            # Apply soft blending
            current_mask = np.any(warped > 0, axis=2)
            for c in range(3):
                mosaic[:, :, c][current_mask] = (
                    alpha[current_mask] * warped[:, :, c][current_mask] +
                    (1 - alpha[current_mask]) * mosaic[:, :, c][current_mask]
                )
        
        return np.clip(mosaic, 0, 255).astype(np.uint8)

    def transformation_sequence(self, sequential_results):
        """
        Compute transformation sequence from image 1 to all other images.
        
        Args:
            sequential_results: Sequential matching results
            
        Returns:
            List of transformation matrices
        """
        homographies = []
        
        for i, result_dict in enumerate(sequential_results):
            if result_dict['success']:
                H = result_dict['result']['homography_ransac'].astype(np.float64)
                homographies.append(H)
            else:
                print(f"Failed pair {i+1}-{i+2}, using identity")
                homographies.append(np.eye(3))
        
        num_images = len(sequential_results) + 1
        H_transforms = {}
        
        # Image 1 reference
        H_transforms[0] = homographies[0] if len(homographies) > 0 else np.eye(3)
        
        for i in range(1, num_images):
            H_chain = np.eye(3)
            # Chain backwards from image i to image 1
            for j in range(i-1, -1, -1):
                if j < len(homographies):
                    H_chain = np.linalg.inv(homographies[j]) @ H_chain
            H_transforms[i] = H_chain
        
        # Convert to list
        transformations = []
        for i in range(num_images):
            if i in H_transforms:
                transformations.append(H_transforms[i])
            else:
                transformations.append(np.eye(3))
        
        return transformations

    def create_mosaic_with_image1_reference(self, images, sequential_results):
        """
        Create mosaic using image 1 as reference (before optimization).
        
        Args:
            images: List of images
            sequential_results: Sequential matching results
            
        Returns:
            Mosaic image
        """
        print("Creating mosaic with Image 1 as reference...")
        
        transformations = self.transformation_sequence(sequential_results)
        canvas_width, canvas_height, translation = self.calculate_canvas_bounds(
            images, transformations
        )
        mosaic = self.warp_images_to_mosaic(
            images, transformations, canvas_width, canvas_height, translation
        )
        
        return mosaic

    def build_pose_transformations(self, optimized_poses, sequential_results):
        """
        Build transformation matrices from GTSAM-optimized poses.
        
        Args:
            optimized_poses: Optimized poses from GTSAM
            sequential_results: Sequential matching results
            
        Returns:
            List of transformation matrices
        """
        relative_transforms = []
        
        # Compute relative transforms between consecutive poses
        for i in range(len(optimized_poses) - 1):
            try:
                pose1 = optimized_poses[i]
                pose2 = optimized_poses[i + 1]
                
                # Inverse of pose1
                x1, y1, theta1 = pose1
                cos_theta1 = np.cos(-theta1)
                sin_theta1 = np.sin(-theta1)
                
                x1_inv = -(cos_theta1 * x1 - sin_theta1 * y1)
                y1_inv = -(sin_theta1 * x1 + cos_theta1 * y1)
                theta1_inv = -theta1
                
                pose1_inv = np.array([x1_inv, y1_inv, theta1_inv])
                
                # Compose with pose2
                rel_pose = self.pose_compose(pose1_inv, pose2)
                
                # Convert to transformation matrix
                x_rel, y_rel, theta_rel = rel_pose
                cos_dt = np.cos(theta_rel)
                sin_dt = np.sin(theta_rel)
                
                rel_transform = np.array([
                    [cos_dt, -sin_dt, -x_rel],
                    [sin_dt,  cos_dt, y_rel],
                    [0,       0,      1]
                ], dtype=np.float64)
                
                relative_transforms.append(rel_transform)
            except Exception as e:
                print(f"Failed to compute relative transform for pair {i} and {i+1}: {e}")
                relative_transforms.append(np.eye(3))
        
        num_images = len(optimized_poses)
        H_transforms = {}
        
        H_transforms[0] = relative_transforms[0] if len(relative_transforms) > 0 else np.eye(3)
        
        for i in range(1, num_images):
            H_chain = np.eye(3)
            for j in range(i - 1, -1, -1):
                if j < len(relative_transforms):
                    H_chain = np.linalg.inv(relative_transforms[j]) @ H_chain
            H_transforms[i] = H_chain
        
        transformations = [H_transforms[i] for i in range(num_images)]
        return transformations

    def create_mosaic_with_gtsam_poses(self, images, optimized_poses, sequential_results):
        """
        Create mosaic using GTSAM-optimized poses.
        
        Args:
            images: List of images
            optimized_poses: Optimized poses from GTSAM
            sequential_results: Sequential matching results
            
        Returns:
            Mosaic image
        """
        print("Creating mosaic with GTSAM-optimized poses...")
        
        transformations = self.build_pose_transformations(optimized_poses, sequential_results)
        
        canvas_width, canvas_height, translation = self.calculate_canvas_bounds(
            images, transformations
        )
        mosaic = self.warp_images_to_mosaic(
            images, transformations, canvas_width, canvas_height, translation
        )
        
        return mosaic