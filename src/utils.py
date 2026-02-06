"""
Utility functions for underwater photomosaicing SLAM
"""

import numpy as np
import os
import glob


def load_images_from_directory(image_dir, pattern="*.tif", sort_key=None):
    """
    Load images from directory with optional custom sorting.
    
    Args:
        image_dir: Directory containing images
        pattern: Glob pattern for image files
        sort_key: Optional sorting function
        
    Returns:
        Sorted list of image paths
    """
    image_paths = glob.glob(os.path.join(image_dir, pattern))
    
    if sort_key is None:
        # Default: sort by filename
        image_paths = sorted(image_paths)
    else:
        image_paths = sorted(image_paths, key=sort_key)
    
    return image_paths


def img_sort_key(filename):
    """
    Custom sorting key for images named like 'img_1.tif', 'img_2.tif', etc.
    
    Args:
        filename: Image filename
        
    Returns:
        Numeric sort key
    """
    basename = os.path.basename(filename)
    
    if basename.startswith('img_') and basename.endswith('.tif'):
        try:
            num = int(basename.split('_')[1].split('.')[0])
            return num
        except:
            return float('inf')
    else:
        return float('inf')


def compute_drift_statistics(initial_poses, optimized_poses):
    """
    Compute drift correction statistics.
    
    Args:
        initial_poses: Initial pose estimates
        optimized_poses: GTSAM-optimized poses
        
    Returns:
        Dictionary with drift statistics
    """
    corrections = []
    
    for i, (initial, optimized) in enumerate(zip(initial_poses, optimized_poses)):
        dx = optimized[0] - initial[0]
        dy = optimized[1] - initial[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        corrections.append({
            'node': i + 1,
            'dx': dx,
            'dy': dy,
            'distance': distance
        })
    
    total_correction = sum([c['distance'] for c in corrections])
    max_correction = max([c['distance'] for c in corrections])
    mean_correction = np.mean([c['distance'] for c in corrections])
    
    return {
        'corrections': corrections,
        'total': total_correction,
        'max': max_correction,
        'mean': mean_correction
    }


def compute_uncertainty_reduction(initial_covariances, optimized_covariances):
    """
    Compute uncertainty reduction from optimization.
    
    Args:
        initial_covariances: Initial covariance matrices
        optimized_covariances: Optimized covariance matrices
        
    Returns:
        Uncertainty reduction percentage
    """
    initial_total = sum([np.linalg.det(cov[:2, :2]) for cov in initial_covariances])
    optimized_total = sum([np.linalg.det(cov[:2, :2]) for cov in optimized_covariances])
    
    reduction = ((optimized_total - initial_total) / initial_total) * 100
    
    return reduction


def print_matching_summary(sequential_results, non_sequential_results):
    """
    Print summary statistics for matching results.
    
    Args:
        sequential_results: Sequential matching results
        non_sequential_results: Non-sequential matching results
    """
    print("\n" + "="*70)
    print("MATCHING SUMMARY")
    print("="*70)
    
    # Sequential pairs
    successful_seq = sum([1 for r in sequential_results if r['success']])
    print(f"\nSequential Pairs: {successful_seq}/{len(sequential_results)} successful")
    
    inlier_ratios = [r['result']['inlier_ratio'] for r in sequential_results if r['success']]
    if inlier_ratios:
        print(f"  Inlier Ratio: {min(inlier_ratios):.1%} - {max(inlier_ratios):.1%} "
              f"(mean: {np.mean(inlier_ratios):.1%})")
    
    # Non-sequential pairs
    loop_closures = sum([1 for r in non_sequential_results if r.get('success', False)])
    print(f"\nLoop Closures: {loop_closures}/{len(non_sequential_results)} validated")
    
    loop_ratios = [r['result']['inlier_ratio'] for r in non_sequential_results 
                   if r.get('success', False)]
    if loop_ratios:
        print(f"  Inlier Ratio: {min(loop_ratios):.1%} - {max(loop_ratios):.1%} "
              f"(mean: {np.mean(loop_ratios):.1%})")
    
    print("="*70 + "\n")


def print_optimization_summary(initial_poses, optimized_poses, 
                               initial_covariances, optimized_covariances):
    """
    Print optimization results summary.
    
    Args:
        initial_poses: Initial poses
        optimized_poses: Optimized poses
        initial_covariances: Initial covariances
        optimized_covariances: Optimized covariances
    """
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    
    # Drift corrections
    drift_stats = compute_drift_statistics(initial_poses, optimized_poses)
    print(f"\nDrift Corrections:")
    print(f"  Maximum: {drift_stats['max']:.1f} pixels (Node {[c for c in drift_stats['corrections'] if c['distance'] == drift_stats['max']][0]['node']})")
    print(f"  Mean: {drift_stats['mean']:.1f} pixels")
    print(f"  Total: {drift_stats['total']:.1f} pixels")
    
    # Uncertainty reduction
    uncertainty_reduction = compute_uncertainty_reduction(
        initial_covariances, optimized_covariances
    )
    print(f"\nUncertainty Reduction: {uncertainty_reduction:.1f}%")
    
    print("="*70 + "\n")


def save_results(output_dir, factor_graph_data, optimized_poses, 
                optimized_covariances, mosaic_before, mosaic_after):
    """
    Save all results to output directory.
    
    Args:
        output_dir: Output directory path
        factor_graph_data: Factor graph data dictionary
        optimized_poses: Optimized poses
        optimized_covariances: Optimized covariances
        mosaic_before: Mosaic before optimization
        mosaic_after: Mosaic after optimization
    """
    import cv2
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mosaics
    cv2.imwrite(os.path.join(output_dir, 'mosaic_before.png'), mosaic_before)
    cv2.imwrite(os.path.join(output_dir, 'mosaic_after.png'), mosaic_after)
    
    # Save poses
    np.save(os.path.join(output_dir, 'initial_poses.npy'), 
           factor_graph_data['vertices'])
    np.save(os.path.join(output_dir, 'optimized_poses.npy'), optimized_poses)
    
    # Save covariances
    np.save(os.path.join(output_dir, 'initial_covariances.npy'), 
           factor_graph_data['vertex_covariances'])
    np.save(os.path.join(output_dir, 'optimized_covariances.npy'), 
           optimized_covariances)
    
    print(f"Results saved to: {output_dir}")