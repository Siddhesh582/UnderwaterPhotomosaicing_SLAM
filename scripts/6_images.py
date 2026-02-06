"""
Run underwater photomosaicing SLAM on 6-image dataset
"""

import sys
import os
import glob
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import ImageFeatureMatching
from src.utils import (
    print_matching_summary,
    print_optimization_summary,
    save_results
)
from src.visualization import plot_mosaic_comparison, plot_drift_corrections


def main():
    print("\nUNDERWATER PHOTOMOSAICING SLAM - 6 Images")
    
    # Initialize matcher
    matcher = ImageFeatureMatching()
    
    # Load images
    image_dir = 'data/skerki_6'
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    
    print(f"\nLoaded {len(image_paths)} images:")
    for i, path in enumerate(image_paths):
        print(f"  {i+1}. {os.path.basename(path)}")
    
    # Read images
    images = matcher.read_image(image_paths, 
                               titles=[f"Image {i+1}" for i in range(len(image_paths))],
                               show=False)
    
    # ==================== SEQUENTIAL MATCHING ====================
    print("\nSEQUENTIAL MATCHING")
    
    sequential_results = []
    successful_pairs = 0
    
    for i in range(len(images) - 1):
        print(f"\nMatching Image {i+1} and Image {i+2}")
        try:
            result = matcher.register_image_pair(images[i], images[i + 1], visualize=False)
            
            success = result['success']
            if success:
                successful_pairs += 1
                print(f"{result['inlier_count']}/{result['total_matches']} inliers "
                      f"({result['inlier_ratio']:.1%})")
            else:
                print(f"Failed: {result['error']}")
            
            sequential_results.append({
                'pair': (i, i+1),
                'pair_name': f"{i+1}-{i+2}",
                'result': result,
                'success': success
            })
        
        except Exception as e:
            print(f"Exception occurred: {e}")
            sequential_results.append({
                'pair': (i, i+1),
                'pair_name': f"{i+1}-{i+2}",
                'result': {'success': False, 'error': str(e)},
                'success': False
            })
    
    # ==================== NON-SEQUENTIAL MATCHING ====================
    print("\nNON-SEQUENTIAL MATCHING (Loop Closure Detection)")
    
    # Define non-sequential pairs for 6 images
    non_sequential_pairs = [(0, 5), (1, 4), (1, 3), (2, 4), (0, 4), (1, 5)]
    non_sequential_results = []
    loop_closure_pairs = 0
    min_inlier_ratio = 0.4
    
    for idx, (i, j) in enumerate(non_sequential_pairs):
        print(f"\n[{idx+1}/{len(non_sequential_pairs)}] Matching non-sequential Image {i+1} and Image {j+1}")
        try:
            result = matcher.register_image_pair(images[i], images[j], visualize=False)
            
            if result['success'] and result['inlier_ratio'] >= min_inlier_ratio:
                loop_closure_pairs += 1
                print(f"  Loop Closure: {result['inlier_count']}/{result['total_matches']} inliers "
                      f"({result['inlier_ratio']:.1%} >= {min_inlier_ratio:.0%})")
                is_loop_closure = True
            elif result['success']:
                print(f"  No Loop Closure: {result['inlier_count']}/{result['total_matches']} inliers "
                      f"({result['inlier_ratio']:.1%} < {min_inlier_ratio:.0%})")
                is_loop_closure = False
            else:
                print(f"  Failed: {result['error']}")
                is_loop_closure = False
            
            non_sequential_results.append({
                'pair': (i, j),
                'pair_name': f"{i+1}-{j+1}",
                'result': result,
                'success': result['success'] and result['inlier_ratio'] >= min_inlier_ratio,
                'is_potential_loop_closure': is_loop_closure
            })
        
        except Exception as e:
            print(f"  Exception occurred: {e}")
            non_sequential_results.append({
                'pair': (i, j),
                'pair_name': f"{i+1}-{j+1}",
                'result': {'success': False, 'error': str(e)},
                'success': False,
                'is_potential_loop_closure': False
            })
    
    # Print summary
    print_matching_summary(sequential_results, non_sequential_results)
    
    # ==================== FACTOR GRAPH CONSTRUCTION ====================
    print("\nFACTOR GRAPH CONSTRUCTION")
    
    factor_graph_data = matcher.plot_complete_factor_graph(
        sequential_results,
        non_sequential_results,
        save_path='results/factor_graph_6_images_before_gtsam.png'
    )
    
    # ==================== GTSAM OPTIMIZATION ====================
    print("\nGTSAM POSE GRAPH OPTIMIZATION")
    
    optimized_poses, optimized_covariances = matcher.solve_with_gtsam(
        factor_graph_data['vertices'],
        factor_graph_data['edges']
    )
    
    matcher.plot_after_gtsam_optimization(
        optimized_poses,
        optimized_covariances,
        factor_graph_data['edges'],
        save_path='results/factor_graph_6_images_after_gtsam.png'
    )
    
    # Print optimization summary
    print_optimization_summary(
        factor_graph_data['vertices'],
        optimized_poses,
        factor_graph_data['vertex_covariances'],
        optimized_covariances
    )
    
    # ==================== MOSAIC GENERATION ====================
    print("\nMOSAIC GENERATION")
    
    print("\nCreating mosaic before optimization...")
    mosaic_before = matcher.create_mosaic_with_image1_reference(images, sequential_results)
    
    print("Creating mosaic after optimization...")
    mosaic_after = matcher.create_mosaic_with_gtsam_poses(images, optimized_poses, sequential_results)
    
    # Plot comparison
    plot_mosaic_comparison(mosaic_before, mosaic_after, 
                          save_path='results/mosaic_comparison_6_images.png')
    
    # Plot drift corrections
    plot_drift_corrections(factor_graph_data['vertices'], optimized_poses,
                          save_path='results/drift_corrections_6_images.png')
    
    # Save all results
    os.makedirs('results', exist_ok=True)
    save_results('results', factor_graph_data, optimized_poses, 
                optimized_covariances, mosaic_before, mosaic_after)
    
    print(f"\nResults saved to: results/")


if __name__ == "__main__":
    main()