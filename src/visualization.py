"""
Visualization utilities for underwater photomosaicing SLAM
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_mosaic_comparison(mosaic_before, mosaic_after, save_path=None):
    """
    Plot side-by-side comparison of mosaics before and after optimization.
    
    Args:
        mosaic_before: Mosaic before optimization
        mosaic_after: Mosaic after optimization
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 12))
    
    ax1.imshow(cv2.cvtColor(mosaic_before, cv2.COLOR_BGR2RGB))
    ax1.set_title('Mosaic: Before GTSAM Optimization', fontsize=16, weight='bold')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(mosaic_after, cv2.COLOR_BGR2RGB))
    ax2.set_title('Mosaic: After GTSAM Optimization', fontsize=16, weight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_drift_corrections(initial_poses, optimized_poses, save_path=None):
    """
    Plot drift corrections as arrows from initial to optimized positions.
    
    Args:
        initial_poses: Initial pose estimates
        optimized_poses: Optimized poses
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot initial trajectory
    initial_xy = np.array([[p[0], p[1]] for p in initial_poses])
    ax.plot(initial_xy[:, 0], initial_xy[:, 1], 'r--', 
           linewidth=2, alpha=0.6, label='Before Optimization')
    
    # Plot optimized trajectory
    optimized_xy = np.array([[p[0], p[1]] for p in optimized_poses])
    ax.plot(optimized_xy[:, 0], optimized_xy[:, 1], 'g-', 
           linewidth=2, alpha=0.8, label='After Optimization')
    
    # Plot correction arrows
    for i, (initial, optimized) in enumerate(zip(initial_poses, optimized_poses)):
        dx = optimized[0] - initial[0]
        dy = optimized[1] - initial[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > 1.0:  # Only show significant corrections
            ax.arrow(initial[0], initial[1], dx, dy, 
                    head_width=5, head_length=8, 
                    fc='blue', ec='blue', alpha=0.5)
            
            # Annotate node
            ax.text(initial[0], initial[1], f'{i+1}', 
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X Position (pixels)', fontsize=14, weight='bold')
    ax.set_ylabel('Y Position (pixels)', fontsize=14, weight='bold')
    ax.set_title('Drift Corrections from GTSAM Optimization', 
                fontsize=16, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_inlier_statistics(sequential_results, non_sequential_results, save_path=None):
    """
    Plot inlier ratio statistics.
    
    Args:
        sequential_results: Sequential matching results
        non_sequential_results: Non-sequential matching results
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sequential pairs
    seq_ratios = [r['result']['inlier_ratio'] * 100 
                 for r in sequential_results if r['success']]
    ax1.bar(range(1, len(seq_ratios)+1), seq_ratios, color='steelblue', alpha=0.7)
    ax1.axhline(y=40, color='r', linestyle='--', linewidth=2, label='Threshold (40%)')
    ax1.set_xlabel('Sequential Pair', fontsize=12, weight='bold')
    ax1.set_ylabel('Inlier Ratio (%)', fontsize=12, weight='bold')
    ax1.set_title('Sequential Pair Inlier Ratios', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Non-sequential pairs
    loop_ratios = [r['result']['inlier_ratio'] * 100 
                  for r in non_sequential_results if r['result'].get('success', False)]
    valid = [r['success'] for r in non_sequential_results]
    colors = ['green' if v else 'red' for v in valid]
    
    ax2.bar(range(1, len(non_sequential_results)+1), 
           [r['result']['inlier_ratio'] * 100 if r['result'].get('success', False) else 0
            for r in non_sequential_results],
           color=colors, alpha=0.7)
    ax2.axhline(y=40, color='r', linestyle='--', linewidth=2, label='Threshold (40%)')
    ax2.set_xlabel('Non-Sequential Pair', fontsize=12, weight='bold')
    ax2.set_ylabel('Inlier Ratio (%)', fontsize=12, weight='bold')
    ax2.set_title('Non-Sequential Pair Inlier Ratios (Loop Closures)', 
                 fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_uncertainty_ellipses(poses, covariances, title, save_path=None):
    """
    Plot poses with uncertainty ellipses.
    
    Args:
        poses: List of poses
        covariances: List of covariance matrices
        title: Plot title
        save_path: Optional path to save figure
    """
    from matplotlib.patches import Ellipse
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot trajectory
    xy = np.array([[p[0], p[1]] for p in poses])
    ax.plot(xy[:, 0], xy[:, 1], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
    
    # Plot uncertainty ellipses
    for i, (pose, cov) in enumerate(zip(poses, covariances)):
        x, y, theta = pose
        
        # Compute ellipse parameters
        eigenvals, eigenvecs = np.linalg.eigh(cov[:2, :2])
        eigenvals = np.maximum(eigenvals, 0.1)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 4 * np.sqrt(eigenvals)
        
        ellipse = Ellipse((x, y), width, height, angle=angle,
                         facecolor='cyan', alpha=0.3, edgecolor='blue',
                         linewidth=1.5, zorder=2)
        ax.add_patch(ellipse)
        
        # Node marker
        ax.plot(x, y, 'ro', markersize=8, zorder=3)
        ax.text(x, y, f'{i+1}', fontsize=10, ha='center', va='center',
               color='white', weight='bold', zorder=4)
    
    ax.set_xlabel('X Position (pixels)', fontsize=14, weight='bold')
    ax.set_ylabel('Y Position (pixels)', fontsize=14, weight='bold')
    ax.set_title(title, fontsize=16, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()