"""
Underwater Photomosaicing with Pose Graph SLAM
"""

from .pipeline import ImageFeatureMatching
from .utils import (
    load_images_from_directory,
    img_sort_key,
    compute_drift_statistics,
    compute_uncertainty_reduction,
    print_matching_summary,
    print_optimization_summary,
    save_results
)
from .visualization import (
    plot_mosaic_comparison,
    plot_drift_corrections,
    plot_inlier_statistics,
    plot_uncertainty_ellipses
)

__version__ = "1.0.0"
__author__ = "Siddhesh Shingate"

__all__ = [
    'ImageFeatureMatching',
    'load_images_from_directory',
    'img_sort_key',
    'compute_drift_statistics',
    'compute_uncertainty_reduction',
    'print_matching_summary',
    'print_optimization_summary',
    'save_results',
    'plot_mosaic_comparison',
    'plot_drift_corrections',
    'plot_inlier_statistics',
    'plot_uncertainty_ellipses'
]