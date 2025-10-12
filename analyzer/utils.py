"""
Utility functions for pose analysis.
Contains mathematical helpers for angle calculation, distance metrics, and smoothing.
"""

import numpy as np
from typing import List, Tuple


def calculate_angle(point_a: Tuple[float, float], 
                   point_b: Tuple[float, float], 
                   point_c: Tuple[float, float]) -> float:
    """
    Calculate angle at point_b formed by points a-b-c.
    
    Uses vector dot product formula: cos(θ) = (u·v) / (|u||v|)
    Returns angle in degrees (0-180).
    
    Args:
        point_a: First point (x, y)
        point_b: Vertex point (x, y)
        point_c: Third point (x, y)
    
    Returns:
        Angle in degrees
    
    Example:
        For elbow angle: calculate_angle(shoulder, elbow, wrist)
    """
    # Create vectors from vertex point
    vector_ba = np.array([point_a[0] - point_b[0], point_a[1] - point_b[1]])
    vector_bc = np.array([point_c[0] - point_b[0], point_c[1] - point_b[1]])
    
    # Calculate angle using dot product
    # Added epsilon to prevent division by zero on perfectly aligned points
    cosine_angle = np.dot(vector_ba, vector_bc) / (
        np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc) + 1e-6
    )
    
    # Clamp to [-1, 1] to handle floating point errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def calculate_distance(point_a: Tuple[float, float], 
                      point_b: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point_a: First point (x, y)
        point_b: Second point (x, y)
    
    Returns:
        Distance as float
    """
    return np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)


def normalize_by_torso(value: float, torso_height: float) -> float:
    """
    Normalize a distance value by torso height for scale-invariance.
    
    This makes pose detection robust across different video resolutions
    and distances from camera. A person closer to camera has larger absolute
    pixel distances, but same relative proportions.
    
    Args:
        value: Raw distance value to normalize
        torso_height: Height of torso (shoulder to hip distance)
    
    Returns:
        Normalized value (typically 0.0 to 2.0 range)
    """
    return value / (torso_height + 1e-6)


def smooth_signal(values: List[float], window_size: int = 5) -> List[float]:
    """
    Apply moving average smoothing to reduce noise.
    
    Used to smooth landmark positions and reduce jitter from MediaPipe.
    Window size of 5 means we average current frame with 2 before and 2 after.
    
    Args:
        values: List of values to smooth
        window_size: Size of moving average window (must be odd)
    
    Returns:
        Smoothed values (same length as input)
    """
    if len(values) < window_size:
        return values
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(values)):
        # Handle edges by using available data
        start_idx = max(0, i - half_window)
        end_idx = min(len(values), i + half_window + 1)
        window = values[start_idx:end_idx]
        smoothed.append(np.mean(window))
    
    return smoothed


def temporal_filter(detections: List[bool], min_consecutive: int = 3) -> List[bool]:
    """
    Filter pose detections to require minimum consecutive frames.
    
    Reduces false positives by ensuring a pose is held for at least N frames.
    For example, if someone briefly passes through T-pose while transitioning,
    we don't want to count it.
    
    Args:
        detections: Boolean list of frame-by-frame detections
        min_consecutive: Minimum consecutive True values required
    
    Returns:
        Filtered boolean list
    """
    filtered = [False] * len(detections)
    
    consecutive_count = 0
    for i, detected in enumerate(detections):
        if detected:
            consecutive_count += 1
            if consecutive_count >= min_consecutive:
                # Mark the last min_consecutive frames as valid
                for j in range(max(0, i - min_consecutive + 1), i + 1):
                    filtered[j] = True
        else:
            consecutive_count = 0
    
    return filtered


def calculate_symmetry_score(left_points: List[Tuple[float, float]], 
                             right_points: List[Tuple[float, float]],
                             center_x: float) -> float:
    """
    Calculate left-right symmetry score for dance movements.
    
    Good dancers often have symmetric movements. This calculates how similar
    left and right side movements are, mirrored across body centerline.
    
    Args:
        left_points: List of left side keypoints (e.g., left wrist, left knee)
        right_points: Corresponding right side keypoints
        center_x: X-coordinate of body center (usually mid-shoulder)
    
    Returns:
        Symmetry score from 0.0 (asymmetric) to 1.0 (perfectly symmetric)
    """
    if len(left_points) != len(right_points):
        return 0.0
    
    total_error = 0.0
    for left, right in zip(left_points, right_points):
        # Mirror left point across center
        mirrored_left_x = 2 * center_x - left[0]
        # Calculate distance between mirrored left and actual right
        distance = calculate_distance((mirrored_left_x, left[1]), right)
        total_error += distance
    
    # Normalize by number of points and convert to 0-1 score
    # Using exponential decay: score = e^(-error)
    avg_error = total_error / len(left_points)
    symmetry_score = np.exp(-avg_error * 10)  # Scale factor of 10 tuned empirically
    
    return float(symmetry_score)