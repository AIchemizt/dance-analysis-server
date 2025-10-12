"""
Movement analysis module for temporal and intensity metrics.
Analyzes how body parts move over time, not just static poses.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from analyzer.utils import calculate_distance, smooth_signal, calculate_symmetry_score


class MovementAnalyzer:
    """
    Analyzes movement patterns across video frames.
    Provides insights beyond static pose detection:
    - Movement intensity heatmap (which body parts moved most)
    - Symmetry score (left vs right side movement balance)
    - Temporal smoothing of pose detections
    - Movement velocity analysis
    
    These metrics are valuable for dance analysis because they capture
    the dynamic quality of movement, not just frozen poses.
    """
    
    def __init__(self):
        self.landmark_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def calculate_movement_intensity(self, 
                                    frame_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate how much each body part moved throughout the video.
        Algorithm:
        1. For each landmark, track position across all frames
        2. Calculate total distance traveled (sum of frame-to-frame distances)
        3. Normalize by number of frames to get average movement per frame
        
        Args:
            frame_data: List of frame dictionaries from PoseDetector
        
        Returns:
            Dictionary mapping body part names to movement intensity scores
            Higher scores = more movement
        
        Use case:
        A dancer doing mostly arm movements will have high intensity for
        wrists/elbows but low for ankles/knees. This helps identify
        the "focus" of the choreography.
        """
        # Initialize movement tracking for key landmarks
        landmark_indices = {
            'left_wrist': 15,
            'right_wrist': 16,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
        
        movement_intensity = {name: 0.0 for name in landmark_indices.keys()}
        
        # Track positions frame-by-frame
        landmark_trajectories = {name: [] for name in landmark_indices.keys()}

        # Collect all positions
        for frame in frame_data:
            if frame['landmarks'] is None:
                continue
            
            for name, idx in landmark_indices.items():
                landmark = frame['landmarks'][idx]
                landmark_trajectories[name].append((landmark.x, landmark.y))
    
        # Calculate total distance traveled for each landmark
        for name, positions in landmark_trajectories.items():
            if len(positions) < 2:
                continue
            
            total_distance = 0.0
            for i in range(1, len(positions)):
                distance = calculate_distance(positions[i-1], positions[i])
                total_distance += distance
            
            # Normalize by number of frames to get average movement per frame
            movement_intensity[name] = total_distance / len(positions)
    
        return movement_intensity

    def calculate_overall_symmetry(self, 
                                 frame_data: List[Dict[str, Any]]) -> float:
        """
        Calculate average left-right movement symmetry across entire video.
        Good dance technique often involves symmetric movements (though not always).
        This metric helps identify balance issues or intentionally asymmetric choreo.
        
        Args:
            frame_data: List of frame dictionaries
        
        Returns:
            Average symmetry score (0.0 = asymmetric, 1.0 = perfectly symmetric)
        
        Implementation note:
        I calculate symmetry frame-by-frame then average, rather than comparing
        total movement distances. This captures temporal symmetry, not just
        overall distance symmetry.
        """
        symmetry_scores = []
        
        for frame in frame_data:
            if frame['landmarks'] is None:
                continue
            
            landmarks = frame['landmarks']
            
            # Get paired landmarks for symmetry comparison
            left_points = [
                (landmarks[15].x, landmarks[15].y),  # left_wrist
                (landmarks[13].x, landmarks[13].y),  # left_elbow
                (landmarks[27].x, landmarks[27].y),  # left_ankle
                (landmarks[25].x, landmarks[25].y),  # left_knee
            ]
            
            right_points = [
                (landmarks[16].x, landmarks[16].y),  # right_wrist
                (landmarks[14].x, landmarks[14].y),  # right_elbow
                (landmarks[28].x, landmarks[28].y),  # right_ankle
                (landmarks[26].x, landmarks[26].y),  # right_knee
            ]
            
            # Calculate body center (midpoint of shoulders)
            center_x = (landmarks[11].x + landmarks[12].x) / 2
            
            frame_symmetry = calculate_symmetry_score(left_points, right_points, center_x)
            symmetry_scores.append(frame_symmetry)
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.0

    def detect_movement_peaks(self, 
                             frame_data: List[Dict[str, Any]], 
                             threshold: float = 0.02) -> List[int]:
        """
        Detect frames with significant movement (peaks in movement velocity).
        
        Useful for identifying key moments in dance: jumps, spins, dramatic gestures.
        
        Args:
            frame_data: List of frame dictionaries
            threshold: Minimum movement to consider a "peak" (normalized coordinates)
        
        Returns:
            List of frame numbers where significant movement occurred
        
        Algorithm:
        1. Calculate center-of-mass (average of all visible landmarks)
        2. Track frame-to-frame movement of center-of-mass
        3. Identify frames where movement exceeds threshold
        """
        movement_velocities = []
        peak_frames = []
        
        # Calculate center of mass for each frame
        centers = []
        for frame in frame_data:
            if frame['landmarks'] is None:
                centers.append(None)
                continue
            
            # Average position of all landmarks
            x_coords = [lm.x for lm in frame['landmarks']]
            y_coords = [lm.y for lm in frame['landmarks']]
            center = (np.mean(x_coords), np.mean(y_coords))
            centers.append(center)
        
        # Calculate frame-to-frame velocity
        for i in range(1, len(centers)):
            if centers[i] is None or centers[i-1] is None:
                movement_velocities.append(0.0)
                continue
            
            velocity = calculate_distance(centers[i-1], centers[i])
            movement_velocities.append(velocity)
        
        # Smooth velocities to reduce noise
        if len(movement_velocities) > 5:
            smoothed_velocities = smooth_signal(movement_velocities, window_size=5)
        else:
            smoothed_velocities = movement_velocities
        
        # Identify peaks
        for i, velocity in enumerate(smoothed_velocities):
            if velocity > threshold:
                peak_frames.append(i)
        
        return peak_frames

    def temporal_pose_filter(self, 
                            pose_detections: Dict[str, List[bool]], 
                            min_consecutive: int = 3) -> Dict[str, List[int]]:
        """
        Apply temporal filtering to reduce false positive pose detections.
        A pose must be held for at least min_consecutive frames to count.
        This prevents fleeting "accidental" poses during transitions.
        
        Args:
            pose_detections: Dict mapping pose names to boolean detection lists
            min_consecutive: Minimum frames to confirm a pose
        
        Returns:
            Dict mapping pose names to lists of confirmed frame numbers
        
        Example:
        Input: {'T-Pose': [False, False, True, True, True, False, True, False]}
        Output: {'T-Pose': [2, 3, 4]}  (frames 2-4 had 3 consecutive detections)
        """
        filtered_results = {}
        
        for pose_name, detections in pose_detections.items():
            confirmed_frames = []
            consecutive_count = 0
            start_frame = 0
            
            for i, detected in enumerate(detections):
                if detected:
                    if consecutive_count == 0:
                        start_frame = i
                    consecutive_count += 1
                    
                    # Once we hit the threshold, mark all frames in the sequence
                    if consecutive_count >= min_consecutive:
                        # Add frames that weren't already added
                        for frame_num in range(start_frame, i + 1):
                            if frame_num not in confirmed_frames:
                                confirmed_frames.append(frame_num)
                else:
                    consecutive_count = 0
            
            filtered_results[pose_name] = sorted(confirmed_frames)
        
        return filtered_results