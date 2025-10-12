"""
Pose detection wrapper using MediaPipe Pose.
Handles video processing and landmark extraction with confidence filtering.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict, Any


class PoseDetector:
    """
    Wrapper around MediaPipe Pose for extracting body landmarks from video.
    
    This class handles the low-level video processing and provides clean
    interfaces for higher-level pose analysis modules.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Pose detector.
        
        Args:
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for frame-to-frame tracking
        
        Note:
            Higher confidence values reduce false positives but may miss
            challenging poses. Values of 0.5 are reasonable defaults based
            on MediaPipe documentation and my testing.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Video mode for temporal consistency
            model_complexity=1,        # Medium complexity (0=lite, 1=full, 2=heavy)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def process_video(self, video_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Process entire video and extract landmarks for each frame.
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of frame data dictionaries, each containing:
                - frame_number: int
                - landmarks: List of 33 landmark objects (or None if detection failed)
                - timestamp: float (seconds)
            Returns None if video cannot be opened.
        
        Implementation Notes:
            - Initially I processed all frames, but for 60fps videos this was
              overkill. Now I process every frame but allow callers to subsample.
            - MediaPipe returns landmarks in normalized coordinates [0, 1] which
              is great for scale-invariance but requires conversion for pixel ops.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_data = []
        frame_number = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Convert BGR (OpenCV format) to RGB (MediaPipe format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(image_rgb)
            
            # Store results even if no pose detected (for temporal analysis)
            frame_data.append({
                'frame_number': frame_number,
                'landmarks': results.pose_landmarks.landmark if results.pose_landmarks else None,
                'timestamp': frame_number / fps if fps > 0 else frame_number * 0.033  # Fallback to ~30fps
            })
            
            frame_number += 1
        
        cap.release()
        self.pose.close()
        
        return frame_data
    
    def get_landmark_position(self, landmarks, landmark_index: int) -> Optional[tuple]:
        """
        Extract (x, y, z) position of a specific landmark.
        
        Args:
            landmarks: MediaPipe landmark list
            landmark_index: Index of desired landmark (use mp.solutions.pose.PoseLandmark)
        
        Returns:
            Tuple of (x, y, z) in normalized coordinates [0, 1], or None if invalid
        
        Note:
            z-coordinate represents depth (distance from camera). Negative z means
            the landmark is closer to camera than the hip center.
        """
        if landmarks is None or landmark_index >= len(landmarks):
            return None
        
        landmark = landmarks[landmark_index]
        
        # Filter out low-confidence landmarks
        # visibility < 0.5 means MediaPipe isn't confident the point is visible
        if landmark.visibility < 0.5:
            return None
        
        return (landmark.x, landmark.y, landmark.z)
    
    def calculate_torso_height(self, landmarks) -> float:
        """
        Calculate torso height for normalization.
        
        Uses average of left and right shoulder-to-hip distances.
        This is more robust than using just one side (occlusion handling).
        
        Args:
            landmarks: MediaPipe landmark list
        
        Returns:
            Torso height in normalized coordinates (typically 0.2-0.4)
        """
        left_shoulder = self.get_landmark_position(
            landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        )
        right_shoulder = self.get_landmark_position(
            landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        )
        left_hip = self.get_landmark_position(
            landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value
        )
        right_hip = self.get_landmark_position(
            landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP.value
        )
        
        # Need at least one shoulder and one hip
        if not (left_shoulder or right_shoulder) or not (left_hip or right_hip):
            return 0.3  # Default fallback value
        
        # Calculate both sides and average
        heights = []
        if left_shoulder and left_hip:
            heights.append(abs(left_shoulder[1] - left_hip[1]))
        if right_shoulder and right_hip:
            heights.append(abs(right_shoulder[1] - right_hip[1]))
        
        return np.mean(heights) if heights else 0.3