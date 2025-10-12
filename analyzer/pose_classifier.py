"""
Multi-pose classification module.
Detects specific dance poses using geometric rules and angle calculations.
"""

import numpy as np
from typing import Dict, List, Optional
from analyzer.utils import calculate_angle, calculate_distance, normalize_by_torso
from analyzer.pose_detector import PoseDetector
import mediapipe as mp


class PoseClassifier:
    """
    Classifies body poses from MediaPipe landmarks.
    
    Currently implements 5 fundamental poses:
    1. T-Pose (arms straight out to sides)
    2. Arms-Up (both arms raised above head)
    3. Squat (knees bent, hips lowered)
    4. Lunge (one leg forward, one back, both knees bent)
    5. Jump (both feet off ground, detected via rapid vertical movement)
    
    Design Philosophy:
    - Use angle-based detection for robustness (angles are scale-invariant)
    - Normalize distances by torso height for different body sizes
    - Allow configurable tolerances for each pose
    - Return confidence scores, not just boolean detections
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
        # Tolerance values tuned through experimentation on test videos
        # These represent "good enough" thresholds that balance precision/recall
        self.angle_tolerance = 20  # degrees
        self.position_tolerance_ratio = 0.15  # as fraction of torso height
        
    def detect_t_pose(self, landmarks, torso_height: float) -> Dict[str, any]:
        """
        Detect T-Pose: arms extended horizontally to sides.
        
        Criteria:
        1. Both elbows should be nearly straight (angle > 160°)
        2. Shoulders-elbows-wrists should form horizontal line
        3. Wrists should be level with shoulders (±15% torso height)
        4. Arms should extend outward from body
        
        Args:
            landmarks: MediaPipe pose landmarks
            torso_height: Normalized torso height for scaling
        
        Returns:
            Dict with 'detected' (bool) and 'confidence' (0.0-1.0)
        
        Debugging notes:
        - Initially used strict y-tolerance of 0.05, but this failed when dancers
          had slight arm tilt. Increased to 0.15 * torso_height.
        - Added elbow angle check after false positives with bent arms.
        """
        # Extract required landmarks
        left_shoulder = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        right_shoulder = (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
        left_elbow = (landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
        right_elbow = (landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
        left_wrist = (landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y)
        right_wrist = (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
        
        # Check 1: Arms should be relatively straight
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        arms_straight = (left_elbow_angle > 160 and right_elbow_angle > 160)
        
        # Check 2: Wrists should be level with shoulders (horizontal arms)
        y_tolerance = torso_height * self.position_tolerance_ratio
        left_level = abs(left_wrist[1] - left_shoulder[1]) < y_tolerance
        right_level = abs(right_wrist[1] - right_shoulder[1]) < y_tolerance
        
        # Check 3: Arms should extend outward (wrists wider than shoulders)
        left_extended = left_wrist[0] < left_shoulder[0]
        right_extended = right_wrist[0] > right_shoulder[0]
        
        # Calculate confidence based on how many criteria are met
        criteria_met = sum([
            arms_straight,
            left_level and right_level,
            left_extended and right_extended
        ])
        confidence = criteria_met / 3.0
        
        detected = criteria_met >= 3  # All criteria must pass
        
        return {'detected': detected, 'confidence': confidence}
    
    def detect_arms_up(self, landmarks, torso_height: float) -> Dict[str, any]:
        """
        Detect Arms-Up pose: both arms raised above head.
        
        Common in dance celebrations, jumps, and expressive movements.
        
        Criteria:
        1. Both wrists above head (nose level or higher)
        2. Arms can be straight or bent (flexible)
        3. Wrists should be relatively close to body centerline
        
        Returns:
            Dict with 'detected' and 'confidence'
        """
        nose = (landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
               landmarks[self.mp_pose.PoseLandmark.NOSE.value].y)
        left_wrist = (landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y)
        right_wrist = (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
        
        # Check if wrists are above nose level (y decreases upward in image coords)
        left_above = left_wrist[1] < nose[1]
        right_above = right_wrist[1] < nose[1]
        
        # Calculate confidence
        if left_above and right_above:
            # Bonus confidence if wrists are close together (classic arms-up)
            wrist_distance = abs(left_wrist[0] - right_wrist[0])
            close_together = wrist_distance < torso_height * 0.5
            confidence = 1.0 if close_together else 0.8
            detected = True
        elif left_above or right_above:
            confidence = 0.5  # Partial detection
            detected = False
        else:
            confidence = 0.0
            detected = False
        
        return {'detected': detected, 'confidence': confidence}
    
    def detect_squat(self, landmarks, torso_height: float) -> Dict[str, any]:
        """
        Detect Squat: hips lowered, knees bent.
        
        Criteria:
        1. Knee angle < 120° (bent knees)
        2. Hip height significantly lower than standing
        3. Torso relatively upright (not leaning forward too much)
        
        Challenge:
        Without a reference "standing" frame, we estimate squat by checking
        if hips are close to knee level (hip.y ≈ knee.y in image space).
        
        Returns:
            Dict with 'detected' and 'confidence'
        """
        left_hip = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y)
        right_hip = (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y)
        left_knee = (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y)
        right_knee = (landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
        left_ankle = (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
        right_ankle = (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
        
        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Check if knees are bent (angle < 120°)
        knees_bent = (left_knee_angle < 120 and right_knee_angle < 120)
        
        # Check if hips are lowered (close to knee level)
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        avg_knee_y = (left_knee[1] + right_knee[1]) / 2
        hip_knee_distance = abs(avg_hip_y - avg_knee_y)
        
        # In a deep squat, hips should be within 0.3 * torso_height of knees
        hips_lowered = hip_knee_distance < torso_height * 0.3
        
        # Calculate confidence
        if knees_bent and hips_lowered:
            # Deeper squat = higher confidence
            depth_ratio = 1.0 - (hip_knee_distance / (torso_height * 0.3))
            confidence = min(1.0, 0.7 + depth_ratio * 0.3)
            detected = True
        elif knees_bent:
            confidence = 0.5  # Partial squat
            detected = False
        else:
            confidence = 0.0
            detected = False
        
        return {'detected': detected, 'confidence': confidence}
    
    def detect_lunge(self, landmarks, torso_height: float) -> Dict[str, any]:
        """
        Detect Lunge: one leg forward with bent knee, one leg back.
        
        Criteria:
        1. Front knee bent (angle < 120°)
        2. Significant difference in knee x-positions (one forward, one back)
        3. Back leg relatively straight (angle > 150°)
        
        Returns:
            Dict with 'detected' and 'confidence'
        
        Note:
        This detects lunge in either direction (left or right leg forward).
        """
        left_hip = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y)
        right_hip = (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y)
        left_knee = (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y)
        right_knee = (landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
        left_ankle = (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
        right_ankle = (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
        
        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Check horizontal separation of knees
        knee_separation = abs(left_knee[0] - right_knee[0])
        significant_separation = knee_separation > torso_height * 0.3
        
        # Determine which leg is forward and check angles
        if left_knee[0] < right_knee[0]:  # Left leg forward
            front_knee_bent = left_knee_angle < 120
            back_leg_straight = right_knee_angle > 150
        else:  # Right leg forward
            front_knee_bent = right_knee_angle < 120
            back_leg_straight = left_knee_angle > 150
        
        # Calculate confidence
        criteria_met = sum([significant_separation, front_knee_bent, back_leg_straight])
        confidence = criteria_met / 3.0
        detected = criteria_met >= 2  # At least 2 of 3 criteria
        
        return {'detected': detected, 'confidence': confidence}
    
    def classify_pose(self, landmarks, torso_height: float) -> Dict[str, Dict[str, any]]:
        """
        Run all pose classifiers on given landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            torso_height: Normalized torso height
        
        Returns:
            Dictionary mapping pose names to detection results:
            {
                'T-Pose': {'detected': bool, 'confidence': float},
                'Arms-Up': {'detected': bool, 'confidence': float},
                ...
            }
        """
        if landmarks is None:
            return {}
        
        results = {
            'T-Pose': self.detect_t_pose(landmarks, torso_height),
            'Arms-Up': self.detect_arms_up(landmarks, torso_height),
            'Squat': self.detect_squat(landmarks, torso_height),
            'Lunge': self.detect_lunge(landmarks, torso_height),
        }
        
        return results