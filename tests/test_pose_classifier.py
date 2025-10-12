"""
Unit tests for PoseClassifier.
Tests individual pose detection logic with synthetic landmark data.
"""

import pytest
import numpy as np
from analyzer.pose_classifier import PoseClassifier


class TestPoseClassifier:
    """Test suite for pose classification logic."""
    
    def create_mock_landmarks(self, positions: dict):
        """
        Helper to create mock MediaPipe landmarks.
        
        Args:
            positions: Dict mapping landmark indices to (x, y) tuples
        
        Returns:
            List of 33 mock landmark objects
        """
        class MockLandmark:
            def __init__(self, x=0.5, y=0.5, z=0.0):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = 0.9
        
        # Initialize all landmarks to default center position
        landmarks = [MockLandmark() for _ in range(33)]
        
        # Override with specified positions
        for idx, (x, y) in positions.items():
            landmarks[idx] = MockLandmark(x, y, 0.0)
        
        return landmarks
    
    def test_t_pose_detection_positive(self):
        """Test T-Pose detection with ideal T-pose configuration."""
        classifier = PoseClassifier()
        
        # Create landmarks for perfect T-pose
        # Arms straight out, horizontal, wrists at shoulder level
        positions = {
            11: (0.5, 0.3),   # left_shoulder
            12: (0.7, 0.3),   # right_shoulder
            13: (0.3, 0.3),   # left_elbow (straight line)
            14: (0.9, 0.3),   # right_elbow (straight line)
            15: (0.1, 0.3),   # left_wrist (extended left)
            16: (1.1, 0.3),   # right_wrist (extended right)
            23: (0.5, 0.6),   # left_hip
            24: (0.7, 0.6),   # right_hip
        }
        
        landmarks = self.create_mock_landmarks(positions)
        torso_height = 0.3  # shoulder to hip distance
        
        result = classifier.detect_t_pose(landmarks, torso_height)
        
        assert result['detected'] == True
        assert result['confidence'] > 0.8
    
    def test_t_pose_detection_negative(self):
        """Test that non-T-pose is correctly rejected."""
        classifier = PoseClassifier()
        
        # Arms down at sides
        positions = {
            11: (0.5, 0.3),   # left_shoulder
            12: (0.7, 0.3),   # right_shoulder
            13: (0.5, 0.4),   # left_elbow
            14: (0.7, 0.4),   # right_elbow
            15: (0.5, 0.5),   # left_wrist (at hip level)
            16: (0.7, 0.5),   # right_wrist (at hip level)
            23: (0.5, 0.6),   # left_hip
        }
        
        landmarks = self.create_mock_landmarks(positions)
        torso_height = 0.3
        
        result = classifier.detect_t_pose(landmarks, torso_height)
        
        assert result['detected'] == False
    
    def test_arms_up_detection_positive(self):
        """Test Arms-Up detection with raised arms."""
        classifier = PoseClassifier()
        
        positions = {
            0: (0.6, 0.1),    # nose
            11: (0.5, 0.3),   # left_shoulder
            12: (0.7, 0.3),   # right_shoulder
            15: (0.55, 0.05), # left_wrist (above nose)
            16: (0.65, 0.05), # right_wrist (above nose)
        }
        
        landmarks = self.create_mock_landmarks(positions)
        torso_height = 0.3
        
        result = classifier.detect_arms_up(landmarks, torso_height)
        
        assert result['detected'] == True
        assert result['confidence'] > 0.7
    
    def test_squat_detection_positive(self):
        """Test Squat detection with bent knees and lowered hips."""
        classifier = PoseClassifier()
        
        # Squat position: hips low, knees bent
        positions = {
            23: (0.5, 0.55),  # left_hip (lowered)
            24: (0.7, 0.55),  # right_hip (lowered)
            25: (0.6, 0.58),  # left_knee (bent forward)
            26: (0.8, 0.58),  # right_knee (bent forward)
            27: (0.5, 0.8),   # left_ankle
            28: (0.7, 0.8),   # right_ankle
        }
        
        landmarks = self.create_mock_landmarks(positions)
        torso_height = 0.25
        
        result = classifier.detect_squat(landmarks, torso_height)
        
        assert result['detected'] == True
    
    def test_lunge_detection_positive(self):
        """Test Lunge detection with one leg forward."""
        classifier = PoseClassifier()
        
        # Lunge: left leg forward with bent knee, right leg back
        positions = {
            23: (0.5, 0.4),   # left_hip
            24: (0.7, 0.4),   # right_hip
            25: (0.3, 0.4),   # left_knee (forward, more bent ~90°)
            26: (0.8, 0.6),   # right_knee (back, straighter ~153°)
            27: (0.3, 0.8),   # left_ankle
            28: (0.9, 0.8),   # right_ankle
        }
        
        landmarks = self.create_mock_landmarks(positions)
        torso_height = 0.3
        
        result = classifier.detect_lunge(landmarks, torso_height)
        
        assert result['detected'] == True
    
    def test_classify_pose_returns_all_poses(self):
        """Test that classify_pose returns results for all implemented poses."""
        classifier = PoseClassifier()
        
        # Use T-pose configuration
        positions = {
            11: (0.5, 0.3), 12: (0.7, 0.3),
            13: (0.3, 0.3), 14: (0.9, 0.3),
            15: (0.1, 0.3), 16: (1.1, 0.3),
            23: (0.5, 0.6),
        }
        
        landmarks = self.create_mock_landmarks(positions)
        results = classifier.classify_pose(landmarks, torso_height=0.3)
        
        # Should return results for all poses
        assert 'T-Pose' in results
        assert 'Arms-Up' in results
        assert 'Squat' in results
        assert 'Lunge' in results
        
        # Each result should have 'detected' and 'confidence'
        for pose_name, result in results.items():
            assert 'detected' in result
            assert 'confidence' in result