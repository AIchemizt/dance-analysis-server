"""
Unit tests for PoseDetector class.
Tests video processing, landmark extraction, and error handling.
"""

import pytest
import os
import numpy as np
from analyzer.pose_detector import PoseDetector


class TestPoseDetector:
    """Test suite for PoseDetector functionality."""
    
    def test_initialization(self):
        """Test that PoseDetector initializes correctly."""
        detector = PoseDetector(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        assert detector is not None
        assert detector.mp_pose is not None
    
    def test_invalid_video_path(self):
        """Test graceful handling of non-existent video file."""
        detector = PoseDetector()
        result = detector.process_video("nonexistent_video.mp4")
        assert result is None
    
    def test_process_video_output_structure(self):
        """
        Test that process_video returns correctly structured data.
        
        Note: This test requires a sample video file at tests/fixtures/sample_video.mp4
        If the file doesn't exist, the test is skipped.
        """
        video_path = "tests/fixtures/sample_video.mp4"
        
        if not os.path.exists(video_path):
            pytest.skip(f"Sample video not found at {video_path}")
        
        detector = PoseDetector()
        frame_data = detector.process_video(video_path)
        
        assert frame_data is not None
        assert len(frame_data) > 0
        
        # Check structure of first frame
        first_frame = frame_data[0]
        assert 'frame_number' in first_frame
        assert 'landmarks' in first_frame
        assert 'timestamp' in first_frame
        assert first_frame['frame_number'] == 0
    
    def test_get_landmark_position_valid(self):
        """Test landmark position extraction with valid data."""
        detector = PoseDetector()
        
        # Create mock landmark with high visibility
        class MockLandmark:
            def __init__(self):
                self.x = 0.5
                self.y = 0.5
                self.z = 0.0
                self.visibility = 0.9
        
        landmarks = [MockLandmark() for _ in range(33)]
        position = detector.get_landmark_position(landmarks, 0)
        
        assert position is not None
        assert len(position) == 3
        assert position[0] == 0.5
        assert position[1] == 0.5
    
    def test_get_landmark_position_low_visibility(self):
        """Test that low visibility landmarks are filtered out."""
        detector = PoseDetector()
        
        class MockLandmark:
            def __init__(self):
                self.x = 0.5
                self.y = 0.5
                self.z = 0.0
                self.visibility = 0.3  # Low visibility
        
        landmarks = [MockLandmark() for _ in range(33)]
        position = detector.get_landmark_position(landmarks, 0)
        
        # Should return None for low visibility
        assert position is None
    
    def test_calculate_torso_height(self):
        """Test torso height calculation."""
        detector = PoseDetector()
        
        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = 0.9
        
        # Create mock landmarks with shoulders at y=0.3 and hips at y=0.6
        landmarks = [MockLandmark(0.5, 0.5, 0.0) for _ in range(33)]
        landmarks[11] = MockLandmark(0.4, 0.3, 0.0)  # left_shoulder
        landmarks[12] = MockLandmark(0.6, 0.3, 0.0)  # right_shoulder
        landmarks[23] = MockLandmark(0.4, 0.6, 0.0)  # left_hip
        landmarks[24] = MockLandmark(0.6, 0.6, 0.0)  # right_hip
        
        torso_height = detector.calculate_torso_height(landmarks)
        
        # Expected: 0.6 - 0.3 = 0.3
        assert abs(torso_height - 0.3) < 0.01