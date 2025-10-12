"""
Unit tests for MovementAnalyzer.
Tests movement intensity, symmetry, and temporal filtering.
"""

import pytest
import numpy as np
from analyzer.movement_analyzer import MovementAnalyzer


class TestMovementAnalyzer:
    """Test suite for movement analysis functionality."""
    
    def create_mock_frame_data(self, num_frames: int, movement_pattern: str = "static"):
        """
        Create mock frame data for testing.
        
        Args:
            num_frames: Number of frames to generate
            movement_pattern: "static", "moving_arms", or "moving_body"
        
        Returns:
            List of frame data dictionaries with mock landmarks
        """
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.z = 0.0
                self.visibility = 0.9
        
        frame_data = []
        
        for i in range(num_frames):
            # Create 33 landmarks (MediaPipe standard)
            if movement_pattern == "static":
                # No movement
                landmarks = [MockLandmark(0.5, 0.5) for _ in range(33)]
            
            elif movement_pattern == "moving_arms":
                # Arms move, body stays still
                landmarks = [MockLandmark(0.5, 0.5) for _ in range(33)]
                # Animate wrists moving up and down
                wrist_y = 0.5 + 0.2 * np.sin(i * 0.1)
                landmarks[15] = MockLandmark(0.3, wrist_y)  # left_wrist
                landmarks[16] = MockLandmark(0.7, wrist_y)  # right_wrist
            
            elif movement_pattern == "moving_body":
                # Whole body moves
                offset_x = 0.1 * np.sin(i * 0.1)
                offset_y = 0.05 * np.cos(i * 0.1)
                landmarks = [MockLandmark(0.5 + offset_x, 0.5 + offset_y) for _ in range(33)]
            
            frame_data.append({
                'frame_number': i,
                'landmarks': landmarks,
                'timestamp': i * 0.033
            })
        
        return frame_data
    
    def test_movement_intensity_static(self):
        """Test that static video has low movement intensity."""
        analyzer = MovementAnalyzer()
        frame_data = self.create_mock_frame_data(30, "static")
        
        intensity = analyzer.calculate_movement_intensity(frame_data)
        
        # All body parts should have near-zero movement
        for body_part, value in intensity.items():
            assert value < 0.01, f"{body_part} should be nearly static"
    
    def test_movement_intensity_moving_arms(self):
        """Test that arm movement is captured with higher intensity."""
        analyzer = MovementAnalyzer()
        frame_data = self.create_mock_frame_data(30, "moving_arms")
        
        intensity = analyzer.calculate_movement_intensity(frame_data)
        
        # Wrists should have higher intensity than ankles
        assert intensity['left_wrist'] > intensity['left_ankle']
        assert intensity['right_wrist'] > intensity['right_ankle']
    
    def test_symmetry_calculation(self):
        """Test symmetry score calculation."""
        analyzer = MovementAnalyzer()
        
        # Symmetric movement (both sides identical)
        frame_data_symmetric = self.create_mock_frame_data(20, "moving_arms")
        symmetry_symmetric = analyzer.calculate_overall_symmetry(frame_data_symmetric)
        
        assert symmetry_symmetric > 0.7, "Symmetric movement should score high"
    
    def test_detect_movement_peaks(self):
        """Test detection of high-movement frames."""
        analyzer = MovementAnalyzer()
        frame_data = self.create_mock_frame_data(50, "moving_body")
        
        # THE FIX IS ON THIS LINE vvv
        peaks = analyzer.detect_movement_peaks(frame_data, threshold=0.005)
        
        # Should detect some peak frames in moving video
        assert len(peaks) > 0
        assert all(isinstance(frame_num, int) for frame_num in peaks)
    
    def test_temporal_pose_filter(self):
        """Test temporal filtering of pose detections."""
        analyzer = MovementAnalyzer()
        
        # Simulate intermittent T-pose detections
        pose_detections = {
            'T-Pose': [False, False, True, True, True, False, False, True, False, True, True, True, True]
            # Only frames 2-4 and 9-12 should pass (3+ consecutive)
        }
        
        filtered = analyzer.temporal_pose_filter(pose_detections, min_consecutive=3)
        
        assert 'T-Pose' in filtered
        t_pose_frames = filtered['T-Pose']
        
        # Should include frames 2, 3, 4 (first sequence) and 9, 10, 11, 12 (second sequence)
        assert 2 in t_pose_frames
        assert 3 in t_pose_frames
        assert 4 in t_pose_frames
        assert 9 in t_pose_frames
        assert 10 in t_pose_frames
        assert 11 in t_pose_frames
        assert 12 in t_pose_frames
        
        # Should NOT include isolated detections
        assert 7 not in t_pose_frames
    
    def test_temporal_filter_no_valid_sequences(self):
        """Test temporal filter with no sequences meeting threshold."""
        analyzer = MovementAnalyzer()
        
        # Only 1-2 frame sequences
        pose_detections = {
            'Squat': [True, False, True, True, False, True, False]
        }
        
        filtered = analyzer.temporal_pose_filter(pose_detections, min_consecutive=3)
        
        # No sequences of 3+ frames
        assert len(filtered['Squat']) == 0