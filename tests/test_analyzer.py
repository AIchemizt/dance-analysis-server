import pytest
import os
from analyzer.process import analyze_dance_video

# Define the path to the sample video in the project's root directory
# This assumes you run pytest from the root folder (e.g., 'dance-analysis-server/')
SAMPLE_VIDEO_PATH = "video.mp4"
T_POSE_VIDEO_PATH = "tests/t_pose_video.mp4"

def test_analyze_video_with_invalid_path():
    """
    Tests if the function correctly returns None for a non-existent file path.
    This checks for graceful error handling.
    """
    result = analyze_dance_video("non_existent_file.mp4")
    assert result is None

def test_analyze_video_output_format_on_valid_video():
    """
    Tests if the function returns a dictionary with the correct top-level keys
    when given a valid video. This checks the output format.
    """
    # Ensure the sample video exists before running the test
    assert os.path.exists(SAMPLE_VIDEO_PATH), f"Sample video not found at: {SAMPLE_VIDEO_PATH}"

    result = analyze_dance_video(SAMPLE_VIDEO_PATH)
    
    # Check that the result is a dictionary
    assert isinstance(result, dict)
    
    # Check for the presence of the required keys
    assert "total_frames" in result
    assert "detected_poses" in result

def test_t_pose_detection_accuracy():
    """
    Tests if a T-Pose is correctly detected in a specific test video.
    This checks for pose detection accuracy.
    """
    # Ensure the T-pose test video exists before running the test
    assert os.path.exists(T_POSE_VIDEO_PATH), f"T-Pose test video not found at: {T_POSE_VIDEO_PATH}"

    result = analyze_dance_video(T_POSE_VIDEO_PATH)
    
    # Check that the result contains the 'detected_poses' key
    assert "detected_poses" in result
    
    # Check that "T-Pose" was detected
    assert "T-Pose" in result["detected_poses"]
    
    # Check that the list of frames where T-Pose was detected is not empty
    assert len(result["detected_poses"]["T-Pose"]) > 0