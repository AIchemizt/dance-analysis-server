import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def detect_t_pose(landmarks):
    """Checks if the landmarks indicate a T-Pose using dynamic tolerance."""
    # Define landmarks for shoulders, wrists, and hips
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

    # Calculate person's approximate torso height in the frame for dynamic tolerance
    torso_height = abs(left_shoulder.y - left_hip.y)
    # A tolerance of 20% of the torso height is robust
    y_tolerance = torso_height * 0.2

    # Check if arms are straight out to the sides
    left_arm_straight = abs(left_wrist.y - left_shoulder.y) < y_tolerance
    right_arm_straight = abs(right_wrist.y - right_shoulder.y) < y_tolerance
    
    left_arm_outstretched = left_wrist.x < left_shoulder.x
    right_arm_outstretched = right_wrist.x > right_shoulder.x

    if left_arm_straight and right_arm_straight and left_arm_outstretched and right_arm_outstretched:
        return True
    return False

def analyze_dance_video(video_path):
    """
    Analyzes a video to detect keypoints and identify standard dance poses.
    Outputs a JSON summary of the detected poses and the frames they appear in.
    """
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    analysis_summary = {"total_frames": 0, "detected_poses": {}}
    frame_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # --- POSE DETECTION LOGIC ---
            if detect_t_pose(landmarks):
                pose_name = "T-Pose"
                if pose_name not in analysis_summary["detected_poses"]:
                    analysis_summary["detected_poses"][pose_name] = []
                analysis_summary["detected_poses"][pose_name].append(frame_count)
            
            # --- You can add more 'elif detect_another_pose(landmarks):' here ---

        frame_count += 1
    
    analysis_summary["total_frames"] = frame_count
    cap.release()
    pose.close()

    return analysis_summary