"""
Flask REST API server for dance movement analysis.
Provides endpoints for video upload, analysis, and health checking.
"""

import os
import uuid
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from analyzer.pose_detector import PoseDetector
from analyzer.pose_classifier import PoseClassifier
from analyzer.movement_analyzer import MovementAnalyzer


app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
UPLOAD_FOLDER = '/tmp/dance_uploads'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for load balancer monitoring.
    Returns 200 OK if service is running.
    No external dependencies checked to ensure fast response.
    """
    return jsonify({
        'status': 'healthy',
        'service': 'dance-analysis-server',
        'version': '2.0.0'
    }), 200


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    Main endpoint for video analysis.
    Accepts: multipart/form-data with 'video' file
    Returns: JSON with comprehensive analysis results
    
    Response format:
    {
        "total_frames": int,
        "duration_seconds": float,
        "detected_poses": {
            "PoseName": {
                "frames": [list of frame numbers],
                "count": int,
                "average_confidence": float
            }
        },
        "movement_analysis": {
            "movement_intensity": {body_part: score},
            "symmetry_score": float,
            "high_movement_frames": [list of frame numbers]
        }
    }
    
    Error codes:
    - 400: No file provided or invalid file type
    - 413: File too large (>100MB)
    - 500: Processing error
    """
    # Validate request has file
    if 'video' not in request.files:
        return jsonify({
            'error': 'No video file provided',
            'message': 'Request must include a video file with key "video"'
        }), 400
    
    video_file = request.files['video']
    
    # Validate filename
    if video_file.filename == '':
        return jsonify({
            'error': 'Empty filename',
            'message': 'Uploaded file has no name'
        }), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'message': f'Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
            'received': video_file.filename.rsplit('.', 1)[1].lower() if '.' in video_file.filename else 'unknown'
        }), 400
    
    # Generate unique filename to prevent collisions
    filename = secure_filename(video_file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    video_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    try:
        # Save uploaded file
        video_file.save(video_path)
        
        # Initialize analysis modules
        detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        classifier = PoseClassifier()
        analyzer = MovementAnalyzer()
        
        # Step 1: Extract pose landmarks from video
        frame_data = detector.process_video(video_path)
        
        if frame_data is None or len(frame_data) == 0:
            return jsonify({
                'error': 'Video processing failed',
                'message': 'Could not extract frames from video. File may be corrupted.'
            }), 500
        
        # Step 2: Classify poses in each frame
        pose_detections_raw = {
            'T-Pose': [],
            'Arms-Up': [],
            'Squat': [],
            'Lunge': []
        }
        pose_confidences = {pose: [] for pose in pose_detections_raw.keys()}
        
        for frame in frame_data:
            if frame['landmarks'] is None:
                # No pose detected in this frame
                for pose_name in pose_detections_raw.keys():
                    pose_detections_raw[pose_name].append(False)
                continue
            
            # Calculate torso height for normalization
            torso_height = detector.calculate_torso_height(frame['landmarks'])
            
            # Run all pose classifiers
            classifications = classifier.classify_pose(frame['landmarks'], torso_height)
            
            for pose_name in pose_detections_raw.keys():
                if pose_name in classifications:
                    detected = classifications[pose_name]['detected']
                    confidence = classifications[pose_name]['confidence']
                    pose_detections_raw[pose_name].append(detected)
                    if detected:
                        pose_confidences[pose_name].append(confidence)
                else:
                    pose_detections_raw[pose_name].append(False)
        
        # Step 3: Apply temporal filtering to reduce false positives
        filtered_poses = analyzer.temporal_pose_filter(pose_detections_raw, min_consecutive=3)
        
        # Step 4: Calculate movement metrics
        movement_intensity = analyzer.calculate_movement_intensity(frame_data)
        symmetry_score = analyzer.calculate_overall_symmetry(frame_data)
        movement_peaks = analyzer.detect_movement_peaks(frame_data, threshold=0.015)
        
        # Step 5: Format results
        detected_poses_summary = {}
        for pose_name, frames in filtered_poses.items():
            if len(frames) > 0:
                avg_confidence = (
                    sum(pose_confidences[pose_name]) / len(pose_confidences[pose_name])
                    if pose_confidences[pose_name] else 0.0
                )
                detected_poses_summary[pose_name] = {
                    'frames': frames,
                    'count': len(frames),
                    'average_confidence': round(avg_confidence, 3)
                }
        
        # Calculate video duration
        duration = frame_data[-1]['timestamp'] if frame_data else 0.0
        
        result = {
            'total_frames': len(frame_data),
            'duration_seconds': round(duration, 2),
            'detected_poses': detected_poses_summary,
            'movement_analysis': {
                'movement_intensity': {k: round(v, 4) for k, v in movement_intensity.items()},
                'symmetry_score': round(symmetry_score, 3),
                'high_movement_frames': movement_peaks
            }
        }
        
        return jsonify(result), 200

    except Exception as e:
        # Log error for debugging (in production, use proper logging)
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred during video processing',
            'details': str(e)
        }), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {video_path}: {e}")


@app.route('/', methods=['GET'])
def index():
    """
    Root endpoint with API documentation.
    """
    docs = {
        'service': 'Dance Movement Analysis Server',
        'version': '2.0.0',
        'endpoints': {
            '/health': {
                'method': 'GET',
                'description': 'Health check endpoint',
                'response': {'status': 'healthy'}
            },
            '/analyze': {
                'method': 'POST',
                'description': 'Analyze dance video for poses and movement patterns',
                'request': {
                    'content_type': 'multipart/form-data',
                    'field': 'video',
                    'accepted_formats': list(ALLOWED_EXTENSIONS),
                    'max_size': '100MB'
                },
                'response': {
                    'total_frames': 'Number of processed frames',
                    'duration_seconds': 'Video duration',
                    'detected_poses': 'Dictionary of detected poses with frame numbers',
                    'movement_analysis': 'Movement intensity and symmetry metrics'
                }
            }
        },
        'example_curl': 'curl -X POST -F "video=@dance.mp4" http://localhost:8080/analyze'
    }
    return jsonify(docs), 200


if __name__ == '__main__':
    # For local development only
    # Production should use Gunicorn (see Dockerfile CMD)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)