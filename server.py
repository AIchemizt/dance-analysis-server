import os
import uuid
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def upload_and_analyze():
    # Import heavy libraries locally, only when this function is called
    from analyzer.process import analyze_dance_video
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_dir = "/tmp"
    filename = str(uuid.uuid4()) + os.path.splitext(video_file.filename)[1]
    video_path = os.path.join(temp_dir, filename)
    video_file.save(video_path)

    try:
        analysis_data = analyze_dance_video(video_path)
        if analysis_data is None:
            return jsonify({"error": "Could not process video"}), 500
        return jsonify(analysis_data)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# --- THIS IS THE CRITICAL MISSING PIECE ---
@app.route('/health', methods=['GET'])
def health_check():
    """A simple health check endpoint with no external dependencies."""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # For local development, read PORT from environment or default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)