# Dance Movement Analysis Server

### A Cloud-Native AI/ML Service for Real-Time Pose Detection

**Author:** Abhishek
**Repository:** [PASTE YOUR GITHUB REPOSITORY URL HERE]

---

## Project Overview

This project is a cloud-deployed, containerized AI/ML server designed to analyze short-form dance videos. It accepts a video file via a REST API endpoint, processes it using MediaPipe to perform pose detection, and returns a JSON summary identifying frames where specific dance poses (like the "T-Pose") occur.

The entire application is built with a production-first mindset, leveraging Docker for containerization and deployed on AWS Elastic Container Service (ECS) with Fargate for a scalable, serverless architecture. This solution demonstrates a complete, end-to-end workflow from Python development to a live, public-facing cloud service.

---

## Key Features

* **AI-Powered Pose Detection:** Utilizes Google's MediaPipe library to accurately detect 33 different body keypoints in each video frame.
* **Specific Pose Recognition:** Includes logic to identify standard poses, with a current implementation for detecting the "T-Pose". The architecture is modular, allowing for easy addition of more pose detectors.
* **RESTful API:** A simple and robust API built with Flask accepts video uploads (`multipart/form-data`) and returns structured JSON results.
* **Containerized & Portable:** Fully containerized with Docker, ensuring consistent behavior across development and production environments.
* **Scalable Cloud Deployment:** Deployed on AWS ECS with Fargate, allowing the service to automatically scale based on demand without managing underlying servers.
* **Unit Tested:** Core analysis logic is validated with unit tests using `pytest` to ensure reliability and accuracy.

---

## Tech Stack

* **Backend:** Python 3.11, Flask
* **AI / ML:** MediaPipe, OpenCV, NumPy
* **Server:** Gunicorn with Gevent workers
* **Containerization:** Docker
* **Cloud Provider:** Amazon Web Services (AWS)
* **Deployment Service:** AWS ECS with Fargate & Application Load Balancer
* **CI/CD Tooling:** AWS CLI

---

## Local Setup and Installation

To run this project on your local machine, ensure you have Docker installed.

1.  **Clone the repository:**
    ```sh
    git clone [PASTE YOUR GITHUB REPOSITORY URL HERE]
    cd dance-analysis-server
    ```

2.  **Build the Docker image:**
    ```sh
    docker build -t dance-analysis-server .
    ```

3.  **Run the Docker container:**
    This will start the server on `http://localhost:8080`.
    ```sh
    docker run -p 8080:8080 dance-analysis-server
    ```

---

## API Usage

The server exposes a single endpoint for video analysis.

* **Endpoint:** `/analyze`
* **Method:** `POST`
* **Request Body:** `multipart/form-data`
    * **Key:** `video`
    * **Value:** The video file you want to analyze (e.g., `my_dance.mp4`).

### Example `curl` Request

Use this command from your terminal to test the running server with a sample video.

```sh
curl -X POST -F "video=@/path/to/your/video.mp4" http://localhost:8080/analyze

Example Success Response (200 OK)
The server returns a JSON object summarizing the analysis, including the total frames processed and a dictionary of detected poses with the frame numbers where they appeared.

{
  "detected_poses": {
    "T-Pose": [
      170,
      171,
      172,
      173
    ]
  },
  "total_frames": 1460
}

Cloud Deployment Architecture
The application is deployed on AWS using a serverless container orchestration pattern.

ECR (Elastic Container Registry): The Docker image is stored in a private ECR repository.

ECS (Elastic Container Service): An ECS Task Definition specifies the Docker image, CPU, and memory requirements.

Fargate: The task runs on Fargate, a serverless compute engine, so we don't need to manage EC2 instances.

ALB (Application Load Balancer): An ALB receives public traffic on port 80 and routes it to the running container's port 8080. It also handles health checks using the /health endpoint.

ECS Service: The service ensures that the specified number of tasks is always running and connects it to the ALB. It's configured for rolling updates to ensure zero-downtime deployments.

This architecture is secure, scalable, and cost-effective, as resources are only consumed when the task is running.

Public Endpoint URL: http://dance-analysis-lb-1218962309.us-east-1.elb.amazonaws.com

Thought Process and Vision
My approach to this task was to build a solution that is not just functional but also robust, scalable, and production-ready. I chose AWS ECS with Fargate over a simple EC2 instance because it better represents modern cloud-native design patterns. This choice abstracts away server management, provides built-in scalability, and integrates seamlessly with other AWS services like the Application Load Balancer for high availability and zero-downtime deployments. This significantly reduces long-term operational overhead.

The Python code is structured in a modular way, separating the core analysis logic (analyzer/process.py) from the web server (server.py). This separation of concerns makes the code easier to maintain, test, and extend. For example, adding new pose detection functions would only require modifying the process.py file without touching the API layer, which is critical for rapid feature development. The inclusion of unit tests with pytest further ensures the reliability of the core logic.

For Callus Company, this architecture provides a strong and flexible foundation. The current T-Pose detection can be expanded to a library of dozens of standard dance poses (e.g., arabesque, plié, pirouette). The JSON output can be fed into a frontend application to provide dancers with real-time feedback or used for advanced analytics to track a dancer's progress over time. The scalable nature of the deployment means the system can handle a growing user base, from a few dancers to a global community, without requiring manual intervention to provision new servers. This project serves as a solid proof-of-concept for a much larger and more feature-rich platform.