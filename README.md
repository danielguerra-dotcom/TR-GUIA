# TR GUIA

GUIA (Guide Using Intelligent Algorithms) is a research-driven assistive system developed as part of a high-school thesis project, focused on exploring how artificial intelligence can enhance real-world accessibility.

The project follows a progressive, multi-stage development approach that evolves from fundamental perception modules to a fully integrated multimodal assistant. Early stages focus on core computer vision capabilities such as object detection (YOLO) and depth estimation (MiDaS), providing environmental awareness. These components are later combined and extended into a final system powered by a language-based AI assistant (Gemini), capable of interpreting visual input and interacting with the user through natural, voice-driven communication.

Rather than being a single static application, GUIA represents an iterative exploration of how independent AI technologies can be understood, implemented, and ultimately unified into a coherent assistive solution. The final outcome demonstrates the transition from low-level perception to high-level reasoning, aiming to support users—particularly those with visual impairments—in understanding and navigating their surroundings.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/danielguerra-dotcom/TR-GUIA.git
   cd TR-GUIA
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. For the Gemini assistant scripts, obtain a Gemini API key from Google AI Studio and set it as an environment variable:
   ```
   export GEMINI_API_KEY=your_api_key_here
   ```
   Or create a `.env` file in the project root with:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

The project includes several scripts in the `scripts/` directory. Each script can be run independently.

### Depth Mapping

- **depth_map.py**: Generates a depth map from camera input using MiDaS.
  ```
  python scripts/depth_map.py --cam 0 --profile fast
  ```
  Options:
  - `--cam`: Camera index (default: 0)
  - `--profile`: Quality profile, 'fast' or 'quality' (default: fast)
  - `--imgsz`: Input image size (default: 512)
  - `--cap_w`, `--cap_h`: Capture resolution (default: 640x480)

- **depth_map + object_detection.py**: Combines depth mapping with object detection.
  ```
  python scripts/depth_map\ +\ object_detection.py --cam 0 --model yolov8n.pt
  ```
  Options: Similar to above, plus object detection parameters like `--conf` for confidence threshold.

### Object Detection

- **object_detection_prototype.py**: Basic object detection using YOLOv8.
  ```
  python scripts/object_detection_prototype.py --cam 0
  ```

- **object_detection_final.py**: Enhanced object detection with optimizations.
  ```
  python scripts/object_detection_final.py --cam 0 --model yolov8n.pt --conf 0.5
  ```
  Options:
  - `--cam`: Camera index
  - `--imgsz`: Image size for inference
  - `--conf`: Confidence threshold
  - `--model`: Path to YOLO model (tries candidates if not specified)
  - `--no_preview`: Run without displaying preview

### Gemini Assistant

- **Gemini_assistant_prototype.py**: Prototype multimodal assistant with audio and video.
  ```
  python scripts/Gemini_assistant_prototype.py
  ```

- **Gemini_assistant_final.py**: Final version with improved audio processing and duplex communication.
  ```
  python scripts/Gemini_assistant_final.py
  ```
  This script requires the GEMINI_API_KEY environment variable or .env file.

## External Libraries

The project depends on the following external libraries (see `requirements.txt` for versions):

- opencv-python: Computer vision tasks
- numpy: Numerical computations
- torch: Deep learning framework
- ultralytics: YOLO object detection
- google-genai: Google Gemini API client
- sounddevice: Audio input/output
- scipy: Scientific computing
- librosa: Audio processing
- python-dotenv: Environment variable management
- httpx: HTTP client
- loguru: Logging

## Screenshots

Screenshots of the application in action are available in the `screenshots/` directory.

## License

See LICENSE file for details.
