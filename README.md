# Cover Drive Analyzer (Real-Time)

This project analyzes cricket cover drives in **real-time** or from **video files** using [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) for human pose estimation. It overlays **joint angles**, **phase detection**, and **feedback cues** onto the video, and produces both an annotated video and a JSON evaluation report.

---

## Setup

1. **Clone / Download** the project folder.

2. **Install dependencies** (Python ≥ 3.8 recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. **Install ffmpeg** (needed for YouTube video download & processing):
   - **macOS (Homebrew):**
     ```bash
     brew install ffmpeg
     ```
   - **Windows:** Download FFmpeg and add it to your PATH.
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```

4. **Install yt-dlp** (for YouTube Shorts/links):
   ```bash
   pip install yt-dlp
   ```

---

## Run Instructions

### 1. From a local video
```bash
python cover_drive_analysis_realtime.py "/path/to/video.mp4" --output output
```

### 2. From a YouTube Short
```bash
python cover_drive_analysis_realtime.py "https://youtube.com/shorts/vSX3IRxGnNY" --output output
```

### 3. From a webcam
```bash
python cover_drive_analysis_realtime.py 0 --output output
```
*(0 is usually the default webcam device index)*

---

## Output Files

After running, the following will be saved in your chosen `--output` directory:

- **`annotated_video.mp4`** → video with pose landmarks, joint angles, and feedback text overlay
- **`evaluation.json`** → structured report with:
  - Overall performance score
  - Phase-wise breakdown
  - Average FPS

---

## Key Features

- **Real-time pose tracking** with MediaPipe
- **Angle calculations** for elbow, shoulder, and hip
- **Phase detection** (stance → backswing → downswing → follow-through)
- **Dynamic feedback cues**
- **Multi-source support:**
  - Local video files
  - YouTube links
  - Live webcam feed

---

## Sample Output

### Evaluation JSON Structure
```json
{
  "overall_score": 8.2,
  "phases": {
    "stance": {"score": 8.5, "feedback": "Good balance"},
    "backswing": {"score": 7.8, "feedback": "Slight over-rotation"},
    "downswing": {"score": 8.4, "feedback": "Excellent timing"},
    "follow_through": {"score": 8.1, "feedback": "Complete extension"}
  },
  "average_fps": 28.5,
  "total_frames": 420
}
```

---

## Notes & Assumptions

### 1. **Assumed Camera Angle**
The analysis works best with a **side-on** or slightly angled view of the batsman. Front-facing videos will yield less accurate joint angle data.

### 2. **Lighting & Resolution**
MediaPipe Pose performs best in good lighting and with ≥720p resolution.

### 3. **Bat & Ball Detection**
This version focuses purely on body pose; bat-ball contact is inferred by motion phases, not explicit bat tracking.

### 4. **No Player Tracking Across Clips**
This code processes a single video; multi-clip tracking is outside the current scope.

### 5. **Evaluation Metrics**
The feedback system is rule-based with preset angle ranges, not a trained machine learning model.

---

## Example Run

```bash
python cover_drive_analysis_realtime.py "https://youtube.com/shorts/vSX3IRxGnNY" --output results
```

**Output:**
- `results/annotated_video.mp4`
- `results/evaluation.json`

---

## Requirements

### Dependencies
Create a `requirements.txt` file with:
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.21.0
yt-dlp>=2023.7.6
argparse
json
```

### System Requirements
- Python 3.8+
- FFmpeg installed
- Webcam (optional, for real-time analysis)
- 4GB+ RAM recommended

---

## License

This project is open-source. Feel free to modify and distribute.

---
