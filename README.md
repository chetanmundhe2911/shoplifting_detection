# Shoplift Detection System

This repository contains a Python-based project that utilizes `YOLOv8` (You Only Look Once) object detection model for detecting objects from videos, RTSP streams, or GIF files. It provides a user-friendly Graphical User Interface (GUI) built using `Tkinter`.

## Demo

Check out the Shoplift Detection System in action below:

![Shoplift Detection Demo](ezgif.com-video-to-gif-converter.gif)

## Features

- Detect objects in videos in real-time.
- Supports detection in video files (`.mp4`, `.avi`) or GIF files (`.gif`).
- Accepts RTSP stream URLs for live detection.
- Displays detection results visually, allows saving annotated frames to the `tested/` directory.

---

## Requirements

Before running the application, ensure you have installed all necessary dependencies.

1. Python 3.7+ installed on your system.
2. Install required Python libraries using `pip`:

```bash
pip install ultralytics opencv-python-headless tk pillow
```

---

## How It Works

1. **Model Integration**: The application uses a pre-trained YOLOv8 model (`best_model_v7.pt`) to detect objects.
2. **GUI Features**:
   - Upload video or GIF file, process it, and display real-time detections.
   - Enter an RTSP stream URL to analyze the live stream.
3. **Frame Output**: The app saves all processed and annotated frames in the `tested` directory.

---

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/chetanmundhe2911/shoplifting_detection.git
   cd shoplift-detection-system
   ```
2. Place the YOLOv8 model file `best_model_v7.pt` in the same directory as the script.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run shoplift_detection.py
   ```

---

## Usage Instructions

### 1. Uploading a Video or GIF:

- Click the **"Upload Video/GIF"** button in the GUI.
- Select a video file or GIF file from your system.
- For videos:
  - The program processes the file frame-by-frame using the YOLOv8 model. 
  - Annotated frames (with detection boxes) are displayed in real-time.
  - Frames with detections are saved in the `tested/` directory.
- For GIFs:
  - The GIF is displayed in a separate window as-is (no detection is currently conducted for GIFs).

### 2. RTSP Stream:

- Enter the RTSP stream URL in the `RTSP Stream URL` input field.
- Click the **"Start RTSP Stream"** button and watch real-time detection results.

---

## Adding GIF Demo Preview on GitHub Repo

To display a demo GIF on your main `README.md` file, follow these steps:

1. Create or record a demo GIF showcasing your application functionalities.
2. Save the GIF (e.g., `demo.gif`) in your repository's root directory.
3. Add the following markdown code in the `README.md` file to embed:

   ```markdown
   ## Demo

   ![Shoplift Detection Demo](demo.gif)
   ```

### Example Embedded Demo

![Shoplift Detection Demo](demo.gif)

---

## Folder Structure

```plaintext
shoplift-detection-system/
├── best_model_v7.pt         # YOLOv8 trained weights
├── shoplift_detection.py    # Main application script
├── requirements.txt         # Python dependencies
├── README.md                # Documentation (current file)
├── demo.gif                 # Demo GIF showcasing app usage
└── tested/                  # Directory for saving annotated frames
```

---

## License

This project is **open-source** under the [MIT License](LICENSE.md). Feel free to use, modify, and distribute as needed.

---

### Author

Developed by **Chetan Mundhe**.

Contributions and feedback are welcome! 😊
