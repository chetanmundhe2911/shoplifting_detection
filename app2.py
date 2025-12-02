import tkinter as tk
from tkinter import filedialog, Label, Entry, Button
import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageTk
import os

# Load the trained YOLOv8 model
model = YOLO('best_model_v7.pt')  # Load your trained model

# Create a directory to save the frames if it doesn't exist
output_dir = 'tested'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def upload_file():
    # Open file dialog to upload a video or GIF
    file_path = filedialog.askopenfilename(filetypes=[("Video/GIF files", "*.mp4;*.avi;*.gif")])
    if file_path:
        if file_path.lower().endswith('.gif'):
            display_gif(file_path)
        else:
            process_video(file_path)

def shoplifting_detection_logic(detected_boxes):
    # Example condition: if more than one person is detected
    return len(detected_boxes) > 1  # Return True if shoplifting is detected

def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make predictions using the YOLO model
        results = model(frame)
        
        # Get the boxes, class labels, and confidence scores
        boxes = results[0].boxes
        boxes_xyxy = boxes.xyxy.cpu().numpy()  # Get the bounding box coordinates
        boxes_cls = boxes.cls.cpu().numpy()  # Get class labels
        boxes_conf = boxes.conf.cpu().numpy()  # Get confidence scores

        # Store detected boxes for shoplifting logic
        detected_boxes = []

        # Draw only the bounding boxes for persons (class ID 0) with confidence > 0.5
        for i in range(len(boxes_xyxy)):
            if boxes_cls[i] == 0 and boxes_conf[i] > 0.5:  # Class ID 0 is for persons
                detected_boxes.append(boxes_xyxy[i])  # Add to detected boxes
                x1, y1, x2, y2 = boxes_xyxy[i]
                confidence = boxes_conf[i]

                # Check if shoplifting is detected to determine box color
                is_shoplifting = shoplifting_detection_logic(detected_boxes)
                color = (0, 0, 255) if is_shoplifting else (0, 255, 0)  # Red if shoplifting, green otherwise

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'Person: {confidence:.2f}', (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if detected_boxes:
            # Apply shoplifting detection logic
            if is_shoplifting:
                cv2.putText(frame, "Shoplifting Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Normal Behavior", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("Detections", frame)

        # Save the frame with detections
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def process_rtsp():
    rtsp_url = rtsp_entry.get()
    if rtsp_url:
        process_video(rtsp_url)

def display_gif(gif_path):
    gif_window = tk.Toplevel(root)
    gif_window.title("GIF Viewer")
    gif_window.geometry("600x400")

    # Load the GIF using PIL
    gif_image = Image.open(gif_path)
    frames = []

    try:
        while True:
            frame = ImageTk.PhotoImage(gif_image.copy())
            frames.append(frame)
            gif_image.seek(len(frames))  # Load next frame
    except EOFError:
        pass

    # Label to display GIF frames
    gif_label = Label(gif_window)
    gif_label.pack(fill="both", expand=True)

    def update_gif(index):
        frame = frames[index]
        gif_label.configure(image=frame)
        index = (index + 1) % len(frames)
        gif_window.after(100, update_gif, index)  # 100 ms delay between frames

    update_gif(0)  # Start displaying the first frame

# Create the GUI
root = tk.Tk()
root.title("Shoplift Detection")
root.geometry("800x600")  # Set the initial size of the window
root.configure(bg="#f0f0f0")  # Set a background color

# Header
header = tk.Label(root, text="Shoplift Detection System", font=("Helvetica", 24, "bold"), bg="#4CAF50", fg="white", pady=10)
header.pack(fill="x")

# RTSP URL Entry
rtsp_label = Label(root, text="RTSP Stream URL:", font=("Helvetica", 14), bg="#f0f0f0")
rtsp_label.pack(pady=10)
rtsp_entry = Entry(root, width=60, font=("Helvetica", 12))
rtsp_entry.pack(pady=5)

# Button to start RTSP stream detection
rtsp_button = Button(root, text="Start RTSP Stream", command=process_rtsp, font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
rtsp_button.pack(pady=10)

# Button to upload a video or GIF file
upload_button = Button(root, text="Upload Video/GIF", command=upload_file, font=("Helvetica", 12), bg="#2196F3", fg="white", padx=10, pady=5)
upload_button.pack(pady=10)

root.mainloop()

#-----------------
#-------------
