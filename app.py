


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

def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make predictions using the YOLO model
        results = model(frame)
        
        # Draw the detections on the frame
        annotated_frame = results[0].plot()

        # Save the frame with detections
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)

        # Display the frame with detections
        cv2.imshow("Detections", annotated_frame)

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
