import tkinter as tk
from tkinter import filedialog, Label, Entry, Button, messagebox
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import os

# Load the trained YOLOv8 model
model = YOLO('best_v2.pt')  # Load your trained model

# Create a directory to save the frames if it doesn't exist
output_dir = 'tested'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class VideoApp:

    def __init__(self, master):
        self.master = master
        self.master.title("Shoplift Detection")
        self.master.attributes('-fullscreen', True)

        # Bind Esc key to exit full screen
        self.master.bind("<Escape>", self.toggle_fullscreen)
        self.master.bind("q", self.quit_app)  # Bind 'q' key to quit the application
        self.master.geometry("800x600")  # Set the initial size of the window
        self.master.configure(bg="#f0f0f0")  # Set a background color

        # Header
        header = tk.Label(master, text="Shoplift Detection System", font=("Helvetica", 24, "bold"), bg="#4CAF50", fg="white", pady=10)
        header.pack(fill="x")

        # RTSP URL Entry
        rtsp_label = Label(master, text="RTSP Stream URL:", font=("Helvetica", 14), bg="#f0f0f0")
        rtsp_label.pack(pady=10)
        self.rtsp_entry = Entry(master, width=60, font=("Helvetica", 12))
        self.rtsp_entry.pack(pady=5)

        # Button to start RTSP stream detection
        rtsp_button = Button(master, text="Start RTSP Stream", command=self.process_rtsp, font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        rtsp_button.pack(pady=10)

        # Button to upload a video or GIF file
        upload_button = Button(master, text="Upload Video/GIF", command=self.upload_file, font=("Helvetica", 12), bg="#2196F3", fg="white", padx=10, pady=5)
        upload_button.pack(pady=10)

        # Label to display video frames
        self.video_label = Label(master)
        self.video_label.pack()

    def toggle_fullscreen(self, event=None):
        is_fullscreen = self.master.attributes('-fullscreen')
        self.master.attributes('-fullscreen', not is_fullscreen)

    def upload_file(self):
        # Open file dialog to upload a video or GIF
        file_path = filedialog.askopenfilename(filetypes=[("Video/GIF files", "*.mp4;*.avi;*.gif;*.mkv")])
        if file_path:
            if file_path.lower().endswith('.gif'):
                self.display_gif(file_path)
            else:
                self.process_video(file_path)

    def process_video(self, video_source):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to open video source: " + video_source)
            return
        self.update_frame(cap)

    def update_frame(self, cap):
        ret, frame = cap.read()
        if ret:
            # Make predictions using the YOLO model
            results = model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    confidence = box.conf[0]  # Get the confidence score
                    if confidence > 0.52:  # Only process boxes with confidence > 0.50
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]

                        if class_name == 'shoplift':
                            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates of the box
                            box_color = (0, 0, 255)  # Red color for 'shoplift'
                            text_color = (0, 0, 255)
                            box_thickness = 10
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
                            label = f"{class_name} {confidence:.2f}"
                            font_scale = 3.0
                            text_thickness = 3
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (600, 400))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection
            self.video_label.configure(image=imgtk)

            self.video_label.after(10, self.update_frame, cap)  # Update frame every 10 ms
        else:
            cap.release()

    def process_rtsp(self):
        rtsp_url = self.rtsp_entry.get()
        if rtsp_url:
            self.process_video(rtsp_url)

    def display_gif(self, gif_path):
        gif_window = tk.Toplevel(self.master)
        gif_window.title("GIF Viewer")
        gif_window.geometry("600x400")

        gif_image = Image.open(gif_path)
        frames = []

        try:
            while True:
                frame = ImageTk.PhotoImage(gif_image.copy())
                frames.append(frame)
                gif_image.seek(len(frames))  # Load next frame
        except EOFError:
            pass

        gif_label = Label(gif_window)
        gif_label.pack(fill="both", expand=True)

        def update_gif(index):
            frame = frames[index]
            gif_label.configure(image=frame)
            index = (index + 1) % len(frames)
            gif_window.after(100, update_gif, index)  # 100 ms delay between frames

        update_gif(0)  # Start displaying the first frame

    def quit_app(self, event=None):
        self.master.quit()



## Updated code.........======eeeee------

# Create the main window
root = tk.Tk()
app = VideoApp(root)
root.mainloop()
