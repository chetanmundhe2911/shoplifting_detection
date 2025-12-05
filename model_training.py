import os
import cv2
import torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8n.pt').to(device)

def get_all_video_files(folder):
    video_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    return video_files

def process_and_train(shoplifting_folder, normal_folder, output_model_path):
    # Load the pre-trained YOLOv8 model on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO('yolov8n.pt').to(device)

    def get_all_video_files(folder):
        video_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.mp4'):
                    video_files.append(os.path.join(root, file))
        return video_files

    video_paths = []
    labels = []

    # Collect video paths and labels from all subfolders
    shoplifting_videos = get_all_video_files(shoplifting_folder)
    normal_videos = get_all_video_files(normal_folder)

    for video_file in shoplifting_videos:
        video_paths.append(video_file)
        labels.append('shoplifting')

    for video_file in normal_videos:
        video_paths.append(video_file)
        labels.append('normal')

    # Split data into training, validation, and test sets
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(video_paths, labels, test_size=0.3, random_state=42)
    val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42)

    # Create temporary directories for training, validation, and test data
    temp_dataset_path = os.path.join(os.getcwd(), 'temp_dataset')
    os.makedirs(os.path.join(temp_dataset_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_path, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_path, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_path, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_path, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_path, 'test', 'labels'), exist_ok=True)

    print(f"Temporary dataset path: {temp_dataset_path}")

    def process_videos(video_paths, labels, split):
        for video_path, label in zip(video_paths, labels):
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame to 640x640
                frame_resized = cv2.resize(frame, (640, 640))
                
                # Convert frame to tensor and add batch dimension
                frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).to(device).float()  # BCHW format
                
                # Get bounding box coordinates from YOLOv8 model
                results = model(frame_tensor)
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
                
                # Skip frame if no detections
                if len(boxes) == 0:
                    continue
                
                # Prepare data for training
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    conf = box[4] if len(box) > 4 else 1.0  # Default confidence to 1.0 if not provided
                    cls = box[5] if len(box) > 5 else 0  # Default class to 0 if not provided
                    if conf > 0.5:  # Confidence threshold
                        # Save frame and bounding box to temporary files
                        img_name = f"{os.path.basename(video_path)}_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"
                        img_path = os.path.join(temp_dataset_path, split, 'images', img_name)
                        label_path = os.path.join(temp_dataset_path, split, 'labels', img_name.replace('.jpg', '.txt'))
                        
                        cv2.imwrite(img_path, frame_resized)
                        print(f"Saved image to: {img_path}")
                        
                        with open(label_path, 'w') as f:
                            # YOLO format: class x_center y_center width height
                            class_id = 0 if label == 'normal' else 1
                            img_height, img_width, _ = frame_resized.shape
                            x_center = (x1 + x2) / 2 / img_width
                            y_center = (y1 + y2) / 2 / img_height
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                        print(f"Saved label to: {label_path}")
                        
            cap.release()

    # Process training, validation, and test videos
    process_videos(train_paths, train_labels, 'train')
    process_videos(val_paths, val_labels, 'val')
    process_videos(test_paths, test_labels, 'test')
    cv2.destroyAllWindows()

    # Create dataset configuration file
    dataset_config = f"""
    path: {temp_dataset_path}
    train: train/images
    val: val/images
    test: test/images
    nc: 2
    names: ['normal', 'shoplifting']
    """
    dataset_yaml_path = os.path.join(temp_dataset_path, 'dataset.yaml')
    with open(dataset_yaml_path, 'w') as f:
        f.write(dataset_config)

    print(f"Created dataset.yaml at: {dataset_yaml_path}")
    print(f"Dataset config:\n{dataset_config}")

    # Train the model for 1 epoch and save the best model based on validation performance
    model.train(data=dataset_yaml_path, epochs=1, imgsz=640, save_period=1, device=device)

    print(f"Training completed. Model saved to: {output_model_path}")

    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dataset_path)
    print(f"Removed temporary dataset directory: {temp_dataset_path}")

# Example usage
shoplifting_folder = 'shoplifting_videos'
normal_folder = 'normal_videos'
output_model_path = 'best_model_v8.pt'

#------------

print(f"Current working directory: {os.getcwd()}")
process_and_train(shoplifting_folder, normal_folder, output_model_path)


