#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Keypoint RCNN Inference Script
Input: Video file
Output: CSV files containing 2D coordinates of 17 human body keypoints for each frame.
"""

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import csv
import os
from tqdm import tqdm

# Check for GPU availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

def initialize_model():
    """
    Initialize the Keypoint RCNN model pre-trained on COCO dataset.
    Returns:
        model: Initialized and eval-mode model.
    """
    # Load a pre-trained model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval() # Set to evaluation mode
    return model

def process_video(video_path, output_dir, model):
    """
    Process each frame of the video to extract keypoints.
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the output CSV files.
        model: The initialized Keypoint RCNN model.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")

    # COCO Keypoint names (17 keypoints)
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    frame_count = 0
    with open(os.path.join(output_dir, 'keypoints_data.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header: frame_num, kp0_x, kp0_y, kp0_score, kp1_x, ...
        header = ['frame_num']
        for i, name in enumerate(keypoint_names):
            header.extend([f'{name}_x', f'{name}_y', f'{name}_score'])
        csv_writer.writerow(header)

        with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR (OpenCV) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to tensor and add batch dimension
                image_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    predictions = model(image_tensor)

                # Assuming we take the first (and only) person detected with the highest score
                # In a controlled setting, this might be sufficient. For multiple people, you need more logic.
                if len(predictions[0]['keypoints']) > 0:
                    # Get the keypoints of the first detected person
                    keypoints = predictions[0]['keypoints'][0].cpu().numpy()
                    scores = predictions[0]['scores'].cpu().numpy()

                    # Prepare row for CSV: [frame_count, kp0_x, kp0_y, kp0_score, ...]
                    row = [frame_count]
                    for kp in keypoints:
                        row.extend([kp[0], kp[1], kp[2]]) # x, y, confidence score
                    csv_writer.writerow(row)
                else:
                    # If no person is detected, write a row of zeros or NaNs
                    row = [frame_count] + [0] * (len(keypoint_names) * 3)
                    csv_writer.writerow(row)

                frame_count += 1
                pbar.update(1)

    cap.release()
    print(f"Inference complete. Data saved to {output_dir}")

if __name__ == "__main__":
    # --- Configuration ---
    INPUT_VIDEO_PATH = "path/to/your/input/video.mp4"
    OUTPUT_DIR = "path/to/your/output/directory"

    # --- Run Inference ---
    kp_model = initialize_model()
    process_video(INPUT_VIDEO_PATH, OUTPUT_DIR, kp_model)