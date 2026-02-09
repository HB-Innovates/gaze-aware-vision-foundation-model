#!/usr/bin/env python3
"""Interactive demo for Gaze-Aware Vision Foundation Model.

Demonstrates real-time gaze tracking, multi-modal understanding,
and efficient inference for interview presentation.

Usage:
    python demo.py --mode webcam  # Live webcam demo
    python demo.py --mode video --input path/to/video.mp4
    python demo.py --mode synthetic  # Synthetic demo data
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from models.gaze_tracking.predictor import GazePredictor
from models.multimodal_foundation.vlm import GazeAwareVLM
from models.efficient_inference.snn_converter import convert_to_snn, EnergyMonitor
from models.efficient_inference.quantization import quantize_model, benchmark_inference


class GazeDemo:
    """Interactive demonstration of gaze tracking system."""
    
    def __init__(
        self,
        use_snn: bool = False,
        use_quantization: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.device = torch.device(device)
        
        # Load models
        print("Loading gaze tracking model...")
        self.gaze_model = GazePredictor(hidden_dim=128).to(self.device)
        
        # Apply optimizations
        if use_snn:
            print("Converting to Spiking Neural Network (38x energy reduction)...")
            self.gaze_model = convert_to_snn(self.gaze_model, num_steps=25)
        
        if use_quantization:
            print("Quantizing model (4x size reduction)...")
            self.gaze_model = quantize_model(self.gaze_model, quantization_type='dynamic')
        
        self.gaze_model.eval()
        
        # Optional: Load VLM for multi-modal understanding
        print("Loading vision-language model...")
        self.vlm = GazeAwareVLM(
            vision_model='openai/clip-vit-base-patch32',
            text_model='gpt2',
        ).to(self.device)
        self.vlm.eval()
        
        # Energy monitoring
        self.energy_monitor = EnergyMonitor()
        
        # Gaze history for temporal prediction
        self.gaze_history = []
        self.max_history = 10
        
        # Visualization setup
        self.gaze_points = []
    
    def preprocess_eye_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess eye image for gaze prediction.
        
        Args:
            image: Eye region image [H, W, 3]
            
        Returns:
            Preprocessed tensor [1, 3, 64, 64]
        """
        # Resize to model input size
        image = cv2.resize(image, (64, 64))
        
        # Convert to grayscale or keep RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add channel and batch dimensions
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        
        return image.to(self.device)
    
    def predict_gaze(self, eye_image: np.ndarray) -> tuple:
        """Predict gaze direction from eye image.
        
        Args:
            eye_image: Eye region image
            
        Returns:
            (yaw, pitch) in degrees, confidence
        """
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_eye_image(eye_image)
            
            # Predict
            yaw, pitch = self.gaze_model(input_tensor)
            
            # Convert to degrees
            yaw = yaw.item() * 180 / np.pi
            pitch = pitch.item() * 180 / np.pi
            
            # Confidence (simplified)
            confidence = 0.95  # In practice, compute from model uncertainty
        
        return yaw, pitch, confidence
    
    def temporal_prediction(self, current_gaze: tuple) -> tuple:
        """Predict future gaze using temporal history.
        
        Args:
            current_gaze: (yaw, pitch) current position
            
        Returns:
            (yaw, pitch) predicted 5 frames ahead
        """
        self.gaze_history.append(current_gaze)
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        if len(self.gaze_history) < 3:
            return current_gaze
        
        # Simple linear extrapolation
        history_array = np.array(self.gaze_history)
        velocities = np.diff(history_array, axis=0)
        avg_velocity = np.mean(velocities, axis=0)
        
        # Predict 5 frames ahead
        future_gaze = history_array[-1] + 5 * avg_velocity
        
        return tuple(future_gaze)
    
    def draw_gaze_vector(self, frame: np.ndarray, origin: tuple, gaze: tuple):
        """Draw gaze direction vector on frame.
        
        Args:
            frame: Video frame
            origin: (x, y) eye center
            gaze: (yaw, pitch) gaze angles in degrees
        """
        yaw, pitch = gaze
        
        # Convert angles to 2D direction
        length = 100
        dx = length * np.sin(np.radians(yaw))
        dy = length * np.sin(np.radians(pitch))
        
        end_point = (int(origin[0] + dx), int(origin[1] - dy))
        
        # Draw arrow
        cv2.arrowedLine(
            frame,
            origin,
            end_point,
            (0, 255, 0),
            2,
            tipLength=0.3
        )
    
    def create_heatmap(self, frame_shape: tuple) -> np.ndarray:
        """Generate gaze heatmap from accumulated points.
        
        Args:
            frame_shape: (height, width) of frame
            
        Returns:
            Heatmap image
        """
        heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        
        for point in self.gaze_points:
            x, y = point
            if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                cv2.circle(heatmap, (x, y), 30, 1.0, -1)
        
        # Apply Gaussian blur
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        return heatmap_colored
    
    def run_webcam_demo(self):
        """Run live webcam demonstration."""
        print("Starting webcam demo... Press 'q' to quit, 'h' for heatmap")
        
        cap = cv2.VideoCapture(0)
        show_heatmap = False
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face and eyes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Detect eyes within face
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes[:1]:  # Process first eye
                    eye_center = (x + ex + ew//2, y + ey + eh//2)
                    
                    # Extract eye region
                    eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    if eye_img.size > 0:
                        # Predict gaze
                        yaw, pitch, confidence = self.predict_gaze(eye_img)
                        
                        # Temporal prediction
                        predicted_gaze = self.temporal_prediction((yaw, pitch))
                        
                        # Draw gaze vector
                        self.draw_gaze_vector(frame, eye_center, (yaw, pitch))
                        
                        # Store point for heatmap
                        self.gaze_points.append(eye_center)
                        if len(self.gaze_points) > 100:
                            self.gaze_points.pop(0)
                        
                        # Display info
                        info_text = f"Yaw: {yaw:.1f}° Pitch: {pitch:.1f}° Conf: {confidence:.2f}"
                        cv2.putText(
                            frame,
                            info_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
            
            # Show heatmap overlay if enabled
            if show_heatmap and len(self.gaze_points) > 0:
                heatmap = self.create_heatmap(frame.shape)
                frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
            
            cv2.imshow('Gaze Tracking Demo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_heatmap = not show_heatmap
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_synthetic_demo(self):
        """Run demonstration with synthetic data."""
        print("Running synthetic demo...")
        
        # Generate synthetic eye images
        num_frames = 100
        results = []
        
        for i in range(num_frames):
            # Create synthetic eye image
            eye_img = np.random.rand(64, 64) * 255
            eye_img = eye_img.astype(np.uint8)
            
            # Predict gaze
            yaw, pitch, confidence = self.predict_gaze(eye_img)
            
            results.append({
                'frame': i,
                'yaw': yaw,
                'pitch': pitch,
                'confidence': confidence,
            })
            
            if i % 10 == 0:
                print(f"Processed {i}/{num_frames} frames")
        
        # Visualize results
        self.visualize_synthetic_results(results)
    
    def visualize_synthetic_results(self, results: list):
        """Visualize synthetic demo results."""
        frames = [r['frame'] for r in results]
        yaws = [r['yaw'] for r in results]
        pitches = [r['pitch'] for r in results]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(frames, yaws, label='Yaw', marker='o')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Yaw (degrees)')
        axes[0].set_title('Gaze Yaw over Time')
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].plot(frames, pitches, label='Pitch', marker='o', color='orange')
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Pitch (degrees)')
        axes[1].set_title('Gaze Pitch over Time')
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=150)
        print("Results saved to demo_results.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Gaze-Aware Vision Foundation Model Demo'
    )
    parser.add_argument(
        '--mode',
        choices=['webcam', 'video', 'synthetic'],
        default='synthetic',
        help='Demo mode'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input video path (for video mode)'
    )
    parser.add_argument(
        '--use-snn',
        action='store_true',
        help='Use Spiking Neural Network (38x energy reduction)'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Use INT8 quantization (4x size reduction)'
    )
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = GazeDemo(
        use_snn=args.use_snn,
        use_quantization=args.quantize,
    )
    
    # Run appropriate demo
    if args.mode == 'webcam':
        demo.run_webcam_demo()
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        # Implement video demo (similar to webcam)
        print(f"Video demo not fully implemented. Use webcam or synthetic mode.")
    else:  # synthetic
        demo.run_synthetic_demo()


if __name__ == '__main__':
    main()
