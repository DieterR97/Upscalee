import argparse
import os
from PIL import Image
from inference_realesrgan import load_model, download_model_weights, process_image
from realesrgan import RealESRGANer
import cv2
import pyiqa
import torch

class UpscaleModel:
    def __init__(
        self,
        model_path,
        model_name="RealESRGAN_x4plus",
        scale=4,
        gpu_id=0,
        face_enhance=False,
    ):
        """Initialize the model with the necessary parameters."""
        # Store initialization parameters for the upscaling model
        self.model_name = model_name  # Name of the RealESRGAN model to use
        self.model_path = model_path  # Path to store/load model weights
        self.scale = scale  # Upscaling factor (e.g., 4 means 4x resolution)
        self.gpu_id = gpu_id  # GPU device ID to use
        self.face_enhance = face_enhance  # Whether to enhance faces in images

        # # Try CUDA first, fall back to CPU if it fails
        # try:
        #     if torch.cuda.is_available():
        #         self.device = 'cuda'
        #         self.gpu_id = gpu_id
        #         print("Using CUDA for inference")
        #     else:
        #         raise RuntimeError("CUDA not available")
        # except:
        #     self.device = 'cpu'
        #     self.gpu_id = None
        #     print("Falling back to CPU for inference")

        # Load model architecture and get weights URL
        self.model, self.netscale, self.file_url = load_model(
            argparse.Namespace(model_name=model_name)
        )

        # Download weights if not already present
        self.model_path = download_model_weights(self.file_url, self.model_name)

        # Initialize the RealESRGAN upsampler with configuration
        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=self.model_path,
            model=self.model,
            tile=0,  # 0 means process whole image at once
            tile_pad=10,  # Padding pixels for tiles if used
            pre_pad=0,
            half=True,  # Use half-precision (FP16) for better performance
            gpu_id=self.gpu_id,
        )

        # Initialize face enhancement if requested
        self.face_enhancer = None
        if self.face_enhance:
            from gfpgan import GFPGANer
            # Set up GFPGAN model for face enhancement
            self.face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=self.scale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,  # Use RealESRGAN for background
            )

        # Get list of available image quality assessment metrics
        self.available_metrics = pyiqa.list_models()

    def upscale(self, image, args):
        """
        Upscale the given image using RealESRGAN and optionally enhance faces.
        Returns the path to the processed image.
        """
        # Process the image using RealESRGAN (and GFPGAN if face_enhance=True)
        save_path = process_image(args, image, self.upsampler, self.face_enhancer)

        # Load the processed image for color channel correction
        image = cv2.imread(save_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load processed image at {save_path}")

        # Convert BGR to RGB (OpenCV loads as BGR by default)
        image_swapped = image[..., ::-1]

        # Save the color-corrected image
        filename = os.path.basename(save_path)
        final_results_dir = "pre_swapped_channels_results"
        swapped_image_path = os.path.join(final_results_dir, filename)
        cv2.imwrite(swapped_image_path, image_swapped)

        return swapped_image_path

    def evaluate_image_quality(self, original_path, upscaled_path, metrics):
        """
        Compare original and upscaled images using specified quality metrics.
        Returns a dictionary of metric scores and their valid ranges.
        """
        results = {}
        
        for metric_name in metrics:
            try:
                metric = pyiqa.create_metric(metric_name)
                
                # Calculate quality scores for both images
                original_score = metric(original_path).item()
                upscaled_score = metric(upscaled_path).item()
                
                # Store scores and valid range for interpretation
                results[metric_name] = {
                    'original': original_score,
                    'upscaled': upscaled_score,
                    'score_range': metric.score_range
                }
            except Exception as e:
                print(f"Error evaluating metric {metric_name}: {str(e)}")
                
        return results


