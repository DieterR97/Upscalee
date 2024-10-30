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
        # Store initialization parameters
        self.model_name = model_name
        self.model_path = model_path
        self.scale = scale
        self.gpu_id = gpu_id
        self.face_enhance = face_enhance

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

        # Load the appropriate model and get the URL(s) for the model weights
        self.model, self.netscale, self.file_url = load_model(
            argparse.Namespace(model_name=model_name)
        )

        # Download model weights
        self.model_path = download_model_weights(self.file_url, self.model_name)

        # Create the upsampler
        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=self.model_path,
            model=self.model,
            tile=0,  # Set tile size if you want to control memory usage
            tile_pad=10,
            pre_pad=0,
            half=True,  # Use FP16 if supported for faster processing
            # half=self.device == 'cuda',  # Only use half precision on CUDA
            gpu_id=self.gpu_id,
        )

        # Optional face enhancer
        self.face_enhancer = None
        if self.face_enhance:
            from gfpgan import GFPGANer

            self.face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=self.scale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,
            )

        # Initialize available IQA metrics
        self.available_metrics = pyiqa.list_models()

    def upscale(self, image, args):
        """Upscale the given image."""
        # Pass the PIL image and the args dictionary
        save_path = process_image(args, image, self.upsampler, self.face_enhancer)

        # Load the processed image
        image = cv2.imread(save_path)

        # Check if the image was loaded successfully
        if image is None:
            raise FileNotFoundError(f"Failed to load processed image at {save_path}")

        # Swap red and blue channels (convert BGR to RGB)
        image_swapped = image[..., ::-1]  # This reverses the last dimension (channels)

        # Get filename and create final path
        filename = os.path.basename(save_path)
        final_results_dir = "pre_swapped_channels_results"  # Use the same directory
        swapped_image_path = os.path.join(final_results_dir, filename)
        
        # Save the RGB image
        cv2.imwrite(swapped_image_path, image_swapped)

        # Return the path to the saved image
        return swapped_image_path

    def evaluate_image_quality(self, original_path, upscaled_path, metrics):
        """Evaluate image quality using specified metrics."""
        results = {}
        
        for metric_name in metrics:
            try:
                # Create metric
                metric = pyiqa.create_metric(metric_name)
                
                # Calculate scores
                original_score = metric(original_path).item()
                upscaled_score = metric(upscaled_path).item()
                
                # Get score range for context
                score_range = metric.score_range
                
                results[metric_name] = {
                    'original': original_score,
                    'upscaled': upscaled_score,
                    'score_range': score_range
                }
            except Exception as e:
                print(f"Error evaluating metric {metric_name}: {str(e)}")
                
        return results


