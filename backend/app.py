from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
import os
from werkzeug.utils import secure_filename
from pyiqa import create_metric
from PIL import Image
from model import (
    UpscaleModel,
)
import pyiqa
import torch
from pathlib import Path
import json
from contextlib import contextmanager
import logging
import numpy as np
from PIL.ExifTags import TAGS
import math
from get_image_info import get_image_info
from functools import lru_cache
import sys
from threading import Lock
import re
from spandrel_inference import SpandrelUpscaler
from model_watcher import setup_model_watcher
from flask_socketio import SocketIO
from typing import Dict, List
import base64
import io
import pkg_resources
import subprocess
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import hashlib
from tqdm import tqdm
from threading import Event
import time
import shutil

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})  # Enable CORS for all routes

# Directory structure for handling image processing pipeline
UPLOAD_FOLDER = 'temp_uploads'  # Stores uploaded images temporarily
TEMP_RESULTS_DIR = 'pre_swapped_channels_results'  # Intermediate processing results
FINAL_RESULTS_DIR = 'final_results'  # Final processed images

def cleanup_temp_directories():
    """Clean up temporary directories on startup."""
    directories = [UPLOAD_FOLDER, TEMP_RESULTS_DIR, FINAL_RESULTS_DIR]
    for directory in directories:
        if os.path.exists(directory):
            # Remove all files in the directory
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            os.makedirs(directory)

# Clean up directories on startup
cleanup_temp_directories()

# Create all necessary directories (you can remove this since cleanup_temp_directories handles creation)
# for directory in [UPLOAD_FOLDER, TEMP_RESULTS_DIR, FINAL_RESULTS_DIR]:
#     if not os.path.exists(directory):
#         os.makedirs(directory)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define device for PyTorch operations
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

# iqa_metric = create_metric('lpips', device='cpu')

# # Initialize the model once when the server starts
# # TODO: Initialize only after input image is received and analysed
# # TODO: Update the model path and model name dynamically based on input image classification
# # TODO: check if gpu is present and can do 'cuda' or related and set gpu_id appropriately
# model = UpscaleModel(
#     model_path="",
#     model_name="",
#     # model_name="RealESRGAN_x4plus_anime_6B",  # Model specialized for anime-style images
#     # model_name="RealESRGAN_x4plus",  # Alternative model for general images
#     scale=4,  # Upscaling factor
#     # gpu_id=None,  # Or specify the GPU ID if you're using one
#     gpu_id=0,
# )

# Initialize the IQA metric (e.g., LPIPS)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# iqa_metric = pyiqa.create_metric('lpips', device=device)
# Get available metrics directly from pyiqa
AVAILABLE_METRICS = pyiqa.list_models()

# Define supported models with their capabilities and characteristics
AVAILABLE_MODELS = {
    "RealESRGAN_x4plus": {
        "name": "RealESRGAN x4+ (General)",
        "description": "General purpose x4 upscaling",
        "scale": 4,
        "variable_scale": False,
        "architecture": "Real-ESRGAN",
        "source_url": "https://github.com/xinntao/Real-ESRGAN"
    },
    "RealESRGAN_x2plus": {
        "name": "RealESRGAN x2+ (General)",
        "description": "General purpose x2 upscaling",
        "scale": 2,
        "variable_scale": False,
        "architecture": "Real-ESRGAN",
        "source_url": "https://github.com/xinntao/Real-ESRGAN"
    },
    "RealESRNet_x4plus": {
        "name": "RealESRNet x4+ (Smooth)",
        "description": "General purpose x4 upscaling with MSE loss (over-smooth effects)",
        "scale": 4,
        "variable_scale": False,
        "architecture": "Real-ESRGAN",
        "source_url": "https://github.com/xinntao/Real-ESRGAN"
    },
    "realesr-general-x4v3": {
        "name": "RealESR General v3 (Fast)",
        "description": "General purpose x4 (can also be used for x1, x2, x3) upscaling, a tiny small model (consume much fewer GPU memory and time); not too strong deblur and denoise capacity",
        "scale": 4,
        "variable_scale": True,
        "architecture": "Real-ESRGAN",
        "source_url": "https://github.com/xinntao/Real-ESRGAN"
    },
    "RealESRGAN_x4plus_anime_6B": {
        "name": "RealESRGAN x4+ Anime 6B",
        "description": "Optimized for anime/artwork/illustrations x4 upscaling",
        "scale": 4,
        "variable_scale": False,
        "architecture": "Real-ESRGAN",
        "source_url": "https://github.com/xinntao/Real-ESRGAN"
    },
    "realesr-animevideov3": {
        "name": "RealESR AnimeVideo v3",
        "description": "Optimized for anime video x4 (can also be used for x1, x2, x3) upscaling",
        "scale": 4,
        "variable_scale": True,
        "architecture": "Real-ESRGAN",
        "source_url": "https://github.com/xinntao/Real-ESRGAN"
    }
}

# Add this new route
@app.route("/models", methods=["GET"])
def get_models():
    """Get all models and handle notifications about removed models."""
    scan_result = scan_model_directory()
    
    if "error" in scan_result:
        return jsonify(scan_result), 404
    
    # Merge default models with registered custom models
    all_models = AVAILABLE_MODELS.copy()  # Start with default models
    all_models.update(scan_result["registered"])  # Add custom registered models
    
    # Update the scan result with merged models
    scan_result["registered"] = all_models
        
    # If any models were removed, include this in the response
    if scan_result["removed"]:
        removed_models = scan_result["removed"]
        scan_result["message"] = f"Removed {len(removed_models)} model(s) that no longer exist: {', '.join(removed_models)}"
    
    return jsonify(scan_result)

@app.route("/upscale", methods=["POST"])
def upscale_image():
    # Check if an image file is provided in the request
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream)

    max_dimension = 1024  # Maximum size for width or height
    # Resize the image if it exceeds the maximum dimension
    image = resize_image(image, max_dimension)

    print("Loaded image size:", image.size)

    # Prepare filename and model information for output
    filename = image_file.filename
    filename_no_ext = filename.rsplit(".", 1)[0]

    # Get the selected model and scale from the request
    selected_model = request.form.get("model")
    if not selected_model:
        return jsonify({"error": "No model selected"}), 400
    
    # Check if this is a custom model that should use Spandrel
    registered_models = load_registered_models()
    is_spandrel_model = selected_model in registered_models
    
    # Get the default scale based on whether it's a built-in or custom model
    if is_spandrel_model:
        default_scale = registered_models[selected_model].get("scale", 4)
    else:
        default_scale = AVAILABLE_MODELS[selected_model]["scale"]
    
    # Get the selected scale from the request, or use the default
    selected_scale = int(request.form.get("scale", default_scale))
    
    # Initialize the appropriate model
    if is_spandrel_model:
        config = load_config()
        model = SpandrelUpscaler(
            model_path=config["modelPath"],
            model_name=selected_model,
            scale=selected_scale,
            gpu_id=0,
        )
    else:
        model = UpscaleModel(
            model_path="",
            model_name=selected_model,
            scale=selected_scale,
            gpu_id=0,
        )

    # Define arguments for the upscaling process
    args = {
        "face_enhance": False,  # Enable/disable face enhancement
        "outscale": selected_scale,  # Use the selected scale
        "output": TEMP_RESULTS_DIR,  # Use the constant
        "suffix": f"{filename_no_ext}_with_{model.model_name}_x{selected_scale}",  # Suffix for output filename
        "ext": "png",  # Output file extension
    }

    # Upscale the image using the model
    upscaled_image_path = model.upscale(image, args)

    # Verify the file exists before sending
    if not os.path.exists(upscaled_image_path):
        return jsonify({"error": "Failed to generate upscaled image"}), 500

    # Return the upscaled image to the client
    return send_file(upscaled_image_path, mimetype="image/png")


# @app.route('/assess_quality', methods=['POST'])
# def assess_quality():
#     # Assume the request contains paths to two images to compare
#     data = request.json
#     image1_path = data['image1']
#     image2_path = data['image2']
    
#     # Compute the IQA score
#     score = iqa_metric(image1_path, image2_path)
    
#     return jsonify({'score': score.item()})

def resize_image(image, max_size):
    """
    Resize the input image if it exceeds the maximum dimension while maintaining aspect ratio.
    """
    width, height = image.size

    # Check if resizing is necessary
    if width <= max_size and height <= max_size:
        # Return the original image if it's smaller than the max size
        return image

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Resize based on the maximum size while maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image

# Add this near your other global variables
MODEL_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

@app.route("/model-status/<model_name>", methods=["GET"])
def check_model_status(model_name):
    """Check if a model's weights are already downloaded."""
    # First check if this is a custom (Spandrel) model
    registered_models = load_registered_models()
    if model_name in registered_models and registered_models[model_name].get("is_spandrel", False):
        # Spandrel models are already downloaded in the custom directory
        return jsonify({"downloaded": True})
    
    # For RealESRGAN models, check the weights directory
    model_path = Path(MODEL_WEIGHTS_DIR) / f"{model_name}.pth"
    return jsonify({"downloaded": model_path.exists()})

# Get available metrics and their types
def get_metric_info():
    """Get information about available metrics."""
    catalog = load_metrics_catalog()
    selected = get_selected_metrics()
    
    # Filter catalog to only include selected metrics
    selected_metrics = selected['nr'] + selected['fr']
    return {k: v for k, v in catalog.items() if k in selected_metrics}

# @app.route('/available-metrics', methods=['GET'])
# def get_available_metrics():
#     return jsonify(get_metric_info())

# Add this near your other global variables
METRIC_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metric_weights")
os.makedirs(METRIC_WEIGHTS_DIR, exist_ok=True)

# Define supported quality metrics with detailed metadata
SUPPORTED_METRICS = {
    'musiq': {
        'name': 'Multi-scale Image Quality Transformer (musiq)',
        'description': 'State-of-the-art transformer-based model that evaluates image quality across multiple scales, trained on multiple IQA databases',
        'score_range': [0, 100],
        'type': 'nr',
        'higher_better': True
    },
    'nima': {
        'name': 'Neural Image Assessment (nima)',
        'description': 'Predicts aesthetic and technical quality scores using deep CNN, trained on AVA dataset',
        'score_range': [1, 10],
        'type': 'nr',
        'higher_better': True
    },
    'brisque': {
        'name': 'Blind/Referenceless Image Spatial Quality Evaluator (brisque)',
        'description': 'Classic NR-IQA method using natural scene statistics, fast and reliable',
        'score_range': [0, 100],
        'type': 'nr',
        'higher_better': False
    },
    'hyperiqa': {
        'name': 'Hypernetwork for Image Quality Assessment (hyperiqa)',
        'description': 'Uses hypernetworks to generate quality-aware features, good at handling various distortion types',
        'score_range': [0, 1],
        'type': 'nr',
        'higher_better': True
    },
    'maniqa': {
        'name': 'Multi-dimension Attention Network for IQA (maniqa)',
        'description': 'Uses multi-dimensional attention mechanism to capture quality-relevant features',
        'score_range': [0, 1],
        'type': 'nr',
        'higher_better': True
    },
    'lpips': {
        'name': 'Learned Perceptual Image Patch Similarity (lpips)',
        'description': 'Deep learning-based perceptual similarity metric that correlates well with human perception',
        'score_range': [0, 1],
        'type': 'fr',
        'higher_better': False
    },
    'dists': {
        'name': 'Deep Image Structure and Texture Similarity (dists)',
        'description': 'Combines structure and texture features for similarity assessment',
        'score_range': [0, 1],
        'type': 'fr',
        'higher_better': False
    },
    'fsim': {
        'name': 'Feature Similarity Index Measure (fsim)',
        'description': 'Based on phase congruency and gradient magnitude features',
        'score_range': [0, 1],
        'type': 'fr',
        'higher_better': True
    },
    'pieapp': {
        'name': 'Perceptual Image-Error Assessment through Pairwise Preference (pieapp)',
        'description': 'Trained on human perceptual preferences between distorted image pairs',
        'score_range': [0, 1],
        'type': 'fr',
        'higher_better': False
    },
    'ms_ssim': {
        'name': 'Multi-Scale Structural Similarity Index (ms_ssim)',
        'description': 'Enhanced version of SSIM that evaluates images at multiple scales',
        'score_range': [0, 1],
        'type': 'fr',
        'higher_better': True
    }
}

@app.route('/available-metrics', methods=['GET'])
def get_available_metrics():
    return jsonify(SUPPORTED_METRICS)

# Configure environment for metric weights storage and management
os.environ['TORCH_HOME'] = METRIC_WEIGHTS_DIR
os.environ['TORCH_HUB_DIR'] = METRIC_WEIGHTS_DIR
torch.hub.set_dir(METRIC_WEIGHTS_DIR)

# Add this cache for loaded metrics
@lru_cache(maxsize=None)
def get_cached_metric(metric_name):
    """
    Cache loaded metrics to improve performance by preventing unnecessary reloading.
    Uses LRU cache to store metrics in memory after first load.
    """
    return pyiqa.create_metric(metric_name, device=device)

@contextmanager
def temporary_torch_home():
    """Temporarily set TORCH_HOME to our metric weights directory."""
    original_torch_home = os.environ.get('TORCH_HOME')
    os.environ['TORCH_HOME'] = METRIC_WEIGHTS_DIR
    try:
        yield
    finally:
        if original_torch_home:
            os.environ['TORCH_HOME'] = original_torch_home
        else:
            del os.environ['TORCH_HOME']

def check_metric_weights_available(metric_name):
    """Check if metric weights are available."""
    try:
        with temporary_torch_home():
            # Try to create metric - this will use existing weights if available
            metric = get_cached_metric(metric_name)
            return True
    except Exception as e:
        print(f"Weights not found for {metric_name}: {str(e)}")
        return False

@contextmanager
def download_detector():
    """Context manager to detect when downloads start."""
    class DownloadStartedException(Exception):
        def __init__(self, metric_name):
            self.metric_name = metric_name

    class Detector:
        def __init__(self):
            self.original_stdout = sys.__stdout__
            
        def write(self, text):
            # Check for download message
            if "Downloading:" in text:
                # Extract metric name from the path
                if "pyiqa/" in text:
                    metric_name = text.split("pyiqa/")[1].split("_")[0].lower()
                    raise DownloadStartedException(metric_name)
            self.original_stdout.write(text)
            
        def flush(self):
            self.original_stdout.flush()

    detector = Detector()
    original_stdout = sys.stdout
    sys.stdout = detector
    
    try:
        yield detector
    finally:
        sys.stdout = original_stdout

def check_metric_downloads(metrics):
    """Check if any metrics need downloading and return download status if needed."""
    class DownloadChecker:
        def __init__(self, metric_name):
            self._original_stdout = sys.stdout
            self._original_write = sys.stdout.write
            self.metric_name = metric_name
            self.needs_download = False
            
        def __enter__(self):
            def new_write(text):
                if "Downloading:" in text:
                    self.needs_download = True
                return self._original_write(text)
            
            sys.stdout.write = new_write
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.write = self._original_write
            return False

    # Check each metric
    for metric_name in metrics:
        try:
            with DownloadChecker(metric_name) as checker:
                # Just try to create the metric to see if it needs downloading
                _ = get_cached_metric(metric_name)
                if checker.needs_download:
                    return jsonify({
                        'downloading': {
                            'status': True,
                            'metrics': [metric_name],
                            'message': f'Downloading weights for {metric_name}... This is a one-time download for future use.'
                        }
                    })
        except Exception as e:
            print(f"Error checking metric {metric_name}: {str(e)}")
            continue
    
    return None

# Global download tracking
download_in_progress = False
download_lock = Lock()

# Define exception class at module level
class DownloadStartedException(Exception):
    """
    Custom exception to handle metric weight download detection.
    Raised when a new metric download is initiated.
    """
    def __init__(self, metric_name):
        self.metric_name = metric_name

def get_cached_metric_with_download_check(metric_name):
    """Try to create metric and detect if download is needed."""
    global download_in_progress
    
    class DownloadDetector:
        def __init__(self):
            self._original_stdout = sys.stdout
            self._original_write = sys.stdout.write
            
        def __enter__(self):
            def new_write(text):
                global download_in_progress
                if "Downloading:" in text and not download_in_progress:
                    with download_lock:
                        download_in_progress = True
                    self._original_write(text)
                    raise DownloadStartedException(metric_name)
                return self._original_write(text)
            
            sys.stdout.write = new_write
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.write = self._original_write
            return False

    with DownloadDetector():
        return get_cached_metric(metric_name)
    
@app.route('/evaluate-quality', methods=['POST'])
def evaluate_quality():
    """
    Comprehensive image quality evaluation endpoint.
    Supports both full-reference (FR) and no-reference (NR) metrics:
    - FR metrics compare original and upscaled images
    - NR metrics evaluate single images independently
    
    Features:
    - Automatic metric weight downloading
    - Download status tracking
    - Concurrent request handling
    - Error handling and reporting
    
    Includes cleanup of temporary files after evaluation.
    """
    global download_in_progress
    temp_files = []
    
    try:
        original_file = request.files['original_image']
        upscaled_file = request.files['upscaled_image']
        metrics = json.loads(request.form['metrics'])
        
        # Save temporary files and track their paths
        original_path = os.path.join(UPLOAD_FOLDER, secure_filename(original_file.filename))
        upscaled_path = os.path.join(UPLOAD_FOLDER, secure_filename(upscaled_file.filename))
        temp_files.extend([original_path, upscaled_path])
        
        original_file.save(original_path)
        upscaled_file.save(upscaled_path)
        
        # Load and process images
        original_img = Image.open(original_file)
        upscaled_img = Image.open(upscaled_file)
        
        # Resize original image to match upscaled dimensions for FR metrics
        original_img_resized = original_img.resize(upscaled_img.size, Image.LANCZOS)
        
        # Save the images
        original_img_resized.save(original_path)
        upscaled_img.save(upscaled_path)

        # Check metrics for downloads
        for metric_name in metrics:
            try:
                metric = get_cached_metric_with_download_check(metric_name)
            except DownloadStartedException as e:
                return jsonify({
                    'downloading': {
                        'status': True,
                        'metrics': [e.metric_name],
                        'message': f'Downloading weights for {e.metric_name}... This is a one-time download for future use.'
                    }
                })
            except Exception as e:
                print(f"Error with metric {metric_name}: {str(e)}")
                continue

        # Reset download flag after successful evaluation
        with download_lock:
            download_in_progress = False

        # Evaluate metrics
        results = {}
        for metric_name in metrics:
            try:
                metric = get_cached_metric(metric_name)
                metric_info = SUPPORTED_METRICS[metric_name]
                
                print(f"\n{'='*50}")
                print(f"Calculating: {metric_name}")
                print(f"Type: {metric_info['type']}")
                print(f"Expected Range: {metric_info['score_range']}")
                print(f"Higher is better: {metric_info['higher_better']}")
                
                if metric_info['type'] == 'fr':
                    score = metric(upscaled_path, original_path).item()
                    print(f"Score: {score:.4f}")
                    results[metric_name] = {
                        'score': score,
                        'type': 'fr',
                        'score_range': metric_info['score_range'],
                        'higher_better': metric_info['higher_better']
                    }
                else:  # No-reference metric
                    original_score = metric(original_path).item()
                    upscaled_score = metric(upscaled_path).item()
                    print(f"Original Score: {original_score:.4f}")
                    print(f"Upscaled Score: {upscaled_score:.4f}")
                    results[metric_name] = {
                        'original': original_score,
                        'upscaled': upscaled_score,
                        'type': 'nr',
                        'score_range': metric_info['score_range'],
                        'higher_better': metric_info['higher_better']
                    }
                print(f"{'='*50}\n")
            except Exception as e:
                print(f"\nError calculating {metric_name}:")
                print(f"Error: {str(e)}")
                print(f"{'='*50}\n")
                results[metric_name] = {
                    'error': str(e),
                    'type': metric_info['type'],
                    'score_range': metric_info['score_range'],
                    'higher_better': metric_info['higher_better']
                }
        
        # Clean up temporary files only after successful evaluation
        cleanup_temp_files(temp_files)
        return jsonify(results)
        
    except Exception as e:
        # Clean up temporary files in case of error
        cleanup_temp_files(temp_files)
        print(f"Error in evaluate_quality: {str(e)}")
        return jsonify({'error': str(e)}), 500

def cleanup_temp_files(file_paths):
    """Clean up temporary files used during quality evaluation."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up temporary file {file_path}: {str(e)}")

# Add this to disable Flask's reloader for specific paths
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

@app.route("/image-info", methods=["POST"])
def get_image_details():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_file = request.files["image"]
        image = Image.open(image_file)
        
        # Get detailed image information
        info = get_image_info(image)
        
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add these near the top of the file
CONFIG_FILE = Path(__file__).parent / 'config.json'

def load_config():
    """Load configuration from JSON file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default configuration
        default_config = {
            "maxImageDimension": 1024,
            "modelPath": os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        }
        save_config(default_config)
        return default_config

def save_config(config):
    """Save configuration to JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

# Add this new route
@app.route("/config", methods=["GET", "POST"])
def handle_config():
    if request.method == "GET":
        return jsonify(load_config())
    elif request.method == "POST":
        new_config = request.json
        save_config(new_config)
        return jsonify({"message": "Configuration saved successfully"})

# Update the resize_image function to use the config
def resize_image(image, max_size=None):
    """Resize the input image if it exceeds the maximum dimension while maintaining aspect ratio."""
    if max_size is None:
        config = load_config()
        max_size = config["maxImageDimension"]
    width, height = image.size

    # Check if resizing is necessary
    if width <= max_size and height <= max_size:
        # Return the original image if it's smaller than the max size
        return image

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Resize based on the maximum size while maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image

def load_registered_models():
    """Load registered models from JSON file."""
    try:
        with open('registered_models.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_registered_models(models):
    """Save registered models to JSON file."""
    with open('registered_models.json', 'w') as f:
        json.dump(models, f, indent=2)

def scan_model_directory():
    """Scan custom model directory for .pth files and validate registered models."""
    config = load_config()
    model_dir = Path(config["modelPath"])
    registered_models = load_registered_models()
    
    unregistered_models = []
    models_to_remove = []
    
    if not model_dir.exists():
        return {"error": f"Model directory not found: {model_dir}"}
    
    # Get all .pth files in the directory
    existing_model_files = {f.stem: f for f in model_dir.glob("*.pth")}
    
    # First, validate registered models
    for model_name, model_info in list(registered_models.items()):
        if model_name not in existing_model_files:
            models_to_remove.append(model_name)
            continue
    
    # Remove models that no longer exist
    for model_name in models_to_remove:
        del registered_models[model_name]
    
    # Save the updated registered models after removal
    save_registered_models(registered_models)
    
    # Find unregistered models
    for model_name, model_file in existing_model_files.items():
        if model_name not in registered_models:
            # Try to get Spandrel info
            spandrel_info = SpandrelUpscaler.get_model_info(str(model_file))
            print(f"Spandrel info for {model_name}:", spandrel_info)
            
            # Try to determine scale from filename if Spandrel info not available
            scale_match = re.search(r'x(\d+)', model_name.lower())
            scale = int(scale_match.group(1)) if scale_match else None
            
            model_data = {
                "name": model_name,
                "file_name": model_file.name,
                "file_pattern": model_file.name,
                "scale": spandrel_info.get("scale", scale),
                "path": str(model_file),
                "spandrel_info": spandrel_info if spandrel_info.get("is_supported") else None
            }
            print(f"Model data being sent to frontend:", model_data)
            unregistered_models.append(model_data)
    
    return {
        "registered": registered_models,
        "unregistered": unregistered_models,
        "removed": models_to_remove
    }

@app.route("/register-model", methods=["POST"])
def register_model():
    """Register a new model with provided information."""
    try:
        model_info = request.json
        registered_models = load_registered_models()
        
        # Get the full path to the model file
        config = load_config()
        model_path = str(Path(config["modelPath"]) / model_info["file_pattern"])
        
        # Try to get Spandrel-specific information
        spandrel_info = SpandrelUpscaler.get_model_info(model_path)
        
        # Create the model registration info
        model_registration = {
            "name": model_info["display_name"],
            "description": model_info["description"],
            "scale": model_info["scale"],
            "variable_scale": model_info["variable_scale"],
            "architecture": model_info["architecture"],
            "source_url": model_info.get("source_url", ""),  # Add this field
            "file_pattern": model_info["file_pattern"],
            "is_spandrel": True
        }
        
        # Add Spandrel-specific info if available
        if spandrel_info:
            model_registration.update({
                "spandrel_info": spandrel_info
            })
        
        # Add the new model to registered models
        registered_models[model_info["name"]] = model_registration
        
        save_registered_models(registered_models)
        return jsonify({"success": True, "message": "Model registered successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

config = load_config()
socketio = setup_model_watcher(app, config["modelPath"])

# Add these constants near your other global variables
METRICS_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_catalog.json")
DEFAULT_METRICS = {
    "nr": ["musiq", "nima", "brisque", "niqe"],  # Added brisque and niqe as they are classic baselines
    "fr": ["psnr", "ssim", "lpips", "dists"]     # Added lpips and dists as they are modern perceptual metrics
}

def load_metrics_catalog() -> Dict:
    """Load the complete metrics catalog from JSON file."""
    with open(METRICS_CATALOG_PATH, 'r') as f:
        return json.load(f)

def get_selected_metrics() -> Dict[str, List[str]]:
    """Get currently selected metrics from config."""
    config = load_config()
    return config.get('selectedMetrics', DEFAULT_METRICS)

def save_selected_metrics(metrics: Dict[str, List[str]]) -> None:
    """Save selected metrics to config file."""
    config = load_config()
    config['selectedMetrics'] = metrics
    save_config(config)

# Add these new routes
@app.route('/metrics/catalog', methods=['GET'])
def get_metrics_catalog():
    """Get the complete metrics catalog."""
    return jsonify(load_metrics_catalog())

@app.route('/metrics/selected', methods=['GET'])
def get_selected_metrics_route():
    """Get currently selected metrics."""
    return jsonify(get_selected_metrics())

@app.route('/metrics/selected', methods=['POST'])
def update_selected_metrics():
    """Update selected metrics."""
    metrics = request.json
    if not isinstance(metrics, dict) or not all(k in metrics for k in ['nr', 'fr']):
        return jsonify({'error': 'Invalid metrics format'}), 400
    
    save_selected_metrics(metrics)
    return jsonify({'status': 'success'})

@app.route('/calculate-metrics', methods=['POST'])
def calculate_metrics():
    try:
        data = request.json
        original_image_b64 = data.get('original_image')
        upscaled_image_b64 = data.get('upscaled_image')
        metrics_to_calculate = data.get('metrics', {'nr': [], 'fr': []})

        # Load metrics catalog for score ranges
        with open('metrics_catalog.json', 'r') as f:
            metrics_catalog = json.load(f)

        def base64_to_image(base64_str):
            if 'data:image' in base64_str:
                base64_str = base64_str.split(',')[1]
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image, np.array(image)

        original_image, original_array = base64_to_image(original_image_b64)
        upscaled_image, upscaled_array = base64_to_image(upscaled_image_b64)

        # Add minimum size check and resizing
        def ensure_min_size(image, min_size=224):
            """Ensure image meets minimum size requirements."""
            width, height = image.size
            if width < min_size or height < min_size:
                # Calculate new size maintaining aspect ratio
                if width < height:
                    new_width = min_size
                    new_height = int(height * (min_size / width))
                else:
                    new_height = min_size
                    new_width = int(width * (min_size / height))
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return image

        # List of metrics that require minimum 224px size
        min_size_metrics = ['liqe', 'liqe_mix']

        # Create temporary directories for inception_score
        temp_original_dir = 'temp_original_inception'
        temp_upscaled_dir = 'temp_upscaled_inception'


        results = {}
        temp_files = []  # Track all temporary files

        # Calculate No-Reference metrics
        for metric_id in metrics_to_calculate['nr']:
            try:

                metric_info = metrics_catalog[metric_id]
                score_range = metric_info['score_range']
                higher_better = metric_info['higher_better']
                
                print(f"\n{'='*50}")
                print(f"Calculating metric: {metric_id}")
                print(f"Type: {metric_info['type']}")
                print(f"Category: {metric_info['category']}")

                print(f"Score range: [{score_range[0]}, {score_range[1]}]")

                print(f"Expected Range: {metrics_catalog[metric_id]['score_range']}")
                print(f"Higher is better: {metrics_catalog[metric_id]['higher_better']}")
                
                if metric_id == 'inception_score':
                    # Special handling for inception_score
                    os.makedirs(temp_original_dir, exist_ok=True)
                    os.makedirs(temp_upscaled_dir, exist_ok=True)
                    
                    orig_path = os.path.join(temp_original_dir, 'image.png')
                    upsc_path = os.path.join(temp_upscaled_dir, 'image.png')
                    original_image.save(orig_path)
                    upscaled_image.save(upsc_path)
                    
                    metric = pyiqa.create_metric(metric_id).to(device)
                    with torch.no_grad():
                        original_result = metric(temp_original_dir)
                        upscaled_result = metric(temp_upscaled_dir)
                        # Print the result structure to debug
                        print(f"Original result type: {type(original_result)}")
                        print(f"Original result structure: {original_result}")
                        
                        # Extract the inception score mean
                        original_score = original_result['inception_score_mean']
                        upscaled_score = upscaled_result['inception_score_mean']
                        
                        # Convert to float if needed
                        original_score = float(original_score) if hasattr(original_score, 'item') else float(original_score)
                        upscaled_score = float(upscaled_score) if hasattr(upscaled_score, 'item') else float(upscaled_score)
                    
                    # Clean up inception directories
                    shutil.rmtree(temp_original_dir, ignore_errors=True)
                    shutil.rmtree(temp_upscaled_dir, ignore_errors=True)
                else:
                    # Normal metric calculation
                    metric = pyiqa.create_metric(metric_id).to(device)
                    temp_original = f'temp_original_{metric_id}.png'
                    temp_upscaled = f'temp_upscaled_{metric_id}.png'
                    temp_files.extend([temp_original, temp_upscaled])
                    
                    # Apply resizing only for metrics that need it
                    if metric_id in min_size_metrics:
                        temp_orig_img = ensure_min_size(original_image)
                        temp_upsc_img = ensure_min_size(upscaled_image)
                    else:
                        temp_orig_img = original_image
                        temp_upsc_img = upscaled_image
                    
                    temp_orig_img.save(temp_original)
                    temp_upsc_img.save(temp_upscaled)
                    
                    with torch.no_grad():
                        original_score = metric(temp_original).item()
                        upscaled_score = metric(temp_upscaled).item()
                
                print(f"Original Score: {original_score:.4f}")
                print(f"Upscaled Score: {upscaled_score:.4f}")
                print(f"{'='*50}\n")
                
                # Store raw scores without normalization
                results[metric_id] = {
                    'type': 'nr',
                    'category': metric_info['category'],
                    'original': float(original_score),
                    'upscaled': float(upscaled_score),
                    'higher_better': higher_better,
                    'score_range': score_range
                }

            except Exception as e:
                print(f"\nError calculating {metric_id}: {str(e)}")
                print(f"Type: No-Reference")
                print(f"Error: {str(e)}")
                print(f"{'='*50}\n")
                results[metric_id] = {'type': 'nr', 'error': str(e)}

        # Calculate Full-Reference metrics
        for metric_id in metrics_to_calculate['fr']:
            try:
                metric_info = metrics_catalog[metric_id]
                print(f"\n{'='*50}")
                print(f"Calculating metric: {metric_id}")
                print(f"Type: Full-Reference")
                print(f"Expected Range: {metrics_catalog[metric_id]['score_range']}")
                print(f"Higher is better: {metrics_catalog[metric_id]['higher_better']}")

                if metric_id == 'psnr':
                    score = calculate_psnr(original_array, upscaled_array)
                elif metric_id == 'ssim':
                    score = calculate_ssim(original_array, upscaled_array)
                elif metric_id == 'fid':
                    # Special handling for FID metric
                    metric = pyiqa.create_metric(metric_id).to(device)
                    temp_orig_dir = 'temp_original_fid'
                    temp_upsc_dir = 'temp_upscaled_fid'
                    os.makedirs(temp_orig_dir, exist_ok=True)
                    os.makedirs(temp_upsc_dir, exist_ok=True)
                    
                    try:
                        original_image.save(os.path.join(temp_orig_dir, 'image.png'))
                        upscaled_image.save(os.path.join(temp_upsc_dir, 'image.png'))
                        score = metric(temp_orig_dir, temp_upsc_dir).item()
                    finally:
                        shutil.rmtree(temp_orig_dir, ignore_errors=True)
                        shutil.rmtree(temp_upsc_dir, ignore_errors=True)
                else:
                    # Handle other FR metrics
                    metric = pyiqa.create_metric(metric_id).to(device)
                    
                    # Create temporary files
                    temp_orig = f'temp_original_{metric_id}.png'
                    temp_upsc = f'temp_upscaled_{metric_id}.png'
                    temp_files.extend([temp_orig, temp_upsc])
                    
                    # Ensure minimum size and matching dimensions
                    orig_resized = ensure_min_size(original_image)
                    upsc_resized = ensure_min_size(upscaled_image)
                    orig_resized = orig_resized.resize(upsc_resized.size, Image.Resampling.LANCZOS)
                    
                    # Save temporary files
                    orig_resized.save(temp_orig)
                    upsc_resized.save(temp_upsc)
                    
                    try:
                        score = metric(temp_upsc, temp_orig).item()
                    except Exception as e:
                        print(f"Error with metric {metric_id}: {str(e)}")
                        raise

                print(f"Score: {score:.4f}")
                
                results[metric_id] = {
                    'type': 'fr',
                    'score': float(score),
                    'score_range': metric_info['score_range'],
                    'higher_better': metric_info['higher_better']
                }

            except Exception as e:
                print(f"\nError calculating {metric_id}:")
                print(f"Type: Full-Reference")
                print(f"Error: {str(e)}")
                results[metric_id] = {'type': 'fr', 'error': str(e)}

        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error removing temporary file {temp_file}: {str(e)}")

        return jsonify(results)

    except Exception as e:
        # Clean up all temporary files in case of error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        print(f"Error in calculate_metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Update metric calculation functions to handle numpy arrays properly
def calculate_psnr(img1, img2):
    # Ensure same shape
    if img1.shape != img2.shape:
        # Resize img2 to match img1 if necessary
        img2 = np.array(Image.fromarray(img2).resize(
            (img1.shape[1], img1.shape[0]), 
            Image.Resampling.LANCZOS
        ))
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return float(20 * np.log10(PIXEL_MAX / np.sqrt(mse)))

def calculate_ssim(img1, img2):
    from skimage.metrics import structural_similarity as ssim
    
    # Ensure same shape
    if img1.shape != img2.shape:
        img2 = np.array(Image.fromarray(img2).resize(
            (img1.shape[1], img1.shape[0]), 
            Image.Resampling.LANCZOS
        ))
    
    return float(ssim(img1, img2, channel_axis=2))

# Add more metric calculation functions as needed

# Add a new route to check metric loading status
@app.route('/metrics/status', methods=['GET'])
def get_metrics_status():
    return jsonify({
        'loading': LOADING_METRICS,
        'loaded_metrics': LOADED_METRICS
    })

# Add these at the top of the file with other imports
from threading import Lock
LOADING_METRICS = set()  # Set of metrics currently being downloaded
LOADED_METRICS = set()   # Set of metrics that are ready to use
metrics_lock = Lock()

def initialize_metric(metric_id):
    """Initialize a metric and handle its loading state with robust download handling"""
    try:
        with metrics_lock:
            if metric_id not in LOADING_METRICS and metric_id not in LOADED_METRICS:
                LOADING_METRICS.add(metric_id)
        
        # Get the metric weights path and URL from pyiqa
        weights_path = pyiqa.get_metric_weights_path(metric_id)
        weights_url = pyiqa.get_metric_weights_url(metric_id)
        
        # Download weights if needed
        if weights_url and not os.path.exists(weights_path):
            success = download_file_with_retry(
                url=weights_url,
                destination=weights_path,
                max_retries=3
            )
            if not success:
                raise RuntimeError(f"Failed to download weights for metric {metric_id}")

        metric = pyiqa.create_metric(metric_id).to(device)
        
        with metrics_lock:
            LOADING_METRICS.remove(metric_id)
            LOADED_METRICS.add(metric_id)
            
        return metric
    except Exception as e:
        with metrics_lock:
            if metric_id in LOADING_METRICS:
                LOADING_METRICS.remove(metric_id)
        print(f"Error initializing metric {metric_id}: {str(e)}")
        raise

# Add this function near the top
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to import packaging, install if not present
try:
    import packaging
except ImportError:
    install_package('packaging')

# Update the metrics catalog to correctly categorize CKDN
METRIC_TYPES = {
    'ckdn': 'fr',
    'dists': 'fr',
    'lpips': 'fr',
    'musiq': 'nr',
    'nima': 'nr',
    'brisque': 'nr',
    'niqe': 'nr'
    # ... other metrics ...
}

def download_file_with_retry(url, destination, expected_sha256=None, max_retries=3):
    """
    Download a file with retry logic and progress tracking
    """
    # Setup retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # If file exists and hash matches, skip download
    if os.path.exists(destination) and expected_sha256:
        with open(destination, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash == expected_sha256:
                return True

    # Download with progress bar
    try:
        response = session.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

        # Verify hash if provided
        if expected_sha256:
            with open(destination, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash != expected_sha256:
                    os.remove(destination)
                    raise ValueError("Downloaded file hash doesn't match expected hash")

        return True

    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        print(f"Error downloading file: {str(e)}")
        return False

# Add these near the top with other global variables
startup_complete = Event()
startup_status = {
    "ready": False,
    "services": {
        "cuda": False,
        "models_directory": False,
        "temp_directories": False
    },
    "message": "Starting up..."
}

def initialize_backend():
    """Initialize all backend services and track their status."""
    global startup_status
    
    try:
        # Check CUDA availability
        startup_status["message"] = "Checking CUDA availability..."
        cuda_available = torch.cuda.is_available()
        startup_status["services"]["cuda"] = cuda_available
        
        # Initialize directories
        startup_status["message"] = "Setting up directories..."
        try:
            cleanup_temp_directories()
            startup_status["services"]["temp_directories"] = True
        except Exception as e:
            startup_status["message"] = f"Error setting up directories: {str(e)}"
            return False
            
        # Check models directory
        startup_status["message"] = "Checking models directory..."
        config = load_config()
        model_dir = Path(config["modelPath"])
        if not model_dir.exists():
            os.makedirs(model_dir)
        startup_status["services"]["models_directory"] = True
        
        # Mark startup as complete
        startup_status["ready"] = True
        startup_status["message"] = "Backend ready"
        startup_complete.set()
        return True
        
    except Exception as e:
        startup_status["message"] = f"Startup error: {str(e)}"
        return False

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint that returns backend status."""
    return jsonify({
        "status": "ready" if startup_status["ready"] else "starting",
        "services": startup_status["services"],
        "message": startup_status["message"],
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": str(device),
        "version": "1.0.0"  # Add version tracking as needed
    })

# Update the main startup
if __name__ == '__main__':
    # Initialize backend in a separate thread
    import threading
    startup_thread = threading.Thread(target=initialize_backend)
    startup_thread.start()
    
    # Start the Flask app
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)

