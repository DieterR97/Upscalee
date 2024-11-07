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

# Create all necessary directories
for directory in [UPLOAD_FOLDER, TEMP_RESULTS_DIR, FINAL_RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    "RealESRGAN_x4plus": {"description": "General purpose x4 upscaling", "scale": 4, "variable_scale": False},
    "RealESRGAN_x2plus": {"description": "General purpose x2 upscaling", "scale": 2, "variable_scale": False},
    "RealESRNet_x4plus": {"description": "General purpose x4 upscaling with MSE loss (over-smooth effects)", "scale": 4, "variable_scale": False},
    "realesr-general-x4v3": {"description": "General purpose x4 (can also be used for x1, x2, x3) upscaling, a tiny small model (consume much fewer GPU memory and time); not too strong deblur and denoise capacity", "scale": 4, "variable_scale": True},
    "RealESRGAN_x4plus_anime_6B": {"description": "Optimized for anime/artwork/illustrations x4 upscaling", "scale": 4, "variable_scale": False},
    "realesr-animevideov3": {"description": "Optimized for anime video x4 (can also be used for x1, x2, x3) upscaling", "scale": 4, "variable_scale": True}
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
    # strip extension from filename
    filename_no_ext = filename.rsplit(".", 1)[0]

    # Get the selected model and scale from the request
    selected_model = request.form.get("model", "RealESRGAN_x4plus_anime_6B")
    selected_scale = int(request.form.get("scale", AVAILABLE_MODELS[selected_model]["scale"]))
    
    # Initialize the model with the selected type and scale
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
    model_path = Path(MODEL_WEIGHTS_DIR) / f"{model_name}.pth"
    return jsonify({"downloaded": model_path.exists()})

# Get available metrics and their types
def get_metric_info():
    metrics = {}
    for metric_name in pyiqa.list_models():
        metric = pyiqa.create_metric(metric_name)
        metrics[metric_name] = {
            'name': metric_name,
            'type': 'fr' if metric.metric_mode == 'fr' else 'nr',  # 'fr' for full-reference, 'nr' for no-reference
            'score_range': metric.score_range
        }
    return metrics

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
    """
    global download_in_progress
    
    try:
        original_file = request.files['original_image']
        upscaled_file = request.files['upscaled_image']
        metrics = json.loads(request.form['metrics'])
        
        original_path = os.path.join(UPLOAD_FOLDER, secure_filename(original_file.filename))
        upscaled_path = os.path.join(UPLOAD_FOLDER, secure_filename(upscaled_file.filename))
        
        original_file.save(original_path)
        upscaled_file.save(upscaled_path)
        # Load images
        original_img = Image.open(original_file)
        upscaled_img = Image.open(upscaled_file)
        
        # Resize original image to match upscaled dimensions for FR metrics
        original_img_resized = original_img.resize(upscaled_img.size, Image.LANCZOS)
        
        # Save the images
        original_img_resized.save(original_path)
        upscaled_img.save(upscaled_path)

        # Check each metric for downloads
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

        # If we get here, proceed with evaluation
        results = {}
        for metric_name in metrics:
            try:
                metric = get_cached_metric(metric_name)
                metric_info = SUPPORTED_METRICS[metric_name]
                
                if metric_info['type'] == 'fr':
                    score = metric(upscaled_path, original_path).item()
                    results[metric_name] = {
                        'score': score,
                        'type': 'fr',
                        'score_range': metric_info['score_range'],
                        'higher_better': metric_info['higher_better']
                    }
                else:  # No-reference metric
                    original_score = metric(original_path).item()
                    upscaled_score = metric(upscaled_path).item()
                    results[metric_name] = {
                        'original': original_score,
                        'upscaled': upscaled_score,
                        'type': 'nr',
                        'score_range': metric_info['score_range'],
                        'higher_better': metric_info['higher_better']
                    }
            except Exception as e:
                print(f"Error processing metric {metric_name}: {str(e)}")
                results[metric_name] = {
                    'error': str(e),
                    'type': metric_info['type'],
                    'score_range': metric_info['score_range'],
                    'higher_better': metric_info['higher_better']
                }
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in evaluate_quality: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        # Check if the model file still exists
        if model_name not in existing_model_files:
            # Model is registered but file is missing
            models_to_remove.append(model_name)
            continue
    
    # Remove models that no longer exist
    for model_name in models_to_remove:
        del registered_models[model_name]
    
    # If any models were removed, save the updated registered models
    if models_to_remove:
        save_registered_models(registered_models)
    
    # Find unregistered models
    for model_name, model_file in existing_model_files.items():
        if model_name not in registered_models:
            # Try to determine scale from filename
            scale_match = re.search(r'x(\d+)', model_name.lower())
            scale = int(scale_match.group(1)) if scale_match else None
            
            unregistered_models.append({
                "name": model_name,
                "file_pattern": model_file.name,
                "scale": scale,
                "path": str(model_file)
            })
    
    return {
        "registered": registered_models,
        "unregistered": unregistered_models,
        "removed": models_to_remove  # Include information about removed models
    }

@app.route("/register-model", methods=["POST"])
def register_model():
    """Register a new model with provided information."""
    try:
        model_info = request.json
        registered_models = load_registered_models()
        
        # Add the new model to registered models
        registered_models[model_info["name"]] = {
            "name": model_info["display_name"],
            "description": model_info["description"],
            "scale": model_info["scale"],
            "variable_scale": model_info["variable_scale"],
            "architecture": model_info["architecture"],
            "file_pattern": model_info["file_pattern"]
        }
        
        save_registered_models(registered_models)
        return jsonify({"success": True, "message": "Model registered successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000, extra_files=[
        # Only watch your application files
        './app.py',
        './model.py'
    ])

