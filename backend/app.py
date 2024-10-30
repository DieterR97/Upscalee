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

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})  # Enable CORS for all routes

# Define all directory constants
UPLOAD_FOLDER = 'temp_uploads'
TEMP_RESULTS_DIR = 'pre_swapped_channels_results'
FINAL_RESULTS_DIR = 'final_results'

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

# Add this near the top of the file
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
    return jsonify(AVAILABLE_MODELS)

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

# Define supported metrics with their metadata
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

# Create a temporary environment context manager
@contextmanager
def temporary_torch_home():
    original_torch_home = os.environ.get('TORCH_HOME')
    try:
        os.environ['TORCH_HOME'] = METRIC_WEIGHTS_DIR
        yield
    finally:
        if original_torch_home:
            os.environ['TORCH_HOME'] = original_torch_home
        else:
            os.environ.pop('TORCH_HOME', None)

# Add this cache for loaded metrics
@lru_cache(maxsize=None)
def get_cached_metric(metric_name):
    """Cache the loaded metric to prevent reloading."""
    return pyiqa.create_metric(metric_name, device=device)

@app.route('/evaluate-quality', methods=['POST'])
def evaluate_quality():
    try:
        # Get files from request
        original_file = request.files['original_image']
        upscaled_file = request.files['upscaled_image']
        metrics = json.loads(request.form['metrics'])
        
        # Save files temporarily
        original_path = os.path.join(UPLOAD_FOLDER, secure_filename(original_file.filename))
        upscaled_path = os.path.join(UPLOAD_FOLDER, secure_filename(upscaled_file.filename))
        
        original_file.save(original_path)
        upscaled_file.save(upscaled_path)
        
        # Resize original to match upscaled dimensions
        with Image.open(upscaled_path) as upscaled_img:
            upscaled_size = upscaled_img.size
            
        with Image.open(original_path) as original_img:
            original_img = original_img.resize(upscaled_size, Image.Resampling.LANCZOS)
            resized_original_path = os.path.join(UPLOAD_FOLDER, 'resized_' + secure_filename(original_file.filename))
            original_img.save(resized_original_path)
        
        results = {}
        downloading_metrics = []
        
        try:
            # First attempt to create all metrics to trigger downloads
            for metric_name in metrics:
                try:
                    with temporary_torch_home():
                        # This will trigger the download if needed
                        metric = get_cached_metric(metric_name)
                except Exception as e:
                    print(f"Error loading metric {metric_name}: {str(e)}")
                    downloading_metrics.append(metric_name)
            
            # If we're still downloading, inform the client
            if downloading_metrics:
                return jsonify({
                    'downloading': {
                        'status': True,
                        'metrics': downloading_metrics,
                        'message': f'Downloading weights for {", ".join(downloading_metrics)}...'
                    }
                })
            
            # Process all metrics
            for metric_name in metrics:
                with temporary_torch_home():
                    metric = get_cached_metric(metric_name)
                    metric_info = SUPPORTED_METRICS[metric_name]
                    
                    if metric_info['type'] == 'fr':
                        score = metric(upscaled_path, resized_original_path).item()
                        results[metric_name] = {
                            'score': score,
                            'type': 'fr',
                            'score_range': metric_info['score_range'],
                            'higher_better': metric_info['higher_better']
                        }
                    else:  # No-reference metric
                        original_score = metric(resized_original_path).item()
                        upscaled_score = metric(upscaled_path).item()
                        results[metric_name] = {
                            'original': original_score,
                            'upscaled': upscaled_score,
                            'type': 'nr',
                            'score_range': metric_info['score_range'],
                            'higher_better': metric_info['higher_better']
                        }
            
            # Add downloading status to results
            results['downloading'] = {
                'status': False,
                'metrics': []
            }
            
        except Exception as e:
            print(f"Error processing metrics: {str(e)}")
            results['error'] = str(e)
        
        finally:
            # Clean up temporary files
            for path in [original_path, upscaled_path, resized_original_path]:
                if os.path.exists(path):
                    os.remove(path)
        
        return jsonify(results)
        
    except Exception as e:
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

if __name__ == "__main__":
    app.run(debug=True, port=5000, extra_files=[
        # Only watch your application files
        './app.py',
        './model.py'
    ])
