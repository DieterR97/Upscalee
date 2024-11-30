# Add this to your existing config structure
DEFAULT_CONFIG = {
    "maxImageDimension": 1024,
    "modelPath": os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
    "selectedMetrics": {
        "nr": [
            "musiq",    # Modern transformer-based
            "nima",     # Classic aesthetic quality
            "brisque", # Traditional baseline
            "niqe",    # Opinion-unaware baseline
            "hyperiqa", # Recent high-performer
            "maniqa"   # Multi-dimensional attention
        ],
        "fr": [
            "psnr",    # Traditional baseline
            "ssim",    # Traditional structural
            "lpips",   # Learned perceptual
            "dists",   # Deep structure/texture
            "fsim",    # Feature similarity
            "vif"      # Information fidelity
        ]
    },
    "metricCategories": {
        "traditional": [
            "psnr", "ssim", "ms_ssim", "fsim", "vif", "brisque", "niqe"
        ],
        "deepLearning": [
            "musiq", "nima", "lpips", "dists", "hyperiqa", "maniqa"
        ]
    }
}