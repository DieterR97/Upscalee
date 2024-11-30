# Add this to your existing config structure
DEFAULT_CONFIG = {
    "maxImageDimension": 1024,
    "modelPath": os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
    "selectedMetrics": {
        "nr": [
            "brisque", # Traditional baseline
            "hyperiqa", # Recent high-performer
            "maniqa",   # Multi-dimensional attention
            "musiq",    # Modern transformer-based
            "nima",     # Classic aesthetic quality
            "niqe"     # Opinion-unaware baseline
        ],
        "fr": [
            "dists",   # Deep structure/texture
            "fsim",    # Feature similarity
            "lpips",   # Learned perceptual
            "psnr",    # Traditional baseline
            "ssim",    # Traditional structural
            "vif"      # Information fidelity
        ]
    },
    "metricCategories": {
        "traditional": [
            "brisque", "fsim", "ms_ssim", "niqe", "psnr", "ssim", "vif"
        ],
        "deepLearning": [
            "dists", "hyperiqa", "lpips", "maniqa", "musiq", "nima"
        ]
    }
}