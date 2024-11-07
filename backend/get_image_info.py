import os
from PIL import Image
from PIL.ExifTags import TAGS
import struct
import imghdr
import math

def get_image_info(image):
    """
    Get detailed information about an image.
    
    :param image: PIL Image object
    :return: Dictionary containing image information including:
            - Basic format and mode info
            - Dimensions and aspect ratio
            - Color information and channels
            - File and compression details
            - EXIF metadata (if available)
            - Statistical analysis of color channels
    """
    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    common_ratios = {
        1: "1:1 (Square)",
        1.33: "4:3 (Standard)",
        1.78: "16:9 (Widescreen)",
        1.5: "3:2 (Classic Photo)",
        1.6: "16:10 (Wide)",
        2.39: "2.39:1 (Anamorphic)",
    }
    
    # Find closest common ratio
    closest_ratio = min(common_ratios.keys(), key=lambda x: abs(x - aspect_ratio))
    ratio_name = common_ratios[closest_ratio] if abs(closest_ratio - aspect_ratio) < 0.1 else f"{aspect_ratio:.2f}:1"

    info = {
        "basic_info": {
            "format": image.format,
            "mode": image.mode,
            "color_space": get_color_space_info(image.mode),
            "format_description": get_format_description(image.format),
            "animation": getattr(image, "is_animated", False),
            "n_frames": getattr(image, "n_frames", 1) if getattr(image, "is_animated", False) else 1,
        },
        "dimensions": {
            "width": image.width,
            "height": image.height,
            "megapixels": (image.width * image.height) / 1_000_000,
            "aspect_ratio": ratio_name,
            "orientation": get_orientation(image),
            "resolution_category": get_resolution_category(image.width, image.height),
        },
        "color_info": {
            "color_depth": get_color_depth(image.mode),
            "bits_per_pixel": get_bits_per_pixel(image.mode),
            "channels": len(image.getbands()),
            "channel_names": image.getbands(),
            "has_alpha": image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info),
            "palette_size": len(image.getcolors(maxcolors=256)) if image.mode == 'P' else None,
            "transparency": "transparency" in image.info,
            "background": image.info.get("background", None),
        },
        "file_info": {
            "estimated_memory_mb": (image.width * image.height * len(image.getbands()) * 8) / (8 * 1024 * 1024),
            "compression": image.info.get("compression", "None"),
            "compression_details": get_compression_info(image),
            "dpi": image.info.get("dpi", "Not specified"),
            "format_details": get_format_specific_info(image),
        },
        "exif": {},
        "statistics": get_image_statistics(image),
    }

    # Get EXIF data with more organized categories
    try:
        exif_data = image.getexif()
        if exif_data:
            exif_categories = {
                "camera_info": ["Make", "Model", "Software"],
                "capture_settings": ["ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"],
                "image_settings": ["ColorSpace", "WhiteBalance", "Contrast", "Saturation", "Sharpness"],
                "datetime": ["DateTime", "DateTimeOriginal", "DateTimeDigitized"],
                "other": []
            }
            
            categorized_exif = {cat: {} for cat in exif_categories.keys()}
            
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, str(tag_id))
                tag_value = str(value)
                
                # Categorize EXIF data
                categorized = False
                for category, tags in exif_categories.items():
                    if tag in tags:
                        categorized_exif[category][tag] = tag_value
                        categorized = True
                        break
                
                if not categorized:
                    categorized_exif["other"][tag] = tag_value
            
            info["exif"] = {k: v for k, v in categorized_exif.items() if v}
    except Exception as e:
        info["exif_error"] = str(e)

    return info

def get_color_space_info(mode):
    """
    Get detailed information about the color space.
    Translates PIL image modes into human-readable color space descriptions.
    """
    color_spaces = {
        '1': "Binary (Black and White)",
        'L': "Grayscale",
        'LA': "Grayscale with Alpha",
        'P': "8-bit Palette",
        'RGB': "RGB Color",
        'RGBA': "RGB with Alpha",
        'CMYK': "CMYK Color",
        'YCbCr': "YCbCr Color",
        'LAB': "LAB Color",
        'HSV': "HSV Color",
    }
    return color_spaces.get(mode, f"Unknown ({mode})")

def get_color_depth(mode):
    """Get the color depth in bits per pixel."""
    color_depths = {
        '1': 1,
        'L': 8,
        'P': 8,
        'RGB': 24,
        'RGBA': 32,
        'CMYK': 32,
        'YCbCr': 24,
        'LAB': 24,
        'HSV': 24,
        'LA': 16,
    }
    return color_depths.get(mode, None)

def get_format_description(format):
    """Get detailed description of image format."""
    formats = {
        'JPEG': "Joint Photographic Experts Group - Lossy compression format best for photographs",
        'PNG': "Portable Network Graphics - Lossless compression with alpha channel support",
        'GIF': "Graphics Interchange Format - Limited to 256 colors, supports animation",
        'BMP': "Bitmap - Uncompressed raster format",
        'TIFF': "Tagged Image File Format - Flexible format supporting multiple compression types",
        'WEBP': "Web Picture format - Modern format supporting both lossy and lossless compression",
    }
    return formats.get(format, f"Unknown format: {format}")

def get_orientation(image):
    """Determine image orientation."""
    w, h = image.size
    if w == h:
        return "Square"
    elif w > h:
        return "Landscape"
    else:
        return "Portrait"

def get_resolution_category(width, height):
    """
    Categorize image resolution based on total pixel count.
    Returns the highest matching category (4K, Full HD, HD, SD, or Low Resolution).
    """
    pixels = width * height
    categories = [
        (8294400, "4K (3840×2160 or higher)"),
        (2073600, "Full HD (1920×1080)"),
        (921600, "HD (1280×720)"),
        (307200, "SD (640×480)"),
        (0, "Low Resolution")
    ]
    for threshold, category in categories:
        if pixels >= threshold:
            return category

def get_bits_per_pixel(mode):
    """Calculate bits per pixel."""
    mode_bits = {
        '1': 1,
        'L': 8,
        'P': 8,
        'RGB': 24,
        'RGBA': 32,
        'CMYK': 32,
        'YCbCr': 24,
        'LAB': 24,
        'HSV': 24,
        'LA': 16,
    }
    return mode_bits.get(mode, 0)

def get_compression_info(image):
    """Get detailed compression information."""
    if image.format == 'JPEG':
        return "DCT-based lossy compression"
    elif image.format == 'PNG':
        return "DEFLATE lossless compression"
    elif image.format == 'TIFF':
        return f"Compression: {image.info.get('compression', 'Unknown')}"
    return "Unknown compression"

def get_format_specific_info(image):
    """
    Get format-specific details that only apply to certain image types.
    JPEG: subsampling and progressive encoding info
    PNG: interlace and optimization settings
    """
    info = {}
    if image.format == 'JPEG':
        info['subsampling'] = image.info.get('subsampling', 'Unknown')
        info['progressive'] = image.info.get('progressive', False)
    elif image.format == 'PNG':
        info['interlace'] = image.info.get('interlace', False)
        info['optimization'] = image.info.get('optimize', False)
    return info

def get_image_statistics(image):
    """
    Calculate basic image statistics for each color channel.
    For RGB images: calculates min, max, and mean values for each channel.
    For Grayscale images: calculates min, max, and mean values.
    Returns empty dict if calculations fail or for unsupported modes.
    """
    try:
        if image.mode == 'RGB':
            r, g, b = image.split()
            return {
                "red_channel": {
                    "min": min(r.getdata()),
                    "max": max(r.getdata()),
                    "mean": sum(r.getdata()) / (image.width * image.height)
                },
                "green_channel": {
                    "min": min(g.getdata()),
                    "max": max(g.getdata()),
                    "mean": sum(g.getdata()) / (image.width * image.height)
                },
                "blue_channel": {
                    "min": min(b.getdata()),
                    "max": max(b.getdata()),
                    "mean": sum(b.getdata()) / (image.width * image.height)
                }
            }
        elif image.mode == 'L':
            data = list(image.getdata())
            return {
                "grayscale": {
                    "min": min(data),
                    "max": max(data),
                    "mean": sum(data) / (image.width * image.height)
                }
            }
        return {}
    except:
        return {}
