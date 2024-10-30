import argparse
import cv2
import glob
import os
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def load_model(args):
    """Load the appropriate model based on the model name provided."""
    model, netscale, file_url = None, None, None

    model_mappings = {
        "RealESRGAN_x4plus": {
            "model": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            ),
            "netscale": 4,
            "file_url": [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            ],
        },
        "RealESRNet_x4plus": {
            "model": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            ),
            "netscale": 4,
            "file_url": [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
            ],
        },
        "RealESRGAN_x4plus_anime_6B": {
            "model": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            ),
            "netscale": 4,
            "file_url": [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            ],
        },
        "RealESRGAN_x2plus": {
            "model": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            ),
            "netscale": 2,
            "file_url": [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            ],
        },
        "realesr-animevideov3": {
            "model": SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=4,
                act_type="prelu",
            ),
            "netscale": 4,
            "file_url": [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
            ],
        },
        "realesr-general-x4v3": {
            "model": SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            ),
            "netscale": 4,
            "file_url": [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            ],
        },
    }

    if args.model_name in model_mappings:
        model_info = model_mappings[args.model_name]
        model, netscale, file_url = (
            model_info["model"],
            model_info["netscale"],
            model_info["file_url"],
        )

    return model, netscale, file_url


def download_model_weights(file_url, model_path):
    """Download model weights if they do not exist."""
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    for url in file_url:
        model_path = load_file_from_url(
            url=url,
            model_dir=os.path.join(ROOT_DIR, "weights"),
            progress=True,
            file_name=None,
        )
    return model_path


def process_image(args, image, upsampler, face_enhancer=None):
    """Process a single image for upscaling."""

    # Convert PIL Image to NumPy array
    img = np.array(image)

    # Determine image mode
    img_mode = "RGBA" if img.ndim == 3 and img.shape[2] == 4 else None

    try:
        if args["face_enhance"]:
            # Call the face enhancer if enabled
            _, _, output = face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            # Upscale the image using the upsampler
            output, _ = upsampler.enhance(img, outscale=args["outscale"])
    except RuntimeError as error:
        print(f"Error processing image: {error}")
        return None  # Return None in case of an error

    # Determine the output file extension
    extension = "png" if img_mode == "RGBA" else args["ext"]
    suffix = f"_{args['suffix']}" if args["suffix"] else ""

    # Construct the save path (You might want to adapt this depending on your needs)
    save_path = os.path.join(args["output"], f"upscaled{suffix}.{extension}")
    print("save_path: " + save_path)

    # Save the output image using OpenCV
    cv2.imwrite(save_path, output)

    return save_path  # Optionally return the save path


def main():
    """Main function for the Real-ESRGAN inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="inputs", help="Input image or folder"
    )
    parser.add_argument(
        "-n", "--model_name", type=str, default="RealESRGAN_x4plus", help="Model name"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="results", help="Output folder"
    )
    parser.add_argument(
        "-dn", "--denoise_strength", type=float, default=0.5, help="Denoise strength"
    )
    parser.add_argument(
        "-s", "--outscale", type=float, default=4, help="Upsampling scale"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Custom model path"
    )
    parser.add_argument("--suffix", type=str, default="out", help="Output suffix")
    parser.add_argument("-t", "--tile", type=int, default=0, help="Tile size")
    parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding")
    parser.add_argument("--pre_pad", type=int, default=0, help="Pre-padding size")
    parser.add_argument(
        "--face_enhance", action="store_true", help="Enhance faces using GFPGAN"
    )
    parser.add_argument(
        "--fp32", action="store_true", help="Use fp32 precision during inference"
    )
    parser.add_argument(
        "--alpha_upsampler",
        type=str,
        default="realesrgan",
        help="Alpha channel upsampler",
    )
    parser.add_argument(
        "--ext", type=str, default="auto", help="Image extension (auto, jpg, png)"
    )
    parser.add_argument("-g", "--gpu-id", type=int, default=None, help="GPU device ID")

    args = parser.parse_args()

    # Load model and get file URL
    model, netscale, file_url = load_model(args)

    # Set model path
    model_path = args.model_path or os.path.join("weights", args.model_name + ".pth")
    if not os.path.isfile(model_path):
        model_path = download_model_weights(file_url, model_path)

    # Adjust for denoise strength if using general-x4v3
    dni_weight = None
    if args.model_name == "realesr-general-x4v3" and args.denoise_strength != 1:
        wdn_model_path = model_path.replace(
            "realesr-general-x4v3", "realesr-general-wdn-x4v3"
        )
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # Create upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id,
    )

    # Optional face enhancer
    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer

        face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=args.outscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )

    # Process input paths
    os.makedirs(args.output, exist_ok=True)
    paths = (
        [args.input]
        if os.path.isfile(args.input)
        else sorted(glob.glob(os.path.join(args.input, "*")))
    )

    # Process images
    for idx, path in enumerate(paths):
        print(f"Processing image {idx}: {path}")
        process_image(args, path, upsampler, face_enhancer)


if __name__ == "__main__":
    main()
