import torch
import numpy as np
from PIL import Image
from pathlib import Path
from spandrel import ModelLoader, ImageModelDescriptor
import cv2
import os

class SpandrelUpscaler:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        scale: int = 4,
        gpu_id: int = 0,
    ):
        """Initialize the model using Spandrel's ModelLoader."""
        self.model_name = model_name
        self.model_path = Path(model_path) / f"{model_name}.pth"
        self.scale = scale
        self.gpu_id = gpu_id

        # Initialize model loader
        self.loader = ModelLoader()
        
        # Load the model
        self.model_descriptor = self.loader.load_from_file(str(self.model_path))
        
        # Verify it's an image model
        if not isinstance(self.model_descriptor, ImageModelDescriptor):
            raise ValueError("Loaded model is not an image model")
            
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model_descriptor.cuda()
        
        # Set to evaluation mode
        self.model_descriptor.eval()

    def upscale(self, image, args):
        """Upscale the given image using the loaded model."""
        try:
            # Convert PIL Image to tensor
            input_tensor = self._prepare_input(image)
            
            # Process image
            with torch.no_grad():
                output_tensor = self.model_descriptor(input_tensor)
            
            # Convert tensor back to PIL Image
            output_image = self._tensor_to_pil(output_tensor)
            
            # Save the processed image
            save_path = os.path.join(
                args["output"],
                f"{args['suffix']}.{args['ext']}"
            )
            
            # Save the PIL image directly
            output_image.save(save_path)
            
            # Convert to RGB format for consistency
            image = cv2.imread(save_path)
            if image is None:
                raise FileNotFoundError(f"Failed to load processed image at {save_path}")
            # image_swapped = image[..., ::-1]
            # Convert BGR to RGB
            image_swapped = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Save the color-corrected image
            filename = os.path.basename(save_path)
            final_results_dir = "pre_swapped_channels_results"
            swapped_image_path = os.path.join(final_results_dir, filename)
            # cv2.imwrite(swapped_image_path, image_swapped)
            cv2.imwrite(swapped_image_path, cv2.cvtColor(image_swapped, cv2.COLOR_RGB2BGR))
            
            return swapped_image_path
            
        except Exception as e:
            print(f"Error during upscaling: {str(e)}")
            raise

    def _prepare_input(self, pil_image):
        """Convert PIL image to tensor in the correct format."""
        # Convert to RGB if image has alpha channel
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL image to tensor (assuming RGB input)
        tensor = torch.from_numpy(np.array(pil_image)).float()
        
        # Rearrange dimensions from HWC to BCHW and normalize
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Move to same device as model
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        return tensor

    def _tensor_to_pil(self, tensor):
        """Convert output tensor back to PIL Image."""
        # Denormalize and convert to uint8
        array = (tensor.squeeze(0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def get_model_info(model_path: str) -> dict:
        """Get model information using Spandrel."""
        try:
            loader = ModelLoader()
            model = loader.load_from_file(model_path)
            
            if not isinstance(model, ImageModelDescriptor):
                raise ValueError("Not an image model")
            
            # Get the actual model architecture name from the model class
            architecture = model.model.__class__.__name__
            
            return {
                "architecture": architecture,  # This will now be the actual architecture name
                "is_supported": True,
                "input_channels": model.input_channels,
                "output_channels": model.output_channels,
                "supports_half": model.supports_half,
                "supports_bfloat16": model.supports_bfloat16,
                "size_requirements": model.size_requirements.__dict__ if model.size_requirements else None,
                "tiling": str(model.tiling),
                "scale": model.scale
            }
        except Exception as e:
            # If loading fails, it likely means the architecture is not supported
            return {
                "is_supported": False,
                "error": str(e)
            }
