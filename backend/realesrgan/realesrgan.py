class RealESRGANer:
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        with torch.no_grad():
            # ... existing setup code ...

            # Calculate total tiles
            self.total_tiles = ((h_pad - 1) // self.tile + 1) * ((w_pad - 1) // self.tile + 1)
            self.tiles_processed = 0

            # Process tiles
            for h_idx in range(0, h_pad - self.tile + 1, self.tile):
                for w_idx in range(0, w_pad - self.tile + 1, self.tile):
                    # ... existing tile processing code ...
                    
                    # Update progress
                    self.tiles_processed += 1
                    progress = (self.tiles_processed / self.total_tiles) * 100
                    print(f"PROGRESS:{progress:.2f}")

            # ... rest of the enhancement code ...

            return output, img_mode 