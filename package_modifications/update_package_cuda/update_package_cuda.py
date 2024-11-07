import shutil
from pathlib import Path

def update_package_files():
    # Define source and destination paths
    source_files = {
        'utils.py': Path('package_modifications/update_package_cuda/utils.py')
    }
    
    dest_paths = {
        'utils.py': Path('venv/Lib/site-packages/realesrgan/utils.py')
    }
    
    # Copy each file
    for filename, source in source_files.items():
        dest = dest_paths[filename]
        try:
            print(f"Copying {filename}...")
            # Create parent directories if they don't exist
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Copy the file, overwriting if it exists
            shutil.copy2(source, dest)
            print(f"Successfully copied {filename}")
        except Exception as e:
            print(f"Error copying {filename}: {e}")

if __name__ == "__main__":
    print("Starting package update for CUDA...")
    update_package_files()
    print("Package update for CUDA complete!")
