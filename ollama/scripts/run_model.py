import subprocess
import sys

CONFIGURATIONS = {
    "chat": {
        "model": "dolphin-llama3",
        "description": "General chat mode"
    },
    "vision": {
        "model": "llama3.2-vision",
        "description": "Image analysis mode"
    },
    "code": {
        "model": "mistral",
        "description": "Code assistance mode"
    },
    "fast": {
        "model": "gemma2",
        "description": "Fast response mode"
    }
}

def print_usage():
    print("\nUsage: python run_model.py <mode>")
    print("\nAvailable modes:")
    for mode, config in CONFIGURATIONS.items():
        print(f"  {mode:<10} - {config['description']} (using {config['model']})")

def run_model(mode):
    if mode not in CONFIGURATIONS:
        print(f"Unknown mode: {mode}")
        print_usage()
        return
    
    config = CONFIGURATIONS[mode]
    print(f"\nStarting {config['description']} with {config['model']}...")
    
    try:
        subprocess.run(f"ollama run {config['model']}", shell=True)
    except KeyboardInterrupt:
        print("\nStopping model...")
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ['-h', '--help']:
        print_usage()
        return

    run_model(sys.argv[1])

if __name__ == "__main__":
    main()
