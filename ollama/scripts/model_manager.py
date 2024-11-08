import subprocess
import json
import sys

def run_command(command):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            errors='replace'
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error running command: {e}")
        return None

def list_models():
    """List all installed models with details"""
    models = run_command("ollama list")
    print("\nInstalled Models:")
    print(models if models else "No models found")

def pull_recommended_models():
    """Pull recommended models for different use cases"""
    models = {
        "llama3.2-vision": "Vision and image analysis",
        "dolphin-llama3": "General chat and queries",
        "gemma2": "Fast and efficient processing",
        "mistral": "All-around performance",
        "llava": "Alternative vision model"
    }
    
    for model, description in models.items():
        print(f"\nPulling {model} ({description})...")
        result = run_command(f"ollama pull {model}")
        print(result if result else f"Failed to pull {model}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python model_manager.py list          - List installed models")
        print("  python model_manager.py pull          - Pull recommended models")
        return

    command = sys.argv[1]
    if command == "list":
        list_models()
    elif command == "pull":
        pull_recommended_models()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
