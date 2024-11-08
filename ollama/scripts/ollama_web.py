from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat

def initialize_model():
    """Initialize the model with fallbacks"""
    models_to_try = [
        ("llama3.2-vision", "Best for image analysis"),
        ("dolphin-llama3", "Best for general chat"),
        ("gemma2", "Fast and efficient"),
        ("mistral", "Good all-around model"),
        ("llava", "Alternative vision model")
    ]
    
    for model_name, description in models_to_try:
        try:
            print(f"Trying to initialize {model_name} ({description})...")
            llm = OllamaChat(model=model_name)
            print(f"Successfully initialized {model_name}")
            return llm
        except Exception as e:
            print(f"Failed to initialize {model_name}: {str(e)}")
    
    raise Exception("No available models found. Please install a model using 'ollama pull model_name'")

def search_web():
    # Initialize the model and agent
    try:
        llm = initialize_model()
        online_agent = OnlineAgent(llm)
        print("\nModel initialized successfully! Available commands:")
        print("- Regular search: Just type your question")
        print("- Image analysis: Include an image path in your query")
        print("- URL analysis: Include a URL to analyze its content")
        print("- Type 'quit' to exit")
    except Exception as e:
        print(f"Error initializing: {str(e)}")
        return

    while True:
        # Get user input
        print("\nEnter your query (or 'quit' to exit):")
        print("For general search: What is quantum computing?")
        print("For URL analysis: https://example.com")
        print("For specific URL questions: What are the main points in https://example.com")
        prompt = input("> ")
        
        if prompt.lower() == 'quit':
            break
            
        try:
            # Extract URL if present in the prompt
            words = prompt.split()
            urls = [word for word in words if word.startswith(("http://", "https://"))]
            
            if urls:
                url = urls[0]  # Take the first URL found
                # Remove the URL from the prompt to get the question
                question = prompt.replace(url, "").strip()
                
                print("\nAnalyzing URL:", url)
                if question:
                    print("With specific question:", question)
                    # First get the content from URL
                    content = online_agent.search_url(url)
                    # Then ask the specific question about the content
                    response = online_agent.ask(question, context=content)
                else:
                    # If no specific question, just analyze the URL
                    response = online_agent.search_url(url)
            else:
                print("\nSearching the web...")
                response = online_agent.search(prompt)
                
            print("\nResponse:", response)
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    print("Web Search Tool (using llm-axe)")
    print("--------------------------------")
    search_web()