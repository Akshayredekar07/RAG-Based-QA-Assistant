import subprocess
import sys
import os

def run_streamlit():
    """Run the Streamlit app"""
    try:
        if not os.path.exists('.env'):
            print(".env file not found!")
            print("Please create a .env file with your GOOGLE_API_KEY")
            return

        subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend.py"])
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting Gemini RAG Chatbot...")
    print("Make sure you have set your GOOGLE_API_KEY in the .env file")
    run_streamlit()