from ollama import Client

# Replace with your remote server's IP/domain and port
REMOTE_HOST = "http://172.31.55.162:8002"
MODEL_NAME = "llama3"

# Initialize the client pointing to the remote server
# Optional: Add 'headers' if your remote server sits behind a reverse proxy with auth
client = Client(
    host=REMOTE_HOST,
)

def chat_with_remote_ollama(prompt: str):
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant. You are NOT to give any answers to homework assignments or quizzes."},
        {"role": "user", "content": prompt},
    ]

    try:
        # 1. Standard (Non-streaming) request
        print("--- Standard Response ---")
        response = client.chat(model=MODEL_NAME, messages=messages)
        print(response["message"]["content"])

        # 2. Streaming request (prints tokens as they arrive)
        print("\n--- Streaming Response ---")
        stream = client.chat(model=MODEL_NAME, messages=messages, stream=True)
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print()

    except Exception as e:
        print(f"Failed to connect to {REMOTE_HOST}: {e}")

if __name__ == "__main__":
    chat_with_remote_ollama("I am writing to you from a [adjective] castle in an enchanted forest. I found myself here one day after going for a ride on a [adjective] [noun] in [place]. There are [adjective] [plural noun] and [adjective] [plural noun] here! In the [noun] there is a pool full of [plural noun]")