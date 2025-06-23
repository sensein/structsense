## 🔧 Setting Up

1. **Install Docker**  
   If Docker is not installed on your system, [download and install it](https://docs.docker.com/get-docker/).

2. **Start the Container**  
   From the project directory, run:
   ```bash
   docker compose up
3. **Test the Setup**
    Use the following curl command to verify that the model is responding correctly:
    ```bash
    curl -s http://localhost:11434/api/chat \
      -H "Content-Type: application/json" \
      -d '{
        "model": "deepseek-r1:7b",
        "messages": [
          { "role": "user", "content": "Why is the sky blue?" }
        ]
      }' | jq -r 'select(.message.role=="assistant") | .message.content' | tr -d "\n"
    
    ```