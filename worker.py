import asyncio
import json
import logging
import argparse
import uuid
import sys
import time
from typing import List, Dict, Any, Optional
import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KubliWorker")

class KubliWorker:
    """
    Kubli Compute Worker Node.
    Connects to the Kubli server via WebSocket and processes local AI tasks (prompts and document context) using Ollama.
    """
    def __init__(self, server_url: str, worker_id: str, model: str, ollama_url: str):
        self.server_url = server_url.rstrip("/")
        self.worker_id = worker_id
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.is_busy = False
        self.supported_models = [self.model]

    async def ensure_ollama_model(self, session: aiohttp.ClientSession) -> bool:
        """
        Checks if the designated model exists in local Ollama storage without pulling or downloading.
        """
        try:
            async with session.get(f"{self.ollama_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    installed_models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                    
                    installed_names_and_short = []
                    for name in installed_models:
                        installed_names_and_short.append(name)
                        if ":" in name:
                            installed_names_and_short.append(name.split(":")[0])
                            
                    return self.model in installed_names_and_short
        except Exception as e:
            logger.error(f"Failed to check Ollama models: {e}")
        return False

    async def execute_ai_task(
        self, 
        model: str, 
        prompt: str, 
        messages: list, 
        documents: list,
        images: list,
        parameters: dict, 
        chunk_callback=None
    ) -> str:
        """
        Executes local AI generation using Ollama streaming /api/chat endpoint with memory and document context support.
        """
        logger.info(f"Executing task on model '{model}' with {len(messages)} history messages, {len(documents)} documents, and {len(images)} images")

        formatted_messages = list(messages) if messages else []
        
        # Build prompt content combining documents and prompt text
        doc_context_str = ""
        if documents:
            doc_context_str += "\n\n--- ATTACHED DOCUMENTS & CONTEXT ---\n"
            for doc in documents:
                doc_name = doc.get("name", "Document")
                doc_content = doc.get("content", "")
                doc_context_str += f"\n[File: {doc_name}]\n{doc_content}\n"
            doc_context_str += "--- END OF DOCUMENTS ---\n\n"

        full_prompt_text = f"{doc_context_str}{prompt}".strip()

        if not formatted_messages:
            user_msg = {"role": "user", "content": full_prompt_text}
            if images:
                user_msg["images"] = images
            formatted_messages.append(user_msg)
        else:
            # Append document context to the last user message if present
            if formatted_messages[-1].get("role") == "user":
                last_user_content = formatted_messages[-1].get("content", "")
                formatted_messages[-1]["content"] = f"{doc_context_str}{last_user_content}".strip()
                if images:
                    formatted_messages[-1]["images"] = images

        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": True,
            "options": parameters.get("options", {})
        }

        full_response = []
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.ollama_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    raise RuntimeError(f"Ollama error (HTTP {resp.status}): {err_text}")

                async for line in resp.content:
                    line_str = line.decode('utf-8').strip()
                    if not line_str:
                        continue
                    try:
                        chunk_json = json.loads(line_str)
                        delta = chunk_json.get("message", {}).get("content", "")
                        if delta:
                            full_response.append(delta)
                            if chunk_callback:
                                await chunk_callback(delta)
                    except json.JSONDecodeError:
                        continue

                result_text = "".join(full_response)
                if not result_text:
                    raise RuntimeError("Ollama returned an empty response.")

                return result_text

    async def start(self):
        ws_url = f"{self.server_url}/ws/worker"
        
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    has_model = await self.ensure_ollama_model(session)
                    if not has_model:
                        logger.warning(f"Worker missing required local model '{self.model}'. Retrying check in 10 seconds...")
                        await asyncio.sleep(10)
                        continue

                    logger.info(f"Connecting to Kubli Server at {ws_url}...")
                    async with session.ws_connect(ws_url) as ws:
                        logger.info("Connected to Kubli network! Registering capabilities...")

                        reg_payload = {
                            "type": "register",
                            "worker_id": self.worker_id,
                            "models": self.supported_models
                        }
                        await ws.send_json(reg_payload)

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                msg_type = data.get("type")

                                if msg_type == "register_ack":
                                    logger.info(f"Registered! Worker active for model '{self.model}'. Awaiting tasks.")

                                elif msg_type == "assign_task":
                                    task_id = data.get("task_id")
                                    model = data.get("model")
                                    prompt = data.get("prompt", "")
                                    messages = data.get("messages", [])
                                    documents = data.get("documents", [])
                                    images = data.get("images", [])
                                    parameters = data.get("parameters", {})

                                    logger.info(f"Received task assignment: {task_id} ({model})")
                                    self.is_busy = True

                                    async def stream_chunk_handler(chunk_text):
                                        await ws.send_json({
                                            "type": "task_chunk",
                                            "task_id": task_id,
                                            "chunk": chunk_text
                                        })

                                    try:
                                        result = await self.execute_ai_task(
                                            model=model,
                                            prompt=prompt,
                                            messages=messages,
                                            documents=documents,
                                            images=images,
                                            parameters=parameters,
                                            chunk_callback=stream_chunk_handler
                                        )
                                        
                                        await ws.send_json({
                                            "type": "task_complete",
                                            "task_id": task_id,
                                            "result": result,
                                            "error": None
                                        })
                                        logger.info(f"Completed task {task_id} and returned output.")

                                    except Exception as e:
                                        logger.error(f"Error executing task {task_id}: {e}")
                                        await ws.send_json({
                                            "type": "task_complete",
                                            "task_id": task_id,
                                            "result": None,
                                            "error": str(e)
                                        })
                                    finally:
                                        self.is_busy = False

                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                logger.warning("WebSocket connection closed or error encountered.")
                                break

            except aiohttp.ClientError as e:
                logger.error(f"Network error: {e}. Retrying in 5 seconds...")
            except Exception as e:
                logger.error(f"Unexpected worker exception: {e}. Retrying in 5 seconds...")

            await asyncio.sleep(5)

async def fetch_ollama_models(ollama_url: str) -> List[str]:
    """Queries Ollama REST API to retrieve all locally installed models."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ollama_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception as e:
        logger.error(f"Failed to fetch installed models from Ollama at {ollama_url}: {e}")
    return []

def select_model_interactively(installed_models: List[str]) -> str:
    """Presents a CLI menu for the user to select one of their installed Ollama models."""
    if not installed_models:
        print("\n[ERROR] No installed Ollama models found locally on this machine.")
        print("Please install a model first using: ollama run <model_name>\n")
        sys.exit(1)

    print("\n==============================================")
    print(" Installed Ollama Models Found on Machine:")
    print("==============================================")
    for idx, name in enumerate(installed_models, 1):
        print(f"  [{idx}] {name}")
    print("==============================================")

    while True:
        try:
            choice = input(f"\nSelect a model number to serve (1-{len(installed_models)}) [Default: 1]: ").strip()
            if not choice:
                selected = installed_models[0]
                print(f"Selected default model: {selected}")
                return selected
            
            if choice.isdigit():
                num = int(choice)
                if 1 <= num <= len(installed_models):
                    selected = installed_models[num - 1]
                    print(f"Selected model: {selected}")
                    return selected
            
            if choice in installed_models:
                print(f"Selected model: {choice}")
                return choice
            
            print(f"Invalid selection '{choice}'. Enter a number between 1 and {len(installed_models)}.")
        except (KeyboardInterrupt, EOFError):
            print("\nModel selection cancelled by user.")
            sys.exit(0)

def select_server_url(default_url: str = "http://localhost:13500") -> str:
    """Presents a CLI prompt for the user to enter or confirm the Kubli server URL."""
    print("\n==============================================")
    print(" Kubli Server Connection Setup:")
    print("==============================================")
    try:
        url_input = input(f"Enter Kubli Server URL [Default: {default_url}]: ").strip()
        if not url_input:
            print(f"Using default server URL: {default_url}")
            return default_url
        
        if not url_input.startswith("http://") and not url_input.startswith("https://"):
            url_input = f"http://{url_input}"
        
        print(f"Server URL set to: {url_input}")
        return url_input
    except (KeyboardInterrupt, EOFError):
        print("\nServer URL configuration cancelled by user.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kubli Distributed AI Compute Worker")
    parser.add_argument("--server", type=str, default=None, help="Kubli Server URL (if omitted, interactive prompt will ask)")
    parser.add_argument("--id", type=str, default=f"worker-{uuid.uuid4().hex[:6]}", help="Unique worker ID")
    parser.add_argument("--model", type=str, default=None, help="Specific model name (if omitted, an interactive menu will pop up)")
    parser.add_argument("--ollama", type=str, default="http://localhost:11434", help="Local Ollama instance URL")

    args = parser.parse_args()

    installed = asyncio.run(fetch_ollama_models(args.ollama))

    # 1. Select AI Model
    selected_model = args.model
    if not selected_model:
        selected_model = select_model_interactively(installed)
    else:
        installed_names_and_short = []
        for name in installed:
            installed_names_and_short.append(name)
            if ":" in name:
                installed_names_and_short.append(name.split(":")[0])

        if selected_model not in installed_names_and_short:
            print(f"\n[ERROR] Model '{selected_model}' is not installed in local Ollama.")
            print(f"Installed models: {', '.join(installed) if installed else 'None'}\n")
            sys.exit(1)

    # 2. Select / Confirm Kubli Server URL
    server_url = args.server
    if not server_url:
        server_url = select_server_url("http://localhost:13500")

    # 3. Instantiate and run worker node
    worker = KubliWorker(
        server_url=server_url,
        worker_id=args.id,
        model=selected_model,
        ollama_url=args.ollama
    )

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logger.info("Kubli Worker stopped by user.")