import asyncio
import json
import logging
import uuid
import time
from typing import Dict, List, Optional, Any
from aiohttp import web, WSMsgType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KubliServer")

class Task:
    """Represents an AI generation request queued or processing in Kubli with document attachments."""
    def __init__(
        self, 
        task_id: str, 
        model: str, 
        prompt: str, 
        parameters: dict, 
        messages: Optional[list] = None,
        documents: Optional[list] = None,
        images: Optional[list] = None
    ):
        self.task_id = task_id
        self.model = model
        self.prompt = prompt
        self.messages = messages or []
        self.documents = documents or []
        self.images = images or []
        self.parameters = parameters
        self.status = "queued"  # queued, processing, completed, failed
        self.created_at = time.time()
        self.assigned_worker_id: Optional[str] = None
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.completion_future = asyncio.Future()
        self.chunk_queue: asyncio.Queue = asyncio.Queue()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "prompt": self.prompt,
            "messages": self.messages,
            "documents": self.documents,
            "images": self.images,
            "parameters": self.parameters,
            "status": self.status,
            "created_at": self.created_at,
            "assigned_worker_id": self.assigned_worker_id,
            "result": self.result,
            "error": self.error
        }

class WorkerConnection:
    """Represents an active WebSocket connection to a Kubli processing worker."""
    def __init__(self, worker_id: str, models: List[str], ws: web.WebSocketResponse):
        self.worker_id = worker_id
        self.models = set(models)
        self.ws = ws
        self.is_busy = False
        self.current_task_id: Optional[str] = None
        self.connected_at = time.time()

class KubliServer:
    def __init__(self):
        self.task_queue: List[Task] = []
        self.tasks: Dict[str, Task] = {}
        self.workers: Dict[str, WorkerConnection] = {}
        self.lock = asyncio.Lock()

    async def dispatch_pending_tasks(self):
        """Iterates through queued tasks and assigns them to available matching workers."""
        async with self.lock:
            if not self.task_queue:
                return

            i = 0
            while i < len(self.task_queue):
                task = self.task_queue[i]
                
                candidate_worker = None
                for worker in self.workers.values():
                    if not worker.is_busy and task.model in worker.models:
                        candidate_worker = worker
                        break

                if candidate_worker:
                    self.task_queue.pop(i)
                    task.status = "processing"
                    task.assigned_worker_id = candidate_worker.worker_id
                    candidate_worker.is_busy = True
                    candidate_worker.current_task_id = task.task_id

                    logger.info(f"Assigning task {task.task_id} ({task.model}) with {len(task.documents)} docs and {len(task.images)} images to worker '{candidate_worker.worker_id}'")

                    payload = {
                        "type": "assign_task",
                        "task_id": task.task_id,
                        "model": task.model,
                        "prompt": task.prompt,
                        "messages": task.messages,
                        "documents": task.documents,
                        "images": task.images,
                        "parameters": task.parameters
                    }
                    try:
                        await candidate_worker.ws.send_json(payload)
                    except Exception as e:
                        logger.error(f"Failed sending task {task.task_id} to worker {candidate_worker.worker_id}: {e}")
                        task.status = "queued"
                        task.assigned_worker_id = None
                        candidate_worker.is_busy = False
                        candidate_worker.current_task_id = None
                        self.task_queue.insert(0, task)
                else:
                    i += 1

    async def handle_get_models(self, request: web.Request) -> web.Response:
        """Returns list of models currently supported by connected workers."""
        async with self.lock:
            models_info = {}
            for worker in self.workers.values():
                for model in worker.models:
                    if model not in models_info:
                        models_info[model] = {"total_workers": 0, "available_workers": 0}
                    models_info[model]["total_workers"] += 1
                    if not worker.is_busy:
                        models_info[model]["available_workers"] += 1

        return web.json_response({
            "models": models_info,
            "total_active_workers": len(self.workers)
        })

    async def handle_submit_prompt(self, request: web.Request) -> web.Response:
        """Accepts prompt and document requests from clients and streams execution token-by-token."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        model = data.get("model")
        prompt = data.get("prompt", "")
        messages = data.get("messages", [])
        documents = data.get("documents", [])
        images = data.get("images", [])
        parameters = data.get("parameters", {})
        stream_requested = data.get("stream", True)

        if not model or (not prompt and not messages and not documents and not images):
            return web.json_response({"error": "Missing 'model' or task payload"}, status=400)

        async with self.lock:
            model_available = any(model in w.models for w in self.workers.values())

        if not model_available:
            return web.json_response({
                "error": f"No active worker currently supports model '{model}'. Use GET /api/models to check available models."
            }, status=404)

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        task = Task(
            task_id=task_id, 
            model=model, 
            prompt=prompt, 
            parameters=parameters, 
            messages=messages,
            documents=documents,
            images=images
        )

        async with self.lock:
            self.tasks[task_id] = task
            self.task_queue.append(task)

        logger.info(f"New prompt request queued [{task_id}] for model '{model}' with {len(documents)} docs / {len(images)} images")
        
        asyncio.create_task(self.dispatch_pending_tasks())

        if stream_requested:
            response = web.StreamResponse(
                status=200,
                reason='OK',
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                }
            )
            await response.prepare(request)

            try:
                while True:
                    chunk = await asyncio.wait_for(task.chunk_queue.get(), timeout=300.0)
                    if chunk is None:
                        break
                    if isinstance(chunk, dict) and "error" in chunk:
                        payload = json.dumps({"error": chunk["error"]})
                        await response.write(f"data: {payload}\n\n".encode('utf-8'))
                        break
                    
                    payload = json.dumps({"chunk": chunk})
                    await response.write(f"data: {payload}\n\n".encode('utf-8'))
                
                await response.write(b"data: [DONE]\n\n")
            except asyncio.TimeoutError:
                task.status = "failed"
                task.error = "Processing timed out"
                await response.write(b"data: {\"error\": \"Processing timed out\"}\n\n")
            
            return response
        else:
            try:
                await asyncio.wait_for(task.completion_future, timeout=300.0)
                return web.json_response({
                    "status": "success",
                    "task": task.to_dict()
                })
            except asyncio.TimeoutError:
                task.status = "failed"
                task.error = "Processing timed out"
                return web.json_response({"error": "Task execution timed out"}, status=504)

    async def handle_get_queue(self, request: web.Request) -> web.Response:
        """Returns queue and processing status."""
        async with self.lock:
            queued = [t.to_dict() for t in self.task_queue]
            processing = [t.to_dict() for t in self.tasks.values() if t.status == "processing"]
        return web.json_response({"queued": queued, "processing": processing})

    async def handle_worker_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Manages WebSocket connections and lifecycle for compute workers."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        worker_id = None
        logger.info("New worker connection initialized...")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type")

                    if msg_type == "register":
                        worker_id = data.get("worker_id", f"worker-{uuid.uuid4().hex[:6]}")
                        models = data.get("models", [])

                        async with self.lock:
                            if worker_id in self.workers:
                                logger.warning(f"Worker '{worker_id}' re-registered. Overwriting connection.")
                            
                            self.workers[worker_id] = WorkerConnection(worker_id, models, ws)

                        logger.info(f"Worker '{worker_id}' registered with models: {models}")
                        await ws.send_json({"type": "register_ack", "status": "ok", "worker_id": worker_id})
                        
                        asyncio.create_task(self.dispatch_pending_tasks())

                    elif msg_type == "task_chunk":
                        task_id = data.get("task_id")
                        chunk = data.get("chunk", "")
                        task = self.tasks.get(task_id)
                        if task:
                            await task.chunk_queue.put(chunk)

                    elif msg_type == "task_complete":
                        task_id = data.get("task_id")
                        result = data.get("result")
                        error = data.get("error")

                        async with self.lock:
                            worker = self.workers.get(worker_id)
                            if worker:
                                worker.is_busy = False
                                worker.current_task_id = None

                            task = self.tasks.get(task_id)
                            if task:
                                if error:
                                    task.status = "failed"
                                    task.error = error
                                    await task.chunk_queue.put({"error": error})
                                    logger.error(f"Task {task_id} failed on worker {worker_id}: {error}")
                                else:
                                    task.status = "completed"
                                    task.result = result
                                    logger.info(f"Task {task_id} successfully completed by worker {worker_id}")

                                await task.chunk_queue.put(None)
                                if not task.completion_future.done():
                                    task.completion_future.set_result(True)

                        asyncio.create_task(self.dispatch_pending_tasks())

                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket connection exception: {ws.exception()}")

        finally:
            async with self.lock:
                if worker_id and worker_id in self.workers:
                    disconnected_worker = self.workers.pop(worker_id)
                    logger.warning(f"Worker '{worker_id}' disconnected.")

                    if disconnected_worker.current_task_id:
                        task = self.tasks.get(disconnected_worker.current_task_id)
                        if task and task.status == "processing":
                            logger.info(f"Re-queuing interrupted task {task.task_id} from worker {worker_id}")
                            task.status = "queued"
                            task.assigned_worker_id = None
                            self.task_queue.insert(0, task)

            asyncio.create_task(self.dispatch_pending_tasks())

        return ws

@web.middleware
async def cors_middleware(request, handler):
    """Enables CORS headers for all incoming REST requests and OPTIONS preflights."""
    if request.method == "OPTIONS":
        response = web.Response(status=200)
    else:
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            response = ex
    
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

def create_app() -> web.Application:
    server = KubliServer()
    app = web.Application(middlewares=[cors_middleware])
    
    app.router.add_get("/api/models", server.handle_get_models)
    app.router.add_post("/api/generate", server.handle_submit_prompt)
    app.router.add_get("/api/queue", server.handle_get_queue)
    app.router.add_route("OPTIONS", "/{tail:.*}", lambda r: web.Response(status=200))
    
    app.router.add_get("/ws/worker", server.handle_worker_ws)
    
    return app

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Kubli Central Server on http://localhost:8000")
    web.run_app(app, host="0.0.0.0", port=8000)