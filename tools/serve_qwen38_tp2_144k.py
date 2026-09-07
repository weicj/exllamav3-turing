"""Small single-request OpenAI-compatible server for the verified 144K TP2 lane."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock

# This is the verified TP2 lane's operational baseline. Operators may override these
# values explicitly, but a bare manual start must not silently fall back to FlashInfer's
# 16 MiB workspace and lose a TP rank on an ordinary prefill request.
os.environ.setdefault("EXL3_FLASHINFER_WORKSPACE_MB", "128")
os.environ.setdefault("EXL3_NGRAM_STREAM", "1")
os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "Simple")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import ComboSampler, GreedySampler


MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
MODEL_ID = "qwen38-flash-next-exl3-2.05bpw"
MAX_CONTEXT = 147456
CACHE_TOKENS = 147712
GPU_UUIDS = (
    "GPU-1c2b8831-227f-96d1-0c0f-92a189726072",
    "GPU-4da87347-023e-7014-a837-6b1fb649a42e",
)


class Runtime:
    def __init__(self):
        self.config = Config.from_directory(MODEL_DIR)
        self.model = Model.from_config(self.config)
        self.cache = Cache(self.model, max_num_tokens=CACHE_TOKENS, max_batch_size=1)
        self.tokenizer = Tokenizer.from_config(self.config)
        self.generator: Generator | None = None
        self.lock = Lock()
        self.loaded = False

    def load(self):
        if torch.cuda.device_count() != 2:
            raise RuntimeError(f"expected two visible GPUs, got {torch.cuda.device_count()}")
        self.model.load(
            tensor_p=True,
            tp_backend="nccl",
            use_per_device=[22.0, 22.0],
            max_chunk_size=256,
            max_batch_size=1,
            verbose=True,
        )
        self.generator = Generator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
            max_batch_size=1,
            max_chunk_size=512,
        )
        self.loaded = True

    def unload(self):
        if self.generator is not None:
            self.generator = None
        if self.loaded:
            self.model.unload()
            self.loaded = False

    def healthy(self) -> bool:
        if not self.loaded:
            return False
        # The output rank is a pseudo-worker in this process. Every actual CUDA
        # worker must remain alive; otherwise the listener cannot serve TP2 safely.
        for child in getattr(self.model, "mp_children", []):
            if child is None or child.__class__.__name__ == "PseudoChild":
                continue
            if not child.is_alive():
                return False
        return True

    def _messages_to_ids(self, messages: list[dict], enable_thinking: bool) -> torch.Tensor:
        normalized = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system", "user", "assistant", "tool",
            }:
                raise HTTPException(status_code=400, detail="messages must contain valid chat roles")
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "".join(text_parts)
            if not isinstance(content, str):
                raise HTTPException(status_code=400, detail="only text message content is supported")
            normalized.append({"role": message["role"], "content": content})
        if not normalized:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        try:
            return self.tokenizer.hf_chat_template(
                normalized,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"chat template failed: {exc}") from exc

    @staticmethod
    def _thinking_enabled(payload: dict) -> bool:
        """Accept both the direct ExLlama option and OpenAI server conventions."""
        direct = payload.get("enable_thinking")
        if isinstance(direct, bool):
            return direct
        template_kwargs = payload.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict):
            template_value = template_kwargs.get("enable_thinking")
            if isinstance(template_value, bool):
                return template_value
        return False

    @staticmethod
    def _timings(prompt_tokens: int, completion_tokens: int, finished: dict) -> dict:
        prefill_seconds = float(finished.get("time_prefill", 0.0))
        decode_seconds = float(finished.get("time_generate", 0.0))
        decoded_tokens = max(0, completion_tokens - 1)
        return {
            "prompt_n": prompt_tokens,
            "predicted_n": completion_tokens,
            "prompt_per_second": prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0,
            "predicted_per_second": (
                decoded_tokens / decode_seconds
                if decoded_tokens > 0 and decode_seconds > 0
                else 0.0
            ),
        }

    def _build_job(self, payload: dict) -> tuple[Job, int]:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a list")
        max_tokens = int(payload.get("max_tokens", 256))
        if not 1 <= max_tokens <= MAX_CONTEXT:
            raise HTTPException(status_code=400, detail="max_tokens must be between 1 and 147456")
        ids = self._messages_to_ids(messages, self._thinking_enabled(payload))
        prompt_tokens = int(ids.shape[1])
        if prompt_tokens + max_tokens > MAX_CONTEXT:
            raise HTTPException(
                status_code=400,
                detail=f"prompt ({prompt_tokens}) + max_tokens ({max_tokens}) exceeds {MAX_CONTEXT}",
            )

        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.8))
        top_k = int(payload.get("top_k", 0))
        min_p = float(payload.get("min_p", 0.0))
        if temperature <= 0:
            sampler = GreedySampler()
        else:
            sampler = ComboSampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )
        stop_conditions = list(self.tokenizer.config.eos_token_id_list)
        return Job(
            input_ids=ids,
            max_new_tokens=max_tokens,
            sampler=sampler,
            stop_conditions=stop_conditions,
        ), prompt_tokens

    def generate(self, payload: dict) -> dict:
        if self.generator is None:
            raise RuntimeError("model is not loaded")
        job, prompt_tokens = self._build_job(payload)
        self.generator.enqueue(job)
        text = ""
        finished = None
        with torch.inference_mode():
            while self.generator.num_remaining_jobs():
                for result in self.generator.iterate():
                    text += result.get("text", "")
                    if result.get("eos"):
                        finished = result
        if finished is None:
            raise RuntimeError("generation completed without a final result")
        completion_tokens = int(finished["new_tokens"])
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timings": self._timings(prompt_tokens, completion_tokens, finished),
            "finish_reason": "stop" if finished.get("eos_reason") == "stop_token" else "length",
        }

    def generate_stream_events(
        self,
        payload: dict,
        request_id: str,
        created: int,
        cancel_event: threading.Event | None = None,
    ):
        """Yield OpenAI-compatible SSE event strings from the synchronous generator."""
        if self.generator is None:
            raise RuntimeError("model is not loaded")
        job, prompt_tokens = self._build_job(payload)
        self.generator.enqueue(job)
        finished = None
        sent_role = False
        in_thinking = self._thinking_enabled(payload)
        pending_text = ""
        model_id = payload.get("model", MODEL_ID)

        def cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        def event(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"

        def chunk(
            *, reasoning_content: str = "", content: str = "", finish_reason: str | None = None
        ) -> str | None:
            nonlocal sent_role
            delta = {}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            if reasoning_content:
                delta["reasoning_content"] = reasoning_content
            if content:
                delta["content"] = content
            if not delta and finish_reason is None:
                return None
            return event({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            })

        try:
            with torch.inference_mode():
                while self.generator.num_remaining_jobs():
                    if cancelled():
                        return
                    for result in self.generator.iterate():
                        # A CUDA operation cannot be preempted, but no subsequent
                        # thinking or decode step runs after the client cancels.
                        if cancelled():
                            return
                        pending_text += result.get("text", "")
                        while pending_text:
                            if in_thinking:
                                marker_index = pending_text.find("</think>")
                                if marker_index >= 0:
                                    reasoning_piece = pending_text[:marker_index]
                                    pending_text = pending_text[marker_index + len("</think>"):]
                                    in_thinking = False
                                    item = chunk(reasoning_content=reasoning_piece)
                                    if item is not None:
                                        yield item
                                    continue

                                # Retain a short suffix so a split </think> marker is
                                # never leaked into the browser's reasoning panel.
                                safe_length = max(0, len(pending_text) - len("</think>") + 1)
                                if safe_length == 0:
                                    break
                                item = chunk(reasoning_content=pending_text[:safe_length])
                                pending_text = pending_text[safe_length:]
                                if item is not None:
                                    yield item
                                continue

                            item = chunk(content=pending_text)
                            pending_text = ""
                            if item is not None:
                                yield item

                        if result.get("eos"):
                            finished = result
        finally:
            if cancelled() and self.generator is not None:
                self.generator.cancel(job)

        if cancelled():
            return

        if pending_text:
            item = chunk(
                reasoning_content=pending_text if in_thinking else "",
                content="" if in_thinking else pending_text,
            )
            if item is not None:
                yield item

        if finished is None:
            raise RuntimeError("generation completed without a final result")
        completion_tokens = int(finished.get("new_tokens", 0))
        finish_reason = "stop" if finished.get("eos_reason") == "stop_token" else "length"
        item = chunk(finish_reason=finish_reason)
        if item is not None:
            yield item
        yield event({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "timings": self._timings(prompt_tokens, completion_tokens, finished),
        })
        yield "data: [DONE]\n\n"


runtime = Runtime()


async def acquire_runtime_lock(request: Request) -> bool:
    """Acquire the single-flight lock without retaining it after a disconnect."""
    while True:
        if await request.is_disconnected():
            return False
        acquired = await asyncio.to_thread(runtime.lock.acquire, True, 0.1)
        if acquired:
            if await request.is_disconnected():
                runtime.lock.release()
                return False
            return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(runtime.load)
    try:
        yield
    finally:
        await asyncio.to_thread(runtime.unload)


app = FastAPI(title="Qwen3.8 Flash-Next TP2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    healthy = runtime.healthy()
    return JSONResponse(status_code=200 if healthy else 503, content={
        "status": "ok" if healthy else ("degraded" if runtime.loaded else "loading"),
        "model": MODEL_ID,
        "context_tokens": MAX_CONTEXT,
        "cache_tokens": CACHE_TOKENS,
        "parallelism": "NCCL TP2",
    })


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "owned_by": "exllama",
            "context_length": MAX_CONTEXT,
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict, request: Request):
    started = time.time()
    # The generator/cache are deliberately single-flight. This also keeps the measured
    # 144K capacity meaningful: two long requests cannot overcommit the single cache.
    if not await acquire_runtime_lock(request):
        raise HTTPException(status_code=499, detail="client disconnected")
    if payload.get("stream", False):
        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        # Keep all CUDA calls on one worker thread while forwarding events to the
        # async response. A disconnect stops the job at the next generator boundary.
        events: queue.Queue[str | BaseException | None] = queue.Queue()
        cancel_event = threading.Event()

        def worker() -> None:
            try:
                for item in runtime.generate_stream_events(
                    payload,
                    request_id,
                    int(started),
                    cancel_event,
                ):
                    if cancel_event.is_set():
                        break
                    events.put(item)
            except BaseException as exc:  # surfaced by the response iterator
                events.put(exc)
            finally:
                runtime.lock.release()
                events.put(None)

        threading.Thread(target=worker, name="qwen-stream", daemon=True).start()

        async def event_iterator():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.to_thread(events.get, True, 0.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                cancel_event.set()

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        result = await asyncio.to_thread(runtime.generate, payload)
    finally:
        runtime.lock.release()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(started),
        "model": payload.get("model", MODEL_ID),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": result["finish_reason"],
        }],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
        "timings": result["timings"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8037)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
