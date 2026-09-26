# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import asyncio
import logging
import threading
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
import websockets
import websockets.asyncio.server
import websockets.frames

from . import msgpack_numpy


def _to_serializable(obj: Any) -> Any:
    """Recursively convert non-serializable objects (e.g. PIL Image) into
    msgpack-friendly types so the response can be packed by msgpack_numpy.

    PIL Image objects are converted to numpy arrays (which msgpack_numpy
    supports) and the downstream client uses ``np.asarray`` on them anyway.
    """
    # Lazy import to avoid a hard dependency if PIL is unavailable.
    try:
        from PIL import Image as _PILImage
    except Exception:  # pragma: no cover - PIL should be available
        _PILImage = None

    if _PILImage is not None and isinstance(obj, _PILImage.Image):
        return np.asarray(obj)
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    return obj


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # seconds; -1 means never shutdown
        metadata: Optional[dict] = None,
        ping_interval: Optional[float] = 30.0,
        ping_timeout: Optional[float] = 300.0,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()

        # WebSocket keepalive tuning (server side!)
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

        # Thread lock for concurrent inference protection
        self._inference_lock = threading.Lock()

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            close_timeout=10,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server: websockets.asyncio.server.Server) -> None:
        """Monitor server-level idle time; if no requests for idle_timeout seconds, shut down server."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send metadata as first binary frame
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                raw = await websocket.recv()
                msg = msgpack_numpy.unpackb(raw)
                self._last_active = time.time()

                # ★ CRITICAL: do NOT block event loop with inference
                ret = await asyncio.to_thread(self._route_message, msg)

                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # Send traceback as text, then close with 1011
                tb = traceback.format_exc()
                try:
                    await websocket.send(tb)
                except Exception:
                    pass
                try:
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except Exception:
                    pass
                raise

    def _route_message(self, msg: Any) -> Dict[str, Any]:
        """
        Fault-tolerant routing:
        Accepts either:
          1) protocol envelope:
             {"type": "ping|infer|reset|...", "request_id": "...", "payload": {...}}
          2) legacy style:
             { ...payload... }  (treated as infer)
        """
        # Normalize
        if isinstance(msg, dict):
            req_id = msg.get("request_id", "default")
            mtype = msg.get("type", "infer")
            payload = msg.get("payload", msg)  # if no payload wrapper, treat whole dict as payload
        else:
            req_id = "default"
            mtype = "infer"
            payload = msg

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer / predict_action
        if mtype in ("infer", "predict_action"):
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
                }

            try:
                # IMPORTANT: only pass payload into policy, not control fields
                # Use lock to ensure thread-safe concurrent inference
                with self._inference_lock:
                    output_dict = self._policy.predict_action(**payload, is_run_parallel_eval_submission=True)
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "inference_result",
                    "request_id": req_id,
                    "data": _to_serializable(output_dict),
                }
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }

        # unknown request type
        return {
            "status": "error",
            "ok": False,
            "type": "unknown",
            "request_id": req_id,
            "error": {"message": f"Unsupported message type '{mtype}'"},
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise NotImplementedError("This module is not intended to be run directly.")
