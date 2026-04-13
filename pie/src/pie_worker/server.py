"""
FFI server for PIE backend communication.

This module provides the RPC endpoint that handles requests from the Rust runtime
using direct FFI calls via PyO3.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import msgpack

from .runtime import Runtime

# Status codes for FFI dispatch (must match Rust)
STATUS_OK = 0
STATUS_METHOD_NOT_FOUND = 1
STATUS_INVALID_PARAMS = 2
STATUS_INTERNAL_ERROR = 3


# D1 IPC sub-bucket profiling — Python side. Reuses the same env var as
# WorkerProfiler so a single PIE_WORKER_PROFILING=1 turns everything on.
_IPC_PROFILE_ENABLED = os.environ.get("PIE_WORKER_PROFILING", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
    "off",
)


def poll_ffi_queue(
    ffi_queue, service: Runtime, stop_event: threading.Event, poll_timeout_ms: int = 100
) -> None:
    """Poll the Rust FfiQueue and process requests.

    This is the new high-performance worker loop that polls a Rust queue
    directly without Python queue overhead. Should be called from a dedicated
    Python thread that owns all CUDA state.

    Args:
        ffi_queue: _pie.FfiQueue instance from start_server_with_ffi
        service: Runtime instance to dispatch calls to
        stop_event: Event to signal shutdown
        poll_timeout_ms: How long to block waiting for requests (ms)
    """
    # Method dispatch table
    methods = {
        "handshake": service.handshake_rpc,
        "query": service.query_rpc,
        "fire_batch": service.fire_batch,
        "embed_image": service.embed_image_rpc,
        "initialize_adapter": service.initialize_adapter_rpc,
        "update_adapter": service.update_adapter_rpc,
        "upload_adapter": service.upload_adapter_rpc,
        "download_adapter": service.download_adapter_rpc,
    }

    try:
        while not stop_event.is_set():
            # Poll the Rust queue (releases GIL while waiting)
            request = ffi_queue.poll_blocking(poll_timeout_ms)
            if request is None:
                continue  # Timeout, try again

            # D1 — capture the moment poll_blocking returned. Anything
            # before this point is wire transit + ipc-channel internal
            # deserialization (Rust → Python). Anything after is Python
            # work attributable to the dispatch loop.
            t_recv = time.perf_counter() if _IPC_PROFILE_ENABLED else 0.0

            request_id, method, payload = request

            try:
                # Unpack args
                args = msgpack.unpackb(payload)
                t_unpack = time.perf_counter() if _IPC_PROFILE_ENABLED else 0.0

                # Get handler
                fn = methods.get(method)
                if fn is None:
                    response = msgpack.packb(f"Method not found: {method}")
                    ffi_queue.respond(request_id, response)
                    continue

                # Call handler
                if isinstance(args, dict):
                    result = fn(**args)
                elif isinstance(args, (list, tuple)):
                    result = fn(*args)
                else:
                    result = fn(args)

                t_handler_done = time.perf_counter() if _IPC_PROFILE_ENABLED else 0.0

                # Pack and respond
                response = msgpack.packb(result)
                t_pack = time.perf_counter() if _IPC_PROFILE_ENABLED else 0.0
                ffi_queue.respond(request_id, response)
                t_respond = time.perf_counter() if _IPC_PROFILE_ENABLED else 0.0

                if _IPC_PROFILE_ENABLED:
                    # Emit a structured per-call line that pairs with Rust's
                    # [IPC-PROFILE]. The "handler_ms" here matches the Python
                    # side of the wire_python_ms bucket on the Rust side.
                    print(
                        f"[IPC-PROFILE-PY] method={method} "
                        f"req_bytes={len(payload)} resp_bytes={len(response)} "
                        f"unpack_ms={(t_unpack - t_recv) * 1000.0:.3f} "
                        f"handler_ms={(t_handler_done - t_unpack) * 1000.0:.3f} "
                        f"pack_ms={(t_pack - t_handler_done) * 1000.0:.3f} "
                        f"respond_ms={(t_respond - t_pack) * 1000.0:.3f} "
                        f"total_ms={(t_respond - t_recv) * 1000.0:.3f}",
                        flush=True,
                    )

            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                print(f"[FFI Queue Error] {method}: {e}\n{tb}")
                response = msgpack.packb(str(e))
                ffi_queue.respond(request_id, response)
    finally:
        # Ensure cleanup when thread stops
        print("[FFI Worker] Shutting down Runtime...")
        service.shutdown()


def start_ffi_worker(
    ffi_queue, service: Runtime, thread_name: str = "pie-ffi-worker"
) -> tuple[threading.Thread, threading.Event]:
    """Start the FFI worker thread that polls the Rust queue.

    Args:
        ffi_queue: _pie.FfiQueue instance
        service: Runtime instance to dispatch calls to
        thread_name: Name for the worker thread (for debugging)

    Returns:
        tuple (thread, stop_event) where thread is already started.
    """
    stop_event = threading.Event()

    def worker():
        poll_ffi_queue(ffi_queue, service, stop_event)

    thread = threading.Thread(target=worker, name=thread_name, daemon=True)
    thread.start()
    return thread, stop_event
