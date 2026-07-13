"""Blocking bridge from the HF trainer thread to the server event loop."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
import inspect
import threading
from typing import Any, Coroutine, TypeVar


ResultT = TypeVar("ResultT")


class BlockingAsyncBridge:
    """Run async engine work on its owner loop and block only the caller thread."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        owner_thread_id: int,
    ) -> None:
        self._loop = loop
        self._owner_thread_id = int(owner_thread_id)
        self._active_lock = threading.Lock()
        self._active: set[Future[Any]] = set()

    def call(
        self,
        coroutine: Coroutine[Any, Any, ResultT],
        *,
        timeout: float | None = None,
    ) -> ResultT:
        """Submit one coroutine and synchronously return its result."""

        if threading.get_ident() == self._owner_thread_id:
            coroutine.close()
            raise RuntimeError(
                "BlockingAsyncBridge.call() cannot run on the server event-loop thread"
            )
        if not inspect.iscoroutine(coroutine):
            raise TypeError("BlockingAsyncBridge.call() requires a coroutine object")
        if self._loop.is_closed():
            coroutine.close()
            raise RuntimeError("server event loop is closed")

        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        with self._active_lock:
            self._active.add(future)
        try:
            return future.result(timeout=timeout)
        finally:
            with self._active_lock:
                self._active.discard(future)

    @property
    def active_calls(self) -> int:
        with self._active_lock:
            return len(self._active)


__all__ = ["BlockingAsyncBridge"]
