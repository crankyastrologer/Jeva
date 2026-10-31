from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


class QueueFullError(RuntimeError):
    pass


class InferenceEngine:
    def __init__(
        self,
        executor: Callable[[Sequence[Any]], Sequence[Any]],
        *,
        queue_size: int = 32,
        default_timeout: float = 180.0,
        batch_window: float = 0.005,
        max_batch: int = 8,
    ):
        self._executor = executor
        self.batch_window = max(0.0, float(batch_window))
        self.max_batch = max(1, int(max_batch))
        self.default_timeout = default_timeout
        self._queue: asyncio.Queue[tuple[asyncio.Future, Any]] = asyncio.Queue(maxsize=queue_size)
        self._worker: asyncio.Task | None = None
        self._closed = False
        self._batches = 0
        self._jobs = 0

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="jeva-inference-worker")

    async def stop(self) -> None:
        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
            self._worker = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def queue_size(self) -> int:
        return self._queue.maxsize

    @property
    def stats(self) -> dict[str, int]:
        return {"batches": self._batches, "jobs": self._jobs}

    async def submit(self, payload: Any, *, timeout: float | None = None) -> Any:
        if self._closed:
            raise RuntimeError("inference engine is shutting down")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        try:
            self._queue.put_nowait((future, payload))
        except asyncio.QueueFull as exc:
            raise QueueFullError("inference queue is full") from exc
        limit = self.default_timeout if timeout is None else timeout
        try:
            return await asyncio.wait_for(future, timeout=limit)
        except TimeoutError:
            future.cancel()
            raise

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            future, payload = await self._queue.get()
            batch: list[tuple[asyncio.Future, Any]] = [(future, payload)]
            deadline = loop.time() + self.batch_window
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                except TimeoutError:
                    break
            try:
                results = await asyncio.to_thread(self._executor, [item for _, item in batch])
            except BaseException as exc:
                for pending, _ in batch:
                    if not pending.done():
                        pending.set_exception(exc)
            else:
                for (pending, _), result in zip(batch, results, strict=True):
                    if not pending.done():
                        pending.set_result(result)
            finally:
                self._batches += 1
                self._jobs += len(batch)
                for _ in batch:
                    self._queue.task_done()
