from __future__ import annotations

import threading
from collections import defaultdict

BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str], int] = defaultdict(int)
        self._latency_sum: dict[str, float] = defaultdict(float)
        self._latency_count: dict[str, int] = defaultdict(int)
        self._buckets: dict[tuple[str, float], int] = defaultdict(int)
        self._inflight = 0
        self._queue_depth = 0
        self._batches = 0
        self._batch_jobs = 0

    def inc_inflight(self) -> None:
        with self._lock:
            self._inflight += 1

    def dec_inflight(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = depth

    def inc_request(self, endpoint: str, status: str) -> None:
        with self._lock:
            self._requests[(endpoint, status)] += 1

    def observe_batch(self, size: int) -> None:
        with self._lock:
            self._batches += 1
            self._batch_jobs += size

    def observe(self, endpoint: str, seconds: float) -> None:
        with self._lock:
            self._latency_sum[endpoint] += seconds
            self._latency_count[endpoint] += 1
            for bucket in BUCKETS:
                if seconds <= bucket:
                    self._buckets[(endpoint, bucket)] += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            lines.append("# HELP jeva_requests_total Total HTTP requests by endpoint and status.")
            lines.append("# TYPE jeva_requests_total counter")
            for (endpoint, status), count in sorted(self._requests.items()):
                lines.append(
                    f'jeva_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}'
                )
            lines.append("# HELP jeva_request_duration_seconds Request latency.")
            lines.append("# TYPE jeva_request_duration_seconds histogram")
            for endpoint in sorted(self._latency_count):
                cumulative = 0
                for bucket in BUCKETS:
                    cumulative = self._buckets.get((endpoint, bucket), 0)
                    lines.append(
                        "jeva_request_duration_seconds_bucket"
                        f'{{endpoint="{endpoint}",le="{bucket:g}"}} {cumulative}'
                    )
                total = self._latency_count[endpoint]
                lines.append(
                    "jeva_request_duration_seconds_bucket"
                    f'{{endpoint="{endpoint}",le="+Inf"}} {total}'
                )
                lines.append(
                    f'jeva_request_duration_seconds_sum{{endpoint="{endpoint}"}} '
                    f"{self._latency_sum[endpoint]:.6f}"
                )
                lines.append(
                    f'jeva_request_duration_seconds_count{{endpoint="{endpoint}"}} {total}'
                )
            lines.append("# HELP jeva_inflight_requests Requests currently being handled.")
            lines.append("# TYPE jeva_inflight_requests gauge")
            lines.append(f"jeva_inflight_requests {self._inflight}")
            lines.append("# HELP jeva_queue_depth Pending inference jobs.")
            lines.append("# TYPE jeva_queue_depth gauge")
            lines.append(f"jeva_queue_depth {self._queue_depth}")
            lines.append("# HELP jeva_batches_total Micro-batches executed by the engine.")
            lines.append("# TYPE jeva_batches_total counter")
            lines.append(f"jeva_batches_total {self._batches}")
            lines.append("# HELP jeva_batch_jobs_total Jobs executed inside micro-batches.")
            lines.append("# TYPE jeva_batch_jobs_total counter")
            lines.append(f"jeva_batch_jobs_total {self._batch_jobs}")
        return "\n".join(lines) + "\n"
