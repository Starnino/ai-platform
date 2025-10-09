from __future__ import annotations
import time, contextlib
from functools import lru_cache
from fastapi import FastAPI, Request
from starlette.responses import Response
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,
    ProcessCollector, PlatformCollector, GCCollector
)

EXCLUDE_PATHS = {"/metrics", "/ready", "/healthz", "/health", "/favicon.ico", "/openapi.json"}

class BaseMetrics:
    """Base metrics manager."""

    def __init__(
        self,
        *,
        http_latency_buckets: tuple[float, ...] | None = None,
        req_size_buckets:   tuple[int,   ...] | None = None,
        resp_size_buckets:  tuple[int,   ...] | None = None
    ) -> None:
        http_latency_buckets = http_latency_buckets or (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2, 3, 5, 8, 13, 20)
        req_size_buckets  = req_size_buckets or (200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000)
        resp_size_buckets = resp_size_buckets or (200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000)

        # ---------- Process Metrics --------
        try: 
            ProcessCollector()
        except ValueError: 
            pass
        try: 
            PlatformCollector()
        except ValueError: 
            pass
        try: 
            GCCollector()
        except ValueError: 
            pass

        # -------- HTTP Metrics --------
        self.http_inflight = Gauge(
            "http_inflight_requests", "In-flight HTTP requests",
            ["path"]
        )
        self.http_latency = Histogram(
            "http_request_duration_seconds", "HTTP request latency (s)",
            ["method", "path", "status"],
            buckets=http_latency_buckets
        )
        self.http_requests = Counter(
            "http_requests_total", "HTTP requests",
            ["method", "path", "status"]
        )
        self.app_exceptions = Counter(
            "app_exceptions_total", "Unhandled exceptions",
            ["path", "type"]
        )
        self.http_req_size = Histogram(
            "http_request_size_bytes", "HTTP request size (bytes)",
            ["path"], buckets=req_size_buckets
        )
        self.http_resp_size = Histogram(
            "http_response_size_bytes", "HTTP response size (bytes)",
            ["path"], buckets=resp_size_buckets
        )

    # ---------- Helpers ----------
    @staticmethod
    def _templated_path(request: Request) -> str:
        route = request.scope.get("route")
        return getattr(route, "path", request.url.path)

    async def http_middleware(self, request: Request, call_next):
        """Middleware pronto: latenza, inflight, req/resp size, eccezioni."""
        path = self._templated_path(request)
        
        if path in EXCLUDE_PATHS:
            return await call_next(request)

        # request size (best-effort)
        with contextlib.suppress(Exception):
            body = await request.body()
            self.http_req_size.labels(path=path).observe(len(body))

        self.http_inflight.labels(path=path).inc()
        t0 = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as e:
            self.app_exceptions.labels(path=path, type=e.__class__.__name__).inc()
            self.http_latency.labels(request.method, path, "500").observe(time.perf_counter() - t0)
            self.http_requests.labels(request.method, path, "500").inc()
            self.http_inflight.labels(path=path).dec()
            raise

        dt = time.perf_counter() - t0
        status = str(response.status_code)
        self.http_latency.labels(request.method, path, status).observe(dt)
        self.http_requests.labels(request.method, path, status).inc()

        with contextlib.suppress(Exception):
            size = None
            if hasattr(response, "body") and response.body is not None:
                size = len(response.body)
            elif response.headers.get("content-length"):
                size = int(response.headers["content-length"])
            if size is not None:
                self.http_resp_size.labels(path=path).observe(size)

        self.http_inflight.labels(path=path).dec()
        return response

    # ----- /metrics endpoint -----
    def attach(self, app: FastAPI) -> None:
        @app.middleware("http")
        async def _mw(request: Request, call_next):
            return await self.http_middleware(request, call_next)

        @app.get("/metrics", include_in_schema=False)
        def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # -----------------------------
    # Factory singleton
    # -----------------------------
    @classmethod
    @lru_cache(maxsize=1)
    def get_metrics(cls) -> "BaseMetrics":
        return cls()