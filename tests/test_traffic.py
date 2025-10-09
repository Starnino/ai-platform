"""
Chaotic traffic generator for AI serving gateway with per-request logging.

Usage example:
python tools/traffic_random.py --gateway http://localhost:8080 --config config/models.yaml --concurrency 16 --requests-max 100 --batch-max 1000 --err-prob 0.15 --jitter-max 400 --verbose
"""

import argparse
import asyncio
import contextlib
import itertools
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import httpx

logger = logging.getLogger("traffic_random")

# ---------------- Color helpers ----------------
def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")

class Palette:
    def __init__(self, enable: bool):
        self.enable = enable and _supports_color()
    def wrap(self, s: str, code: str) -> str:
        if not self.enable: return s
        return f"\033[{code}m{s}\033[0m"
    def cyan(self, s):   return self.wrap(s, "36")
    def green(self, s):  return self.wrap(s, "32")
    def yellow(self, s): return self.wrap(s, "33")
    def red(self, s):    return self.wrap(s, "31")
    def bold(self, s):   return self.wrap(s, "1")

# ---------------- Random payload utils ----------------
def _rand_scalar(t):
    if t is float: return random.uniform(-20.0, 60.0)
    if t is int:   return random.randint(0, 365)
    if t is bool:  return bool(random.getrandbits(1))
    if t is str:   return random.choice(["foo","bar","baz","qux"])
    raise TypeError(f"Unsupported scalar type: {t}")

def _as_shape(seq) -> List[int]:
    if seq is None: return []
    if isinstance(seq, (list, tuple)): return [int(x) for x in seq]
    raise TypeError(f"Unexpected shape type: {type(seq)}")

def rand_by_shape(t, shape: Sequence[int]):
    shape = list(shape or [])
    if not shape: return _rand_scalar(t)
    size, *rest = shape
    return [rand_by_shape(t, rest) for _ in range(size)]

def build_instance(input_schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    inst = {}
    for block in input_schema:
        inst[block["name"]] = rand_by_shape(block["type"], _as_shape(block.get("shape")))
    return inst

def corrupt_instance(inst: Dict[str, Any], schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    bad = dict(inst)
    if not schema: return bad
    b = random.choice(schema); name = b["name"]
    what = random.choice(["drop","wrong_type","wrong_shape"])
    if what == "drop":
        bad.pop(name, None)
    elif what == "wrong_type":
        bad[name] = "INVALID"
    else:
        bad[name] = [inst[name], inst[name]]
    return bad

async def jitter(ms_min: int, ms_max: int):
    if ms_max > 0:
        await asyncio.sleep(random.uniform(ms_min, ms_max) / 1000.0)

# ---------------- Tracking structures ----------------
@dataclass
class ReqInfo:
    task_id: int
    model: str
    batch: int
    corrupted: bool
    status: int | None = None
    latency_ms: float | None = None

@dataclass
class Report:
    requests: Dict[int, ReqInfo] = field(default_factory=dict)

    def add_send(self, info: ReqInfo):
        self.requests[info.task_id] = info

    def add_recv(self, task_id: int, status: int, latency_ms: float):
        if task_id in self.requests:
            self.requests[task_id].status = status
            self.requests[task_id].latency_ms = latency_ms

    def summarize(self, palette: "Palette", example_limit: int = 10) -> str:
        total = len(self.requests)
        valid = sum(1 for r in self.requests.values() if not r.corrupted)
        invalid = total - valid

        status_counts: Dict[int, int] = {}
        mismatches: List[ReqInfo] = []
        server_errors: List[ReqInfo] = []   # 5xx
        network_errors: List[ReqInfo] = []  # status == -1

        ok_valid = err_valid = ok_invalid = err_invalid = 0

        for r in self.requests.values():
            st = r.status if r.status is not None else -1
            status_counts[st] = status_counts.get(st, 0) + 1

            if st == -1:
                network_errors.append(r)
            elif 500 <= st < 600:
                server_errors.append(r)

            if r.corrupted:
                if st == 422:
                    ok_invalid += 1
                else:
                    err_invalid += 1
                    mismatches.append(r)
            else:
                if 200 <= st < 300:
                    ok_valid += 1
                else:
                    err_valid += 1
                    mismatches.append(r)

        lines: List[str] = []
        lines.append(palette.bold("=== Traffic Report ==="))
        lines.append(f"Total: {total} | Valid: {valid} | Corrupted: {invalid}")

        if status_counts:
            dist = " ".join([f"{k}:{v}" for k, v in sorted(status_counts.items())])
            lines.append(f"Status distribution: {dist}")

        # expectation summary
        lines.append(palette.green(f"Valid→200: {ok_valid}"))
        lines.append(palette.red  (f"Valid→!200: {err_valid}"))
        lines.append(palette.green(f"Corrupted→422: {ok_invalid}"))
        lines.append(palette.red  (f"Corrupted→!422: {err_invalid}"))

        # per-model summary
        per_model: Dict[str, int] = {}
        for r in self.requests.values():
            per_model[r.model] = per_model.get(r.model, 0) + 1
        if per_model:
            lines.append("Per model: " + ", ".join([f"{m}={n}" for m, n in per_model.items()]))

        # errors
        if server_errors:
            lines.append(palette.red(f"Server errors (5xx): {len(server_errors)}"))
            for r in server_errors[:example_limit]:
                lines.append(f"  id={r.task_id} model={r.model} batch={r.batch} status={r.status} {r.latency_ms:.1f}ms")
            if len(server_errors) > example_limit:
                lines.append(f"  (+{len(server_errors)-example_limit} more)")

        if network_errors:
            lines.append(palette.red(f"Network/Timeout failures: {len(network_errors)}"))
            for r in network_errors[:example_limit]:
                lines.append(f"  id={r.task_id} model={r.model} batch={r.batch} status=-1 {r.latency_ms:.1f}ms")
            if len(network_errors) > example_limit:
                lines.append(f"  (+{len(network_errors)-example_limit} more)")

        # mismatches 
        if mismatches:
            lines.append(palette.yellow(f"Mismatches vs expectation: {len(mismatches)}"))
            for r in mismatches[:example_limit]:
                exp = "422" if r.corrupted else "2xx"
                lines.append(f"  id={r.task_id} model={r.model} batch={r.batch} expected={exp} got={r.status}")
            if len(mismatches) > example_limit:
                lines.append(f"  (+{len(mismatches)-example_limit} more)")
        else:
            lines.append(palette.green("All results matched expectations ✅"))

        return "\n".join(lines)

# ---------------- Core request with colored logs ----------------
async def send_predict(
    client: httpx.AsyncClient,
    palette: Palette,
    report: Report,
    task_id: int,
    model: str,
    schema_in: List[Dict[str, Any]],
    batch_min: int,
    batch_max: int,
    err_prob: float,
):
    bs = random.randint(batch_min, batch_max)
    instances = [build_instance(schema_in) for _ in range(bs)]
    corrupted = (random.random() < err_prob) and bs > 0
    if corrupted:
        idx = random.randrange(bs)
        instances[idx] = corrupt_instance(instances[idx], schema_in)

    report.add_send(ReqInfo(task_id=task_id, model=model, batch=bs, corrupted=corrupted))
    logger.info("%s", palette.cyan(f"[{task_id:>3}] SEND {model} batch={bs}{' (X)' if corrupted else ''}"))

    t0 = time.perf_counter()
    try:
        r = await client.post(f"/models/{model}/predict", json={"instances": instances})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        report.add_recv(task_id, r.status_code, dt_ms)
        color = palette.green if 200 <= r.status_code < 300 else (palette.yellow if 400 <= r.status_code < 500 else palette.red)
        logger.info("%s", color(f"[{task_id:>3}] RECV {model} {r.status_code} {dt_ms:.1f}ms"))
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        report.add_recv(task_id, -1, dt_ms)
        logger.error("%s", palette.red(f"[{task_id:>3}] FAIL {model} {type(e).__name__} {dt_ms:.1f}ms"))

async def run_for_model(
    client: httpx.AsyncClient,
    palette: Palette,
    report: Report,
    name: str,
    cfg: Dict[str, Any],
    req_min: int,
    req_max: int,
    batch_min: int,
    batch_max: int,
    err_prob: float,
    jitter_min: int,
    jitter_max: int,
    sem: asyncio.Semaphore,
    id_counter: itertools.count,
):
    with contextlib.suppress(Exception):
        await client.get(f"/models/{name}", timeout=5.0)

    n = random.randint(req_min, req_max)
    schema_in = cfg.get("schema", {}).get("input", [])

    async def _task():
        task_id = next(id_counter)
        async with sem:
            await jitter(jitter_min, jitter_max)
            await send_predict(client, palette, report, task_id, name, schema_in, batch_min, batch_max, err_prob)

    await asyncio.gather(*[_task() for _ in range(n)], return_exceptions=True)

# ---------------- Entrypoint ----------------
async def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    for noisy in ["httpx", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    random.seed(args.seed if args.seed is not None else (int(time.time()) % 10_000))

    from src.shared.loader import load_models_config
    models_cfg = load_models_config(args.config)
    if not models_cfg:
        logger.warning("No models found in config: %s", args.config)
        return

    palette = Palette(enable=not args.no_color)
    report = Report()

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.gateway, timeout=args.timeout, limits=limits) as client:
        sem = asyncio.Semaphore(args.concurrency)
        id_counter = itertools.count(1)
        tasks = [
            run_for_model(
                client, palette, report, name, cfg,
                args.requests_min, args.requests_max,
                args.batch_min, args.batch_max,
                args.err_prob, args.jitter_min, args.jitter_max,
                sem, id_counter
            )
            for name, cfg in models_cfg.items()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # --- Final report ---
    print("\n" + report.summarize(palette))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate random traffic with colored logs and final report.")
    p.add_argument("--gateway", default="http://localhost:8080", help="Gateway base URL")
    p.add_argument("--config", default="config/models.yaml", help="Path to models YAML config")
    p.add_argument("--concurrency", type=int, default=16, help="Max concurrent requests")
    p.add_argument("--requests-min", type=int, default=1)
    p.add_argument("--requests-max", type=int, default=100)
    p.add_argument("--batch-min", type=int, default=1)
    p.add_argument("--batch-max", type=int, default=1000)
    p.add_argument("--err-prob", type=float, default=0.15, help="Probability of invalid payload (0..1)")
    p.add_argument("--jitter-min", type=int, default=5)
    p.add_argument("--jitter-max", type=int, default=400)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = p.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
