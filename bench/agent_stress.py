"""Agent-traffic stress test for an OpenAI-compatible server (dsv41).

Simulates N coding agents at once, each running a multi-turn tool-calling conversation the way agent clients do:
streamed chat completions with a large tool list, the assistant's tool calls and reasoning sent back, tool results
(real source files) appended, so the prompt grows every turn and every turn re-reads the previous one from the prefix
cache. Sampling settings vary per agent like real clients (temperature, top_p, top_k, penalties, max_tokens, stop,
seed, parallel tool calls, an occasional structured-output or non-streamed turn). Some turns are abandoned
mid-stream (the client closes the connection), as an agent does on interrupt or timeout.

Reports and exits 1 on: HTTP 5xx or transport errors, a stream that stalls (no chunk for --stall s), malformed tool
calls, degenerate output (one token repeated), the server's /health failing, or empty answers.
   python3 -I agent_stress.py --url http://127.0.0.1:8106/v1 --agents 8 --minutes 20 [--max-context 200000]
"""
import argparse, glob, json, os, random, socket, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8106/v1")
ap.add_argument("--agents", type=int, default=8)
ap.add_argument("--minutes", type=float, default=20)
ap.add_argument("--max-context", type=int, default=200_000, help="restart an agent's conversation past ~this many tokens")
ap.add_argument("--stall", type=float, default=180, help="seconds without a streamed chunk that count as a hang")
ap.add_argument("--abort-rate", type=float, default=0.12)
ap.add_argument("--files", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                help="directory whose source files serve as tool results")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--log", default="")
a = ap.parse_args()
BASE = a.url.rstrip("/")
ROOT = BASE[:-3] if BASE.endswith("/v1") else BASE
MODEL = json.load(urllib.request.urlopen(BASE + "/models", timeout=30))["data"][0]["id"]
FILES = [f for f in glob.glob(os.path.join(a.files, "**", "*"), recursive=True)
         if os.path.isfile(f) and f.endswith((".py", ".sh", ".md", ".cu", ".cuh", ".env", ".json")) and 500 < os.path.getsize(f) < 120_000]
assert FILES, "no source files for tool results"
T0 = time.time()
lock = threading.Lock()
stats = {"turns": 0, "ok": 0, "aborted": 0, "tool_calls": 0, "completion_tokens": 0, "prompt_tokens_max": 0}
problems = []
logf = open(a.log, "a") if a.log else None


def note(kind, agent, detail):
    with lock:
        problems.append((kind, agent, detail))
        line = f"[{time.time() - T0:7.1f}s] PROBLEM {kind} agent {agent}: {detail}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()


def log(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    if logf:
        logf.write(line + "\n"); logf.flush()


TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path relative to the repo"},
                    "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents with a regular expression.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                    "glob": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace an exact string in a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
                    "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return its output.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                    "required": ["command"]}}},
    {"type": "function", "function": {"name": "web_fetch", "description": "Fetch a URL and return it as markdown.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["url"]}}},
] + [{"type": "function", "function": {"name": f"mcp_tool_{i}", "description": f"Auxiliary integration tool number {i}: "
      "looks up records in an external system by key and returns them as JSON. " * 3,
      "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "fields": {"type": "array", "items": {"type": "string"}}},
                     "required": ["key"]}}} for i in range(24)]

SYSTEM = ("You are a coding agent working in a repository of GPU inference code. Use the tools to inspect files, run "
          "commands and make edits. Call tools whenever you need information; when you are done, answer briefly. " +
          "Follow the repository conventions. " * 40)

TASKS = ["Find where the decode step graph is captured and explain how the batch size is chosen.",
         "Look for any place that could read a stale prompt length and tell me how it is guarded.",
         "Read the profile files and summarize every environment switch with its purpose.",
         "Find the code that handles speculative drafts and list its configuration knobs.",
         "Check the launcher script for hard-coded paths and suggest a fix.",
         "Search for TODO or FIXME comments and triage them by risk.",
         "Explain how the sidecar communicates with the main server, with file references.",
         "Write a short design note on how prefill chunks are scheduled, reading the relevant files first."]

SAMPLING = [
    {"temperature": 1.0, "top_p": 0.95},                       # model defaults
    {"temperature": 0.6, "top_p": 0.95},
    {"temperature": 0.0},
    {"temperature": 0.7, "top_k": 40},
    {"temperature": 1.0, "top_p": 0.9, "presence_penalty": 0.3},
    {"temperature": 0.8, "frequency_penalty": 0.2, "seed": 1234},
    {"temperature": 1.0, "repetition_penalty": 1.05},
    {"temperature": 0.3, "top_p": 0.8},           # (min_p > 0 is refused with speculative decoding: README)
]


def post_stream(agent, body, abort_after):
    """Streams one request; returns (message dict, finish_reason, usage, aborted). abort_after: chunks before closing."""
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=a.stall)
    content, reasoning, calls, finish, usage, n = [], [], {}, None, None, 0
    try:
        sock_timeout_set = False
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break
            d = json.loads(line[5:])
            if d.get("error"):
                raise RuntimeError(f"error event in the stream: {json.dumps(d['error'])[:300]}")
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    reasoning.append(delta.get("reasoning_content") or delta.get("reasoning"))
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
            n += 1
            if abort_after is not None and n >= abort_after:
                resp.close()
                return None, "aborted", None, True
    finally:
        resp.close()
    msg = {"role": "assistant", "content": "".join(content) or None}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = [{"id": s["id"] or f"call_{i}", "type": "function", "function": {"name": s["name"], "arguments": s["args"]}}
                             for i, s in sorted(calls.items())]
    return msg, finish, usage, False


def post_plain(body):
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=a.stall * 3))
    c = r["choices"][0]
    m = c["message"]
    msg = {"role": "assistant", "content": m.get("content")}
    if m.get("reasoning_content") or m.get("reasoning"):
        msg["reasoning_content"] = m.get("reasoning_content") or m.get("reasoning")
    if m.get("tool_calls"):
        msg["tool_calls"] = m["tool_calls"]
    return msg, c.get("finish_reason"), r.get("usage"), False


def degenerate(text):
    if not text or len(text) < 400:
        return False
    tail = text[-400:]
    for k in (1, 2, 3, 4, 6, 8):
        unit = tail[-k:]
        if unit.strip() and unit * (400 // k) == tail[-(400 // k) * k:]:
            return True
    return False


def tool_result(rng, call):
    name = call["function"]["name"]
    if name in ("read_file", "grep", "run_shell", "list_dir") or rng.random() < 0.5:
        f = rng.choice(FILES)
        try:
            text = open(f, errors="replace").read()
        except OSError:
            text = "error: unreadable"
        if name == "grep":
            text = "\n".join(l for l in text.splitlines() if rng.random() < 0.3)
        return text[: rng.choice([2_000, 8_000, 20_000, 60_000])]
    if name == "edit_file":
        return "The file was edited."
    return json.dumps({"result": "ok", "records": [{"k": rng.random()} for _ in range(rng.randint(1, 30))]})


def agent(i, stop_at):
    rng = random.Random(a.seed * 1000 + i)
    while time.time() < stop_at:
        samp = dict(rng.choice(SAMPLING))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rng.choice(TASKS)}]
        log(f"agent {i}: new conversation, sampling {samp}")
        prompt_tokens = 0
        while time.time() < stop_at and prompt_tokens < a.max_context:
            body = {"model": MODEL, "messages": msgs, "tools": TOOLS, "max_tokens": rng.choice([1024, 4096, 8192, 16384]),
                    **samp}
            r = rng.random()
            if r < 0.08:
                body["response_format"] = {"type": "json_object"}
                body.pop("tools")
                body["messages"] = msgs + [{"role": "user", "content": "Reply with a JSON object summarizing progress so far."}]
            elif r < 0.15:
                body["tool_choice"] = "required"
            elif r < 0.2:
                body["parallel_tool_calls"] = False
            if rng.random() < 0.1:
                body["stop"] = ["</answer>", "\n\n\n\n"]
            if rng.random() < 0.15:
                body["reasoning_effort"] = rng.choice(["minimal", "low", "medium", "high", "none"])
            streamed = rng.random() < 0.85
            abort_after = rng.randint(1, 300) if streamed and rng.random() < a.abort_rate else None
            if streamed:
                body["stream"] = True
                body["stream_options"] = {"include_usage": True}
            t = time.time()
            with lock:
                stats["turns"] += 1
            try:
                msg, finish, usage, aborted = post_stream(i, body, abort_after) if streamed else post_plain(body)
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code >= 500:
                    note("http5xx", i, f"{e.code} after {time.time() - t:.1f}s: {detail}")
                elif "maximum context length" in detail or "too long" in detail.lower():
                    break
                else:
                    note(f"http{e.code}", i, detail)
                break
            except (socket.timeout, TimeoutError):
                note("stall", i, f"no data for {a.stall:.0f}s (turn {len(msgs)} messages, ~{prompt_tokens} prompt tokens)")
                break
            except Exception as e:  # noqa: BLE001 - transport errors are the finding
                note("transport", i, f"{type(e).__name__}: {e} after {time.time() - t:.1f}s")
                break
            if aborted:
                with lock:
                    stats["aborted"] += 1
                continue                                   # the agent retries the same turn
            if finish == "length":
                with lock:
                    stats["length"] = stats.get("length", 0) + 1
            if body.get("response_format"):
                try:
                    json.loads(msg.get("content") or "")
                except ValueError:
                    if finish != "length":           # cut off at max_tokens: the client's limit, not a server fault
                        note("bad-json", i, f"finish {finish}: {(msg.get('content') or '')[:200]!r}")
                continue
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                with lock:
                    stats["completion_tokens"] += usage.get("completion_tokens", 0)
                    stats["prompt_tokens_max"] = max(stats["prompt_tokens_max"], prompt_tokens)
            text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
            if degenerate(text):
                note("degenerate", i, f"...{text[-160:]!r} (~{prompt_tokens} prompt tokens, {samp})")
            calls = msg.get("tool_calls") or []
            for c in calls:
                try:
                    json.loads(c["function"]["arguments"] or "{}")
                except ValueError:
                    if finish != "length":
                        note("bad-tool-args", i, f"finish {finish}, {c['function']['name']}: {c['function']['arguments'][:200]!r}")
                if c["function"]["name"] not in {t["function"]["name"] for t in TOOLS}:
                    note("unknown-tool", i, c["function"]["name"])
            if not calls and not (msg.get("content") or "").strip() and finish != "length":
                note("empty", i, f"finish {finish}, reasoning {len(msg.get('reasoning_content') or '')} chars")
            with lock:
                stats["ok"] += 1
                stats["tool_calls"] += len(calls)
            msgs.append(msg)
            if calls:
                for c in calls:
                    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": tool_result(rng, c)})
            else:
                msgs.append({"role": "user", "content": rng.choice(["Continue: dig deeper and check the related files.",
                                                                    "Good. Now verify it by reading the code it depends on.",
                                                                    "Run a quick check with the shell tool, then report."])})
            time.sleep(rng.uniform(0, 3))                  # client-side think time


def health():
    try:
        return urllib.request.urlopen(ROOT + "/health", timeout=10).status == 200
    except Exception:  # noqa: BLE001
        return False


stop_at = T0 + a.minutes * 60
threads = [threading.Thread(target=agent, args=(i, stop_at), daemon=True) for i in range(a.agents)]
for k, th in enumerate(threads):
    th.start()
    time.sleep(2)
while any(th.is_alive() for th in threads):
    time.sleep(30)
    with lock:
        s = dict(stats)
    ok = health()
    log(f"status: {s}, problems {len(problems)}, health {'ok' if ok else 'FAILED'}")
    if not ok:
        note("health", -1, "/health failed")
        break
    if time.time() > stop_at + 900:
        note("stuck", -1, "agents still running 15 min after the end")
        break
kinds = {}
for k, _, _ in problems:
    kinds[k] = kinds.get(k, 0) + 1
log(f"done: {stats}; problems by kind: {kinds or 'none'}")
raise SystemExit(1 if problems else 0)
