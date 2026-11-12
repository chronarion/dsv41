"""Warm the server for real traffic: capture the whole-step graphs clients will replay before they arrive (from
shieldstar's warm-up in issue #5). A graph key holds the batch size, the draft width and the sampler's branches
(greedy or not, top-k on and its max, top-p on, ...), so the warm-up sends what clients send:
  defaults   no sampling fields: the server's generation config (temperature 1.0, top_p 0.95)
  top_k      the same plus top_k=20 (agent clients with Qwen-style sampling)
  mixed      both in one batch (top-k on with max = vocab size is a key family of its own)
Each family runs at batch 1 (several prompts, so batch 1 meets several draft widths and copy drafting's wide verify)
and as a batch of 8 with staggered lengths, so the batch steps down through every smaller size. Idle graphs are evicted
when memory runs short (DSV41_WS_IDLE_S), so later traffic still gets graphs; warming only makes the first minutes
fast.   python3 warm.py [URL, default http://127.0.0.1:8106] [LEVELS, default 1,8]"""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8106"
LEVELS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,8").split(",")]
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))["data"][0]["id"]
PROMPTS = ["Write a Python class for an LRU cache with get and put, with docstrings and type hints.",
           "Explain how TCP congestion control works, step by step.",
           "Repeat this line 40 times, one per line: the quick brown fox jumps over the lazy dog",
           "Refactor this function to be iterative and explain the change:\n"
           "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"]
FAMILIES = {"defaults": lambda i: {}, "top_k": lambda i: {"top_k": 20},
            "mixed": lambda i: {"top_k": 20} if i % 2 else {}}


def one(i, max_tokens, extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            "max_tokens": max_tokens, "ignore_eos": True, "chat_template_kwargs": {"thinking": False}, **extra}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=600).read()


def burst(c, fam):
    # staggered lengths: requests finish one by one, so the batch steps down c, c-1, ..., 1
    ts = [threading.Thread(target=one, args=(i, 96 + 48 * i, FAMILIES[fam](i))) for i in range(c)]
    [x.start() for x in ts]
    [x.join() for x in ts]


t0 = time.time()
for fam in FAMILIES:
    for c in LEVELS:
        if c == 1 and fam == "mixed":
            continue
        if c == 1:
            for i in range(len(PROMPTS)):
                one(i, 192, FAMILIES[fam](0))
        else:
            burst(c, fam)
print(f"warm-up done in {time.time() - t0:.0f} s", flush=True)
