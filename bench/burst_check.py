"""Concurrent long-prompt check: N unique long prompts sent at once (several agents starting work together), so the
server prefills several requests in the same steps. Every request must finish with a sensible answer and the server
must stay up. Regression test for the replay-skip crash (a skipped request with zero replay rows sorted first as a
zero-token "decode": ZeroDivisionError in vLLM's sparse indexer, 2026-11-11).
   python3 -I burst_check.py URL [--n 16] [--tokens 30000]"""
import argparse, json, random, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8106/v1")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--tokens", type=int, default=30000)
a = ap.parse_args()
model = json.load(urllib.request.urlopen(a.url + "/models", timeout=30))["data"][0]["id"]
WORDS = "river stone lamp orbit garden copper silent harbor violet engine marble canyon ledger falcon prism meadow".split()
res, lock = [], threading.Lock()


def one(i):
    r = random.Random(i)
    secret = r.randint(10000, 99999)
    words = [r.choice(WORDS) for _ in range(int(a.tokens * 0.75))]
    words.insert(len(words) // 2, f"The secret number is {secret}.")
    body = {"model": model, "max_tokens": 600, "temperature": 0,
            "messages": [{"role": "user", "content": " ".join(words) + "\n\nWhat is the secret number? Reply with the number only."}],
            "chat_template_kwargs": {"thinking": False}}
    t = time.time()
    try:
        req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=1200))
        text = out["choices"][0]["message"].get("content") or ""
        ok = str(secret) in text
        r_ = ("ok" if ok else "wrong", time.time() - t, text.strip()[:60])
    except Exception as e:  # noqa: BLE001
        r_ = (f"error {type(e).__name__}: {e}", time.time() - t, "")
    with lock:
        res.append((i, *r_))


th = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
for t in th:
    t.start()
for t in th:
    t.join()
errors = [x for x in res if x[1].startswith("error")]
wrong = [x for x in res if x[1] == "wrong"]
print(f"burst: {a.n} x {a.tokens} tokens at once: {a.n - len(errors) - len(wrong)} correct, {len(wrong)} wrong, "
      f"{len(errors)} errors; slowest {max(x[2] for x in res):.1f} s", flush=True)
for x in (errors + wrong)[:5]:
    print("  ", x, flush=True)
raise SystemExit(1 if errors else 0)
