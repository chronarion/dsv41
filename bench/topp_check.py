"""top_p must hold inside batch-2 decode graphs, whatever request first captured the graph. Needs a fresh server: first a greedy request
beside a T=1 top_p 1.0 request, then a greedy request beside T=1 requests on a high-entropy prompt: top_p 0.01 stays near one output,
top_p 1.0 scatters; greedy on the same prompt is the control for batch numerics.
   python3 -I topp_check.py [URL] [--n 12]"""
import argparse, concurrent.futures as cf, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8106/v1")
ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
LONG = "Write a long, detailed essay about the history of the Roman Empire. Be thorough."
OPEN = "Write fifteen unrelated random English words, separated by spaces, and nothing else."


def chat(prompt, **p):
    req = {"model": model, "messages": [{"role": "user", "content": prompt}], "chat_template_kwargs": {"thinking": False}, **p}
    r = urllib.request.urlopen(urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(),
                                                      {"Content-Type": "application/json"}), timeout=1200)
    return json.load(r)["choices"][0]["message"].get("content") or ""


def beside(temperature=1.0, **p):
    with cf.ThreadPoolExecutor(2) as ex:
        g = ex.submit(chat, LONG, temperature=0, max_tokens=400)
        o = ex.submit(chat, OPEN, temperature=temperature, max_tokens=60, **p)
        g.result()
        return o.result()


with cf.ThreadPoolExecutor(2) as ex:                  # capture the batch-2 graph from {greedy, T=1 top_p 1.0}
    [f.result() for f in [ex.submit(chat, LONG, temperature=0, max_tokens=800),
                          ex.submit(chat, LONG, temperature=1.0, top_p=1.0, max_tokens=800)]]
greedy = {beside(temperature=0) for _ in range(a.n)}     # batch numerics alone: near-ties flip even at temperature 0
tight = {beside(top_p=0.01) for _ in range(a.n)}
loose = {beside(top_p=1.0) for _ in range(a.n)}
ok = len(tight) <= len(greedy) + 2 and len(loose) >= a.n // 2 and len(tight) < len(loose)
print(f"TOP_P CHECK {'PASS' if ok else 'FAIL'}: distinct of {a.n}: greedy {len(greedy)}, top_p 0.01 {len(tight)}, top_p 1.0 {len(loose)}")
