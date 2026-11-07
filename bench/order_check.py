"""A request's answer must not depend on the request before it: greedy prompts of mixed lengths in an order that puts each one after
longer and shorter neighbours, on both parities of the runner's request count. Fails on a wrong or changed answer or a stalled request.
With --c N the same runs as N concurrent streams, so long prompts prefill between other requests' decode steps.
   python3 -I order_check.py [URL] [--rounds 8] [--long 3000,30000] [--c 1]"""
import argparse, json, random, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8106/v1")
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--long", default="3000,30000")
ap.add_argument("--slow", type=float, default=5.0, help="seconds beyond which a short request counts as stalled")
ap.add_argument("--c", type=int, default=1)
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
QA = [("Count from 1 to 40, separated by spaces. Output only the numbers.", "38 39 40"),
      ("Write the English alphabet in capital letters, separated by spaces. Output only the letters.", "X Y Z"),
      ("List the first 20 even numbers, separated by commas. Output only the numbers.", "38, 40"),
      ("Count down from 30 to 1, separated by spaces. Output only the numbers.", "3 2 1"),
      ("List the twelve months of the year in order, separated by commas. Output only the months.", "November, December"),
      ("List the seven days of the week, starting on Monday, three times over, separated by commas. Output only the days.", "Saturday, Sunday")]
PAD = "The harbour master logged the tide tables, the ferry times and the names of the boats that came in that week. "


def ask(pad, q, tokens=160):
    req = {"model": model, "messages": [{"role": "user", "content": PAD * pad + q}], "temperature": 0, "max_tokens": tokens,
           "chat_template_kwargs": {"thinking": False}}
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=3600))
    return r["choices"][0]["message"].get("content") or "", time.perf_counter() - t, r["usage"]["prompt_tokens"]


short = [(p, i) for i in range(len(QA)) for p in (0, 2, 6, 12)]          # the same question at four prompt lengths
longs = [int(x) // 24 for x in a.long.split(",")]
seen, bad, worst, n, lock = {}, [], [0.0], [0], threading.Lock()


def stream(seed):
    rng = random.Random(seed)
    for _ in range(a.rounds):
        order = short[:]
        rng.shuffle(order)
        for j, (p, i) in enumerate(order):
            if j % 3 == 0:                                               # period 3: what follows a long prompt alternates parity
                ask(rng.choice(longs), QA[i][0], 4)
            text, dt, _ = ask(p, QA[i][0])
            with lock:
                n[0] += 1
                worst[0] = max(worst[0], dt)
                if QA[i][1] not in text or seen.setdefault((p, i), text) != text:
                    bad.append((p, i, round(dt, 2), text[:60]))
                elif dt > a.slow * a.c:                                  # a short request can queue behind others' long prefill
                    bad.append((p, i, round(dt, 2), "stalled"))


th = [threading.Thread(target=stream, args=(s + 1,)) for s in range(a.c)]
[t.start() for t in th]; [t.join() for t in th]
print(f"ORDER CHECK c={a.c} {n[0] - len(bad)}/{n[0]} consistent, slowest short request {worst[0]:.2f} s")
for b in bad[:12]:
    print("  pad %d question %d: %.2f s, %r" % b)
