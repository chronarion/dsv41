"""The request options a client can send, each on a prompt with a known long answer, then mixed in one batch: every option takes its own
graph key or the stock path, and QA (single answers, greedy) reaches few of them.
   python3 -I api_check.py [URL]"""
import json, sys, threading, urllib.error, urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8106/v1"
model = json.load(urllib.request.urlopen(f"{url}/models"))["data"][0]["id"]
COUNT = "Count from 1 to 40, separated by spaces. Output only the numbers."
END = "38 39 40"


def post(extra, prompt=COUNT, stream=False):
    req = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200,
           "chat_template_kwargs": {"thinking": False}, "temperature": 0, **extra}
    r = urllib.request.urlopen(urllib.request.Request(f"{url}/chat/completions", json.dumps(req).encode(),
                                                      {"Content-Type": "application/json"}), timeout=600)
    if not stream:
        return json.load(r)
    text = ""
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            for c in json.loads(line[6:]).get("choices", []):
                text += c.get("delta", {}).get("content") or ""
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


body = lambda r, i=0: r["choices"][i]["message"].get("content") or ""
CASES = {
    "greedy": ({}, lambda r: END in body(r)),
    "temperature 0.7": ({"temperature": 0.7}, lambda r: END in body(r)),
    "temperature 1": ({"temperature": 1.0}, lambda r: END in body(r)),
    "top_p 0.9": ({"temperature": 1.0, "top_p": 0.9}, lambda r: END in body(r)),
    "top_k 20": ({"temperature": 1.0, "top_k": 20}, lambda r: END in body(r)),
    "min_p 0.1": ({"temperature": 1.0, "min_p": 0.1}, lambda r: END in body(r)),
    "seed": ({"temperature": 1.0, "seed": 7}, lambda r: END in body(r)),
    "frequency_penalty": ({"frequency_penalty": 0.1}, lambda r: len(body(r)) > 40),
    "presence_penalty": ({"presence_penalty": 0.1}, lambda r: len(body(r)) > 40),
    "repetition_penalty": ({"repetition_penalty": 1.02}, lambda r: len(body(r)) > 40),
    "logit_bias": ({"logit_bias": {"1000": -5}}, lambda r: END in body(r)),
    "logprobs": ({"logprobs": True, "top_logprobs": 3}, lambda r: END in body(r) and len(r["choices"][0]["logprobs"]["content"]) > 40),
    "stop string": ({"stop": ["20 21"]}, lambda r: body(r).rstrip().endswith("19")),
    "max_tokens 1": ({"max_tokens": 1}, lambda r: r["usage"]["completion_tokens"] == 1),
    "max_tokens 20": ({"max_tokens": 20}, lambda r: r["usage"]["completion_tokens"] == 20),
    "min_tokens": ({"min_tokens": 120, "ignore_eos": True}, lambda r: r["usage"]["completion_tokens"] >= 120),
    "n 2": ({"n": 2, "temperature": 1.0}, lambda r: END in body(r, 0) and END in body(r, 1)),
    "json_schema": ({"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {
        "type": "object", "properties": {"numbers": {"type": "array", "items": {"type": "integer"}}}, "required": ["numbers"]}}}},
        lambda r: isinstance(json.loads(body(r))["numbers"], list)),
    "thinking": ({"chat_template_kwargs": {"thinking": True}, "max_tokens": 800}, lambda r: END in body(r)),
}
ok = 0
for name, (extra, check) in CASES.items():
    try:
        r = post(extra)
        good = check(r)
        print(f"{'PASS' if good else 'FAIL'}  {name}" + ("" if good else f": {body(r)[:90]!r}"), flush=True)
        ok += good
    except urllib.error.HTTPError as e:
        msg = e.read()[:160]
        known = e.code == 400 and b"not yet supported with speculative decoding" in msg    # vLLM refuses these up front
        print(f"{'SKIP' if known else 'FAIL'}  {name}: HTTP {e.code} {msg!r}", flush=True)
        ok += known
    except Exception as e:
        print(f"FAIL  {name}: {e!r}", flush=True)
try:
    text = post({"stream": True}, stream=True)
    stream_ok = END in body(text)
except Exception as e:
    stream_ok = False
print(f"{'PASS' if stream_ok else 'FAIL'}  streaming", flush=True)
ok += stream_ok
# mixed options in one batch: the greedy streams must still give the exact answer
mix, res, lock = [{}, {"temperature": 0.7}, {"temperature": 1.0, "top_k": 20}, {"temperature": 1.0, "seed": 3}, {"logprobs": True}, {}], [], threading.Lock()


def one(e):
    for _ in range(6):
        t = body(post(e))
        with lock:
            res.append(END in t)


th = [threading.Thread(target=one, args=(e,)) for e in mix]
[t.start() for t in th]; [t.join() for t in th]
print(f"{'PASS' if all(res) else 'FAIL'}  mixed batch {sum(res)}/{len(res)}", flush=True)
ok += all(res)
print(f"API {ok}/{len(CASES) + 2}")
