"""The chat-path tokenizer patch (patches/vllm/deepseek_v41.py, dsv41 A3) against the stock tokenizer: the same ids on
source code, prose, CJK, random Unicode, special-token spellings and JSON, at forced small chunk sizes, cached and not,
and through the renderer's call (truncation / max_length / add_special_tokens). Run inside the server container:
   docker cp bench/tokenizer_check.py dsv41:/tmp/ && docker exec dsv41 python3 /tmp/tokenizer_check.py"""
import glob, importlib.util, json, random
from transformers import TokenizersBackend

spec = importlib.util.spec_from_file_location("vllm.tokenizers._dsv41_a3", "/engine/patches/vllm/deepseek_v41.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base = TokenizersBackend.from_pretrained("/model")
wrapped = m.get_deepseek_v41_tokenizer(TokenizersBackend.from_pretrained("/model"))
added = base.get_added_vocab()
ref = lambda t: base.backend_tokenizer.encode(t, add_special_tokens=False).ids
rng = random.Random(7)
texts = [open(f, encoding="utf-8", errors="replace").read()
         for f in sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/**/*.py", recursive=True))[:300]]
texts += [open(f, encoding="utf-8", errors="replace").read()
          for f in sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/**/*.md", recursive=True))[:100]]
texts.append("".join(chr(rng.randint(0x4e00, 0x9fa5)) for _ in range(3000)))
alphabet = "abcXYZ  \n\r\t.,;:!?'\"()[]{}<>-_=+*/\\|0123456789中文日本語ひらがなカタカナéüßøñ€😀́"
texts += ["".join(rng.choice(alphabet) for _ in range(rng.randint(100, 4000))) for _ in range(200)]
specials = list(added)
texts += [rng.choice(specials).join(t[:rng.randint(0, 3000)] for t in rng.sample(texts, 4)) for _ in range(150)]
texts.append(json.dumps({"tools": [{"name": "f", "description": "x " * 500, "parameters": {"a": [1, 2, 3] * 300}}]}))
bad = 0
for chunk in (200, 1000, 32000):
    m._CHUNK_CHARS = chunk
    enc = m._SegmentEncoder(base, added)
    for t in texts:
        for _ in range(2):
            bad += enc.encode(t) != ref(t)
calls = ({"truncation": True, "max_length": 1048577, "add_special_tokens": False}, {"add_special_tokens": False},
         {"add_special_tokens": True}, {"truncation": True, "max_length": 50, "add_special_tokens": False})
for t in texts[:300]:
    for kw in calls:
        bad += wrapped(t, **kw)["input_ids"] != base(t, **kw)["input_ids"]
print(f"tokenizer check: {len(texts)} texts x 3 chunk sizes x 2 passes + 300 x {len(calls)} calls: {bad} mismatches")
raise SystemExit(1 if bad else 0)
