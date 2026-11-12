# vLLM 0.32 patches (mounted over the image's files by `dsv41.sh`)

Backport of vLLM #61161 ("Avoid reusing random draws across block verification rounds", merged 2026-10-15, after
0.32): block verification and DSpark's drafting noise take a per-draft-step seed, `spec_step_seed(seed, i) = seed ^
(i << 32)`, so a position verified again in a later round gets fresh randomness. Without it vLLM 0.32's sampled (T > 0)
speculative decoding with `rejection_sample_method: block` is not exactly distribution-preserving (vLLM issue #61114;
measured here with the PR's multi-round test: P = 0.419 for a target of 0.5 at the third position).

| file | image path | change |
|---|---|---|
| `gumbel.py` | `vllm/v1/worker/gpu/sample/gumbel.py` | adds `spec_step_seed`; drafting draws (`gumbel_block_argmax` with a logits cache) use the step seed |
| `rejection_sampler_utils.py` | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | block verification's uniforms use the step seed |

Both are the image's files (md5 06738070... / 9ef4edc4...) plus the hunks marked `dsv41 C1`. ds41's fused Markov loop
(`kernels/vendor/ds41/w6-moe/markov/w6mk.cu`) and W5 loop (`fastpath/w5_markov.py`) carry the same step seed.

`deepseek_v41_encoding.py` (`vllm/tokenizers/deepseek_v41_encoding.py`, the image's file plus the hunk marked `dsv41 A1`):
`reasoning_effort` also accepts OpenAI's `minimal` (low's budget, 50) and `medium` (62, halfway between low and high).
vLLM 0.32 accepted only low / high / xhigh / max and an integer, so agent clients that send `medium` got HTTP 400.

`sampling_params.py` (`vllm/sampling_params.py`, the image's file plus the hunk marked `dsv41 A2`): `min_p` and
`logit_bias` are accepted with speculative decoding. vLLM 0.32 refused both up front for every model. The GPU model
runner V4.1 uses applies both to every verified position before rejection sampling, the same way it already applies
top-k, top-p, `min_tokens` and `allowed_token_ids` there, so the output keeps the requested distribution.
`bench/api_check.py` checks it: a +100 bias holds at every position, `min_p` 0.999 at temperature 1 equals greedy, and
no sampled token falls below `min_p`'s bound.

`deepseek_v41.py` (`vllm/tokenizers/deepseek_v41.py`, the image's file plus the hunk marked `dsv41 A3`): the chat path
tokenizes the rendered prompt one special-token-delimited piece at a time, with an LRU cache of pieces (8M tokens),
and encodes a long new piece as parallel chunks (`encode_batch`), cut only where no pre-tokenizer pattern can span the
cut (a space between two ASCII letters, or a newline before an ASCII letter). The tokenizer has an empty normalizer and
no added token strips whitespace, so the ids equal one `encode` of the whole prompt; checked on ~1,400 texts and the
renderer's call arguments. The encoder keeps its own Rust tokenizer: HF's `__call__` leaves truncation set on the shared
one. Requests with `truncate_prompt_tokens` or other arguments take the stock call.

`detokenizer.py` (`vllm/v1/engine/detokenizer.py`, the image's file plus the hunk marked `dsv41 A4`): the incremental
detokenizer's decode stream is primed with the prompt's last 64 tokens instead of all of them. Its first step decoded
everything it was primed with (46 ms at 157K tokens) on the API server's output loop, which serves every stream; the
generated text is the same (checked on 1,672 prompt / continuation pairs, `skip_special_tokens` on and off).

