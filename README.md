# dsv41: DeepSeek-V4.1-Flash-1017 on a DGX Station GB300

An OpenAI-compatible inference server for DeepSeek-V4.1-Flash-1017 on one DGX Station GB300 (GB300 with 252 GB HBM,
plus the station's RTX PRO 6000 for the experts that do not fit in HBM), built on vLLM 0.32 and the ds41 kernels.
Thinking, tool calls, DSpark speculative decoding and contexts up to 1M tokens are supported.

The default profile is lossless. It runs at the official precision (FP4 experts, FP8 and BF16 elsewhere), and its
logits match the official inference code on the same weights, with KLD at the run-to-run noise floor. Lossy options
are separate profiles. Each one lists the speed it buys and the KLD it adds against that reference.

## What is in here

| path | what |
|---|---|
| `dsv41_engine/runner.py` | whole-step runner: every decode step (input prep, target forward, LM head, rejection, drafter) replays one captured CUDA graph per (batch, draft width, sampler branch) |
| `dsv41_engine/conf_width.py` | per-step verify width from the current drafts' confidences. At batch 1 the runner replays a graph of exactly that many drafts (one per width); at batch 2 and up each request's rows beyond its width are masked inside the graph, skipping their experts. Stops at the first position it refuses, so the output distribution is unchanged |
| `dsv41_engine/replay_skip.py` | intermediate prefill chunks skip the decoder replay nothing reads |
| `dsv41_engine/prefill_indices.py` | fixes a vLLM 0.32 bug: sparse prefill read stale index rows, which put NaN into short requests batched with a long one |
| `dsv41_engine/dynk.py` | ds41's DynK estimator (5 or 7 drafts per batch-1 step); off in the shipped profiles, which use `conf_width.py` |
| `dsv41_engine/suffix.py` | copy (suffix) drafting: when the request's history repeats, batch-1 steps verify 31 copied tokens (one-hot draft distribution, so the output distribution is unchanged) |
| `dsv41_engine/{ll_gemm,dflash_prep,fp8_head,capture,worker}.py` | low-latency MXFP8 GEMMs (pinned tactics at M <= 8; FlashInfer's low-latency kernel compiled with 16/32-row tiles for 8 < M <= 32, after vLLM #63921), drafter input prep, the optional FP8 verify head, capture helpers, the vLLM worker |
| `kernels/vendor/ds41/fastpath/hook/rowmap-warm-v1.json` | the expert split: 256 hot experts per layer in GB300 HBM, 128 cold on the RTX, 24 of the cold ones also in HBM (warm: prefill, and the router's per-call redirect) |
| `kernels/vendor/ds41` | ds41's kernels and vLLM components (MIT) with our vLLM 0.32 ports and changes; `VENDORED.md` lists every change |
| `kernels/vendor/b12x` | b12x (Apache-2.0), used by the RTX sidecar |
| `build.sh` | builds the kernels inside the vLLM 0.32 image and assembles `runtime/` |
| `dsv41.sh` | launcher: the RTX sidecar, then the server; `profiles/*.env` select the options |
| `harness/` | the captured-step harness and the decode-path KLD tool used for every measurement |
| `bench/` | correctness checks: request order (`order_check.py`), the request options (`api_check.py`), top-p inside batch-2 graphs (`topp_check.py`) |

## Requirements

- DGX Station GB300 (GB300 + RTX PRO 6000), Docker with the NVIDIA Container Toolkit's CDI devices
  (`docker info | grep -A6 'Discovered Devices'`). `dsv41.sh` finds the two GPUs by name; set `GB` and `RTX` to their CDI
  UUIDs to override it.
- Images `vllm/vllm-openai:v0.32.0-aarch64` (server) and `vllm/vllm-openai:v0.31.0` (sidecar).
- The checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `MODEL` (a directory of real files).

## Run

    ./build.sh                       # once: kernels + runtime/ (a few minutes)
    MODEL=/path/to/DeepSeek-V4.1-Flash-1017 ./dsv41.sh up            # default (lossless) profile, port 8106
    curl http://127.0.0.1:8106/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model": "dsv41", "messages": [{"role": "user", "content": "Hello"}]}'
    ./dsv41.sh down

The first boot JIT-compiles kernels (cached under `STATE`); later boots take about 8 minutes.

## Profiles and measured speed

Served on one DGX Station GB300 through `dsv41.sh`, vLLM 0.32, decode tok/s per stream unless marked aggregate (agg).
"fixed" and "coding" are greedy 2,048-token generations from a 2.6K / 4.4K-token prompt (median of 3); T=1 GSM8K is
sampled reasoning; prose is bench.py's ten prompts. Single runs of the c=1 shapes vary by about +-5% (the per-step draft
width depends on timing). ds41 (the dgx-station-recipes DeepSeek-V4.1-Flash stack, v0.31) on the same box for reference.

| shape | default (lossless) | `fast` | ds41 |
|---|---|---|---|
| decode KLD | 0.021 | 0.034 | |
| fixed, c=1 | 737 | 769 | 734 |
| coding, c=1 | 787 | 893 | 810 |
| T=1 GSM8K, c=1 | 598 | 659 | 501 |
| T=1 GSM8K, c=4 | 1,044 agg | 1,168 agg | 1,000 agg |
| prose, c=1 / c=2 / c=4 | 437 / 588 agg / 851 agg | 470 / 629 agg / 949 agg | 405 / 546 agg / 759 agg |
| prefill 2K / 8K / 32K / 128K | 24.1K / 48.6K / 60.7K / 60.0K tok/s | 23.4K / 48.7K / 61.6K / 60.2K tok/s | 15.4K / 33.7K / 38.2K / 36.2K tok/s |

Release check of both profiles (2026-11-06): QA 6/6; tool calls 2/2 with thinking on and off; needle retrieval at 129K /
511K / 975K tokens, 6/6; a request-order check (the same greedy prompts after longer and shorter ones, at c=1 and c=4 among
3K and 30K prompts) with no changed answers; the request options below; GSM8K, 200 questions at c=4: default 194 greedy /
195 at T=1, `fast` 194 / 193, stock vLLM 0.32 on the same questions 195 / 196.

Request options: temperature, top_p, top_k, seed, stop, max_tokens, min_tokens, n, logprobs, penalties, JSON schema,
streaming and thinking all work, also mixed in one batch. vLLM refuses `min_p` and `logit_bias` with speculative decoding.

Decode KLD is against the official inference code on the same weights. The official code's own batch-size dependence is
~0.028 (its batch-2 against its batch-16 runs), so the default sits at the noise floor.

Lossy options, each priced against that reference:

| option | step time (batch 1) | decode KLD added | in a profile |
|---|---|---|---|
| singleton drop-2: a verify row's two lowest expert picks are dropped when no other row of the step uses those experts | -10% | +0.012 | `fast` |
| singleton drop-1: the same for a row's lowest pick only | -3.9% | +0.005 | no |
| MXFP8 verify head: the LM head on the drafter's MXFP8 copy of the shared lm_head | -1.0% | +0.002 | `fast` |
| plain drop-1: every row's lowest pick dropped | -4.9% | +0.011 | no |
| plain drop-2: every row's two lowest picks dropped | -12% | +0.036 | no |
| position-gated routing at K=7 (rows 6-7 restricted to the experts rows 0-5 load) | none | +0.038 | no |

Over the default, `fast` gains on every decode shape: prose +7 to +12%, T=1 +10 to +12%, coding +13%, fixed +4%. Start it with
`./dsv41.sh up fast`.

Prefill is limited by power, not by the kernels: with the RTX installed the GB300 module reads a 1250 W limit (NVIDIA's static
allocation for a 76-300 W add-in card), and sustained MoE runs at about 1.2 GHz. 16K prefill chunks instead of 8K lift 8K-token
prompts by 13% and leave the others about flat, but halve the KV capacity (2.15M tokens instead of 4.24M), so they are not
the default.

## Credits

The kernels and most vLLM components come from ds41 (`kernels/vendor/ds41`, MIT; see its `LICENSE`, `NOTICE` and
`THIRD_PARTY.md`). This engine is built on vLLM (Apache-2.0) and uses b12x (Apache-2.0) in the sidecar.
