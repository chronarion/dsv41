# dsv41: DeepSeek-V4.1-Flash-1017 on a DGX Station GB300

An OpenAI-compatible inference server for DeepSeek-V4.1-Flash-1017 on one DGX Station GB300 (GB300 with 252 GB HBM,
plus the station's RTX PRO 6000 for the experts that do not fit in HBM), built on vLLM 0.32 and the ds41 kernels.
Thinking, tool calls, DSpark speculative decoding and contexts up to 1M tokens are supported.

This engine is mostly a port of original-el8's work: [deepseek-v4.1-flash in
original-el8/dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research/tree/main/deepseek-v4.1-flash).

The default profile is lossless. It runs at the official precision (FP4 experts, FP8 and BF16 elsewhere), and its
logits match the official inference code on the same weights, with KLD at the run-to-run noise floor. Lossy options
are separate profiles. Each one lists the speed it buys and the KLD it adds against that reference.

## Requirements

- DGX Station GB300 (GB300 + RTX PRO 6000), Docker with the NVIDIA Container Toolkit's CDI devices
  (`docker info | grep -A6 'Discovered Devices'`). `dsv41.sh` finds the two GPUs by name; set `GB` and `RTX` to their CDI
  UUIDs to override it.
- Images `vllm/vllm-openai:v0.32.0-aarch64` (server) and `vllm/vllm-openai:v0.31.0` (sidecar).
- The checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `MODEL` (a directory of real files).
- About 46 GB of free host memory beyond the sidecar's, for copies of the cold experts that prefill can move onto the
  GB300 (1.10 and later; `DSV41_WARM_ADAPT=0` turns this off).

## Run

    ./build.sh                       # once: kernels + runtime/ (a few minutes)
    MODEL=/path/to/DeepSeek-V4.1-Flash-1017 ./dsv41.sh up            # default (lossless) profile, port 8106
    curl http://127.0.0.1:8106/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model": "dsv41", "messages": [{"role": "user", "content": "Hello"}]}'
    ./dsv41.sh down

The first boot JIT-compiles kernels (cached under `STATE`); later boots take about 8 minutes. After updating the checkout
(`git pull`), run `./build.sh` again: `runtime/` is build output and must match the checkout. Once the server answers,
`dsv41.sh` sends a short warm-up (`bench/warm.py`, about 15 s) shaped like real client requests, so the first live
requests already find their decode-step graphs (`DSV41_WARM=0` skips it).

To keep prompt KV in host memory as well (vLLM's native KV offload), set `KV_OFFLOAD_GB`, for example
`KV_OFFLOAD_GB=32 MODEL=... ./dsv41.sh up`. A prompt pushed out of GPU memory then comes back from host RAM instead of
being computed again: a 40K-token prompt answered in 0.05 s instead of 0.6 s. Decode speed is unchanged (1.8 and later:
the decode-step graphs keep running with the offload hooks around them). 32 GB of host memory holds about 14M tokens.

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
streaming and thinking all work, also mixed in one batch. `reasoning_effort` takes none, minimal, low, medium, high,
xhigh or max (or 1-100); medium sits halfway between low and high. `min_p` and `logit_bias` work too (stock vLLM 0.32
refuses them with speculative decoding; see `patches/vllm/README.md`).


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

## Local Inference Lab benchmark

[Local Inference Lab's](https://github.com/local-inference-lab/llm-inference-bench) standard benchmark (`lil-bench`,
standard/1), default profile, run inside the server container with
`python3 -m lil_bench --no-upload --no-p2pmark --url http://127.0.0.1:8106` (llm-inference-bench 1.7.1; p2pmark skipped:
one GPU serves the API). Measured 2026-11-12 on the release code (1.10).

| | 1 request | 8 requests (agg) | 16 requests (agg) |
|---|---:|---:|---:|
| context 0 | 378 | 1,338 | 1,933 |
| context 64K | 374 | 1,330 | 1,843 |
| context 128K | 364 | 1,339 | 1,867 |

Prefill: 75,035 tok/s on a 32K prompt (time to first token 0.43 s), 76,475 tok/s on 128K (1.68 s). 1.10 lets the 48 warm
expert slots of the prefill layers follow the text: each prefill step counts how many rows every cold expert received and
the next step moves the most-used ones onto the GB300, copying their weights in from host memory. Before, a fixed set
chosen on a tuning corpus left more than half of this text's rows with a cold expert waiting for the RTX; now about one
in twenty: prefill +14-16% (65.9K / 66.1K in 1.8). Decode is unchanged; it never uses the warm slots. 1.5 moved the cold experts that prefill computes on
the GB300 to the layers that prefill runs in full (+11-12%; the RTX was holding the GB300 up on this text), and stopped
capping per-request draft widths at batches 7-8, where the cap saved no work (8 requests +4-6%). 1.6 stopped the
output detokenizer from decoding the whole prompt before a request's first token (46 ms at 128K): +2-3% prefill, and a
repeated 128K prompt answers in 0.06 s.

The `fast` profile (lossy, KLD 0.034; see Profiles) on 1.9: 1 request 397 / 388 / 382 tok/s at context 0 / 64K /
128K, 8 requests 1,517 / 1,480 / 1,434, 16 requests 2,270 / 2,085 / 2,054, prefill 65.5K / 65.3K tok/s. Since 1.9 its
decode-step graphs, and with them its lossy options, cover batches up to 16 instead of 8 (16 requests +11 to +20%; decode
KLD at batch 16 0.039). This needed 1.8's shared graph memory: before it, graphs up to batch 16 ran out of memory under
agent traffic.

The benchmark restarts each stream with the same long prompt, so at 64K-128K part of what it measures is how fast the
server turns the chat request back into tokens. 1.3 tokenizes chat prompts in pieces split at the template's special
tokens, keeps recently seen pieces, and splits a long new piece across CPU threads (`patches/vllm/deepseek_v41.py`;
the token ids are identical). A repeated 128K prompt is ready in 0.13 s instead of 0.33 s, and a new one tokenizes
about 9x faster. Against 1.2: prefill +7-8%, 1 request at 128K +9%, 16 requests at 64K-128K +1-2%.

Its scheduler tests (`--scheduler all`), which exercise admission, mixed prefill and decode, aborts and overload from the
client side, all pass: burst (64 prompts of 30K tokens at once), agents (8 agents in 40-100K-token conversations for 10
minutes), arrivals, short-under-long, overload (an open queue at twice the server's capacity, up to 992 requests waiting:
2,056 of 2,056 completed, drained in 172 s), abort-storm and soak-mix. Overload holds about 1,000 connections open at
once, so raise the client's open-file limit first (`ulimit -n 65536`): with the default 1,024 the benchmark's own
connections fail with `ConnectError` while the server stays healthy. (burst, agents and overload were run on the release
code; arrivals, short-under-long, abort-storm and soak-mix on the 2026-11-11 build before the issue #1-#3 fixes.)

## Agent traffic

`bench/agent_stress.py` runs simulated coding agents against the server: streamed tool calls with a 30-tool list, real file
contents as tool results, contexts that grow past 200K tokens and are re-read from the prefix cache every turn, mixed
sampling settings and `reasoning_effort` values, structured-output turns, and streams the client abandons midway.
`bench/burst_check.py` sends 16 unique 30K-token prompts at once. On the release code: 12 agents for 8 minutes, 1,418 turns, contexts up
to 144K tokens, 55 abandoned streams, no server error, hang or broken output; the burst check 16 of 16 correct. Before the
fixes in 1.1, several long prompts arriving together crashed the server, and so did more than about 20
concurrent requests.

Each decode step at batch 8 or below replays a captured CUDA graph, one per combination of batch size, draft width and
sampling options, and the graphs share a memory budget (`DSV41_WS_POOL_GB`, 6 GB). Until 1.4 the graphs captured first
kept the budget for good, so on a server that first saw tests or benchmarks, live traffic with other sampling settings
ran mostly on the slower uncaptured path (issue #5: 31% of steps replayed). Since 1.4 a graph unused for two minutes
(`DSV41_WS_IDLE_S`) is freed when a new one needs the room, a request shape refused for memory is retried after a
minute, and the server log's step counters name those refusals. Since 1.5 the graphs live in their own memory pool with
2 GB set aside at the first capture (`DSV41_WS_POOL_RESERVE_GB`): long prompts can leave less than 1 GB of device
memory free for hours, and in 1.4 that stopped new captures while idle graphs were still being freed.
1.7 turns the freeing of idle graphs off by default (`DSV41_WS_IDLE_S=0`): three crashes in testing (illegal memory
access) followed freed graphs. Refused shapes are still retried every minute, and new graphs are not captured while
less than 1 GB of device memory is free, so the prefill path keeps the room it needs.

Since 1.8 every graph is captured on one CUDA stream (issue #7). Before, each capture ran on a new stream, and PyTorch
only reuses free memory on the stream that freed it, so no graph could use the memory of another, of a freed graph or of
the reserve: the pool grew with every capture. Now graphs share their scratch memory, about 100 graphs fit in the 2 GB
reserve, and the budget is checked against the pool's real size. Under the forced-eviction test that crashed 1.4 (graph
budget 0.4 GB, graphs freed after 5 s idle) 1.8 ran clean, as it did in shieldstar's tests with 375 evictions. Freeing
idle graphs stays off by default because the pool now rarely fills; `DSV41_WS_IDLE_S=120` turns it on.
`DSV41_WS_ONE_STREAM=0` restores the old behavior for comparison.

A short agent turn on a long cached conversation is computed in one step together with a 128-token replay of its last
cached block. Up to 256 rows that step runs as a captured graph (about 40 ms to the first token); above it, uncaptured
(about 90 ms), which happens once a turn adds more than about 120 tokens (issue #9). The same happens on Local Inference
Lab's benchmark when 16 streams restart at 64K context (steps of 280-285 rows). `EXTRA_CAPTURE_SIZES` adds graph
sizes, at a cost in KV cache, so the default leaves it off:

| `EXTRA_CAPTURE_SIZES` | turns covered | measured | KV cache |
|---|---|---|---|
| (none) | up to ~120 new tokens | | 2.84M tokens |
| `320` | up to ~190 | 16 requests at 64K: 1,800 -> 1,900 tok/s (+5.5%) | 2.64M (-7%) |
| `384` | up to ~250 | turns of 130-250 new tokens: 89 -> 45 ms to the first token | 2.50M (-12%) |

Other cells and prefill are unchanged. `CAPTURE_SIZES` replaces the whole list. Since 1.8 the decode-step graphs use
little of the memory kept outside vLLM's budget, so the graphs can also be paid for from that headroom instead of the KV
cache: `EXTRA_CAPTURE_SIZES=320 GPU_MEM_UTIL=0.952` kept 2.86M tokens of KV and passed the full validation (agent stress
included), but on the standard benchmark the 9 decode cells summed the same as without it (10,745 vs 10,766 tok/s),
so it stays an option. The server log's step-stats line times
the uncaptured steps per reason ("stock steps since the last line").

## Reproducibility

Greedy output (temperature 0) is deterministic for a given path through the server, but not across paths. A prompt
that hits the prefix cache recomputes only its last cached block, while a cold prompt is computed in 8K-token chunks; the
different row counts change which GPU computes which expert and the summation order of some matrix products, so the
logits differ in their last bits and a near-tie between two tokens can flip (issue #10). Since 1.10 the experts that
prefill computes on the GB300 also depend on the prompts prefilled before, so a cold prompt is no longer bit-identical
from run to run either. `DSV41_WARM_ADAPT=0` restores the fixed set of 1.9 and with it run-to-run reproducible cold
prefill.

## Layout

- `dsv41.sh` starts the server (with `profiles/*.env` choosing the options); `build.sh` builds the kernels once.
- `dsv41_engine/` is the engine: the decode-step graphs, speculative decoding and the vLLM fixes.
- `kernels/vendor/` holds the kernels this builds on (ds41, b12x), with every change listed in `VENDORED.md`.
- `bench/` and `harness/` are the correctness checks and measurement tools used for the numbers above.

<details><summary>File by file</summary>

| path | what |
|---|---|
| `dsv41_engine/runner.py` | whole-step runner: every decode step (input prep, target forward, LM head, rejection, drafter) replays one captured CUDA graph per (batch, draft width, sampler branch) |
| `dsv41_engine/conf_width.py` | per-step verify width from the current drafts' confidences. At batch 1 the runner replays a graph of exactly that many drafts (one per width); at batch 2 and up each request's rows beyond its width are masked inside the graph, skipping their experts. Stops at the first position it refuses, so the output distribution is unchanged |
| `dsv41_engine/replay_skip.py` | intermediate prefill chunks skip the decoder replay nothing reads |
| `dsv41_engine/prefill_indices.py` | fixes a vLLM 0.32 bug: sparse prefill read stale index rows, which put NaN into short requests batched with a long one |
| `dsv41_engine/sampler_scratch.py` | fixes a vLLM 0.32 bug: the top-k sampler's scratch buffer was freed when a larger batch replaced it, while graphs captured earlier kept writing into it (dormant with this server's traffic, where the first use already creates the largest buffer) |
| `dsv41_engine/dynk.py` | ds41's DynK estimator (5 or 7 drafts per batch-1 step); off in the shipped profiles, which use `conf_width.py` |
| `dsv41_engine/suffix.py` | copy (suffix) drafting: when the request's history repeats, batch-1 steps verify 31 copied tokens (one-hot draft distribution, so the output distribution is unchanged) |
| `dsv41_engine/{ll_gemm,dflash_prep,fp8_head,capture,worker}.py` | low-latency MXFP8 GEMMs (pinned tactics at M <= 8; FlashInfer's low-latency kernel compiled with 16/32-row tiles for 8 < M <= 32, after vLLM #63921), drafter input prep, the optional FP8 verify head, capture helpers, the vLLM worker |
| `kernels/vendor/ds41/fastpath/hook/warm_adapt.py` | warm experts that follow the text: at prefill, the 48 warm slots of layers 0-19 hold the cold experts the current prompts use most (weights copied in from host memory), so fewer rows wait for the RTX |
| `kernels/vendor/ds41/fastpath/hook/rowmap-warm-v2.json` | the expert split: 256 hot experts per layer in GB300 HBM, 128 cold on the RTX, 48 of the cold ones on layers 0-19 also in HBM for prefill (warm; `rowmap-warm-v1.json`: 24 on every layer) |
| `kernels/vendor/ds41` | ds41's kernels and vLLM components (MIT) with our vLLM 0.32 ports and changes; `VENDORED.md` lists every change |
| `kernels/vendor/b12x` | b12x (Apache-2.0), used by the RTX sidecar |
| `build.sh` | builds the kernels inside the vLLM 0.32 image and assembles `runtime/` |
| `dsv41.sh` | launcher: the RTX sidecar, then the server; `profiles/*.env` select the options |
| `harness/` | the captured-step harness and the decode-path KLD tool used for every measurement |
| `bench/` | correctness checks: request order (`order_check.py`), the request options (`api_check.py`), the chat tokenizer patch (`tokenizer_check.py`), top-p inside batch-2 graphs (`topp_check.py`), several long prompts at once (`burst_check.py`), simulated coding agents (`agent_stress.py`); the boot warm-up (`warm.py`) |

</details>

## Credits

Most of this engine is a port of original-el8's DeepSeek-V4.1-Flash work for the DGX Station GB300
([original-el8/dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research/tree/main/deepseek-v4.1-flash)).
The kernels and most vLLM components come from ds41 (`kernels/vendor/ds41`, MIT; see its `LICENSE`, `NOTICE` and
`THIRD_PARTY.md`). This engine is built on vLLM (Apache-2.0) and uses b12x (Apache-2.0) in the sidecar.

The fixes for crashes under concurrent and agent traffic follow shieldstar's issue reports (#1-#5), including their
root-cause analysis, proposed patches and the warm-up script; the shared graph pool (#7) and graph replay under KV
offload (#8) in 1.8 are their patches. The load and agent tests use Local Inference Lab's llm-inference-bench.
