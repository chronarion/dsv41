# dsv41: DeepSeek-V4.1-Flash-1017 on a DGX Station GB300

A server for DeepSeek-V4.1-Flash-1017 that speaks the OpenAI API and runs on one DGX Station GB300. Most of the model
lives in the GB300's 252 GB of memory; the experts that do not fit run on the station's second GPU, the RTX PRO 6000.
Thinking, tool calls and contexts up to 1M tokens work. It is built on vLLM 0.32 and the ds41 kernels, and is mostly a
port of [original-el8's work](https://github.com/original-el8/dgx-station-gb300-research/tree/main/deepseek-v4.1-flash).

On Local Inference Lab's standard benchmark it generates about 375 tokens per second for one request, about 1,340 in
total for 8 requests and about 1,900 for 16, and reads prompts at 75,000-76,000 tokens per second.

The default profile gives the same output as DeepSeek's own inference code: the remaining difference is as small as the
difference between two runs of DeepSeek's code. A second profile, `fast`, trades a small, measured difference for
4-17% more speed.

## Requirements

- A DGX Station GB300 (GB300 + RTX PRO 6000) with Docker and the NVIDIA Container Toolkit. `dsv41.sh` finds the two
  GPUs by name; set `GB` and `RTX` to their CDI UUIDs to choose them yourself
  (`docker info | grep -A6 'Discovered Devices'` lists them).
- The images `vllm/vllm-openai:v0.32.0-aarch64` (server) and `vllm/vllm-openai:v0.31.0` (the RTX side).
- The model `deepseek-ai/DeepSeek-V4.1-Flash-1017` as a directory of real files, given as `MODEL`.
- About 46 GB of free CPU memory, for copies of the experts that prompt reading moves onto the GB300
  (`DSV41_WARM_ADAPT=0` turns this off).

## Run

    ./build.sh                                                   # once, a few minutes
    MODEL=/path/to/DeepSeek-V4.1-Flash-1017 ./dsv41.sh up        # default profile, port 8106
    curl http://127.0.0.1:8106/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model": "dsv41", "messages": [{"role": "user", "content": "Hello"}]}'
    ./dsv41.sh down

`./dsv41.sh up fast` starts the `fast` profile. The first start compiles kernels and takes longer; later starts take
about 8 minutes. After `git pull`, run `./build.sh` again before starting.

All the usual request options work, also mixed in one batch: temperature, top_p, top_k, min_p, seed, stop, max_tokens,
min_tokens, n, logprobs, penalties, logit_bias, JSON schemas, streaming and thinking. `reasoning_effort` takes none,
minimal, low, medium, high, xhigh or max (or 1-100). (Stock vLLM 0.32 refuses min_p and logit_bias together with
speculative decoding; see `patches/vllm/README.md`.)

## Speed

Local Inference Lab's [standard benchmark](https://github.com/local-inference-lab/llm-inference-bench) (`lil-bench`,
standard/1), version 1.10, generated tokens per second (8 and 16 requests: all requests together):

| context | default: 1 request | 8 requests | 16 requests | `fast`: 1 request | 8 requests | 16 requests |
|---|---:|---:|---:|---:|---:|---:|
| empty | 378 | 1,338 | 1,933 | 397 | 1,517 | 2,270 |
| 64K tokens | 374 | 1,330 | 1,843 | 388 | 1,480 | 2,085 |
| 128K tokens | 364 | 1,339 | 1,867 | 382 | 1,434 | 2,054 |

Prompt reading: 75,000 tokens per second for a 32K-token prompt (first token after 0.43 s), 76,500 for 128K (1.68 s).
The `fast` columns were measured on 1.9, before prompt reading got faster.

<details><summary>How it was measured, and other request shapes</summary>

Run inside the server container with `python3 -m lil_bench --no-upload --no-p2pmark --url http://127.0.0.1:8106`
(llm-inference-bench 1.7.1; p2pmark skipped because one GPU serves the API), on 2026-11-12. The GPU runs at its power
limit while reading prompts: with the RTX installed the GB300 module reads a 1,250 W limit (NVIDIA's fixed allocation
for a 76-300 W add-in card).

Other shapes, 2026-11-06 (an earlier version; prompt reading has become faster since). Per request unless marked
"total". "fixed" and "coding" are greedy 2,048-token answers to 2.6K / 4.4K-token prompts (median of 3), "T=1 GSM8K" is
sampled reasoning, "prose" ten prose prompts. Single runs at 1 request vary by about 5%. ds41 is the dgx-station-recipes
DeepSeek-V4.1-Flash stack (vLLM 0.31) on the same machine.

| shape | default | `fast` | ds41 |
|---|---|---|---|
| fixed, 1 request | 737 | 769 | 734 |
| coding, 1 request | 787 | 893 | 810 |
| T=1 GSM8K, 1 request | 598 | 659 | 501 |
| T=1 GSM8K, 4 requests | 1,044 total | 1,168 total | 1,000 total |
| prose, 1 / 2 / 4 requests | 437 / 588 / 851 total | 470 / 629 / 949 total | 405 / 546 / 759 total |
| prompt reading, 2K / 8K / 32K / 128K | 24.1K / 48.6K / 60.7K / 60.0K | 23.4K / 48.7K / 61.6K / 60.2K | 15.4K / 33.7K / 38.2K / 36.2K |

Reading prompts in 16K-token pieces instead of 8K makes 8K-token prompts 13% faster and leaves the others about the
same, but halves the room for cached context, so it is not the default.

</details>

## Profiles

How close each profile is to DeepSeek's own code is measured as KLD: the difference between the probabilities the two
assign to the next token, averaged over many tokens of real output (0 means identical). DeepSeek's code differs from
itself by about 0.028 between two batch sizes, so anything near that is noise.

| | default | `fast` |
|---|---|---|
| KLD against DeepSeek's code | 0.021 (noise level) | 0.034 (0.039 at 16 requests) |
| what it changes | nothing | skips some low-weight expert picks while checking drafts, and checks with an 8-bit copy of the output layer |
| speed over default | | +4 to +13% at 1 request, +13% at 8, +10 to +17% at 16 |

<details><summary>Every lossy option that was measured</summary>

Step time at 1 request, and KLD added on top of the default.

| option | step time | KLD added | in a profile |
|---|---|---|---|
| a checked token's two lowest-weight expert picks are skipped when no other token of the step uses those experts | -10% | +0.012 | `fast` |
| the same for the lowest pick only | -3.9% | +0.005 | no |
| the final layer from the drafter's 8-bit copy | -1.0% | +0.002 | `fast` |
| every token's lowest pick skipped | -4.9% | +0.011 | no |
| every token's two lowest picks skipped | -12% | +0.036 | no |
| the last 2 of 8 checked tokens limited to the experts the first 6 use | none | +0.038 | no |

</details>

## Tested with

Every release runs the same checks: a set of questions with known answers, tool calls with thinking on and off, the same
prompts answered alone and among other requests, every request option, 16 long prompts at once, Local Inference Lab's
scheduler tests (bursts, agents, overload), 12 simulated coding agents for 8 minutes, and the standard benchmark. The
1.10 run: all pass, 1,255 agent turns with no error, hang or broken output.

<details><summary>Details</summary>

- `bench/agent_stress.py` simulates coding agents: streamed tool calls with a 30-tool list, real file contents as tool
  results, conversations past 200K tokens that are re-read from the cache every turn, mixed sampling settings and
  `reasoning_effort` values, JSON-schema turns, and streams the client abandons midway. `bench/burst_check.py` sends 16
  different 30K-token prompts at once.
- Release check of both profiles on 2026-11-06: finding a hidden fact at 129K / 511K / 975K tokens, 6 of 6; GSM8K, 200
  questions at 4 requests: default 194 right greedy / 195 sampled, `fast` 194 / 193, stock vLLM 0.32 195 / 196.
- Local Inference Lab's scheduler tests (`--scheduler all`) pass: burst (64 prompts of 30K tokens at once), agents (8
  agents in 40-100K-token conversations for 10 minutes), arrivals, short-under-long, overload (twice the server's
  capacity, up to 992 requests waiting: 2,056 of 2,056 completed), abort-storm and soak-mix. Overload opens about 1,000
  connections, so raise the client's open-file limit first (`ulimit -n 65536`); with the default 1,024 the client's own
  connections fail while the server stays healthy.

</details>

## Options

Set them in the environment when starting, for example `KV_OFFLOAD_GB=32 MODEL=... ./dsv41.sh up`.

| setting | what it does |
|---|---|
| `KV_OFFLOAD_GB=32` | also keeps cached prompts in CPU memory (32 GB holds about 14M tokens), so a long conversation pushed out of GPU memory comes back without being read again: a 40K-token prompt answered in 0.05 s instead of 0.6 s. Speed is otherwise unchanged. |
| `EXTRA_CAPTURE_SIZES=320` or `384` | speeds up short agent turns on long conversations (see below), at 7% or 12% less room for cached context |
| `DSV41_WARM_ADAPT=0` | keeps a fixed set of experts on the GB300 for prompt reading: no 46 GB of CPU memory, and repeatable output (see Reproducibility), but prompt reading about 13% slower |
| `GPU_MEM_UTIL` | the share of GPU memory vLLM may use (default 0.95) |
| `DSV41_WARM=0` | skips the 15-second warm-up after start |

<details><summary>Short agent turns and EXTRA_CAPTURE_SIZES</summary>

A short turn on a long cached conversation is computed in one step together with the last 128 cached tokens. Up to 256
tokens that step runs as a prebuilt GPU step (about 40 ms to the first token); above it, it is assembled on the fly
(about 90 ms), which happens once a turn adds more than about 120 tokens (issue #9). The same happens on the benchmark
when 16 streams restart at 64K context.

| `EXTRA_CAPTURE_SIZES` | turns covered | measured | room for cached context |
|---|---|---|---|
| (none) | up to ~120 new tokens | | 2.84M tokens |
| `320` | up to ~190 | 16 requests at 64K: 1,800 -> 1,900 tokens/s | 2.64M |
| `384` | up to ~250 | turns of 130-250 new tokens: 89 -> 45 ms to the first token | 2.50M |

`EXTRA_CAPTURE_SIZES=320 GPU_MEM_UTIL=0.952` takes the memory from the GPU's spare room instead (2.86M tokens kept)
and passed the full test run, but the benchmark's nine cells added up to the same total as without it. `CAPTURE_SIZES`
replaces the whole list. The server log's 30-second step line times the steps that ran on the slow path, per reason.

</details>

## Reproducibility

With temperature 0, the same request on the same path gives the same text, but different paths can differ. A prompt
found in the cache recomputes only its last block, while a new prompt is read in 8K-token pieces; the different amounts
of work change which GPU computes which expert and the order of some sums, so the probabilities differ in their last
digits and two nearly tied tokens can swap (issue #10). Since 1.10 the experts used while reading a prompt also depend on
the prompts read before it, so a new prompt is not repeatable from run to run either. `DSV41_WARM_ADAPT=0` makes it
repeatable again.

## How it works

- **Speculative decoding.** The model's own small drafter proposes several tokens; the main model checks them in one
  step and keeps the ones it agrees with, without changing what the model would say. How many drafts are checked each
  step follows the drafter's confidence, and when a conversation repeats itself, up to 31 tokens are copied from its
  history.
- **One prebuilt GPU step.** Each generation step (checking the drafts, choosing tokens, drafting the next ones) runs
  as a single prebuilt CUDA graph instead of hundreds of separate GPU calls, for up to 8 requests (16 in `fast`).
- **Two GPUs.** The 256 most used experts of each layer live on the GB300; the other 128 run on the RTX PRO 6000. When
  reading a prompt, 48 of those also get a copy on the GB300, chosen from the experts the current prompt uses most,
  so the GB300 rarely waits for the RTX.

<details><summary>Changes by version</summary>

- 1.1 fixed crashes when several long prompts or more than about 20 requests arrived together.
- 1.3 splits chat prompts at the template's special tokens, remembers recent pieces and tokenizes long new pieces on
  several CPU threads (the same token ids): a repeated 128K prompt is ready in 0.13 s instead of 0.33 s; prompt
  reading +7-8%.
- 1.4 frees prebuilt steps that have not been used for a while, so traffic that differs from the first requests a
  server saw is not left on the slow path (issue #5: 31% of steps had used them). They share a
  6 GB budget (`DSV41_WS_POOL_GB`).
- 1.5 keeps the copies of cold experts on the layers that read the whole prompt (+11-12% prompt reading) and stopped
  limiting draft counts at 7-8 requests, where it saved nothing (+4-6% at 8 requests). The prebuilt steps got their own
  memory pool with 2 GB set aside.
- 1.6 stopped decoding the whole prompt back to text before a request's first token (46 ms at 128K): +2-3% prompt
  reading, and a repeated 128K prompt answers in 0.06 s.
- 1.7 stopped freeing unused prebuilt steps by default (`DSV41_WS_IDLE_S=0`): three crashes in testing followed freed
  steps. New ones are not built while less than 1 GB of GPU memory is free.
- 1.8 builds every step on one CUDA stream (issue #7). PyTorch reuses freed memory only on the stream that freed it, so
  before, no step could reuse another's memory and the pool grew with every new one; now about 100 fit in the 2 GB
  reserve. Freeing them was tested clean again; it stays off because the pool rarely fills (`DSV41_WS_IDLE_S=120`
  turns it on; `DSV41_WS_ONE_STREAM=0` restores the old way). Steps also keep their speed with `KV_OFFLOAD_GB` (issue #8).
- 1.9 lets `fast` use prebuilt steps up to 16 requests (+11 to +20% at 16).
- 1.10 lets the copies on the GB300 follow the prompt being read: +14-16% prompt reading.

</details>

## Layout

- `dsv41.sh` starts the server (with `profiles/*.env` choosing the options); `build.sh` builds the kernels once.
- `dsv41_engine/` is the engine: the prebuilt steps, speculative decoding and the vLLM fixes.
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

Ideas from published work:

- Speculative decoding, where a small drafter proposes tokens and the model checks several at once without changing
  the output distribution: Leviathan, Kalman & Matias, [Fast Inference from Transformers via Speculative
  Decoding](https://arxiv.org/abs/2211.17192), and Chen et al., [Accelerating Large Language Model Decoding with
  Speculative Sampling](https://arxiv.org/abs/2302.01318).
- Checking the drafts as a block rather than one token at a time (`rejection_sample_method: block`): Sun et al.,
  [Block Verification Accelerates Speculative Decoding](https://arxiv.org/abs/2403.10444).
- The drafter, and choosing how many drafts to check each step from its confidence head: Cheng et al.,
  [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
  Generation](https://arxiv.org/abs/2607.05147).
- Copy drafting from the request's own history: Oliaro et al., [SuffixDecoding](https://arxiv.org/abs/2411.04975).
- Keeping frequently used experts on the fast device and the rest elsewhere: [PowerInfer](https://arxiv.org/abs/2312.12456),
  [Fiddler](https://arxiv.org/abs/2402.07033) and [HybriMoE](https://arxiv.org/abs/2504.05897). Moving experts between
  the devices at run time from the current workload, as the warm slots do since 1.10, is also the idea of
  [DALI](https://arxiv.org/abs/2602.03495).
- The `fast` profile's dropped expert picks, chosen per batch: Gupta et al., [Lynx: Enabling Efficient MoE Inference
  through Dynamic Batch-Aware Expert Selection](https://arxiv.org/abs/2411.08982).
- Prefetching the shared expert's weights while the GPU waits on something else (`DSV41_SHARED_PF`): Yüzügüler,
  Zhuang & Cavigelli, [PRESERVE](https://arxiv.org/abs/2501.08192).
