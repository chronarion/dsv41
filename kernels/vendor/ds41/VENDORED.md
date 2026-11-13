# Vendored from garnermccloud/dgx-station-recipes (deepseek-v4.1-flash), MIT

Source: https://github.com/garnermccloud/dgx-station-recipes, commit 8043a68 (cloned 2026-11-04), copyright 2026 Garner
McCloud, MIT (LICENSE here; third-party notices in NOTICE and THIRD_PARTY.md). Kernel sources (kernels/w1-attn,
w3-preattn, w4-boundary, w6-moe, w8-persist; fastpath/peer3 csrc + py), the Python fast-path wrappers that document
each kernel's call interface (fastpath/*.py), and the MegaMoE fork (fastpath/megamoe_fork, DeepGEMM MIT).
Used as generation-1 components of our decode-step executor; each is replaced or modified behind
reference-class gates, with changes noted per file.

## Local modifications
- C0a (2026-11-04): a missing RTX acknowledgement is fatal (`printf` + `__trap()`) instead of silently dropping the
  cold-expert rows of that call: `peer3/csrc/peer3_gb300.cuh` `wait_done`, `w4-boundary/src/csrc/boundary.cu`
  `p3_wait_ack`, `fastpath/megamoe_fork/.../sm100_fp8_fp4_mega_moe.cuh` `w6p3::wait_done` and the fold-hint poll.
  `-DDS41_P3_TIMEOUT_DROP` restores the upstream behaviour. The spin bounds (PEER3_SPINS, default 2e7 polls) only
  expire when the sidecar is dead, so a trap (the process loses its CUDA context and the server restarts) is the
  only outcome that cannot serve wrong logits.
- C0c (2026-11-04): the routers' sqrt-softplus and weight normalisation follow the official gate (torch softplus
  `log1p(exp(l))` with threshold 20, IEEE sqrt; `w / (sum + 1e-20) * route_scale` with IEEE division) instead of
  vLLM's `dsv4_topk` recipe (ex2.approx, Triton logf, sqrt.approx.ftz, `div.full(route_scale, sum) * w`):
  `w4-boundary/include/w4/route_core.cuh` (`sqrt_softplus`, new `route_weight`, `topk6_warp`),
  `include/w4/stage_router.cuh`, `src/csrc/router.cu` (also used by `draft_router.cu`). The summation order of the six
  picks is unchanged (the allowed deviation). `-DDS41_ROUTER_VLLM_MATH` restores the upstream (vLLM-matching) math.
- F9/S12 lossy switch (2026-11-05): `DSV41_TOPK_DROP=1` (env, read once per process by `w4::topk_drop()` in
  `include/w4/route_core.cuh`) makes the decode routers `src/csrc/router.cu` (T <= 16) and `src/csrc/router_mega.cu`
  `mgate_k2` (16 < T <= 64) drop each row's lowest-weight routed expert (first slot on ties) from MegaMoE and the cold
  tier, the other five keeping their top-6 normalisation (`ref_logits.py --topk-drop 1`). Prefill (T > 64) and the
  drafter router are unchanged. Off by default; the kernels take it as an extra launch argument, so the Python ABI
  (`RouterParams`) is unchanged.
- G1 / E1 (2026-11-05): MegaMoE fast dispatch at `kNumExperts % 256 == 0` (our h256 rowmap;
  upstream compiled the whole fast path out there). The fast path's "counts ready" flag lived at
  `expert_token_count[kNumExperts]`, i.e. in the array's 1024-B alignment padding, which does not exist at 256, and
  DeepGEMM's host-side smem size (`get_pipeline_config_for_mega_moe`) has no spare word for a new field. So
  `fastpath/megamoe_fork/.../impls/sm100_fp8_fp4_mega_moe.cuh` keeps the padding word when it exists and otherwise
  borrows (a) the last word of the last dispatch send buffer when the all-token head runs (T <= 16: no token pulls;
  the early shared-L1 task can reach the epilogue before the scheduler's fetch, so smem_d is not safe there) or
  (b) the last word of `smem_d` above T = 16 (the epilogue writes it only for tasks published after the fetch; the
  send buffers hold pulls). Both are zeroed in the prologue. The scheduler (`scheduler/mega_moe.cuh`) takes the flag
  pointer separately; the dispatch warps, which did the counting and synced, no longer spin on it (a later
  epilogue write into smem_d cannot strand them). The drafter's 128-expert call keeps the padding word.
  Fix (2026-11-05, found by the first served run: a DeepGEMM epilogue grid-sync timeout while capturing the T=48
  graph): the scheduler fetched the counts a second time ("Wait dispatch's results") after publishing the shared-L1
  tasks; above T = 16 the flag word lives in smem_d, which those tasks' epilogue may already have overwritten, so the
  second spin could wait forever. Under fast dispatch the second fetch no longer spins (the first one did).
- H6 part 1 (2026-11-05): the fork's routed weight and SFB TMA loads carry EVICT_FIRST for
  decode-sized calls (`fast_dispatch`, T <= 64: each tile is read once per launch), as DeepGEMM 2.8's nvfp4 MegaMoE does
  under kDecodeShaped; prefill-sized calls keep EVICT_NORMAL (tiles reused across token blocks).
- H2(b) (2026-11-05): the W8 woa kernel takes the next GEMM's weight and scales (wo_b:
  `W8oArgs::nxt_w / nxt_w_bytes / nxt_sf / nxt_sf_bytes`, ABI entry 8) and every CTA L2-prefetches its 1/grid slice
  (16-KB `cp.async.bulk.prefetch.L2`) before griddepcontrol.wait, while attention leaves HBM idle; null = off (the
  Python side sets them under `DSV41_WOB_PREFETCH=1`). `w8-persist/oproj/{woa_kernel.cuh,w8oproj_lib.cu}`.
- H2(c) (2026-11-05): `SeamParams` gains `pf_ptr / pf_bytes / pf2_ptr / pf2_bytes` (after `dbg`; the Python struct must
  match, the boot ABI check compares sizes); `seam_kernel`'s last thread L2-prefetches each CTA's slice of them before
  griddepcontrol.wait. The attention-side seam passes the layer's wq_b weight and scales under `DSV41_WQB_PREFETCH=1`.
- H2(d) (2026-11-05, woa chains): `w8o_prepare` also prepares the two-chain variants
  (`woa_kernel<S, 1, 2>`: one CTA x 2 chains; `<S, 2, 2>`: 2 CTAs x 2 chains = split 4), ABI entry 3 advertises split 4,
  and every two-chain variant issues both chains from one warp (`McChains::issue_pair`, ds41's single-issuer
  interleave: ~53 clk per MMA vs ~127 for one chain; two issuer warps hung intermittently in their tests). Selected by
  `DSV41_WOA_SPLIT` / `DSV41_WOA_CHAINS` in the M10 runtime copy of fastpath_w8oproj.py; fixed FP32 order (split-K class).
- G6b (2026-11-05): `fastpath/w9_suffix_kernels.py` gains `_match_chain2_kernel` (bounded grid of 2 x SMs programs that
  stride over the blocks the true length needs: one ticket per program instead of max_model_len / 256; the one-hot q
  rows >= ND written by the chain program only when the next step can be wide, max(lflag before, after) >= trigger,
  with `zero_tok / zero_slot` recording each row's single 0) and `_onehot_lo_kernel` (rows < ND of a copy chain);
  `SuffixDrafter(..., trigger=24)` and `DSV41_SUFFIX_V2` (default 1; 0 = ds41's kernels). Checked against ds41's
  kernels step by step (1,200 steps, identical drafts, flags and q rows).
- vLLM 0.32 ports folded in (2026-11-05),
  every change marked `dsv41` in place: `fastpath_boundary.py` (MoE eligibility on `use_native_mega_moe`, no LL-BF16
  gate requirement; G2 `DSV41_MXFP8_OUT`; E3 `DSV41_MIX_OFFPATH` / `DSV41_MEGA_SMS` run-time knobs; H2(c)
  `DSV41_WQB_PREFETCH` (closed: no gain); H11 router flight recorder `DSV41_ROUTER_REC`), `fastpath_engram_side.py`
  (the Engram module moved in 0.32), `fastpath_preattn.py` + `w3preattn/integrate.py` (the fused qkv stage without the
  step memo, which reads `_fused_layouts_ready`, absent in 0.32; forward_mqa's 17th argument), `fastpath_tail_draft_embed.py`
  (`use_native_mega_moe` routing metadata), `fastpath_tail_markov_fused.py` (0.32 keeps `sample_idx_mapping` int64),
  `fastpath_tail_proj_side.py` (`DSV41_PROJ_SIDE_CAPTURE`: the side-stream projection under graph capture too),
  `fastpath_w8oproj.py` (H2(b) `DSV41_WOB_PREFETCH` (closed), H2(d) `DSV41_WOA_SPLIT` / `DSV41_WOA_CHAINS` (closed:
  defaults unchanged)). Defaults reproduce the served M7 behaviour; the closed experiments stay as off-by-default knobs.
- Runtime pieces vendored for a self-contained release (2026-11-05): `fastpath/hook/mega_peer_hook.py` = the vLLM 0.32
  port of ds41's MegaMoE hot/cold hook, `fastpath/hook/
  rowmap-mix-v1-h256.json` (256 hot experts per layer: MegaMoE's fast dispatch needs a multiple of 256; h255 OOMs this
  RTX at layer 38); `peer3/{sidecar,prebuilt,README.md}` from ds41's `fastpath/peer3` (the RTX sidecar, unchanged; it
  runs on the vLLM 0.31 image as ds41 ships it); `../b12x` = ds41's `docker/b12x` (Apache-2.0, its own LICENSE; the
  sidecar's PYTHONPATH).
- R1 / I1 (2026-11-05; measured no gain, kept off by default): `RouterParams::mega_ready` (appended
  after `dbg`; ABI entry 22): router.cu's leader 0 acquires the other leaders' arrivals and releases 1 into it right
  after its outputs, before the cold-row publish. The MegaMoE fork's C11 block gains `ready`, `flag_mode`,
  `r1_fallbacks`: with flag_mode = 1 (set by fastpath_megamoe.c11_ready under `DSV41_R1=1`) decode-sized calls wait on
  the flag (one poller per CTA, `barrier.cta.red.or` on named barrier 15, fence.proxy.async; the scheduler warp polls
  itself) instead of `cudaGridDependencySynchronize`, fall back to it after 2 ms, and SM 0 clears the flag and waits for
  the router grid at the end. `impls/sm100_fp8_fp4_mega_moe.cuh`, `scheduler/mega_moe.cuh`, `w4-boundary/src/csrc/
  router.cu`, `include/w4/params.h`, `src/csrc/boundary.cu`, `fastpath/{fastpath_boundary,fastpath_megamoe}.py`.
- S41 / S44 / S45 / I10 (2026-11-05, lossy options, all off by default): `DSV41_TOPK_DROP=11 / 12` drop a row's 1 / 2
  lowest picks only where no other non-padding row of the launch picked that expert (router.cu at T <= 16; router_mega.cu
  at 16 < T <= 64 with a per-expert count table in CTA 0's shared memory and one extra cluster barrier); `21 / 22`
  additionally refill each dropped pick with the row's next-ranked expert (ranks 7-14 from the mailbox merge) that
  another row keeps, weighted over the original top-6 sum (router.cu only); `DSV41_PGR=r0` (position-gated routing,
  router.cu at T <= 8, i.e. one request's verify rows): rows >= r0 keep their top-1 and replace picks outside the rows < r0's experts by next-ranked ones inside
  them (the drop mode is ignored then). `include/w4/route_core.cuh` (`topk_drop`, `pgr_r0`), `src/csrc/router.cu`,
  `src/csrc/router_mega.cu`.
- I7 (2026-11-05): `fastpath_boundary.py` `DSV41_SEAM_GSIZE` (`auto`: ceil(T / 8) tokens per 16-CTA seam cluster above
  T = 8, at most 8 clusters; an integer fixes it; default unchanged = 1). Bitwise (a token's math and reduction stay in
  one cluster).
- I19 (2026-11-05): `fastpath_boundary.py` `DSV41_ROUTER_MEGA=1` routes 16 < T <= 64 MoE calls (batch
  3-4 verify steps, the 32-row copy verify) through the W4 router too (`router_mega.cu`, ds41's replica of the stock bf16
  Mega gate, unused by ds41's own `_moe_eligible`), with the gate's image mask; bit-exact with `bf16_mega_gate` +
  `prepare_megamoe_inputs` (864/864 cases). Default off (neutral
  alone); `fast` sets it so the singleton drop also runs above 16 rows.
- J1a / J1b (2026-11-05): warm experts. `fastpath/hook/mega_peer_hook.py`: a rowmap layer
  may list `warm` experts (cold experts with a second copy in MegaMoE rows n_hot..); the module holds hot + warm rows,
  the weight loader fills the warm rows, decode keeps `row_map` (the RTX computes warm experts as before) and the
  prefill path (T > FUSED_MAX) uses `row_map_static` (warm experts hot); `warm_row_map` feeds the router.
  `fastpath_coldskip.py` reads warm experts; `fastpath_a12_retire.py` counts hot + warm rows.
  `fastpath/hook/rowmap-warm-v1.json`: the h256 rowmap plus each layer's 24 most-picked cold experts (corpus counts).
  `fastpath/hook/rowmap-warm-v2.json` (the default since 1.5): the same 960 warm slots, 48 per layer on layers 0-19 and
  none on 20-39 (those layers run only the bounded 128-row replay at prefill).
  `w4-boundary`: `RouterParams` appends `warm_row` (int32[384] MegaMoE row or -1) and `redirect_cap` (ABI entry 23);
  router.cu (T <= 16) collects the launch's distinct cold experts after the drops and, above the cap, moves the first
  warm experts in token order to their MegaMoE rows (outputs written after the pass); router_mega.cu (16 < T <= 64)
  does the same through CTA 0's expert mask (DSMEM `red.or`, two cluster barriers), lowest expert ids first.
  `fastpath_boundary.py`: `DSV41_WARM_CAP="c8,c16"` (T <= 8, T <= 16) and `DSV41_WARM_CAP_WIDE="T:c,..."` (exact widths
  above 16); 0 / unset = off. Checked against a Python reference (990/990 cases).
- J7 / J12 switches (2026-11-05): `fastpath_w8oproj.py` `DSV41_WOA_MAX_T` (engage W8's built 9..16-row woa variants;
  default 8 as ds41); `fastpath_boundary.py` `DSV41_COPY_AHEAD=1` (ds41's COPY_AHEAD option d, default off as ds41).
- B1 / S63 (2026-11-05, after TRT-LLM #21695): shared-expert L2 prefetch from the router's HBM-idle window.
  `w4-boundary`: `RouterParams` appends `shared_pf[4]` / `shared_pf_bytes[4]` / `shared_pf_at` (ABI entry 24);
  `include/w4/route_core.cuh` `shared_pf_issue` (each CTA's 1/grid slice of each range, one `cp.async.bulk.prefetch.L2`
  per range); router.cu and router_mega.cu's K1 issue it before (`shared_pf_at` 1) or right after (2) their
  `griddepcontrol.wait`. `fastpath_boundary.py`: `DSV41_SHARED_PF=0/1/2` (default 0), `DSV41_SHARED_PF_PARTS=all/l1`, the
  layer's MegaMoE shared w13 / w2 and scale tensors as 16-byte-aligned ranges (`_shared_pf_ranges`). Cache policy only.
- C19 / S72 (2026-11-05): `RouterParams.no_drop` (ABI entry 25): 1 = the launch applies no lossy routing
  option (`DSV41_TOPK_DROP` / `DSV41_PGR`); the launchers pass drop 0 then. `fastpath_boundary.py`
  `DSV41_DROP_DECODE_ONLY=1` sets it for every launch outside a whole-step decode capture (`WS_DECODE[0]`, set by the
  engine's runner while it records and captures a graph), so prompt prefill chunks of <= 64 tokens and stock steps
  route exactly. Default off.
- C1 / S68 (2026-11-05, vLLM #61161 backport): `w6-moe/markov/w6mk.cu` and `fastpath/w5_markov.py` key draft step s's
  Gumbel noise by `seed ^ (s << 32)` (vLLM's `spec_step_seed(seed, cache column)`), matching the engine's patched vLLM
  `gumbel.py` / `rejection_sampler_utils.py` (`engine/patches/vllm`), so a position drafted again in a later
  verification round gets fresh noise.
- S82 (2026-11-05): MegaMoE's token loads trimmed to a task's real rows. `sm100_fp8_fp4_mega_moe.cuh`: for
  `LOAD_BLOCK_M >= 32` the token load warp clones `tensor_map_l1_acts` / `l2_acts` per 8-row box into
  `dsv41_trim::desc` (`tensormap.replace.tile.box_dim`, fenced by `ptx::tensor_map_release_gpu` / `acquire_gpu`) and
  partial Linear1 / Linear2 tasks load and expect only their half rows. Output bit-identical (`p1/mega_probe.py`).
- S93 (2026-11-05): prefill combine of `kNumRanks == 1` calls without RTX rows (`num_tokens > 64`, not armed): a warp
  owns whole tokens, one readiness wait, 16-byte `__ldcg` of all slots, same per-channel slot order, so the sum is
  unchanged. S82 + S93: 8K chunk 2.17 -> 1.98 ms per layer, 16K 3.93 -> 3.71, 2083 rows 0.99 -> 0.94; y bit-identical.
- K3 (2026-11-06): `w4-boundary/src/csrc/router.cu`: a padded row (graph padding, or a verify row beyond the confidence
  width) publishes no cold pick, unless all rows of the call are padding (`s_allpad`: dummy batches keep the upstream
  behaviour; without that exception startup stalled ~16 s at T=4 and C0a trapped).
- K3 (2026-11-06): `fastpath_boundary.py` `CW_ROWS` / `CW_MASK`: inside whole-step decode graphs, the row counts the engine's
  `conf_width.py` names take its row mask as the router's padding (rows beyond a request's width route to no expert, hot or
  cold); `router_mega.cu` skips the padded rows' cold picks as `router.cu` does. `DSV41_ROUTER_MEGA_MAX_T` bounds the W4
  router at T > 16 (default 64): the masked rows need it at T 18 and 24, while the 32-row copy verify is faster on the stock path.
- Draft placeholders (2026-11-11, chronarion/dsv41#1): `fastpath_tail_embed.py` gathers row 0 for a negative token id, in
  the Triton gather and in the stock fallback. When a batch shrinks (2 requests drafting 5 to 1 request scheduled for 7),
  vLLM's `-1` draft placeholders reach `input_ids`; the verifier stops at them, but the gather read `weight[-1]`, 10 KB
  before the table (an MMU fault, Xid 31, on driver 620). Outputs unchanged.
- Warm experts that follow the text (2026-11-12, 1.10; `DSV41_WARM_ADAPT=1` in the default profile): new
  `fastpath/hook/warm_adapt.py`; `fastpath/hook/mega_peer_hook.py` keeps the raw weights of every cold expert of the
  warm layers (0-19) on the CPU at load, and in the prefill path (T > FUSED_MAX, outside graph capture, T >= 2048)
  transforms them once into MegaMoE's layout in pinned host memory (self-check against the boot's warm rows), counts
  rows per cold expert each call (one scatter-add, no host sync) and, from the previous call's counts, swaps up to 24
  warm slots per call (weights copied into the slot, prefill row map updated, both on the stream). Decode, the decode
  row map and the RTX sidecar are unchanged. Also `DSV41_COLD_DUMP=dir`: a measurement switch that saves each prefill
  call's cold picks (off by default; syncs every layer).
