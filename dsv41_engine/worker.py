"""vLLM worker for the dsv41 engine: `vllm serve ... --worker-cls dsv41_engine.worker.DSV41Worker` (with the
engine directory on PYTHONPATH) runs the V2 model runner subclass of dsv41_engine.runner."""
from vllm.v1.worker.gpu_worker import Worker


class DSV41Worker(Worker):
    def _make_model_runner(self):
        import os
        from .runner import WholeStepRunner
        assert self.use_v2_model_runner, "the dsv41 engine extends vLLM's V2 model runner"
        from .prefill_indices import install
        install()                         # stale sparse prefill indices (vLLM 0.32): not an option
        if os.environ.get("DSV41_SUFFIX") == "1":
            from .suffix import install   # G6: the speculator is built inside the runner's constructor
            install()
        if os.environ.get("DSV41_DFLASH_PREP") == "1":
            from .dflash_prep import install   # H8 tail item 1: the drafter's input prep pads in parallel
            install()
        if os.environ.get("DSV41_DYNK") == "1":
            from .dynk import install     # S27: DSpark draft length per decode step (scheduler side)
            install()
        if any(os.environ.get(k, "0") not in ("", "0") for k in ("DSV41_CONF_WIDTH", "DSV41_CONF_WIDTH_B",
                                                                 "DSV41_CONF_HEAD")):
            from .conf_width import install   # K3: batch-1 verify width from the current drafts' confidences
            install()
        if os.environ.get("DSV41_DRAFT_TOPP") == "1":
            from .draft_topp import install   # D7 / S69: drafts from the request's top-p at T > 0
            install()
        if os.environ.get("DSV41_REPLAY_SKIP") == "1":
            from .replay_skip import install   # A4 / S60: no decoder replay for prefill chunks nothing reads
            install()
        if os.environ.get("DSV41_ACC_LOG"):
            from .acc_log import install   # M5: shadow acceptance log for draft shaping (measurement only)
            install()
        return WholeStepRunner(self.vllm_config, self.device)
