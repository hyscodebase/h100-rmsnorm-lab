# Does preserving precision casts restore the compiled residual RMSNorm reference?

**Diagnostic checkpoint, separate from the registered latency campaign.** At 1,024 rows, the original experiment excluded all four compiled spellings on the denominator tie probe. PyTorch 2.7.1 exposes an Inductor flag for emulating intermediate precision casts. Test whether changing that flag changes the compiled outputs on one H100. [Original run](../results/s101-r1024-w4096.json) · [PyTorch 2.7.1 configuration](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/config.py#L532-L541)

Freeze this plan and its code before running. One parent launches two sequential fresh Python processes, first with `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=0`, then with `=1`. Each receives the setting before importing PyTorch, gets private Inductor and Triton cache directories, and records the effective `torch._inductor.config.emulate_precision_casts` value. Both must use the same software versions, numerical sources, H100 configuration and byte-identical BF16 inputs.

Use 1,024 × 4,096 and fresh seed 104. The random control uses the seed. The two deterministic tie inputs retain their previously frozen values. Test the explicit two-output PyTorch function under `torch.compile(fullgraph=True, dynamic=False, mode="default")`; compare to eager execution, the existing fused Triton kernel and the independent rounded-residual FP64 oracle. Keep both BF16 output tensors, check their storage and values, and consume each separately. At four coordinates publish both returned BF16 values and their 16-bit encodings, along with the expected BF16 encodings and the unrounded FP64 normalized values. Require the frozen bits on each tie probe. [Fixture and oracle](../check.py) · [Fused kernel](../kernel.py)

Report **restored on these probes** only if the eager and fused controls pass every case under both settings, the OFF compiled function fails at least one frozen tie probe, and the ON compiled function passes both tie probes and the random control. Publish all other observed outcomes, including a failed control. A runtime, configuration, source or device mismatch invalidates the paired checkpoint. If the ON setting restores agreement, preregister a fresh latency comparison using that setting before measuring speed. This checkpoint records numerical outcomes only.

Run once from the checkout with the recorded PyTorch 2.7.1 H100 environment. By default the program writes its complete JSON to stdout. `--output` atomically saves a file after both children finish. The report includes source hashes for this plan, diagnostic, fixture, kernel and installed Inductor configuration. The CLI accepts an optional full Git hash in `BENCH_REVISION` for archival provenance.

```bash
python3 experiments/rmsnorm-residual/precision/probe.py --output precision-checkpoint.json
```
