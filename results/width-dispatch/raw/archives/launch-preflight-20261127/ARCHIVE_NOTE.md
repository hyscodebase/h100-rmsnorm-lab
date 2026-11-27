# Launch preflight receipts

Two pilot submissions were attempted on 2026-11-27 under source revision `c1afffb27ef8428888a73ea456a21dfe3bb771b0`. Neither reached the benchmark or allocated H100 work:

- Attempt `acde0a97cff4400a975e15118b3ac87c` exited 255 because the sandboxed SSH connection closed before remote execution.
- Attempt `7dddd051cb7b48378bd0f3ea579a7c74` exited 127 because the remote wrapper called `python3` before sourcing the GPU environment setup script.

Both receipts and their partial audit/source pin are preserved. The wrapper now starts its remote deadline clock with Bash's built-in `SECONDS` before setup, then invokes Python only after setup is sourced. These launch-only receipts are excluded from the fresh source pin's retry count; no CUDA-event time was recorded.
