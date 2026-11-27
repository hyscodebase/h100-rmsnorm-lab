# Audit-rejected H100 checkpoint

The s53-w4102 kernel execution completed, but the checkpoint failed the newly added evidence audit because the audit incorrectly applied a four-backend row count to the 32 fixture/backend matrix. The next-submission path also did not rerun the evidence audit on a later invocation. These files are preserved for provenance and excluded from final analysis. Rerun s53-w4102 under the corrected pinned source before accepting measurements.
