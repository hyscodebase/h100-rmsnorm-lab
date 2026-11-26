"""Numerical and output-ownership checks for the two-output BF16 residual RMSNorm."""
import torch
import torch.nn.functional as F

WIDTH = 4096
EPS = 1e-6
RTOL = 0.005
ATOL = 1e-6
FIXTURES = ("random", "zero_rows", "cancellation", "tiny", "bf16_ties")


def eager_reference(x, residual, weight):
    updated = (x.float() + residual.float()).to(torch.bfloat16)
    z = updated.float()
    inverse_rms = torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
    return updated, (z * inverse_rms * weight.float()).to(torch.bfloat16)


def native_reference(x, residual, weight):
    updated = x + residual
    return updated, F.rms_norm(updated, (WIDTH,), weight, eps=EPS)


def oracle(x, residual, weight):
    """FP64 sum rounded to BF16, followed by a separate FP64 reduction and division."""
    updated = (x.double() + residual.double()).to(torch.bfloat16)
    z = updated.double()
    denominator = (z.square().sum(-1, keepdim=True) / z.shape[-1] + EPS).sqrt()
    return updated, (z / denominator) * weight.double()


def rounding_fixture(rows, device):
    # With these exact BF16 inputs, columns 0 and 3 distinguish the two reduction inputs.
    x = torch.ones(rows, WIDTH, dtype=torch.bfloat16, device=device)
    terms = torch.tensor([1/256, 3/256, 1/256, 3/256], dtype=torch.bfloat16, device=device)
    weights = torch.tensor([1, 1.25, 1.5, 1.75], dtype=torch.bfloat16, device=device)
    return x, terms.repeat(rows, WIDTH // 4), weights.repeat(WIDTH // 4)


def make_inputs(name, rows, generator, device):
    if name == "bf16_ties":
        return rounding_fixture(rows, device)
    def normal(*shape):
        return torch.randn(shape, dtype=torch.float32, generator=generator, device=device)

    scale = 1e-5 if name == "tiny" else 1.0
    x = (normal(rows, WIDTH) * scale).to(torch.bfloat16)
    residual = (normal(rows, WIDTH) * scale).to(torch.bfloat16)
    weight = normal(WIDTH).to(torch.bfloat16)
    if name == "zero_rows":
        x[::2].zero_()
        residual[::2].zero_()
    elif name == "cancellation":
        residual[:, ::2] = -x[:, ::2]
    return x, residual, weight


def ensure(condition, reason):
    if not condition:
        raise AssertionError(reason)


def byte_span(tensor):
    # PyTorch strides are nonnegative. Include offsets for suballocations in a shared storage.
    last = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return tensor.data_ptr(), tensor.data_ptr() + (last + 1) * tensor.element_size()


def verify_outputs(outputs, expected, inputs=()):
    """Check both materialized outputs, including their independently consumed sums."""
    expected_updated, reference = expected
    ensure(isinstance(outputs, (tuple, list)) and len(outputs) == 2, "expected_two_outputs")
    spans = [byte_span(x) for x in inputs]
    for output in outputs:
        ensure(isinstance(output, torch.Tensor) and output.shape == reference.shape
               and output.dtype == torch.bfloat16 and output.device == reference.device, "output_contract")
        start, end = byte_span(output)
        ensure(all(end <= lo or hi <= start for lo, hi in spans), "overlapping_output_storage")
        spans.append((start, end))
    updated, normalized = outputs
    ensure(torch.equal(updated.view(torch.int16), expected_updated.view(torch.int16)), "residual_bits")
    values = normalized.double()
    diff = (values - reference).abs()
    allowed = ATOL + RTOL * reference.abs()
    ensure((diff <= allowed).all().item(), "normalized_tolerance")

    updated_sum = updated.sum(dtype=torch.float64)
    normalized_sum = values.sum()
    ensure(torch.equal(updated_sum, expected_updated.sum(dtype=torch.float64)), "consumed_residual_sum")
    ensure((normalized_sum - reference.sum()).abs().item() <= allowed.sum().item(), "consumed_normalized_sum")
    return {"updated_sum": updated_sum.item(), "normalized_sum": normalized_sum.item(),
            "max_abs_error": diff.max().item()}


def check_rounding_entries(normalized, reference):
    # These correctly rounded outputs are at least 0.22 BF16 ULP from the nearest midpoint.
    # Using the unrounded sum instead gives 0.99609375 and 1.7578125 at the same columns.
    rows = [0, 0, -1, -1]
    cols = [0, 3, WIDTH - 4, WIDTH - 1]
    target = normalized.new_tensor([0.9921875, 1.765625, 0.9921875, 1.765625])
    ensure(torch.equal(reference[rows, cols].to(torch.bfloat16), target), "rounding_oracle_drift")
    ensure(torch.equal(normalized[rows, cols].view(torch.int16), target.view(torch.int16)), "rounding_sensitive_bits")


@torch.inference_mode()
def validate_backend(fn, rows, seed, device):
    if type(rows) is not int or rows < 1:
        raise ValueError("Expected positive row count.")
    generator = torch.Generator(device=device).manual_seed(seed)
    fixtures = []
    for name in FIXTURES:
        inputs = make_inputs(name, rows, generator, device)
        expected = oracle(*inputs)
        originals = [x.clone() for x in inputs]
        outputs = fn(*inputs)
        try:
            ensure(all(torch.equal(x.view(torch.int16), copy.view(torch.int16)) for x, copy in zip(inputs, originals)),
                   "mutated_inputs")
            diagnostics = verify_outputs(outputs, expected, inputs)
            if name == "bf16_ties":
                check_rounding_entries(outputs[1], expected[1])
            fixtures.append({"fixture": name, "passed": True, **diagnostics})
        except AssertionError as exc:
            fixtures.append({"fixture": name, "passed": False, "reason": str(exc)})
    failure = next(({"fixture": f["fixture"], "reason": f["reason"]} for f in fixtures if not f["passed"]), None)
    return {"passed": failure is None, "rows": rows, "width": WIDTH, "seed": seed,
            "eps": EPS, "rtol": RTOL, "atol": ATOL, "fixtures": fixtures, "failure": failure}
