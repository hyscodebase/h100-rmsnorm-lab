"""Conservative FP32 bounds for normal BF16 residual RMSNorm inputs."""
import torch

UNIT = 2 ** -24
APPROXIMATION_MARGIN = 2 ** -20
EPSILON = 1e-6
WIDTH = 4096
LOW_MAGNITUDE, HIGH_MAGNITUDE = 2 ** -40, 32


def scaled_interval(updated, mean_square):
    """Bound the pre-weight FP32 value on both sides of a BF16 midpoint."""
    gamma = WIDTH * UNIT / (1 - WIDTH * UNIT)
    low_variance = (mean_square * (1 - gamma) + EPSILON * (1 - UNIT)) * (1 - UNIT)
    high_variance = (mean_square * (1 + gamma) + EPSILON * (1 + UNIT)) * (1 + UNIT)
    magnitude = updated.abs()
    magnitude_lo = magnitude / high_variance.sqrt() * (1 - APPROXIMATION_MARGIN)
    magnitude_hi = magnitude / low_variance.sqrt() * (1 + APPROXIMATION_MARGIN)
    return torch.where(updated < 0, -magnitude_hi, magnitude_lo), torch.where(updated < 0, -magnitude_lo, magnitude_hi)


def bf16_bits(tensor):
    return tensor.view(torch.int16).to(torch.int32) & 0xffff


def endpoints(updated, weight, mean_square):
    lo, hi = (end.bfloat16() for end in scaled_interval(updated, mean_square))
    adjacent = (bf16_bits(lo) - bf16_bits(hi)).abs() == 1
    return adjacent, (lo.double() * weight).bfloat16(), (hi.double() * weight).bfloat16()


def assess(updated, weight, expected, reference, candidate):
    """Keep the strict tolerance and verify the two-kernel agreement of every exception."""
    if updated.shape[-1] != WIDTH:
        raise ValueError("Expected the registered width.")
    z, w = updated.double(), weight.double()
    for tensor in (z, w):
        magnitude = tensor.abs()
        domain = (magnitude == 0) | ((magnitude >= LOW_MAGNITUDE) & (magnitude <= HIGH_MAGNITUDE))
        if not bool(domain.all()):
            raise ValueError("Inputs are outside the registered normal FP32 calculation domain.")
    tolerance = 1e-6 + 0.005 * expected.abs()
    old_failures = ((reference.double() - expected).abs() > tolerance) | ((candidate.double() - expected).abs() > tolerance)
    if not bool(old_failures.any()):
        return old_failures, old_failures
    mean_square = z.square().mean(-1, keepdim=True)
    rows, columns = old_failures.nonzero(as_tuple=True)
    adjacent, a, b = endpoints(z[rows, columns], w[columns], mean_square[rows, 0])
    wanted, actual = bf16_bits(reference[rows, columns]), bf16_bits(candidate[rows, columns])
    accepted = adjacent & (wanted == actual) & ((actual == bf16_bits(a)) | (actual == bf16_bits(b)))
    rejected = old_failures.clone()
    rejected[rows, columns] = ~accepted
    return old_failures, rejected
