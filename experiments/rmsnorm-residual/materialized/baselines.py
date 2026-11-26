"""Compiled reference expressions with explicit BF16 rounding semantics."""
import torch

EPS = 1e-6


def rounded_add(x, residual):
    return (x.float() + residual.float()).to(torch.bfloat16)


def normalize_updated(updated, weight):
    operand = updated.float()
    inv_rms = torch.rsqrt(operand.square().mean(dim=-1, keepdim=True) + EPS)
    return (operand * inv_rms * weight.float()).to(torch.bfloat16)


def round_float32(value):
    """Encode finite FP32 operands as BF16, preserving ties to even and signed zero."""
    word = value.view(torch.int32)
    high = (word >> 16) & 0xffff
    low = word & 0xffff
    increment = (low > 0x8000) | ((low == 0x8000) & ((high & 1) != 0))
    rounded = high + increment.to(torch.int32)
    return rounded.to(torch.int16).view(torch.bfloat16), (rounded << 16).view(torch.float32)


def bitcast_reference(x, residual, weight):
    updated, operand = round_float32(x.float() + residual.float())
    inv_rms = torch.rsqrt(operand.square().mean(dim=-1, keepdim=True) + EPS)
    normalized = (operand * inv_rms * weight.float()).to(torch.bfloat16)
    return updated, normalized


def build_compiled(style, mode, explicit, native):
    options = {"mode": mode, "fullgraph": True, "dynamic": False}
    if style == "materialized":
        add = torch.compile(rounded_add, **options)
        norm = torch.compile(normalize_updated, **options)

        def call(x, residual, weight):
            updated = add(x, residual)
            return updated, norm(updated, weight)

        return call
    spellings = {"explicit": explicit, "native": native, "bitcast": bitcast_reference}
    return torch.compile(spellings[style], **options)
