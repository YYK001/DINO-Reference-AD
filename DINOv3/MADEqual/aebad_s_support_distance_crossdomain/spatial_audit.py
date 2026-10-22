"""Audit float32 maps separately from ideal float64 stencil descriptions.

The frozen map averages native float32 layers BEFORE half-pixel resize. The
discovery kernel's explanatory stencil sum has different rounding. Its value
is retained, while the strict map audit follows the actual frozen order.
"""
from types import FunctionType
import numpy as np
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic import spatial as base
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check


def map_point(native, pixel, shape):
    """One pixel with exactly numerics.bilinear's float32 operation order."""
    a = np.asarray(native)
    if a.ndim != 2 or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError('finite float32 native map required')
    axes = []
    for position, size, out in zip(pixel, a.shape, shape, strict=True):
        if out <= 0 or not 0 <= position < out:
            raise ValueError('output pixel outside map')
        x = np.maximum(np.float32(0), np.float32(size / out) *
                       (np.asarray([position], np.float32) + np.float32(.5)) - np.float32(.5))
        lower = np.floor(x).astype(np.int64)
        upper = np.minimum(lower + 1, size - 1)
        axes.append((lower, upper, x - lower.astype(np.float32)))
    (y0, y1, fy), (x0, x1, fx) = axes
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bottom = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return float((top * (1 - fy) + bottom * fy)[0])


def instance_analysis(ctx, sample, gt, maps, raw, meta, profiles, checks,
                      write_link, write_stencil):
    # Private globals preserve the shared kernel and concurrent image isolation.
    # Only its final stencil audit is adapted; regions/profiles/peaks are reused.
    native = {}
    for role, key in (('UNION', 'union'), ('SEGMENT', 'segment')):
        values = np.stack([p[key] for p in profiles])
        physical = np.empty_like(values)
        physical[:, raw['positions']] = values
        z = ((physical.reshape(4, *meta['grid']) - ctx['medians'][:, None, None]) /
             ctx['scales'][:, None, None]).astype(np.float32)
        native[role] = z.mean(axis=0)
    pixels = {}

    def emit(row):
        key = row['instance_id'], row['peak_of_branch'], row['region']
        pixels[key] = row['output_y'], row['output_x']
        write_stencil(row)

    def verify(rows, sid, kind, actual, expected, **fields):
        if kind.startswith('stencil_reconstruct_'):
            role = kind.removeprefix('stencil_reconstruct_')
            key = fields['instance_id'], fields['peak_of_branch'], fields['region']
            rounded = map_point(native[role], pixels[key], meta['output_shape'])
            fields.update(
                reconstruction_arithmetic='native_float32_layer_mean_then_float32_bilinear',
                ideal_float64_stencil_response=float(actual),
                ideal_float64_stencil_abs_error=abs(float(actual) - float(expected)),
                ideal_float64_stencil_within_1e5=abs(float(actual) - float(expected)) <= 1e-5,
                ideal_stencil_role='explanatory_rounding_audit_not_frozen_map_arithmetic')
            actual = rounded
        return check(rows, sid, kind, actual, expected, **fields)

    original = base.instance_analysis
    adapted = FunctionType(original.__code__, dict(original.__globals__, check=verify),
                           original.__name__, original.__defaults__, original.__closure__)
    return adapted(ctx, sample, gt, maps, raw, meta, profiles, checks, write_link, emit)
