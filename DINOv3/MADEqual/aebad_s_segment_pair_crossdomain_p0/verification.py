"""Historical score compatibility; never relax same-run numerical checks."""
from __future__ import annotations

import numpy as np
from . import inputs as ip
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check


def historical_score(rows, sample, kind, actual, expected):
    """Keep both the original strict result and the fixed migration admission."""
    actual, expected = float(actual), float(expected)
    finite = bool(np.isfinite(actual) and np.isfinite(expected))
    delta = actual - expected if finite else None
    strict = finite and abs(delta) <= ip.PROTOCOL['score_atol']
    accepted = check(rows, sample['sample_id'], kind, actual, expected,
                     atol=ip.PROTOCOL['historical_score_atol'], rtol=0,
                     domain=sample['domain'], historical_score_check=True,
                     actual_score=actual if np.isfinite(actual) else None,
                     historical_score=expected if np.isfinite(expected) else None,
                     signed_error=delta, strict_atol=ip.PROTOCOL['score_atol'],
                     strict_verified=bool(strict),
                     acceptance_policy=ip.PROTOCOL['historical_score_policy'])
    rows[-1]['compatibility_warning'] = bool(accepted and not strict)
    if not finite:
        rows[-1]['max_abs_error'] = None
    return accepted


def _true(value):
    return value is True or str(value).lower() in ('true', '1')


def audit(rows):
    selected = [r for r in rows if _true(r.get('historical_score_check', False))]
    errors = [float(r['max_abs_error']) for r in selected if r.get('max_abs_error') not in (None, '')]
    strict_failed = sum(not _true(r.get('strict_verified', False)) for r in selected)
    return dict(policy=ip.PROTOCOL['historical_score_policy'],
                accepted_atol=ip.PROTOCOL['historical_score_atol'],
                original_strict_atol=ip.PROTOCOL['score_atol'], rtol=0,
                checks=len(selected), strict_failed_checks=strict_failed,
                compatibility_warning_checks=sum(_true(r.get('compatibility_warning', False)) for r in selected),
                rejected_checks=sum(not _true(r['verified']) for r in selected),
                max_abs_error=max(errors, default=None),
                historical_strict_reproduction=bool(selected) and strict_failed == 0,
                interpretation='compatibility acceptance does not prove identical distances, rankings or metrics')
