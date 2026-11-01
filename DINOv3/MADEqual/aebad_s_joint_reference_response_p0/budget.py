"""The requested 12-hour consent boundary; no RAM/VRAM/disk admission checks."""
import time
from .inputs import st


def elapsed(a):
    return sum(float(st.read_json(p)['seconds']) for p in (a.run_root/'costs').glob('*.json'))+time.perf_counter()-a._started


def observe(a):
    if not a.allow_over_12h and elapsed(a) > 43200:
        st.write_json(a.run_root/'budget_pause.json', dict(status='pending_user_12h_consent', elapsed_seconds=elapsed(a)))
        raise RuntimeError('12h budget reached after committed unit; resume only with explicit --allow-over-12h authorization')


def launch_check(a):
    path = a.run_root/'time_estimate.json'
    r = st.read_json(path)
    phases = r.get('phases', {})
    scopes = dict(capture={'query_forwards','Source_retrieval','Target_retrieval','model_startup',
                          'Source_resident_startup','Target_resident_startup','cached_query_feature_IO','raw_image_IO','split_distance_IO','shared_Source_IO'},
                  predict={'four_native_maps','native_IO'})
    requested = {k:v for k,v in phases.items() if not (a.stage == 'all-cpu' and k == 'AUPRO_jobs')
                 and (a.stage not in scopes or k in scopes[a.stage])}
    if r.get('scope') != 'fixed_normal_asset_smoke' or not requested or any(v['seconds'] is None for v in requested.values()):
        raise RuntimeError('full launch needs fixed normal asset smoke; unknown phases are not zero; see time_estimate.json')
    projected = sum(v['seconds'] for v in requested.values())
    if not a.allow_over_12h and elapsed(a)+projected > 43200:
        raise RuntimeError('projected full cost exceeds12h; explicit user consent is required before launching')
