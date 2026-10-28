"""Explicit two-normal-image benchmark; commits reusable calibration observations."""
import time
from . import inputs as ip,runtime as rt,pipeline as pp
from .inputs import st


def smoke(a):
    searches,_=rt.references(a,source_only=True)
    rows=[r for r in ip.catalog(a)['source'] if r['calibration']][:2];times=[]
    for i,r in enumerate(rows,1):
        t=time.perf_counter();existing=(a.cache_root/'calibration_observations'/f'{i:03d}'/'manifest.json').exists()
        pp.raw_record(a,r,searches,i);elapsed=time.perf_counter()-t
        times.append(dict(image=r['relative_path'],seconds=elapsed,reused=existing))
    fresh=[r['seconds'] for r in times if not r['reused']]
    value=dict(status='normal_Source_calibration_path_only',records=times,GT_used=False,training=False,
        reusable_by='calibrate',mean_seconds=sum(fresh)/len(fresh) if fresh else None,
        unknown=['full Source+K8 Top16 query runtime','AUPRO/CPU on current device','origin recovery if not already committed'],
        warning='Source-only timing cannot be reported as full formal-query throughput')
    st.write_json(a.run_root/'smoke_report.json',value);return value
