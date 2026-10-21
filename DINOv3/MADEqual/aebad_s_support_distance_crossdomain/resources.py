"""CPU-only cgroup working-set accounting; no cache dropping or CUDA calls."""
from pathlib import Path
import psutil


def memory_snapshot(root=Path('/sys/fs/cgroup'),system_available=None):
    system=int(psutil.virtual_memory().available if system_available is None else system_available)
    result=dict(system_available_bytes=system,strict_available_bytes=system,
                effective_available_bytes=system,inactive_file_reclaim_credit_bytes=0,
                accounting='min(system_available, cgroup_limit-current+inactive_file); bounded by limit')
    try:
        value=(root/'memory.max').read_text().strip()
        if value!='max':
            limit=int(value);current=int((root/'memory.current').read_text())
            if limit<=0 or current<0:raise ValueError('invalid cgroup memory counters')
            stats={k:int(v) for k,v in (line.split() for line in (root/'memory.stat').read_text().splitlines())}
            # Count only inactive file-backed cache. Active files, anonymous
            # allocations and reclaimable slab receive no speculative credit.
            credit=max(0,min(stats.get('inactive_file',0),stats.get('file',0),current))
            strict=min(system,max(0,limit-current))
            effective=min(system,limit,max(0,limit-current+credit))
            result.update(cgroup_limit_bytes=limit,cgroup_current_bytes=current,
                cgroup_anon_bytes=stats.get('anon'),cgroup_file_bytes=stats.get('file'),
                cgroup_inactive_file_bytes=stats.get('inactive_file'),
                inactive_file_reclaim_credit_bytes=credit,strict_available_bytes=strict,
                effective_available_bytes=effective)
    except (OSError,ValueError):
        # If stat is absent, retain the strict limit rather than pretending the
        # entire system MemAvailable belongs to this container.
        try:
            value=(root/'memory.max').read_text().strip()
            if value!='max':
                strict=min(system,max(0,int(value)-int((root/'memory.current').read_text())))
                result.update(strict_available_bytes=strict,effective_available_bytes=strict)
        except (OSError,ValueError):pass
    try:
        process=psutil.Process();processes=[process,*process.children(recursive=True)]
        result['process_rss_sum_bytes']=sum(p.memory_info().rss for p in processes if p.is_running())
        result['rss_interpretation']='diagnostic sum; shared pages can be double counted; not admission usage'
    except psutil.Error:pass
    return result


def available_ram():
    return memory_snapshot()['effective_available_bytes']
