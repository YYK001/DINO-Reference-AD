"""Read-only file/header discovery. Archive indexes are hints; SHA is authoritative."""
from __future__ import annotations
from collections import defaultdict
import os
from pathlib import Path
import tarfile
from .inputs import safe

PRUNE={'.git','.sync','.codex','__pycache__','datasets','external_baselines','_external'}


def suffix_match(name,required):
    name=safe(name.rstrip('/'))
    if name in required:return name
    offset=name.rfind('/raw_cache/')
    candidate=name[offset+1:] if offset>=0 else ''
    return candidate if candidate in required else None


def discover(required,roots,cache_roots=(),archives=(),previous_index=None):
    required={r['relative_path']:r for r in required}
    hits=defaultdict(list);packages=set(Path(p).resolve() for p in archives);searched=[]
    old_index={r['path']:r for r in (previous_index or {}).get('archives',[])}
    for cache in cache_roots:
        for relative in required:
            path=Path(cache)/relative
            if path.is_file():hits[relative].append(dict(storage='file',path=str(path.resolve()),member=''))
    for root in roots:
        root=Path(root).resolve();searched.append(str(root))
        if not root.exists():continue
        for folder,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d not in PRUNE]
            for name in files:
                p=Path(folder)/name
                if name.endswith('.npz'):
                    rel=suffix_match(p.relative_to(root).as_posix(),required)
                    if rel:hits[rel].append(dict(storage='file',path=str(p.resolve()),member=''))
                elif name.endswith(('.tar','.tar.gz','.tgz')) and any(x in name.lower() for x in ('aebad','segment_pair','illumination','view')):
                    packages.add(p.resolve())
    index=[]
    for path in sorted(packages):
        entry=dict(path=str(path),size=None,mtime_ns=None,matches=[],status='indexed')
        try:
            stat=path.stat();entry.update(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
            previous=old_index.get(str(path))
            if previous and all(previous.get(k)==entry[k] for k in ('size','mtime_ns')) and previous.get('status')=='indexed':
                entry['matches']=previous['matches'];entry['index_reused']=True
            else:
                with tarfile.open(path,'r:*') as archive:
                    seen=set()
                    for m in archive:
                        if not m.isfile():continue
                        rel=suffix_match(m.name,required)
                        if rel:
                            if rel in seen:raise ValueError('ambiguous duplicate capture members')
                            seen.add(rel);entry['matches'].append(dict(relative_path=rel,member=m.name))
            for r in entry['matches']:
                if r['relative_path'] in required:hits[r['relative_path']].append(dict(storage='tar',path=str(path),member=r['member']))
        except (OSError,ValueError,tarfile.TarError) as exc:
            entry['status']='index_failed';entry['error']=f'{type(exc).__name__}: {exc}'
        index.append(entry)
    found=[];missing=[]
    for relative,record in required.items():
        candidates=list({(r['storage'],r['path'],r['member']):r for r in hits[relative]}.values())
        if candidates:
            # A named source is never accepted by name alone; preflight decodes/hash-checks it.
            selected=sorted(candidates,key=lambda r:(r['storage']!='file',r['path'],r['member']))[0]
            found.append(dict(**record,**selected,candidate_count=len(candidates),candidates=candidates))
        else:missing.append(dict(**record,status='missing_raw_capture',reason='original eight-support capture not found in searched directories/archive indexes'))
    return found,missing,dict(search_roots=searched,archives=index,
        note='header index only; complete SHA/schema checked separately; no extraction, model or server access')


def blobs(found):
    grouped=defaultdict(list)
    for r in found:
        if r['storage']=='file':yield r,Path(r['path']).read_bytes()
        else:grouped[r['path']].append(r)
    for path,rows in grouped.items():
        needed={r['member']:r for r in rows};seen=set()
        with tarfile.open(path,'r|*') as archive:
            for member in archive:
                if member.name not in needed:continue
                if member.name in seen or not member.isfile():raise ValueError('ambiguous selected raw member')
                seen.add(member.name)
                yield needed[member.name],archive.extractfile(member).read()
        if seen!=set(needed):raise FileNotFoundError('indexed captures no longer present: '+path)
