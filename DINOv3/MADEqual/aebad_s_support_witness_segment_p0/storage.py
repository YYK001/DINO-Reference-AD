"""Small sealed artifacts; modified files are rejected, never re-enrolled."""
import json
from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_gap_soft_segment_p0.inputs import sha,digest,read_json,write_csv as _write_csv


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');_write_csv(tmp,rows);tmp.replace(path)


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def seal(path, identity, arrays, metadata, *, compressed=True):
    path=Path(path);record=path.with_suffix('.json')
    if path.exists() or record.exists():raise FileExistsError('uncommitted/existing artifact preserved: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pending')
    with tmp.open('xb') as f:(np.savez_compressed if compressed else np.savez)(f,**arrays)
    tmp.replace(path)
    write_json(record,dict(identity=identity,sha256=sha(path),metadata=metadata))
    return read_json(record)


def restore(path,identity,metadata=None):
    path=Path(path);record=path.with_suffix('.json')
    if not path.exists() and not record.exists():return None
    if not path.is_file() or not record.is_file():raise ValueError('incomplete artifact preserved: '+str(path))
    r=read_json(record)
    if r['identity']!=identity or sha(path)!=r['sha256'] or (metadata is not None and r['metadata']!=metadata):
        raise ValueError('modified/incompatible artifact: '+str(path))
    return r


def arrays(path,identity,metadata=None):
    r=restore(path,identity,metadata)
    if r is None:raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    return data,r


def initialize(out,bundle):
    out=Path(out).resolve()
    for p in bundle.protected_paths:
        p=Path(p).resolve()
        if out==p or out in p.parents or p in out.parents:raise ValueError('output overlaps historical input: '+str(p))
    if out.drive and out.drive.upper()!='G:':raise ValueError('Windows artifacts must stay on G:')
    m=dict(identity=bundle.identity,frozen=bundle.frozen)
    if out.exists():
        if read_json(out/'input_manifest.json')!=m:raise ValueError('output belongs to another input/code configuration')
    else:
        out.mkdir(parents=True);write_json(out/'input_manifest.json',m);write_json(out/'protocol.json',bundle.protocol)
        write_json(out/'status.json',dict(identity=bundle.identity,status='initialized_no_predictions',all_metrics_complete=False))
    return out
