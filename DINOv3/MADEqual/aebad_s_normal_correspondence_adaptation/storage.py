"""Immutable array bundles, stage-local verified mmap, atomic JSON pointers."""
from pathlib import Path
import hashlib
import json
from collections import OrderedDict
import numpy as np
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.storage import (
    sha, digest, read_json, write_json, write_csv)

NEW_ARRAY_BYTES = 0


def seed(*parts):
    blob=json.dumps(parts,ensure_ascii=False,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:16],'little')


def rng(*parts):
    return np.random.Generator(np.random.PCG64(seed(*parts)))


def put(path, arrays, metadata):
    global NEW_ARRAY_BYTES
    path=Path(path)
    if (path/'manifest.json').exists():
        value=get(path)
        if value[1]['metadata']!=metadata: raise ValueError('array inputs changed: '+str(path))
        return value[1]
    path.mkdir(parents=True,exist_ok=True)
    # Uncommitted .npy files are overwritten on resume; committed bundles never are.
    files={}
    for name,value in arrays.items():
        if not name.replace('_','').isalnum(): raise ValueError('invalid array key')
        p=path/(name+'.npy');tmp=p.with_suffix('.pending')
        with tmp.open('wb') as f: np.save(f,np.asarray(value),allow_pickle=False)
        tmp.replace(p)
        files[name]=dict(sha256=sha(p),shape=list(value.shape),dtype=str(value.dtype),bytes=p.stat().st_size)
        NEW_ARRAY_BYTES+=p.stat().st_size
    record=dict(metadata=metadata,files=files);record['seal']=digest(record)
    write_json(path/'manifest.json',record)
    return record


def get(path, *, verify=True):
    path=Path(path);r=read_json(path/'manifest.json')
    if r['seal']!=digest({k:v for k,v in r.items() if k!='seal'}): raise ValueError('modified array manifest')
    out={}
    for name,spec in r['files'].items():
        p=path/(name+'.npy')
        if verify and sha(p)!=spec['sha256']: raise ValueError('modified array: '+str(p))
        x=np.load(p,mmap_mode='r',allow_pickle=False)
        if list(x.shape)!=spec['shape'] or str(x.dtype)!=spec['dtype']: raise ValueError('array layout changed')
        out[name]=x
    return out,r


class Store:
    """Verify each bundle once per stage, never once per minibatch/patch."""
    def __init__(self,max_open=12):
        self.records=OrderedDict();self.verified={};self.max_open=max_open
    def load(self,path):
        path=Path(path)
        if path in self.records:self.records.move_to_end(path)
        else:
            result=get(path,verify=path not in self.verified)
            if path in self.verified and result[1]['seal']!=self.verified[path]:raise ValueError('manifest changed within stage')
            self.verified[path]=result[1]['seal'];self.records[path]=result
            while len(self.records)>self.max_open:self.records.popitem(last=False)
        return self.records[path]


def sealed_json(path,value=None):
    path=Path(path)
    if value is None:
        r=read_json(path)
        if r['seal']!=digest({k:v for k,v in r.items() if k!='seal'}):raise ValueError('modified record: '+str(path))
        return r
    r=dict(value);r['seal']=digest(r)
    if path.exists() and sealed_json(path)!=r:raise ValueError('immutable record differs: '+str(path))
    write_json(path,r);return r
