"""Trusted-local reads by default; payload hashing is an explicit full audit."""
from pathlib import Path
from collections import OrderedDict
import os
import threading
import numpy as np
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.storage import (
    seed,rng,sha,digest,read_json,write_json,write_csv,put as _put,sealed_json)

AUDIT_MODE='fast'
_HASHED={}
_HASH_LOCK=threading.RLock()


def configure(mode):
    global AUDIT_MODE
    if mode not in ('fast','full'):raise ValueError('unknown audit mode')
    AUDIT_MODE=mode;_HASHED.clear()


def full_audit():return AUDIT_MODE=='full'


def verify_file(path,expected=None):
    """Fast: presence only. Full: hash once per unchanged physical file/process."""
    path=Path(path);info=path.stat()
    if not path.is_file():raise ValueError('not a regular input: '+str(path))
    if full_audit() and expected is not None:
        key=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,expected)
        with _HASH_LOCK:
            if key not in _HASHED:
                if sha(path)!=expected:raise ValueError('modified array/file: '+str(path))
                _HASHED[key]=True
    return info


def put(path,arrays,metadata):
    # Keep original once-on-write identities; existing bundles use our read policy.
    if (Path(path)/'manifest.json').exists():
        _,r=get(path)
        if r['metadata']!=metadata:raise ValueError('array inputs changed: '+str(path))
        return r
    return _put(path,arrays,metadata)


class Store:
    def __init__(self,max_open=10):self.cache=OrderedDict();self.verified={};self.stamps={};self.max_open=max_open;self.lock=threading.RLock()
    def load(self,path,keys=None):
        with self.lock:return self._load(path,keys)
    def _load(self,path,keys=None):
        path=Path(path);r=sealed_json(path/'manifest.json')
        keys=tuple(keys or r['files']);cachekey=(path,keys)
        if cachekey in self.cache:
            self.cache.move_to_end(cachekey)
            value=self.cache[cachekey]
            if value[1]['seal']!=r['seal']:raise ValueError('bundle changed within stage')
            unchanged=True
            for name in keys:
                info=verify_file(path/(name+'.npy'),r['files'][name]['sha256'])
                stamp=(info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
                unchanged &= self.stamps.get((path,name))==stamp
            if unchanged:return value
            self.cache.pop(cachekey)
        arrays={}
        for name in keys:
            spec=r['files'][name];p=path/(name+'.npy');key=(path,name)
            if key in self.verified and self.verified[key]!=spec['sha256']:raise ValueError('array identity changed')
            info=verify_file(p,spec['sha256']);self.verified[key]=spec['sha256']
            self.stamps[key]=(info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
            x=np.load(p,mmap_mode='r',allow_pickle=False)
            if list(x.shape)!=spec['shape'] or str(x.dtype)!=spec['dtype']:raise ValueError('array layout differs')
            arrays[name]=x
        self.cache[cachekey]=(arrays,r)
        while len(self.cache)>self.max_open:self.cache.popitem(last=False)
        return arrays,r


def get(path,keys=None):
    # No process-global mmap ownership: callers can close/remove scratch bundles.
    # Full-audit SHA memo is separate and shared across all these short readers.
    return Store(1).load(path,keys)


def import_bundle(source,destination,mapping,metadata):
    """Link selected immutable NPYs; no copied adapters or transformed banks."""
    source,destination=Path(source),Path(destination)
    arrays,r=get(source,tuple(mapping.values()))
    if (destination/'manifest.json').exists():
        x,saved=get(destination)
        if saved['metadata']!=metadata:raise ValueError('existing imported input differs')
        return saved
    destination.mkdir(parents=True,exist_ok=True);files={}
    for name,old in mapping.items():
        p=destination/(name+'.npy');src=source/(old+'.npy')
        if p.exists():
            # Interrupted same-filesystem import can be resumed without reading it.
            if not os.path.samefile(p,src):
                verify_file(p,r['files'][old]['sha256'])
                value=np.load(p,mmap_mode='r',allow_pickle=False)
                if value.shape!=arrays[old].shape or value.dtype!=arrays[old].dtype:raise ValueError('uncommitted imported array layout differs')
        else:
            try:os.link(src.resolve(),p)
            except OSError:
                try:p.symlink_to(src.resolve())
                except OSError:
                    with p.open('xb') as f:np.save(f,arrays[old],allow_pickle=False)
        files[name]=dict(r['files'][old])
    record=dict(metadata=metadata,files=files);record['seal']=digest(record)
    write_json(destination/'manifest.json',record);return record
