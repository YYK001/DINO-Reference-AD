"""Reuse immutable bundle primitives, with selective per-stage verified mmap reads."""
from pathlib import Path
from collections import OrderedDict
import os
import numpy as np
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.storage import (
    seed,rng,sha,digest,read_json,write_json,write_csv,put,sealed_json)


class Store:
    def __init__(self,max_open=10):self.cache=OrderedDict();self.verified={};self.max_open=max_open
    def load(self,path,keys=None):
        path=Path(path);r=sealed_json(path/'manifest.json')
        keys=tuple(keys or r['files']);cachekey=(path,keys)
        if cachekey in self.cache:
            self.cache.move_to_end(cachekey)
            value=self.cache[cachekey]
            if value[1]['seal']!=r['seal']:raise ValueError('bundle changed within stage')
            return value
        arrays={}
        for name in keys:
            spec=r['files'][name];p=path/(name+'.npy');key=(path,name)
            if key not in self.verified:
                if sha(p)!=spec['sha256']:raise ValueError('modified array: '+str(p))
                self.verified[key]=spec['sha256']
            elif self.verified[key]!=spec['sha256']:raise ValueError('array identity changed')
            x=np.load(p,mmap_mode='r',allow_pickle=False)
            if list(x.shape)!=spec['shape'] or str(x.dtype)!=spec['dtype']:raise ValueError('array layout differs')
            arrays[name]=x
        self.cache[cachekey]=(arrays,r)
        while len(self.cache)>self.max_open:self.cache.popitem(last=False)
        return arrays,r


def get(path,keys=None):return Store(1).load(path,keys)


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
            if sha(p)!=r['files'][old]['sha256']:raise ValueError('uncommitted imported array differs')
        else:
            try:os.link(src.resolve(),p)
            except OSError:
                try:p.symlink_to(src.resolve())
                except OSError:
                    with p.open('xb') as f:np.save(f,arrays[old],allow_pickle=False)
        files[name]=dict(r['files'][old])
    record=dict(metadata=metadata,files=files);record['seal']=digest(record)
    write_json(destination/'manifest.json',record);return record
