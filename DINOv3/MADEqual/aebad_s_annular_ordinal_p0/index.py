"""Persisted native binary LSH; no Python scan over the full normal library."""
import os
import time
from pathlib import Path
from collections import OrderedDict
import numpy as np
from . import inputs as ip,pixels as px
from .protocol import CONFIG,PAIRS,VARIANTS
st=ip.st

class Store:
    def __init__(self,a):
        self.a=a;self.rows=ip.assets(a).normal;self.data=None;self.grays=OrderedDict();self.backend=None;self.query_cache=OrderedDict()
        self.references=[dict(r,shape=(r['image_height'],r['image_width']),read_gray=lambda i=i:self.gray(self.rows[i])) for i,r in enumerate(self.rows)]
    def close(self):
        self.grays.clear()
        if self.data:self.data.__exit__(None,None,None);self.data=None
        self.backend=None
    def gray(self,row):
        k=ip.key(row)
        if k in self.grays:self.grays.move_to_end(k);return self.grays[k]
        path=self.a.cache_root/'gray'/k;meta=dict(image=ip.image(row),pixel_contract=ip.pixel_contract())
        if (path/'manifest.json').exists():
            data,record=st.get(path,('gray',))
            if record['metadata']!=meta:raise ValueError('cached original gray identity/decoder differs')
            gray=data['gray']
        else:
            if self.data is None:self.data=ip.Data(self.a.data)
            rgb=self.data.image(row)
            if rgb.size!=(row['image_width'],row['image_height']):raise ValueError('encoded raster dimensions differ')
            gray=px.grayscale(np.asarray(rgb));st.put(path,dict(gray=gray),meta)
        self.grays[k]=gray
        while len(self.grays)>4:self.grays.popitem(last=False)
        return gray
    def descriptor_path(self,row):return self.a.cache_root/'descriptors'/ip.key(row)
    def describe(self,row):
        path=self.descriptor_path(row);blocks=[px.grid_spec((row['image_height'],row['image_width']),v) for v in range(9)]
        meta=dict(image=ip.image(row),pixel_contract=ip.pixel_contract(),pairs=PAIRS.tolist(),variants=list(map(list,VARIANTS)),blocks=blocks)
        if (path/'manifest.json').exists():
            record=st.sealed_json(path/'manifest.json')
            if record['metadata']!=meta:raise ValueError('binary descriptor dependencies differ')
            return record
        gray=self.gray(row);parts=[px.describe_grid(gray,b) for b in blocks]
        return st.put(path,dict(descriptors=np.concatenate(parts)),meta)
    def _cv(self):
        import cv2
        cv2.setNumThreads(self.a.cpu_workers);cv2.setRNGSeed(42);return cv2
    def build(self):
        dest=self.a.cache_root/'index';commit=dest/'manifest.json'
        if commit.exists():
            old=st.sealed_json(commit)
            if old['identity']!=ip.identity(self.a):raise ValueError('index build dependencies differ')
            return old
        started=time.perf_counter();parts=[]
        for i,row in enumerate(self.rows):
            parts.append(self.describe(row));print(f'[annular index pixels] {i+1}/463 {row["kind"]}',flush=True)
            elapsed=time.perf_counter()-started
            if i>=3 and elapsed/(i+1)*463>43200 and not self.a.allow_over_12h:
                st.write_json(self.a.run_root/'budget_pause.json',dict(stage='build-index',known_projection_seconds=elapsed/(i+1)*463,completed_images=i+1,unknown='native index and full-image inference'))
                raise RuntimeError('projected index preparation alone exceeds 12h; cached images retained, explicit consent required')
        cv=self._cv();dest.mkdir(parents=True,exist_ok=True);groups=[];source_count=sum(p['files']['descriptors']['shape'][0] for p in parts[:455])
        path=dest/'source_vectors.npy';vectors=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(source_count,16));offset=0;blocks=[]
        for i,part in enumerate(parts[:455]):
            data,_=st.get(self.descriptor_path(self.rows[i]),('descriptors',));n=len(data['descriptors']);vectors[offset:offset+n]=data['descriptors']
            local=offset
            for block in part['metadata']['blocks']:
                blocks.append(dict(block,ref=i,start=local,end=local+block['count']));local+=block['count']
            offset+=n
        vectors.flush();native=cv.flann_Index(vectors,{k:v for k,v in CONFIG['index'].items() if k in ('algorithm','table_number','key_size','multi_probe_level')})
        native.save(str(dest/'Source.lsh'));del native;del vectors
        groups.append(dict(name='Source',data=str(path),blocks=blocks,rows=source_count))
        for i in range(455,463):
            row=self.rows[i];data,_=st.get(self.descriptor_path(row),('descriptors',));values=data['descriptors'];offset=0;blocks=[]
            for block in parts[i]['metadata']['blocks']:
                blocks.append(dict(block,ref=i,start=offset,end=offset+block['count']));offset+=block['count']
            native=cv.flann_Index(values,{k:v for k,v in CONFIG['index'].items() if k in ('algorithm','table_number','key_size','multi_probe_level')})
            name=f'Target{i-455}';native.save(str(dest/(name+'.lsh')));del native
            groups.append(dict(name=name,data=str(self.descriptor_path(row)/'descriptors.npy'),blocks=blocks,rows=len(values)))
        return st.sealed_json(commit,dict(kind='annular_binary_indices_v1',identity=ip.identity(self.a),native_version=cv.__version__,
            config=CONFIG['index'],parts=[r['seal'] for r in parts],groups=groups,seconds=time.perf_counter()-started,
            descriptor_bytes=sum(r['files']['descriptors']['bytes'] for r in parts),
            native_index_bytes=sum((dest/(g['name']+'.lsh')).stat().st_size for g in groups),
            source_vector_copy_bytes=path.stat().st_size,
            existing_complete_images=463,construction='native LSH; no learned model/index centroid training'))
    def load(self):
        if self.backend is not None:return self.record
        cv=self._cv();self.record=st.sealed_json(self.a.cache_root/'index/manifest.json')
        if self.record['identity']!=ip.identity(self.a) or self.record['native_version']!=cv.__version__ or self.record['config']!=CONFIG['index']:
            raise ValueError('index image/configuration/native backend identity differs')
        self.backend=[];started=time.perf_counter()
        for group in self.record['groups']:
            data=np.load(group['data'],mmap_mode='r',allow_pickle=False)
            if data.shape!=(group['rows'],16) or data.dtype!=np.uint8:raise ValueError('index vector layout differs')
            native=cv.flann_Index()
            if not native.load(data,str(self.a.cache_root/'index'/(group['name']+'.lsh'))):raise RuntimeError('native index load failed; no silent full rebuild')
            self.backend.append(dict(group,native=native,data_owner=data))
        self.load_seconds=time.perf_counter()-started;return self.record
    def lookup_many(self,bits):
        self.load();keys=[row.tobytes() for row in np.asarray(bits,np.uint8)]
        computed={k:self.query_cache[k] for k in keys if k in self.query_cache}
        missing=list(dict.fromkeys(k for k in keys if k not in computed))
        if missing:
            q=np.frombuffer(b''.join(missing),np.uint8).reshape(-1,16)
            found=[[] for _ in missing];flags=np.zeros(len(missing),bool)
            for group in self.backend:
                budget=16 if group['name']=='Source' else 4;k=min(group['rows'],budget*9+1)
                indices,dist=group['native'].knnSearch(q,k,params={})
                blocks=group['blocks'];ends=np.array([b['end'] for b in blocks],np.int64)
                for j in range(len(q)):
                    ids=indices[j];valid=(ids>=0)&(ids<group['rows'])&(dist[j]>=0)&(dist[j]<=128);ids=ids[valid];dd=dist[j][valid]
                    bb=np.searchsorted(ends,ids,side='right');matches=[]
                    for idx,hamming,bindex in zip(ids,dd,bb):
                        b=blocks[int(bindex)];n=int(idx)-b['start'];ref=b['ref']
                        matches.append((int(hamming),ref,b['x0']+(n%b['nx'])*8,b['y0']+(n//b['nx'])*8,b['version']))
                    matches.sort();unique=[];seen=set()
                    for m in matches:
                        physical=(self.rows[m[1]]['instance_id'],m[2],m[3])
                        if physical not in seen:unique.append(m);seen.add(physical)
                    flags[j] |= len(unique)>budget or (len(ids)==k and group['rows']>k)
                    found[j].extend(unique[:budget])
            for key,hits,flag in zip(missing,found,flags):computed[key]=(hits,bool(flag));self.query_cache[key]=computed[key]
            while len(self.query_cache)>8192:self.query_cache.popitem(last=False)
        return [computed[k] for k in keys]
    def matches(self,info,lookups):
        result=[];saturated=False
        for anchor,(hits,flag) in enumerate(lookups):
            saturated |= flag
            for distance,ref,x,y,version in hits:result.append(dict(anchor=anchor,ref=ref,point=[x,y],version=version,hamming=distance))
        return result,saturated
