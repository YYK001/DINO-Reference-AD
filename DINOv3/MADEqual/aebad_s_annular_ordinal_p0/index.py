"""Persisted native binary LSH; no Python scan over the full normal library."""
import os
import time
from pathlib import Path
from collections import OrderedDict
import numpy as np
from . import inputs as ip,pixels as px
from .protocol import CONFIG,PAIRS,VARIANTS
st=ip.st

def index_root(a):return a.cache_root/'index_sharded_v1'

def source_slices(rows):
    """cv::Mat continuity requires its scalar element count to fit signed int."""
    step=CONFIG['index']['source_shard_rows']
    return [(start,min(start+step,rows)) for start in range(0,rows,step)]

def native_build(cv,values):
    if values.dtype!=np.uint8 or values.ndim!=2 or values.shape[1]!=16 or values.size>2**31-1:
        raise ValueError('native LSH needs uint8 Nx16 with scalar element count <= INT_MAX')
    values=np.ascontiguousarray(values)
    cv.setRNGSeed(CONFIG['index']['seed_per_native_index'])
    native=cv.flann_Index()
    # Explicit build preserves the native error instead of masking it in __init__.
    native.build(values,{k:CONFIG['index'][k] for k in ('algorithm','table_number','key_size','multi_probe_level')})
    return native

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
        dest=index_root(self.a);commit=dest/'manifest.json'
        if commit.exists():
            old=st.sealed_json(commit)
            if old['identity']!=ip.identity(self.a):raise ValueError('index build dependencies differ')
            return old
        started=time.perf_counter();parts=[];reused=0
        for i,row in enumerate(self.rows):
            cached=(self.descriptor_path(row)/'manifest.json').exists();reused+=int(cached)
            parts.append(self.describe(row));print(f'[annular index pixels] {i+1}/463 {row["kind"]} {"reused" if cached else "fresh"}',flush=True)
            elapsed=time.perf_counter()-started
            if i>=3 and elapsed/(i+1)*463>43200 and not self.a.allow_over_12h:
                st.write_json(self.a.run_root/'budget_pause.json',dict(stage='build-index',known_projection_seconds=elapsed/(i+1)*463,completed_images=i+1,unknown='native index and full-image inference'))
                raise RuntimeError('projected index preparation alone exceeds 12h; cached images retained, explicit consent required')
        cv=self._cv();dest.mkdir(parents=True,exist_ok=True);groups=[];source_count=sum(p['files']['descriptors']['shape'][0] for p in parts[:455])
        path=dest/'source_vectors.npy';copy_commit=dest/'source_vectors.json'
        copy_inputs=dict(parts=[p['seal'] for p in parts[:455]],rows=source_count,width=16,dtype='uint8')
        if copy_commit.exists():
            copied=st.sealed_json(copy_commit)
            if copied['inputs']!=copy_inputs or path.stat().st_size!=copied['bytes']:raise ValueError('Source vector copy dependencies/size differ')
            vectors=np.load(path,mmap_mode='r',allow_pickle=False)
            if vectors.shape!=(source_count,16) or vectors.dtype!=np.uint8:raise ValueError('Source vector copy layout differs')
            blocks=copied['blocks']
        else:
            vectors=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(source_count,16));offset=0;blocks=[]
            for i,part in enumerate(parts[:455]):
                data,_=st.get(self.descriptor_path(self.rows[i]),('descriptors',));n=len(data['descriptors']);vectors[offset:offset+n]=data['descriptors']
                local=offset
                for block in part['metadata']['blocks']:
                    blocks.append(dict(block,ref=i,start=local,end=local+block['count']));local+=block['count']
                offset+=n
            vectors.flush();copied=st.sealed_json(copy_commit,dict(inputs=copy_inputs,blocks=blocks,bytes=path.stat().st_size))
        for shard,(start,stop) in enumerate(source_slices(source_count)):
            group=dict(name=f'Source{shard:03d}',kind='Source',data=str(path),row_start=start,rows=stop-start,
                blocks=[b for b in blocks if b['start']<stop and b['end']>start])
            self.build_group(cv,dest,group,vectors[start:stop],copied['seal']);groups.append(group)
            print(f'[annular Source shard] {shard+1}/{len(source_slices(source_count))} rows={stop-start}',flush=True)
        del vectors
        for i in range(455,463):
            row=self.rows[i];data,_=st.get(self.descriptor_path(row),('descriptors',));values=data['descriptors'];offset=0;blocks=[]
            for block in parts[i]['metadata']['blocks']:
                blocks.append(dict(block,ref=i,start=offset,end=offset+block['count']));offset+=block['count']
            name=f'Target{i-455}';group=dict(name=name,kind='Target',data=str(self.descriptor_path(row)/'descriptors.npy'),blocks=blocks,row_start=0,rows=len(values))
            self.build_group(cv,dest,group,values,parts[i]['seal']);groups.append(group)
        return st.sealed_json(commit,dict(kind='annular_binary_indices_sharded_v1',identity=ip.identity(self.a),native_version=cv.__version__,
            config=CONFIG['index'],parts=[r['seal'] for r in parts],groups=groups,seconds=time.perf_counter()-started,
            descriptor_bytes=sum(r['files']['descriptors']['bytes'] for r in parts),
            native_index_bytes=sum((dest/(g['name']+'.lsh')).stat().st_size for g in groups),
            source_vector_copy_bytes=path.stat().st_size,
            reused_descriptor_images=reused,fresh_descriptor_images=len(parts)-reused,
            prior_descriptor_preparation_seconds='not stored in original descriptor manifests; preserved previous-run costs remain authoritative',
            existing_complete_images=463,construction='native LSH; no learned model/index centroid training'))
    def build_group(self,cv,dest,group,values,input_seal):
        path=dest/(group['name']+'.lsh');commit=dest/(group['name']+'.json')
        dependencies=dict(group=group,input_seal=input_seal,native_version=cv.__version__,config=CONFIG['index'],identity=ip.identity(self.a))
        if commit.exists():
            saved=st.sealed_json(commit)
            if saved['dependencies']!=dependencies or path.stat().st_size!=saved['bytes']:raise ValueError('native shard dependencies/size differ')
            return
        tick=time.perf_counter();native=native_build(cv,values);tmp=path.with_suffix('.pending')
        native.save(str(tmp));os.replace(tmp,path);del native
        st.sealed_json(commit,dict(dependencies=dependencies,bytes=path.stat().st_size,seconds=time.perf_counter()-tick))
    def load(self):
        if self.backend is not None:return self.record
        cv=self._cv();self.record=st.sealed_json(index_root(self.a)/'manifest.json')
        if self.record['identity']!=ip.identity(self.a) or self.record['native_version']!=cv.__version__ or self.record['config']!=CONFIG['index']:
            raise ValueError('index image/configuration/native backend identity differs')
        self.backend=[];started=time.perf_counter();owners={}
        for group in self.record['groups']:
            if group['data'] not in owners:owners[group['data']]=np.load(group['data'],mmap_mode='r',allow_pickle=False)
            owner=owners[group['data']];start=group['row_start'];data=owner[start:start+group['rows']]
            if data.shape!=(group['rows'],16) or data.dtype!=np.uint8 or data.size>2**31-1:raise ValueError('index vector layout differs')
            native=cv.flann_Index()
            cv.setRNGSeed(CONFIG['index']['seed_per_native_index'])
            if not native.load(data,str(index_root(self.a)/(group['name']+'.lsh'))):raise RuntimeError('native index load failed; no silent full rebuild')
            self.backend.append(dict(group,native=native,data_owner=owner))
        self.load_seconds=time.perf_counter()-started;return self.record
    def lookup_many(self,bits):
        self.load();keys=[row.tobytes() for row in np.asarray(bits,np.uint8)]
        computed={k:self.query_cache[k] for k in keys if k in self.query_cache}
        missing=list(dict.fromkeys(k for k in keys if k not in computed))
        if missing:
            q=np.frombuffer(b''.join(missing),np.uint8).reshape(-1,16)
            found=[[] for _ in missing];flags=np.zeros(len(missing),bool);pools={}
            for group in self.backend:
                is_source=group.get('kind')=='Source' or group['name'].startswith('Source')
                logical='Source' if is_source else group['name'];budget=16 if is_source else 4;raw_limit=budget*9+1
                pool=pools.setdefault(logical,dict(budget=budget,hits=[[] for _ in missing]))
                k=min(group['rows'],raw_limit)
                indices,dist=group['native'].knnSearch(q,k,params={})
                blocks=group['blocks'];ends=np.array([b['end'] for b in blocks],np.int64)
                for j in range(len(q)):
                    ids=indices[j];valid=(ids>=0)&(ids<group['rows'])&(dist[j]>=0)&(dist[j]<=128);ids=ids[valid];dd=dist[j][valid]
                    ids=ids.astype(np.int64)+group.get('row_start',0)
                    bb=np.searchsorted(ends,ids,side='right');matches=[]
                    for idx,hamming,bindex in zip(ids,dd,bb):
                        b=blocks[int(bindex)];n=int(idx)-b['start'];ref=b['ref']
                        matches.append((int(hamming),ref,b['x0']+(n%b['nx'])*8,b['y0']+(n//b['nx'])*8,b['version']))
                    merged=sorted(pool['hits'][j]+matches)
                    flags[j] |= len(merged)>raw_limit or (len(ids)==k and group['rows']>k)
                    pool['hits'][j]=merged[:raw_limit]
            # Source shard outputs compete globally before version/physical-position
            # merging and the original 16-match cap; never keep 16 per shard.
            for pool in pools.values():
                for j,matches in enumerate(pool['hits']):
                    unique=[];seen=set()
                    for m in matches:
                        physical=(self.rows[m[1]]['instance_id'],m[2],m[3])
                        if physical not in seen:unique.append(m);seen.add(physical)
                    flags[j] |= len(unique)>pool['budget']
                    found[j].extend(unique[:pool['budget']])
            for key,hits,flag in zip(missing,found,flags):computed[key]=(hits,bool(flag));self.query_cache[key]=computed[key]
            while len(self.query_cache)>8192:self.query_cache.popitem(last=False)
        return [computed[k] for k in keys]
    def matches(self,info,lookups):
        result=[];saturated=False
        for anchor,(hits,flag) in enumerate(lookups):
            saturated |= flag
            for distance,ref,x,y,version in hits:result.append(dict(anchor=anchor,ref=ref,point=[x,y],version=version,hamming=distance))
        return result,saturated
