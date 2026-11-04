"""Streaming full-grid preparation and region-only LP jobs. No Source retrieval."""
import time
import copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
from . import inputs as ip,regions,numerics,transport
from .protocol import CONFIG,LAYERS
st=ip.st


def features(a,row):
    if ip.feature_ref(a,row) is not None:return ip.load_features(a,row)
    if not a.extract_missing:raise FileNotFoundError('missing complete feature; confirm server paths or explicitly use --extract-missing: '+row['relative_path'])
    # The original extraction, preprocessing and precision are reused verbatim.
    tick=time.perf_counter()
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import host,extraction,reliable_precision
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0.feature_identity import native_extraction,compatible
    compatible(native_extraction(a),ip.assets(a).extraction);reliable_precision(a.device)
    if not hasattr(a,'_model'):a._model=host(a)[0]
    with Data(a.data) as data:q,meta=extraction(a._model,data.image(row),a.device)
    a.feature_forwards=getattr(a,'feature_forwards',0)+1
    if not hasattr(a,'_forward_rates'):a._forward_rates=[]
    a._forward_rates.append(time.perf_counter()-tick)
    # Streaming Q is consumed immediately; persist compact origin + regions, not
    # hundreds of GiB of complete test grids. Existing raw bundles remain read-only.
    origin=st.sealed_json(a.cache_root/'feature_origins'/(ip.key(row)+'.json'),dict(
        image=ip.image(row),extraction=ip.assets(a).extraction,grid=meta['grid'],kind='new_original_complete_forward_streamed',
        layers=list(LAYERS),dimension=1280))
    ref=dict(path=None,seal=origin['seal'],extraction=st.digest(ip.assets(a).extraction),grid=meta['grid'],
        streamed=True)
    return [q[l] for l in LAYERS],np.asarray(meta['positions'],np.int64),ref


def source_rms(a,progress=None):
    def publish(r):
        st.sealed_json(a.run_root/'source_RMS.json',dict(provider_seal=r['seal'],contract=r['contract'],
            records=r['records'],RMS=r['RMS'],total_patches=r['total_patches']))
        return r
    existing=ip.rms_record(a)
    if existing:return publish(existing)
    contract=ip.rms_contract(a);records=[]
    for j,row in enumerate(ip.assets(a).train):
        path=a.cache_root/'RMS_parts'/(ip.key(row)+'.json')
        ref=ip.feature_ref(a,row)
        dep=dict(image=ip.image(row),extraction=st.digest(contract['extraction']),definition=contract['definition'])
        if path.exists():
            part=st.sealed_json(path)
            if part['dependencies']!=dep or (ref is not None and part['feature_seal']!=ref['seal']):raise ValueError('RMS partial input changed')
        else:
            tick=time.perf_counter();q,pos,ref=features(a,row)
            # Positions checked by the complete feature adapter; each row once.
            sums,n=numerics.squared_sums(q)
            part=st.sealed_json(path,dict(dependencies=dep,feature_seal=ref['seal'],patches=n,squared_sums=sums.tolist(),seconds=time.perf_counter()-tick))
        records.append({k:part[k] for k in ('dependencies','feature_seal','patches','squared_sums','seconds')})
        if (j+1)%10==0:print(f'[coverage Source RMS] {j+1}/455 original images',flush=True)
        if progress:progress('RMS',part['seconds'])
    sums=np.sum([r['squared_sums'] for r in records],axis=0);count=sum(r['patches'] for r in records)
    result=st.sealed_json(a.cache_root/'source_RMS.json',dict(contract=contract,records=records,total_patches=count,
        RMS=numerics.rms_from_sums(sums,count).tolist(),feature_forwards=getattr(a,'feature_forwards',0)))
    return publish(result)


def region_path(a,row):return a.cache_root/'regions'/ip.key(row)


def cached_region_path(a,row,rms):
    for root in [a.cache_root,*getattr(a,'region_caches',[])]:
        probe=copy.copy(a);probe.cache_root=Path(root)
        path=region_path(probe,row)
        if (path/'manifest.json').is_file():
            ip.cached_region(probe,row,rms,None)
            return path
    return None


def region(a,row,rms):
    if getattr(a,'lp_profile','cpu_v1')=='mixed_background_v2':
        existing=cached_region_path(a,row,rms)
        if existing is not None:return st.get(existing)
    path=region_path(a,row);ref=ip.region_feature_ref(a,row)
    cached=ip.cached_region(a,row,rms,ref)
    if cached is not None:
        x,r=st.get(path)
        if r['seal']!=cached['seal']:raise ValueError('region manifest changed during read')
        return x,r
    streamed=None
    if ref is None:
        q,pos,ref=features(a,row);streamed=(q,pos)
    dep=ip.region_dependencies(a,row,rms,ref['seal'])
    origin=dict(schema='region_feature_origin_v1',image=ip.image(row),feature_seal=ref['seal'],
        extraction=ref['extraction'],grid=list(ref['grid']),
        kind='original_forward_streamed' if ref.get('streamed') else 'existing_complete_features',
        source_path=ref.get('path'),authority=ref.get('authority'))
    tick=time.perf_counter()
    if streamed is None:q,pos,_=ip.load_features(a,row)
    else:q,pos=streamed
    arrays=regions.partition(q,pos,ref['grid'],rms['RMS'],CONFIG['regions'])
    r=st.put(path,arrays,dict(dependencies=dep,origin=origin,seconds=time.perf_counter()-tick,GT_used=False))
    print(f'[coverage regions] {row["relative_path"]} patches={len(pos)} seconds={r["metadata"]["seconds"]:.3f}',flush=True)
    return arrays,r


def pair_path(a,query,template):return a.run_root/'pairs'/ip.key(query)/(ip.key(template)+'.json')


class Coverage:
    def __init__(self,a,rms):
        self.a=a;self.rms=rms;self.refs={};self.pool=None;self.new_pairs=0
    def __enter__(self):return self
    def __exit__(self,*args):
        if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)
    def reference(self,row):
        k=ip.key(row)
        if k not in self.refs:self.refs[k]=region(self.a,row,self.rms)
        return self.refs[k]
    def template(self,row,query,qregion,qmeta):
        data,meta=self.reference(row)
        dep=dict(query=qmeta['seal'],template=meta['seal'],RMS=self.rms['seal'],
            distance_device=self.a.device,protocol=st.digest(CONFIG),solver_implementation=st.sha(Path(transport.__file__)))
        path=pair_path(self.a,query,row)
        if path.is_file():
            r=st.sealed_json(path)
            if r['dependencies']!=dep:raise ValueError('pair cache dependencies changed')
            if not r['result']['success']:raise ValueError('retained LP failure requires diagnosis; not a free-coverage fallback')
            return dict(record=r,path=path,done=True)
        tick=time.perf_counter();distance=regions.distance_matrix(data['means'],qregion['means'],self.rms['RMS'],self.a.device)
        work=dict(path=path,dependencies=dep,query=ip.image(query),template=ip.image(row),
            distance_seconds=time.perf_counter()-tick,done=False)
        if self.a.cpu_workers>1:
            # Leave one of the requested CPU slots for the orchestration/distance
            # process. Each HiGHS worker itself uses one thread.
            if self.pool is None:self.pool=ProcessPoolExecutor(max_workers=max(1,self.a.cpu_workers-1),mp_context=get_context('spawn'))
            work['future']=self.pool.submit(transport.solve,distance,data['area'],qregion['area'])
        else:work['result']=transport.solve(distance,data['area'],qregion['area'])
        return work
    def score(self,query,templates):
        qregion,qmeta=region(self.a,query,self.rms)
        pending=[self.template(t,query,qregion,qmeta) for t in templates];records=[]
        for work in pending:
            if work['done']:r=work['record']
            else:
                result=work['future'].result() if 'future' in work else work['result']
                r=st.sealed_json(work['path'],{k:v for k,v in work.items() if k not in ('path','done','future','result')}|dict(result=result,GT_used=False))
                if not result['success']:raise ValueError('LP failed; retained '+str(work['path'])+'; no substitute objective')
                self.new_pairs+=1
            records.append(dict(path=str(work['path']),seal=r['seal'],template=r['template'],result=r['result'],distance_seconds=r['distance_seconds']))
        cap=int(np.argmin([r['result']['objective'] for r in records]));free=int(np.argmin([r['result']['free'] for r in records]))
        return dict(cap=records[cap]['result']['objective'],free=records[free]['result']['free'],
            cap_template=records[cap]['template'],free_template=records[free]['template'],templates=records,query_region=qmeta['seal'])
