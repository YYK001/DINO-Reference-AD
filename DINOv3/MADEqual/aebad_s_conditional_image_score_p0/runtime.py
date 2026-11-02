"""Explicit original Exact/SEGMENT supplementation; normal and test paths separate."""
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from . import inputs as ip,numerics as nm
from .inputs import st
from .protocol import LAYERS
from DINOv3.MADEqual.aebad_s_joint_reference_response_p0 import runtime as original
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer


def features(a,row,persist):
    ref=ip.feature_ref(a,row)
    if ref:
        x,r=st.get(ref['path'],tuple('q'+str(l) for l in LAYERS)+('positions',))
        if r['seal']!=ref['seal']:raise ValueError('feature manifest changed')
        return {l:x['q'+str(l)] for l in LAYERS},dict(grid=r['metadata']['grid'],positions=x['positions']),ref
    return original.features(a,dict(row,domain=row.get('domain','background')),persist=persist)


def selected_indices(heldout):
    if heldout is not None and heldout not in range(8):raise ValueError('group-local heldout required')
    return [j for j in range(8) if j!=heldout]


def segment(nn,parts,heldout):
    selected=selected_indices(heldout);result=[]
    for l in LAYERS:
        v=nn[l]
        if v['support_indices']!=selected or v['dt'].shape[0]!=len(selected) or v['indices'].shape!=v['dt'].shape:
            raise ValueError('K7 distances/indices/features must use identical retained support order')
        # Filter FEATURE LIST as well as distance/index rows. Legacy K8_* names
        # are output keys only; the original kernel supports seven references.
        values,_=geometry_layer(v['dt'],v['indices'],v['ds'],[parts[j][l] for j in selected])
        result.append(values['K8_SEGMENT'])
    return np.stack(result)


class Engine:
    def __init__(self,a):self.a=a;self.references={};self.search=None;self.last=None;self.store=st.Store(2)

    def features(self,row,persist):
        if self.last is not None and self.last[0]==ip.image(row):return self.last[1]
        # Source66 rows lack the legacy domain lookup hint, not image identity.
        value=features(self.a,row,persist=persist)
        q,m,ref=value
        if any(q[l].dtype!=np.float32 or not np.isfinite(q[l]).all() for l in LAYERS):raise ValueError('invalid original features')
        self.last=(ip.image(row),value);return value

    def refs(self,d,seed):
        if (d,seed) not in self.references:
            group=ip.supports(self.a,d,seed);parts=[];origins=[]
            for row in group:
                q,m,origin=features(self.a,row,persist=True);parts.append(q);origins.append(origin)
            self.references[d,seed]=SimpleNamespace(parts=parts,origins=origins,support_seals=[st.digest(r) for r in origins])
        return self.references[d,seed]

    def ensure_search(self):
        if self.search is None:
            from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.execution import ResidentExact
            from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision
            reliable_precision(self.a.device);self.search=ResidentExact(self.a.device,int(self.a.reference_cache_gib*2**30))
            self.a.exact_execution=self.search.stats

    def source_distances(self,c,row,q,m,origin,refs):
        a=self.a
        # Query compact Source distances can be reused across both support sets.
        # Do NOT call the test-only baseline reader with a Source/LOO normal.
        if c['kind']=='query':
            try:
                x,r,_=ip.jp.inspect_source(a,c['domain'],row)
                physical=np.empty_like(x['Source_distance']);physical[:,x['positions']]=x['Source_distance']
                if r['metadata']['grid']!=m['grid']:raise ValueError('shared Source grid differs')
                return physical[:,m['positions']]
            except FileNotFoundError:pass
        binding={k:origin[k] for k in ('path','seal')} if 'path' in origin else origin
        source_contract=dict(query=ip.image(row),source=c['source'],extraction=c['extraction'],feature_origin=binding)
        path=a.cache_root/'shared_Source'/ip.key(row)
        if (path/'manifest.json').exists():
            x,r=st.get(path)
            if r['metadata']['contract']!=source_contract or not np.array_equal(x['positions'],m['positions']):raise ValueError('shared Source distance dependency changed')
            return x['Source_distance']
        ref=ip.bank(a);source,r=self.store.load(ref['path'],tuple('L'+str(l) for l in LAYERS))
        values=np.stack([self.search.exact(self.search.query(q[l]),source['L'+str(l)],('Source',ref['seal'],l))[0] for l in LAYERS])
        st.put(path,dict(Source_distance=values,positions=m['positions']),dict(contract=source_contract,grid=m['grid']))
        return values

    def acquire(self,c,row):
        a=self.a;ref=ip.response(a,c)
        if ref:
            x,r=st.get(ref['path'],(ref['field'],'positions'))
            if r['seal']!=ref['seal']:raise ValueError('native SEGMENT changed')
            m=r['metadata'];return x[ref['field']],x['positions'],m['grid'],ref['encoding'],ref
        cached=ip.retrieval(a,c)
        if cached:
            refs=self.refs(c['domain'],c['support_seed']);x,r=st.get(cached['path'])
            k=c['K'];n=len(x['positions']);selected=selected_indices(c['heldout'])
            if x['target_distance'].shape!=(4,k,n) or x['target_index'].shape!=(4,k,n):raise ValueError('per-support retrieval layout differs')
            if x['Source_distance'].shape!=(4,n):raise ValueError('Source retrieval layout differs')
            if any(x[k].dtype!=np.float32 or not np.isfinite(x[k]).all() for k in ('target_distance','Source_distance')):raise ValueError('original scaled FP32 distances required')
            nn={l:dict(ds=x['Source_distance'][j],dt=x['target_distance'][j],indices=x['target_index'][j],support_indices=selected) for j,l in enumerate(LAYERS)}
            return segment(nn,refs.parts,c['heldout']),x['positions'],r['metadata']['grid'],'raw_distance',cached
        if not a.supplement_retrieval:raise FileNotFoundError('correct SEGMENT/BP absent; --supplement-retrieval is explicit: '+row['relative_path'])
        # Require Source entity before any missing feature/model work.
        ip.bank(a)
        q,m,origin=self.features(row,persist=c['kind']!='query');refs=self.refs(c['domain'],c['support_seed']);self.ensure_search()
        ds=self.source_distances(c,row,q,m,origin,refs);selected=selected_indices(c['heldout']);nn={}
        for j,l in enumerate(LAYERS):
            qt=self.search.query(q[l]);target=[];indices=[]
            for s in selected:
                dd,ii=self.search.exact(qt,refs.parts[s][l],('Target',refs.support_seals[s],l));target.append(dd);indices.append(ii)
            nn[l]=dict(ds=ds[j],dt=np.stack(target),indices=np.stack(indices),support_indices=selected)
        return segment(nn,refs.parts,c['heldout']),m['positions'],m['grid'],'raw_distance',dict(kind='new_original_Exact_SEGMENT',
            query_feature=origin,reference_features=refs.origins,reference_indices=selected)


def observe(a,d,seed,kind,row,j,engine):
    c=ip.contract(a,d,seed,kind,row,j);path=ip.record_path(a,c);old=ip.bp(a,c)
    if path.exists():return old
    tick=time.perf_counter()
    if old:
        value={k:v for k,v in old.items() if k not in ('seal','origin')}
        value['origin']=dict(kind='read_only_BP_import',provider=str(a.statistics_provider),
            provider_seal=st.sealed_json(a.statistics_provider)['seal'],original_record_seal=old.get('seal'),original_origin=old.get('origin'))
        value['execution_path']='small_BP_import';value['seconds']=time.perf_counter()-tick
    else:
        raw,pos,grid,encoding,origin=engine.acquire(c,row)
        value=nm.observation(raw,pos,grid,[row['image_height']//4,row['image_width']//4],ip.parent(a).inputs['mad'],encoding)
        value.update(contract=c,origin=origin,GT_used=False,seconds=time.perf_counter()-tick)
        value['execution_path']='fresh_Exact_SEGMENT' if origin.get('kind')=='new_original_Exact_SEGMENT' else 'native_response_statistics' if 'field' in origin else 'cached_geometry'
    value['contract']=c;value['GT_used']=False
    return st.sealed_json(path,value)
