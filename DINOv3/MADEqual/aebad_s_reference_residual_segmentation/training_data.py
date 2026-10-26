"""One actual synthetic RGB forward, shared NN indices, bounded spatial tile cache."""
from pathlib import Path
import numpy as np
from PIL import Image
from . import inputs as ip,storage as st,tiles
from .protocol import CONFIG,DOMAINS,LAYERS,VERSION
from .retrieval import References,exact_with_audit,target_nearest,build_source_folds


def all_items(a):
    c=ip.catalog(a);rows={r['relative_path']:r for r in ip.train_source(c)+sum(c['supports'].values(),[])}
    synthetic=st.sealed_json(a.run_root/'synthetic_manifest.json')
    if synthetic['identity']!=ip.identity(a):raise ValueError('synthetic pool identity differs')
    result=[dict(origin=p,variant=-1,item_id=st.digest([p,'normal'])[:28],status='normal',row=r) for p,r in rows.items()]
    result += [dict(r,row=rows[r['origin']]) for r in synthetic['items'] if r['status']=='valid']
    for path,row in rows.items():
        if not any(r['origin']==path and r['status']=='valid' for r in synthetic['items']):
            raise ValueError('origin has no effective synthetic sample; inspect foreground/preview, no silent image removal: '+path)
    return result


def query_policy(c,item,phase):return ip.policy(c,item['row'],phase)


def prepare_one(a,item,q,geometry,soft,refs,*,catalogue=None,policies=None):
    p=ip.item_path(a,item);grid=geometry['grid'];positions=np.asarray(geometry['positions'],np.int64)
    shape=[item['row']['image_height']//4,item['row']['image_width']//4]
    if not np.array_equal(np.sort(positions),np.arange(np.prod(grid))):raise ValueError('complete raw Q required before tile selection')
    phases=('source',*DOMAINS) if item['row'].get('source_index') is not None else (item['row']['domain'],)
    plans=tiles.plan(soft,grid,item['item_id']);normal=item['variant']==-1
    metadata=dict(identity=ip.identity(a),item=item,grid=grid,positions=positions.tolist(),output_shape=shape,tiles=plans,
        normal_feature_path=str(ip.normal_path(a,item['row'])) if normal else None,
        original_extraction='one_actual_full_RGB_forward_before_any_tile_crop',GT_used=False)
    if normal:
        # Clean complete grid is already immutable. Only labels/tile descriptions are new.
        st.put(p/'labels',dict(soft=soft.astype(np.float32)),dict(identity=ip.identity(a),item=item['item_id']))
    else:
        for tile in plans:
            y0,x0,y1,x1=tile['bounds'];ids=(np.arange(y0,y1)[:,None]*grid[1]+np.arange(x0,x1)).ravel().astype(np.int64)
            inverse=np.argsort(positions)
            st.put(p/'tiles'/str(tile['tile']),{f'q{l}':np.ascontiguousarray(q[l][inverse[ids]]) for l in LAYERS},
                dict(identity=ip.identity(a),item_id=item['item_id'],tile=tile,grid=grid,physical_positions=ids.tolist()))
        st.put(p/'labels',dict(soft=soft.astype(np.float32)),dict(identity=ip.identity(a),item=item['item_id']))
    catalogue=ip.catalog(a) if catalogue is None else catalogue
    policies={} if policies is None else policies
    def policy_for(phase):
        key=(item['origin'],phase)
        if key not in policies:policies[key]=query_policy(catalogue,item,phase)
        return policies[key]
    source_name=policy_for(phases[0])['source'];source_arrays={};source_audit=[];source_seals=[]
    targets={phase:dict(policy=policy_for(phase),arrays={},audits=[],reference_meta=[]) for phase in phases}
    # One upload of this Q layer, shared by Source and every conditioning phase.
    for l in LAYERS:
        query=refs.search.query(q[l]);d,idx,diag,seal=refs.nearest_source(q[l],source_name,l,query)
        source_arrays[f'd{l}']=d;source_arrays[f'i{l}']=idx;source_audit.append(diag);source_seals.append(seal)
        for phase,target in targets.items():
            d,index,diag,seals,offsets=refs.nearest_target(q[l],target['policy']['target_origins'],l,query)
            target['arrays'][f'd{l}']=d;target['arrays'][f'i{l}']=index
            target['audits'].append(diag['support_audits']);target['reference_meta'].append(dict(layer=l,seals=seals,offsets=offsets))
        del query
    sr=st.put(p/'source_nn',source_arrays,dict(identity=ip.identity(a),source=source_name,source_seals=source_seals,vector_audits=source_audit))
    contexts={}
    for phase,target in targets.items():
        record=st.put(p/'target_nn'/phase,target['arrays'],dict(identity=ip.identity(a),policy=target['policy'],
            reference_meta=target['reference_meta'],vector_audits=target['audits']))
        contexts[phase]=dict(policy=target['policy'],target_seal=record['seal'])
    return st.sealed_json(p/'complete.json',dict(**metadata,source_seal=sr['seal'],contexts=contexts,
        labels_seal=st.sealed_json(p/'labels/manifest.json')['seal'],
        query_tile_seals={str(t['tile']):st.sealed_json(p/'tiles'/str(t['tile'])/'manifest.json')['seal'] for t in plans} if not normal else {},
        normal_feature_seal=st.sealed_json(ip.normal_path(a,item['row'])/'manifest.json')['seal'] if normal else None))


def prepare_training(a):
    from .runtime import host,extraction,reliable_precision
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    reliable_precision(a.device);folds=build_source_folds(a);items=all_items(a);model=None;calls=0;records=[];refs=References(a)
    catalogue=ip.catalog(a);policies={}
    for n,item in enumerate(items,1):
        p=ip.item_path(a,item)
        if (p/'complete.json').exists():
            record=st.sealed_json(p/'complete.json')
            if record['identity']!=ip.identity(a) or record['item']!=item:raise ValueError('training query identity differs')
        else:
            if item['variant']==-1:
                x,r=st.get(ip.normal_path(a,item['row']));q={l:x[f'q{l}'] for l in LAYERS};geometry=dict(grid=r['metadata']['grid'],positions=x['positions'])
                soft=np.zeros((item['row']['image_height']//4,item['row']['image_width']//4),np.float32)
            else:
                rgb=a.cache_root/'synthetic_rgb'/item['item_id'];origin=st.sealed_json(rgb/'manifest.json')
                if origin['seal']!=item['record_seal'] or st.sha(rgb/'image.png')!=origin['rgb_sha256']:raise ValueError('synthetic image changed')
                if model is None:model,_=host(a)
                with Image.open(rgb/'image.png') as image:q,geometry=extraction(model,image.convert('RGB'),a.device)
                labels,lr=st.get(rgb/'labels',('soft_output',))
                if lr['seal']!=origin['label_seal']:raise ValueError('synthetic label changed')
                soft=labels['soft_output'];calls+=1
            record=prepare_one(a,item,q,geometry,soft,refs,catalogue=catalogue,policies=policies)
        records.append(dict(item=item,path=str(p),seal=record['seal']))
        if n%10==0:print(f'[residual training cache] {n}/{len(items)}',flush=True)
    st.sealed_json(a.run_root/'training_manifest.json',dict(identity=ip.identity(a),records=records,
        status='all_fixed_training_queries_prepared',GT_used=False))
    return dict(status='training_queries_prepared',queries=len(items),new_synthetic_RGB_forwards=calls,folds=folds,
        Exact_execution=refs.search.stats)


class Dataset:
    def __init__(self,a,phase,use_reference=True,store=None):
        self.a=a;self.phase=phase;self.use_reference=use_reference;self.store=store or st.Store();self.refs=References(a) if use_reference else None
        self.c=ip.catalog(a);self.policies={};manifest=st.sealed_json(a.run_root/'training_manifest.json')
        if manifest['identity']!=ip.identity(a):raise ValueError('training cache identity differs')
        allowed={r['relative_path'] for r in ip.allowed(self.c,phase)};self.by_origin={p:{'normal':[],'synthetic':[]} for p in allowed}
        self.records={}
        for record in manifest['records']:
            item=record['item']
            if item['origin'] not in allowed:continue
            self.records[item['item_id']]=record
            self.by_origin[item['origin']]['normal' if item['variant']==-1 else 'synthetic'].append(item)
        self.source=sorted(r['relative_path'] for r in ip.train_source(self.c));self.target=[] if phase=='source' else [r['relative_path'] for r in self.c['supports'][phase]]
        if any(not v['normal'] or not v['synthetic'] for v in self.by_origin.values()):raise ValueError('missing normal/synthetic query for an origin')
        norm,nr=self.store.load(a.cache_root/'input_normalization');self.norm={k:np.array(v) for k,v in norm.items()}
        self.patch=st.sealed_json(a.run_root/'input_distance_mad.json')['patch'] if use_reference else None
        self.manifest_seal=manifest['seal']
    def select(self,seed,update,source_queries_only=False):
        rng=st.rng(VERSION,'batch',seed,self.phase,update);result=[]
        for i in range(CONFIG['effective_batch_images']):
            population=self.source if self.phase=='source' or source_queries_only or i<2 else self.target
            origin=population[int(rng.integers(len(population)))];kind='normal' if i%2==0 else 'synthetic'
            candidates=self.by_origin[origin][kind];item=candidates[int(rng.integers(len(candidates)))];r=self.records[item['item_id']]
            meta=st.sealed_json(Path(r['path'])/'complete.json')
            positive=[t for t in meta['tiles'] if t['positive']];normal=[t for t in meta['tiles'] if not t['positive']]
            choices=(positive if positive and (not normal or rng.random()>=CONFIG['tile_normal_probability']) else normal)
            tile=choices[int(rng.integers(len(choices)))];result.append((item,tile))
        return result
    def load(self,choice):
        import torch
        from .model import inputs
        item,tile=choice;r=self.records[item['item_id']];p=Path(r['path']);meta=st.sealed_json(p/'complete.json')
        if meta['seal']!=r['seal']:raise ValueError('training item changed')
        y0,x0,y1,x1=tile['bounds'];grid=meta['grid'];ids=(np.arange(y0,y1)[:,None]*grid[1]+np.arange(x0,x1)).ravel()
        qs={};shape=(y1-y0,x1-x0)
        if item['variant']==-1:
            x,rec=self.store.load(ip.normal_path(self.a,item['row']))
            if rec['seal']!=meta['normal_feature_seal']:raise ValueError('normal grid changed')
            inverse=np.argsort(x['positions']);qs={l:np.array(x[f'q{l}'][inverse[ids]]) for l in LAYERS}
        else:
            x,rec=self.store.load(p/'tiles'/str(tile['tile']))
            if rec['seal']!=meta['query_tile_seals'][str(tile['tile'])]:raise ValueError('actual synthetic Q changed')
            qs={l:np.array(x[f'q{l}']) for l in LAYERS}
        reference=None
        if self.use_reference:
            sx,sr=self.store.load(p/'source_nn');tx,tr=self.store.load(p/'target_nn'/self.phase)
            policy=meta['contexts'][self.phase]['policy']
            if sr['seal']!=meta['source_seal'] or tr['seal']!=meta['contexts'][self.phase]['target_seal']:raise ValueError('training NN identity changed')
            if item['origin'] not in self.policies:self.policies[item['origin']]=query_policy(self.c,item,self.phase)
            if policy!=self.policies[item['origin']] or tr['metadata']['policy']!=policy:raise ValueError('training exclusion policy differs')
            inverse=np.argsort(np.asarray(meta['positions']));reference=(sx,tx,policy,inverse[ids])
        values=[]
        for j,l in enumerate(LAYERS):
            q=torch.from_numpy(qs[l].reshape(*shape,1280).transpose(2,0,1).copy());kwargs={}
            if reference:
                sx,tx,policy,compact=reference;si=sx[f'i{l}'][compact];ti=tx[f'i{l}'][compact]
                ss,tt,rm=self.refs.gather(policy['source'],policy['target_origins'],l,si,ti)
                target_meta=tr['metadata']['reference_meta'][j]
                if (sr['metadata']['source']!=policy['source'] or sr['metadata']['source_seals'][j]!=rm['source_seal']
                        or target_meta!=dict(layer=l,seals=rm['target_seals'],offsets=rm['target_offsets'])):raise ValueError('NN references changed after retrieval')
                kwargs=dict(source=torch.from_numpy(ss.reshape(*shape,1280).transpose(2,0,1).copy()),
                    target=torch.from_numpy(tt.reshape(*shape,1280).transpose(2,0,1).copy()),
                    distances=torch.from_numpy(np.stack((sx[f'd{l}'][compact],tx[f'd{l}'][compact])).reshape(2,*shape).copy()),
                    patch=(float(self.patch['medians'][j]),float(self.patch['scales'][j])))
            values.append(inputs(q,torch.from_numpy(self.norm[f'mu{l}']),float(self.norm[f'sigma{l}']),use_reference=self.use_reference,**kwargs)[None])
        labels,lr=self.store.load(p/'labels',('soft',))
        if lr['seal']!=meta['labels_seal']:raise ValueError('soft training label changed')
        return dict(values=values,soft=labels['soft'],origin=[y0,x0],grid=grid,shape=meta['output_shape'],center=tile['center'])
