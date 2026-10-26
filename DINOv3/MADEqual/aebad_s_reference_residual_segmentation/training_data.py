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
    population=[('Source',ip.train_source(c))]+list(c['supports'].items())
    for name,group in population:
        if not any(r['status']=='valid' and r['origin'] in {v['relative_path'] for v in group} for r in synthetic['items']):
            raise ValueError('population has no effective synthetic sample: '+name+'; normal origins retained; no hidden fill')
    return result


def query_policy(c,item,phase):return ip.policy(c,item['row'],phase)


def prepare_one(a,item,q,geometry,soft,refs,*,catalogue=None,policies=None):
    p=ip.item_path(a,item);grid=geometry['grid'];positions=np.asarray(geometry['positions'],np.int64)
    shape=[item['row']['image_height']//4,item['row']['image_width']//4]
    complete=np.array_equal(np.sort(positions),np.arange(np.prod(grid)))
    if not complete and not geometry.get('reused_original_full_forward_tiles'):raise ValueError('complete raw Q required before tile selection')
    phases=('source',*DOMAINS) if item['row'].get('source_index') is not None else (item['row']['domain'],)
    plans=tiles.plan(soft,grid,item['item_id']);normal=item['variant']==-1
    sparse=ip.config(a)['training_nn_layout']=='physical_tile_union_v1'
    if not sparse and not complete:raise ValueError('legacy full-grid NN cannot use partial raw tiles')
    nn_positions=tiles.physical_union(plans,grid) if sparse else positions
    query_rows=tiles.rows_for(positions,nn_positions,grid)
    metadata=dict(identity=ip.identity(a),item=item,grid=grid,positions=positions.tolist(),output_shape=shape,tiles=plans,
        normal_feature_path=str(ip.normal_path(a,item['row'])) if normal else None,
        original_extraction='one_actual_full_RGB_forward_before_any_tile_crop',GT_used=False)
    if sparse:metadata.update(NN_layout='physical_tile_union_v1',NN_physical_positions=nn_positions.tolist(),
        full_grid_patch_count=int(np.prod(grid)),NN_patch_count=len(nn_positions))
    if geometry.get('reused_original_full_forward_tiles'):metadata['reused_original_full_forward_tiles']=geometry['reused_original_full_forward_tiles']
    if normal:
        # Clean complete grid is already immutable. Only labels/tile descriptions are new.
        st.put(p/'labels',dict(soft=soft.astype(np.float32)),dict(identity=ip.identity(a),item=item['item_id']))
    else:
        for tile in plans:
            y0,x0,y1,x1=tile['bounds'];ids=(np.arange(y0,y1)[:,None]*grid[1]+np.arange(x0,x1)).ravel().astype(np.int64)
            take=tiles.rows_for(positions,ids,grid)
            path=p/'tiles'/str(tile['tile']);meta=dict(identity=ip.identity(a),item_id=item['item_id'],tile=tile,grid=grid,physical_positions=ids.tolist())
            if (path/'manifest.json').exists():
                saved,rec=st.get(path)
                if any(rec['metadata'].get(k)!=v for k,v in meta.items()) or any(not np.array_equal(saved[f'q{l}'],q[l][take]) for l in LAYERS):raise ValueError('committed current tile differs from raw query')
            elif geometry.get('reused_original_full_forward_tiles'):
                origin=next(r for r in geometry['reused_original_full_forward_tiles'] if Path(r['path']).name==str(tile['tile']))
                if st.sealed_json(Path(origin['path'])/'manifest.json')['seal']!=origin['seal']:raise ValueError('raw tile origin modified during import')
                st.import_bundle(origin['path'],path,{f'q{l}':f'q{l}' for l in LAYERS},dict(meta,origin=origin['path'],origin_seal=origin['seal']))
            else:st.put(path,{f'q{l}':np.ascontiguousarray(q[l][take]) for l in LAYERS},meta)
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
        qn=np.ascontiguousarray(q[l][query_rows]);query=refs.search.query(qn);d,idx,diag,seal=refs.nearest_source(qn,source_name,l,query)
        source_arrays[f'd{l}']=d;source_arrays[f'i{l}']=idx;source_audit.append(diag);source_seals.append(seal)
        for phase,target in targets.items():
            d,index,diag,seals,offsets=refs.nearest_target(qn,target['policy']['target_origins'],l,query)
            target['arrays'][f'd{l}']=d;target['arrays'][f'i{l}']=index
            target['audits'].append(diag['support_audits']);target['reference_meta'].append(dict(layer=l,seals=seals,offsets=offsets))
        del query
    if sparse:
        source_arrays['positions']=nn_positions
        for target in targets.values():target['arrays']['positions']=nn_positions
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
    items=all_items(a);reliable_precision(a.device);folds=build_source_folds(a);model=None;calls=0;records=[];refs=References(a);resources=[];imported=0
    catalogue=ip.catalog(a);policies={}
    for n,item in enumerate(items,1):
        p=ip.item_path(a,item)
        if (p/'complete.json').exists():
            record=st.sealed_json(p/'complete.json')
            if record['identity']!=ip.identity(a) or record['item']!=item:raise ValueError('training query identity differs')
        else:
            from .reuse import training_item
            record=training_item(a,item,refs,catalogue,policies)
            if record is not None:
                imported+=1
                records.append(dict(item=item,path=str(p),seal=record['seal']))
                resources.append(dict(item_id=item['item_id'],origin=item['origin'],variant=item['variant'],full_grid_patches=int(np.prod(record['grid'])),
                    NN_patches=record.get('NN_patch_count',int(np.prod(record['grid']))),origin_kind='compatible_parent'))
                continue
            if item['variant']==-1:
                x,r=st.get(ip.normal_path(a,item['row']));q={l:x[f'q{l}'] for l in LAYERS};geometry=dict(grid=r['metadata']['grid'],positions=x['positions'])
                soft=np.zeros((item['row']['image_height']//4,item['row']['image_width']//4),np.float32)
            else:
                rgb=a.cache_root/'synthetic_rgb'/item['item_id'];origin=st.sealed_json(rgb/'manifest.json')
                if origin['seal']!=item['record_seal']:raise ValueError('synthetic image identity changed')
                st.verify_file(rgb/'image.png',origin['rgb_sha256'])
                from .reuse import tile_query
                cached=tile_query(a,item)
                if cached:q,geometry,soft=cached
                else:
                    if model is None:model,_=host(a)
                    with Image.open(rgb/'image.png') as image:q,geometry=extraction(model,image.convert('RGB'),a.device)
                    labels,lr=st.get(rgb/'labels',('soft_output',))
                    if lr['seal']!=origin['label_seal']:raise ValueError('synthetic label changed')
                    soft=labels['soft_output'];calls+=1
            record=prepare_one(a,item,q,geometry,soft,refs,catalogue=catalogue,policies=policies)
        records.append(dict(item=item,path=str(p),seal=record['seal']))
        resources.append(dict(item_id=item['item_id'],origin=item['origin'],variant=item['variant'],full_grid_patches=int(np.prod(record['grid'])),
            NN_patches=record.get('NN_patch_count',int(np.prod(record['grid']))),origin_kind='current_or_resumed'))
        if n%10==0:print(f'[residual training cache] {n}/{len(items)}',flush=True)
    st.sealed_json(a.run_root/'training_manifest.json',dict(identity=ip.identity(a),records=records,
        status='all_fixed_training_queries_prepared',GT_used=False))
    st.write_csv(a.run_root/'training_query_resources.csv',resources)
    from .reuse import flush
    flush(a,'prepare_training')
    return dict(status='training_queries_prepared',queries=len(items),new_synthetic_RGB_forwards=calls,folds=folds,
        reused_training_queries=imported,Exact_execution=refs.search.stats)


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
        if any(not v['normal'] for v in self.by_origin.values()):raise ValueError('missing normal query for an origin')
        self.valid_source=[p for p in self.source if self.by_origin[p]['synthetic']]
        self.valid_target=[p for p in self.target if self.by_origin[p]['synthetic']]
        if not self.valid_source or self.target and not self.valid_target:raise ValueError('required training population has no effective synthetic sample')
        self.normal_only=[p for p,v in self.by_origin.items() if not v['synthetic']]
        norm,nr=self.store.load(a.cache_root/'input_normalization');self.norm={k:np.array(v) for k,v in norm.items()}
        self.patch=st.sealed_json(a.run_root/'input_distance_mad.json')['patch'] if use_reference else None
        self.manifest_seal=manifest['seal']
    def select(self,seed,update,source_queries_only=False):
        rng=st.rng(VERSION,'batch',seed,self.phase,update);result=[]
        for i in range(CONFIG['effective_batch_images']):
            population=self.source if self.phase=='source' or source_queries_only or i<2 else self.target
            origin=population[int(rng.integers(len(population)))];kind='normal' if i%2==0 else 'synthetic'
            if kind=='synthetic' and not self.by_origin[origin]['synthetic']:
                valid=self.valid_source if population is self.source else self.valid_target
                origin=valid[int(rng.integers(len(valid)))]
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
            spos=sx.get('positions',np.asarray(meta['positions']));tpos=tx.get('positions',np.asarray(meta['positions']))
            for available,arrays in ((spos,sx),(tpos,tx)):
                if any(len(arrays[f'{k}{l}'])!=len(available) for k in ('d','i') for l in LAYERS):raise ValueError('NN row count differs from physical positions')
                if 'NN_physical_positions' in meta and not np.array_equal(np.sort(available),meta['NN_physical_positions']):raise ValueError('NN physical set differs from frozen tiles')
            reference=(sx,tx,policy,tiles.rows_for(spos,ids,grid),tiles.rows_for(tpos,ids,grid))
        values=[]
        for j,l in enumerate(LAYERS):
            q=torch.from_numpy(qs[l].reshape(*shape,1280).transpose(2,0,1).copy());kwargs={}
            if reference:
                sx,tx,policy,srows,trows=reference;si=sx[f'i{l}'][srows];ti=tx[f'i{l}'][trows]
                ss,tt,rm=self.refs.gather(policy['source'],policy['target_origins'],l,si,ti)
                target_meta=tr['metadata']['reference_meta'][j]
                if (sr['metadata']['source']!=policy['source'] or sr['metadata']['source_seals'][j]!=rm['source_seal']
                        or target_meta!=dict(layer=l,seals=rm['target_seals'],offsets=rm['target_offsets'])):raise ValueError('NN references changed after retrieval')
                kwargs=dict(source=torch.from_numpy(ss.reshape(*shape,1280).transpose(2,0,1).copy()),
                    target=torch.from_numpy(tt.reshape(*shape,1280).transpose(2,0,1).copy()),
                    distances=torch.from_numpy(np.stack((sx[f'd{l}'][srows],tx[f'd{l}'][trows])).reshape(2,*shape).copy()),
                    patch=(float(self.patch['medians'][j]),float(self.patch['scales'][j])))
            values.append(inputs(q,torch.from_numpy(self.norm[f'mu{l}']),float(self.norm[f'sigma{l}']),use_reference=self.use_reference,**kwargs)[None])
        labels,lr=self.store.load(p/'labels',('soft',))
        if lr['seal']!=meta['labels_seal']:raise ValueError('soft training label changed')
        return dict(values=values,soft=labels['soft'],origin=[y0,x0],grid=grid,shape=meta['output_shape'],center=tile['center'])
