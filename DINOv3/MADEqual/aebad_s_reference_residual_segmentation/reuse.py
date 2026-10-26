"""Read-only compatibility imports by actual dependencies, never by run budget."""
import ast
import inspect
from pathlib import Path
import shutil
import numpy as np
from . import inputs as ip,storage as st,protocol as pr,tiles


def rules(config):
    def functions(file,names):
        tree=ast.parse(Path(__file__).with_name(file).read_text(encoding='utf-8'))
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    keys=('synthesis_seed','synthesis_cycle','retry_limit','scales','scratch_width_pixels','poisson_mode','texture_alpha',
        'change_minimum','soft_change_scale','minimum_changed_pixels','foreground','output_downscale')
    random_rule={name:ast.dump(ast.parse(inspect.getsource(getattr(st,name))),include_attributes=False) for name in ('seed','rng','digest')}
    synth=dict(version=pr.SYNTHESIS_VERSION,random_rule=random_rule,config={k:config[k] for k in keys},
        implementation=functions('synthesis.py',('generate','foreground','changed_label','area_resize')))
    geom=dict(random_rule=random_rule,config={k:config[k] for k in ('tile_center','halo','tile_normal_probability','dimension','distance_divisor','precision')},
        implementation=functions('tiles.py',('bounds','plan','output_window')),
        exact=st.sha(ip.ROOT/'DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/model.py'))
    fold={k:config[k] for k in ('bank_rows','sampler_rounds','source_bank_augmentation','fold','sim_supports')}
    return dict(synthesis=st.digest(synth),geometry=st.digest(geom),fold=st.digest(fold))


def compatible_rules(provider,config,scope):
    value=provider['inputs'];saved=value.get('artifact_rules')
    if saved is not None:return saved[scope]==rules(config)[scope]
    old=value.get('config',{})
    if old!=pr.CONFIG:return False
    names={'synthesis':('synthesis.py','protocol.py','runtime.py'),
        'geometry':('training_data.py','tiles.py','protocol.py','retrieval.py'),
        'fold':('retrieval.py','protocol.py','inputs.py')}[scope]
    record=st.read_json(Path(__file__).with_name('legacy_full_v1_compatibility.json'));known=record['code']
    return (all(value['code'].get('DINOv3/MADEqual/aebad_s_reference_residual_segmentation/'+n)==known[n] for n in names)
        and all(st.sha(ip.ROOT/p)==sha for p,sha in record['shared_RNG'].items()))


def sources(a):
    if hasattr(a,'_residual_providers'):return a._residual_providers
    p=a.run_root/'eligible_raw_providers.json'
    records=[r for r in st.sealed_json(p)['providers'] if r['kind']=='reference_residual'] if p.exists() else [dict(run=str(r),parent_identity=None) for r in getattr(a,'recent_run',[])]
    a._residual_providers=[]
    for record in records:
        value=ip.recent(Path(record['run']))
        if value['kind']!='reference_residual':continue
        if record.get('parent_identity') is not None and value['identity']!=record['parent_identity']:raise ValueError('frozen parent provider identity changed')
        a._residual_providers.append(value)
    return a._residual_providers


def note(a,kind,status,parent,reason,**extra):
    if not hasattr(a,'_asset_imports'):a._asset_imports=[]
    a._asset_imports.append(dict(kind=kind,status=status,parent=str(parent),reason=reason,**extra))


def flush(a,stage):st.write_csv(a.run_root/'reuse'/f'{stage}.csv',getattr(a,'_asset_imports',[]))


def content(path,keys=None):
    r=st.sealed_json(Path(path)/'manifest.json')
    return {k:r['files'][k] for k in (keys or r['files'])}


def fold(a,name,expected):
    for parent in sources(a):
        p=parent['cache']/'source_folds'/name
        if not (p/'manifest.json').exists():continue
        x,r=st.get(p);m=r['metadata']
        if not compatible_rules(parent,ip.config(a),'fold') or any(m.get(k)!=v for k,v in expected.items() if k!='identity'):
            note(a,'fold','not_reused',p,'sampler/origin/config dependencies differ');continue
        if m['identity']!=parent['identity']:raise ValueError('parent fold identity differs')
        meta=dict(expected,candidate_ledger=m['candidate_ledger'],origin=str(p),origin_seal=r['seal'])
        st.import_bundle(p,a.cache_root/'source_folds'/name,{k:k for k in r['files']},meta)
        note(a,'fold','reused',p,'same extraction, origin list, augmentation and sampler; run budget ignored')
        return True
    return False


def synthetic(a,item):
    """Includes already committed items of an interrupted old synthesize stage."""
    for parent in sources(a):
        p=parent['cache']/'synthetic_rgb'/item['item_id'];file=p/'manifest.json'
        if not file.exists():continue
        r=st.sealed_json(file)
        if r['identity']!=parent['identity']:raise ValueError('parent RGB identity differs')
        foreground=parent['cache']/'foreground'/ip.key(dict(relative_path=item['origin']))
        if not (foreground/'manifest.json').exists():
            note(a,'synthetic','unconfirmed',p,'missing parent foreground dependency',item_id=item['item_id']);continue
        compatible=(r['item']==item and compatible_rules(parent,ip.config(a),'synthesis') and
            content(foreground,('mask',))==
            content(a.cache_root/'foreground'/ip.key(dict(relative_path=item['origin'])),('mask',)))
        if not compatible:
            note(a,'synthetic','not_reused',p,'variant/donor/generation/foreground dependency differs',item_id=item['item_id']);continue
        dest=a.cache_root/'synthetic_rgb'/item['item_id'];value={k:v for k,v in r.items() if k not in ('seal','identity','label_seal')}
        value.update(identity=ip.identity(a),origin=str(p),origin_seal=r['seal'])
        if r['status']=='valid':
            if st.sha(p/'image.png')!=r['rgb_sha256']:raise ValueError('parent synthetic RGB modified')
            labels,lr=st.get(p/'labels')
            if lr['seal']!=r['label_seal']:raise ValueError('parent synthetic labels modified')
            dest.mkdir(parents=True,exist_ok=True);rgb=dest/'image.png'
            if rgb.exists():
                if st.sha(rgb)!=r['rgb_sha256']:raise ValueError('uncommitted current RGB differs')
            else:
                try:rgb.hardlink_to((p/'image.png').resolve())
                except OSError:shutil.copyfile(p/'image.png',rgb)
            imported=st.import_bundle(p/'labels',dest/'labels',{k:k for k in lr['files']},dict(identity=ip.identity(a),item=item,origin=str(p/'labels'),origin_seal=lr['seal']))
            value['label_seal']=imported['seal']
        st.sealed_json(dest/'manifest.json',value);note(a,'synthetic','reused',p,'same original ID/donor/RGB rule/foreground/label geometry',item_id=item['item_id'])
        return True
    return False


def item_semantics(item):return {k:v for k,v in item.items() if k!='record_seal'}


def tile_query(a,item):
    """Recover only consumed raw rows from committed original-full-forward tiles."""
    if ip.config(a)['training_nn_layout']!='physical_tile_union_v1':return None
    candidates=list(sources(a));own=ip.item_path(a,item)
    if (own/'tiles/0/manifest.json').exists() and not (own/'complete.json').exists():
        candidates.insert(0,dict(kind='reference_residual',run=a.run_root,cache=a.cache_root,identity=ip.identity(a),inputs=st.sealed_json(a.run_root/'inputs.json')))
    for parent in candidates:
        p=parent['cache']/'training_queries'/item['item_id'];first=p/'tiles/0/manifest.json'
        if not first.exists() or not (p/'labels/manifest.json').exists():continue
        if not compatible_rules(parent,ip.config(a),'geometry') or not compatible_rules(parent,ip.config(a),'synthesis'):continue
        oldrgb=parent['cache']/'synthetic_rgb'/item['item_id'];newrgb=a.cache_root/'synthetic_rgb'/item['item_id']
        if not (oldrgb/'manifest.json').exists():continue
        old=st.sealed_json(oldrgb/'manifest.json');new=st.sealed_json(newrgb/'manifest.json')
        if old['item']!=new['item'] or any(item.get(k)!=v for k,v in old['item'].items()):continue
        if old['rgb_sha256']!=new['rgb_sha256'] or st.sha(oldrgb/'image.png')!=old['rgb_sha256']:continue
        labels,lr=st.get(p/'labels');current,cr=st.get(newrgb/'labels',('soft_output',))
        if cr['seal']!=new['label_seal'] or not np.array_equal(labels['soft'],current['soft_output']):raise ValueError('cached tile label differs from current RGB label')
        complete=st.sealed_json(p/'complete.json') if (p/'complete.json').exists() else None
        if complete and (complete['identity']!=parent['identity'] or complete['labels_seal']!=lr['seal'] or item_semantics(complete['item'])!=item_semantics(item)):raise ValueError('parent tile completion identity differs')
        grid=st.sealed_json(first)['metadata']['grid'];plans=tiles.plan(labels['soft'],grid,item['item_id'])
        if any(not (p/'tiles'/str(t['tile'])/'manifest.json').exists() for t in plans):continue
        needed=tiles.physical_union(plans,grid);q={l:np.empty((len(needed),1280),np.float32) for l in pr.LAYERS};seen=np.zeros(len(needed),bool);origins=[]
        for tile in plans:
            path=p/'tiles'/str(tile['tile']);x,r=st.get(path);m=r['metadata']
            if complete and complete['query_tile_seals'][str(tile['tile'])]!=r['seal']:raise ValueError('cached query tile seal differs from parent completion')
            y,x0,yy,xx=tile['bounds'];ids=(np.arange(y,yy)[:,None]*grid[1]+np.arange(x0,xx)).ravel()
            if m['identity']!=parent['identity'] or m['item_id']!=item['item_id'] or m['tile']!=tile or m['physical_positions']!=ids.tolist():raise ValueError('cached original RGB tile geometry differs')
            rows=tiles.rows_for(needed,ids,grid)
            for l in pr.LAYERS:
                v=x[f'q{l}']
                if v.shape!=(len(ids),1280) or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('cached original tile feature layout differs')
                if not np.array_equal(q[l][rows[seen[rows]]],v[seen[rows]]):raise ValueError('overlapping cached tile feature rows differ')
                q[l][rows]=v
            seen[rows]=True;origins.append(dict(path=str(path),seal=r['seal']))
        if not seen.all():raise ValueError('cached raw tiles do not cover required NN positions')
        note(a,'raw_tiles','reused',p,'same RGB/extraction/label/geometry; only missing NN recomputed',item_id=item['item_id'])
        return q,dict(grid=grid,positions=needed,reused_original_full_forward_tiles=origins),np.array(labels['soft'])
    return None


def training_item(a,item,refs,catalogue,policies):
    """Import real tile/label/NN payloads; full old NN can provide sparse rows."""
    cfg=ip.config(a);dest=ip.item_path(a,item)
    for parent in sources(a):
        p=parent['cache']/'training_queries'/item['item_id'];file=p/'complete.json'
        if not file.exists():continue
        old=st.sealed_json(file)
        if old['identity']!=parent['identity']:raise ValueError('parent training item identity differs')
        if (item_semantics(old['item'])!=item_semantics(item) or not compatible_rules(parent,cfg,'geometry') or
                not compatible_rules(parent,cfg,'synthesis')):
            note(a,'training_query','not_reused',p,'item/geometry/synthesis dependencies differ',item_id=item['item_id']);continue
        phases=('source',*pr.DOMAINS) if item['row'].get('source_index') is not None else (item['row']['domain'],)
        if not (p/'source_nn/manifest.json').exists() or any(not (p/'target_nn'/phase/'manifest.json').exists() for phase in phases):
            note(a,'training_query','unconfirmed',p,'missing NN; compatible raw tiles may be reused independently',item_id=item['item_id']);continue
        normal=item['variant']==-1;origin=item['origin'];grid=old['grid'];positions=np.asarray(old['positions'],np.int64)
        if normal:
            original=parent['cache']/'normal_features'/ip.key(item['row'])
            if st.sealed_json(original/'manifest.json')['seal']!=old['normal_feature_seal']:raise ValueError('parent clean feature changed')
            if content(original)!=content(ip.normal_path(a,item['row'])):continue
        else:
            original=parent['cache']/'synthetic_rgb'/item['item_id'];rgb=st.sealed_json(original/'manifest.json')
            current=st.sealed_json(a.cache_root/'synthetic_rgb'/item['item_id']/'manifest.json')
            if rgb['rgb_sha256']!=current['rgb_sha256'] or st.sha(original/'image.png')!=rgb['rgb_sha256']:continue
            if content(original/'labels',('soft_output',))['soft_output']!=content(p/'labels',('soft',))['soft']:raise ValueError('parent training label differs from generated RGB label')
        labels,lr=st.get(p/'labels')
        if lr['seal']!=old['labels_seal']:raise ValueError('parent training label modified')
        if old['tiles']!=tiles.plan(labels['soft'],grid,item['item_id']):continue
        needed=tiles.physical_union(old['tiles'],grid) if cfg['training_nn_layout']=='physical_tile_union_v1' else positions
        sx,sr=st.get(p/'source_nn')
        if sr['seal']!=old['source_seal']:raise ValueError('parent Source NN modified')
        policies0={phase:ip.policy(catalogue,item['row'],phase) for phase in phases};source_name=policies0[phases[0]]['source']
        oldbank=parent['cache']/('source_original' if source_name=='formal' else 'source_folds/'+source_name)
        oldnormals={parent['cache']/'normal_features'/ip.key(dict(relative_path=path)) for v in policies0.values() for path in v['target_origins']}
        if any(not (path/'manifest.json').exists() for path in (oldbank,*oldnormals)):
            note(a,'training_query','unconfirmed',p,'parent reference manifest missing; raw tiles may still be reused',item_id=item['item_id']);continue
        compatible=sr['metadata']['source']==source_name;source_seals=[];target_states={}
        for j,l in enumerate(pr.LAYERS):
            oldbank=parent['cache']/('source_original' if source_name=='formal' else 'source_folds/'+source_name)
            if st.sealed_json(oldbank/'manifest.json')['seal']!=sr['metadata']['source_seals'][j]:raise ValueError('parent NN Source reference changed')
            _,seal=refs.source(source_name,l);source_seals.append(seal)
            if content(oldbank,(f'L{l}',))!=content(a.cache_root/('source_original' if source_name=='formal' else 'source_folds/'+source_name),(f'L{l}',)):compatible=False
        for phase,policy in policies0.items():
            tx,tr=st.get(p/'target_nn'/phase)
            if tr['seal']!=old['contexts'][phase]['target_seal']:raise ValueError('parent Target NN modified')
            if tr['metadata']['policy']!=policy:compatible=False
            new_meta=[]
            for j,l in enumerate(pr.LAYERS):
                _,seals,offsets=refs.parts(policy['target_origins'],l);new_meta.append(dict(layer=l,seals=seals,offsets=offsets))
                oldmeta=tr['metadata']['reference_meta'][j]
                if offsets!=oldmeta['offsets']:compatible=False
                for k,path in enumerate(policy['target_origins']):
                    oldnormal=parent['cache']/'normal_features'/ip.key(dict(relative_path=path))
                    if st.sealed_json(oldnormal/'manifest.json')['seal']!=oldmeta['seals'][k]:raise ValueError('parent NN Target reference changed')
                    if content(oldnormal,(f'q{l}',))!=content(ip.normal_path(a,dict(relative_path=path)),(f'q{l}',)):compatible=False
            target_states[phase]=(tx,tr,new_meta,policy)
        if not compatible:
            note(a,'training_query','not_reused',p,'reference content or exclusion policy differs',item_id=item['item_id']);continue
        source_rows=tiles.rows_for(sx.get('positions',positions),needed,grid)
        arrays={k:np.asarray(v[source_rows]) for k,v in sx.items() if k!='positions'};arrays['positions']=needed
        source_record=st.put(dest/'source_nn',arrays,dict(identity=ip.identity(a),source=source_name,source_seals=source_seals,
            origin=str(p/'source_nn'),origin_seal=sr['seal'],vector_audits=sr['metadata'].get('vector_audits')))
        contexts={}
        for phase,(tx,tr,new_meta,policy) in target_states.items():
            rows=tiles.rows_for(tx.get('positions',positions),needed,grid);arrays={k:np.asarray(v[rows]) for k,v in tx.items() if k!='positions'};arrays['positions']=needed
            rec=st.put(dest/'target_nn'/phase,arrays,dict(identity=ip.identity(a),policy=policy,reference_meta=new_meta,origin=str(p/'target_nn'/phase),origin_seal=tr['seal']))
            contexts[phase]=dict(policy=policy,target_seal=rec['seal'])
        label=st.import_bundle(p/'labels',dest/'labels',{'soft':'soft'},dict(identity=ip.identity(a),item=item['item_id'],origin=str(p/'labels'),origin_seal=lr['seal']))
        tile_seals={}
        if not normal:
            for tile in old['tiles']:
                oldpath=p/'tiles'/str(tile['tile']);tr=st.sealed_json(oldpath/'manifest.json')
                if tr['seal']!=old['query_tile_seals'][str(tile['tile'])]:raise ValueError('parent query tile modified')
                rec=st.import_bundle(oldpath,dest/'tiles'/str(tile['tile']),{f'q{l}':f'q{l}' for l in pr.LAYERS},dict(tr['metadata'],identity=ip.identity(a),origin=str(oldpath),origin_seal=tr['seal']))
                tile_seals[str(tile['tile'])]=rec['seal']
        meta=dict(old);meta.pop('seal');meta.update(identity=ip.identity(a),item=item,source_seal=source_record['seal'],contexts=contexts,
            labels_seal=label['seal'],query_tile_seals=tile_seals,normal_feature_path=str(ip.normal_path(a,item['row'])) if normal else None,
            normal_feature_seal=st.sealed_json(ip.normal_path(a,item['row'])/'manifest.json')['seal'] if normal else None,
            NN_layout=cfg['training_nn_layout'],NN_physical_positions=needed.tolist(),NN_patch_count=len(needed),full_grid_patch_count=int(np.prod(grid)),
            origin=str(p),origin_seal=old['seal'])
        note(a,'training_query','reused',p,'RGB/features/labels/geometry/reference content/policy verified; select old NN physical rows',item_id=item['item_id'])
        return st.sealed_json(dest/'complete.json',meta)
    return None
