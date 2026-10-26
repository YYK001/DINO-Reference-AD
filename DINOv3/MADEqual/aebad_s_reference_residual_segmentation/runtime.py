"""Explicit original RGB feature extraction and raw-only asset reuse."""
from pathlib import Path
import io
import numpy as np
from PIL import Image
from . import inputs as ip,storage as st
from .protocol import CONFIG,LAYERS,DOMAINS,VERSION
from . import synthesis as synth
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import (
    host,extraction,reliable_precision,source_scales)


def extraction_record(a):
    from importlib.metadata import version
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import PRECISION
    known=None
    if not st.full_audit():
        if (a.run_root/'inputs.json').exists():known=st.sealed_json(a.run_root/'inputs.json')['extraction']['checkpoint_sha256']
        else:
            for path in getattr(a,'recent_run',[]):
                try:provider=ip.recent(path)
                except (OSError,ValueError,KeyError):continue
                value=provider.get('inputs',{})
                known=value.get('extraction',{}).get('checkpoint_sha256') if provider['kind']=='reference_residual' else value.get('checkpoint_sha256')
                if known:break
    st.verify_file(a.checkpoint,known)
    return dict(checkpoint_sha256=known or st.sha(a.checkpoint),host_config=st.read_json(a.official_root/'config.json'),precision=PRECISION,
        software={n:version(n) for n in ('numpy','scipy','Pillow','torch','torchvision')},
        code={rel:st.sha(ip.ROOT/rel) for rel in ('DINOv3/relation_reliability/superadd_host.py',
            'DINOv3/ShiftADD/aebad_s_p0/memory.py','external_baselines/superadd_external/memory_variants.py',
            'DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/runtime.py')},
        external_code={name+'/'+p.relative_to(root).as_posix():st.sha(p)
            for name,root in [('dinov3',a.dinov3_root),('host',a.official_root)] for p in sorted(root.rglob('*.py'))})


def compatible(a,r,c,record):
    if r['catalog']['source']!=c['source'] or r['catalog']['supports']!=c['supports']:raise ValueError('Source/K8 order or image identity differs')
    if r['kind']=='reference_residual':
        if r['inputs']['extraction']!=record:raise ValueError('recent original extraction dependencies differ')
    elif r['kind']=='correspondence':
        old=r['inputs']
        for k in ('checkpoint_sha256','host_config','external_code'):
            if old[k]!=record[k]:raise ValueError('recent extraction identity differs: '+k)
        for n,v in record['software'].items():
            if old['software'].get(n)!=v:raise ValueError('recent extraction software differs: '+n)
        for rel,sha in record['code'].items():
            if old['code'].get(rel)!=sha:raise ValueError('recent extraction implementation differs: '+rel)
    else:
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reuse_compatible
        reuse_compatible(a,r['legacy'],c,record)


def raw_layers(x,m):
    grid=list(m['grid']);p=np.asarray(x['positions'],np.int64);n=int(np.prod(grid))
    if not np.array_equal(np.sort(p),np.arange(n)):raise ValueError('sampled/incomplete grid cannot replace full raw features')
    for l in LAYERS:
        q=x[f'q{l}']
        if q.shape!=(n,1280) or q.dtype!=np.float32 or st.full_audit() and not np.isfinite(q).all():raise ValueError('invalid raw feature layer')
    return grid,p


def physical(x,l,grid):
    q=np.empty_like(x[f'q{l}']);q[x['positions']]=x[f'q{l}']
    return q.reshape(*grid,1280)


def import_normal(a,row,providers):
    dest=ip.normal_path(a,row);meta0=dict(identity=ip.identity(a),image=row,kind='original_complete_grid')
    for r in providers:
        if r['kind']=='reference_residual':
            p=r['cache']/'normal_features'/ip.key(row)
            if not (p/'manifest.json').exists():continue
            x,record=st.get(p);raw_layers(x,record['metadata'])
            if record['metadata'].get('kind')!='original_complete_grid':raise ValueError('adapted/sampled features cannot replace original complete grids')
            if record['metadata']['image']!=row or st.sealed_json(r['run']/'prepared.json')['normal_features'][row['relative_path']]['seal']!=record['seal']:raise ValueError('parent original grid identity differs')
            st.import_bundle(p,dest,{**{f'q{l}':f'q{l}' for l in LAYERS},'positions':'positions'},
                dict(meta0,grid=record['metadata']['grid'],output_shape=record['metadata']['output_shape'],origin=str(p),origin_seal=record['seal']))
            return True
        if r['kind']=='correspondence':
            p=r['cache']/'features'/ip.key(row)
            if not (p/'manifest.json').exists():continue
            record=st.sealed_json(p/'manifest.json')
            if not all(f'full_L{l}' in record['files'] for l in LAYERS):continue  # Never reuse 2048 sampled rows.
            if record['metadata']['image']!=row:raise ValueError('normal origin differs')
            if st.sealed_json(r['run']/'prepared.json')['features'][row['relative_path']]!=record['seal']:raise ValueError('parent full feature not in frozen preparation')
            meta=dict(meta0,grid=record['metadata']['grid'],output_shape=record['metadata']['output_shape'],origin=str(p),origin_seal=record['seal'])
            st.import_bundle(p,dest,{**{f'q{l}':f'full_L{l}' for l in LAYERS},'positions':'full_positions'},meta);return True
        if r['kind']=='raw_manifest':
            value=r['record']['normal_features'].get(row['relative_path'])
            if not value:continue
            p=Path(value['path']);_,record=st.get(p)
            if record['seal']!=value['seal'] or record['metadata']['image']!=row:raise ValueError('explicit raw feature identity differs')
            st.import_bundle(p,dest,{**{f'q{l}':f'q{l}' for l in LAYERS},'positions':'positions'},
                dict(meta0,grid=record['metadata']['grid'],output_shape=record['metadata']['output_shape'],origin=str(p),origin_seal=record['seal']));return True
        if r['kind']=='legacy' and row.get('support_index'):
            p=r['cache']/'supports'/row['domain']/f"{row['support_index']:03d}.npz"
            if not p.exists():continue
            from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.storage import arrays
            x,record=arrays(p,r['identity']);grid=x['grid'].tolist()
            if str(x['relative_path'].item())!=row['relative_path']:raise ValueError('legacy support order differs')
            st.put(dest,{**{f'q{l}':x[f'layer_{l}'] for l in LAYERS},'positions':np.arange(np.prod(grid),dtype=np.int64)},
                dict(meta0,grid=grid,output_shape=[row['image_height']//4,row['image_width']//4],origin=str(p),origin_sha256=record['sha256']));return True
    return False


def fit_input_mad(a,c):
    """Original distance MAD only; network logits get a different later state."""
    import torch
    from DINOv3.relation_reliability.calibration import fit_calibrators
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.model import exact
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import fit_image
    source,_=st.get(a.cache_root/'source_original');banks={l:torch.tensor(np.array(source[f'L{l}']),device=a.device) for l in LAYERS}
    raws=[];records=[]
    for i,row in enumerate([r for r in c['source'] if r['calibration']],1):
        x,m=st.get(ip.normal_path(a,row));meta=m['metadata'];d=np.stack([exact(x[f'q{l}'],banks[l],a.device)[0] for l in LAYERS]);raws.append(d)
        records.append((d,dict(positions=x['positions'].tolist(),grid=meta['grid'],output_shape=meta['output_shape'])))
        print(f'[residual original calibration] {i}/66',flush=True)
    _,mad=fit_calibrators({l:np.concatenate([x[j] for x in raws]) for j,l in enumerate(LAYERS)})
    patch=dict(medians=mad.medians.tolist(),scales=mad.scales.tolist())
    return dict(patch=patch,image=fit_image(records,patch),scope='original_Source_only_distances',origin='current_common_original_bank')


def prepare(a):
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs as fi,fresh_runtime as fr
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import storage as oldst
    from types import SimpleNamespace,FunctionType
    c=fi.prepare_data(a.run_root/'data_index',a.data,a.cache_root);record=extraction_record(a);ip.freeze(a,c,record)
    ident=ip.identity(a);providers=[];audit=[]
    for p in a.recent_run:
        try:
            r=ip.recent(p);compatible(a,r,c,record);providers.append(r);audit.append(dict(run=str(p),status='compatible_raw_inputs'))
        except (OSError,ValueError,KeyError) as e:audit.append(dict(run=str(p),status='not_reused',reason=str(e)))
    for p in a.asset_manifest:
        rr=ip.raw_manifest(p)
        if rr['extraction']!=record or rr['catalog']['source']!=c['source'] or rr['catalog']['supports']!=c['supports']:raise ValueError('explicit raw assets are incompatible')
        providers.append(dict(kind='raw_manifest',record=rr,run=p.parent))
    st.write_json(a.run_root/'reuse_audit.json',dict(records=audit))
    st.sealed_json(a.run_root/'eligible_raw_providers.json',dict(identity=ident,providers=[dict(kind=r['kind'],run=str(r['run']),
        record=r['record'] if r['kind']=='raw_manifest' else None,cache=str(r.get('cache','')),parent_identity=r.get('identity')) for r in providers]))
    bank=a.cache_root/'source_original';baseline=None;chosen_provider=None
    if not (bank/'manifest.json').exists():
        for r in providers:
            if r['kind'] in ('correspondence','reference_residual','raw_manifest'):
                p=r['cache']/'source_original' if r['kind'] in ('correspondence','reference_residual') else Path(r['record']['source_bank']['path'])
                if not (p/'manifest.json').exists():continue
                _,br=st.get(p)
                if r['kind']=='reference_residual' and br['metadata'].get('kind')!='original_source':raise ValueError('adapted bank cannot replace original Source')
                if r['kind'] in ('correspondence','reference_residual') and st.sealed_json(r['run']/'prepared.json')['source_seal']!=br['seal']:raise ValueError('parent original Source bank changed')
                if r['kind']=='raw_manifest' and (br['seal']!=r['record']['source_bank']['seal'] or br['metadata'].get('kind')!='original_source'):raise ValueError('not an original Source bank')
                st.import_bundle(p,bank,{f'L{l}':f'L{l}' for l in LAYERS},dict(identity=ident,kind='original_source',origin=str(p),origin_seal=br['seal']));chosen_provider=r;break
            p=r['cache']/'source_bank.npz'
            if p.exists():
                x,br=oldst.arrays(p,r['identity']);st.put(bank,{f'L{l}':x[f'bank_{l}'] for l in LAYERS},
                    dict(identity=ident,kind='original_source',origin=str(p),origin_sha256=br['sha256']));chosen_provider=r;break
        else:
            build=a.run_root/'formal_source_build';st.write_json(build/'setup.json',dict(identity=ident));st.write_json(build/'fresh_catalog.json',c)
            args=SimpleNamespace(**vars(a));args.run_root=build
            fn=fr.source;FunctionType(fn.__code__,dict(fn.__globals__,host=host),fn.__name__,fn.__defaults__)(args)
            x,br=oldst.arrays(a.cache_root/'source_bank.npz',ident)
            st.put(bank,{f'L{l}':x[f'bank_{l}'] for l in LAYERS},dict(identity=ident,kind='original_source',origin='fresh_original_builder',builder=br['metadata']))
    else:
        _,br=st.get(bank);origin=br['metadata'].get('origin')
        chosen_provider=next((r for r in providers if r['kind'] in ('correspondence','reference_residual') and str(r['cache']/'source_original')==origin),None)
    # Baseline state can only come from the provider of the SAME original bank.
    if chosen_provider and chosen_provider['kind'] in ('correspondence','reference_residual'):
        parent_mad=chosen_provider['run']/('input_distance_mad.json' if chosen_provider['kind']=='reference_residual' else 'calibration/source/UNION_K8.json')
        # A missing optional state uses fit_input_mad below, after normal grids
        # are ready. Existing invalid records must still fail sealed validation.
        if parent_mad.exists() or parent_mad.is_symlink():
            old=st.sealed_json(parent_mad)
            baseline=dict(patch=old['patch'],image=old['image'],scope='original_Source_only_distances',
                origin=str(chosen_provider['run']),old_state_seal=old['seal'])
            st.sealed_json(a.run_root/'baseline_parent.json',dict(run=str(chosen_provider['run']),cache=str(chosen_provider['cache']),identity=chosen_provider['identity'],state=old))
    model=None;calls=0;reused=0;existing=0
    with fi.Data(Path(c['data_root'])) as data:
        rows=c['source']+sum(c['supports'].values(),[])
        for i,row in enumerate(rows,1):
            p=ip.normal_path(a,row)
            if not (p/'manifest.json').exists():
                if import_normal(a,row,providers):reused+=1
                else:
                    if model is None:model,_=host(a)
                    q,m=extraction(model,data.image(row),a.device);calls+=1
                    st.put(p,{**{f'q{l}':q[l] for l in LAYERS},'positions':np.asarray(m['positions'],np.int64)},
                        dict(identity=ident,image=row,kind='original_complete_grid',grid=m['grid'],
                            output_shape=[row['image_height']//4,row['image_width']//4],origin='fresh_original_host'))
            else:existing+=1
            x,r=st.get(p);raw_layers(x,r['metadata'])
            if r['metadata']['identity']!=ident or r['metadata']['image']!=row:raise ValueError('normal feature identity changed')
            print(f'[residual normal grids] {i}/{len(rows)} {row["relative_path"]} existing={existing} imported={reused} new_forwards={calls} audit={st.AUDIT_MODE}',flush=True)
    baseline_path=a.run_root/'input_distance_mad.json'
    if not baseline_path.exists():
        if baseline is None:baseline=fit_input_mad(a,c)
        st.sealed_json(baseline_path,dict(identity=ident,**baseline))
    source,br=st.get(bank);stats={}
    for l in LAYERS:
        if source[f'L{l}'].shape!=(CONFIG['bank_rows'],1280) or source[f'L{l}'].dtype!=np.float32:raise ValueError('formal original bank dimensions differ')
        mu,sigma=source_scales(source[f'L{l}']);stats[f'mu{l}']=mu;stats[f'sigma{l}']=np.asarray(sigma)
    # Fixed power iteration of Source PC1, not fitted on Target/test/GT.
    sampled=np.array(source['L31'][::CONFIG['foreground']['bank_stride']]);sampled-=stats['mu31']
    axis=st.rng(VERSION,'PC1').normal(size=1280).astype(np.float32);axis/=np.linalg.norm(axis)
    for _ in range(CONFIG['foreground']['iterations']):
        axis=sampled.T@(sampled@axis);norm=np.linalg.norm(axis)
        if not np.isfinite(norm) or norm<=0:raise ValueError('unavailable Source PCA foreground axis')
        axis/=norm
    stats['axis']=axis;st.put(a.cache_root/'input_normalization',stats,dict(identity=ident,source_seal=br['seal']))
    foreground_rows=[]
    for row in ip.train_source(c)+sum(c['supports'].values(),[]):
        x,r=st.get(ip.normal_path(a,row),('q31','positions'));grid=r['metadata']['grid']
        mask,diag=synth.foreground(physical(x,31,grid),axis,stats['mu31'])
        st.put(a.cache_root/'foreground'/ip.key(row),dict(mask=mask),dict(identity=ident,origin=row['relative_path'],diagnostics=diag))
        foreground_rows.append(dict(relative_path=row['relative_path'],**diag))
    st.write_csv(a.run_root/'foreground_audit.csv',foreground_rows)
    value=dict(identity=ident,source_seal=br['seal'],normal_features={row['relative_path']:dict(path=str(ip.normal_path(a,row)),
        seal=st.sealed_json(ip.normal_path(a,row)/'manifest.json')['seal']) for row in rows})
    st.sealed_json(a.run_root/'prepared.json',value)
    st.sealed_json(a.run_root/'raw_assets.json',dict(format='reference_residual_raw_assets_v1',catalog=c,extraction=record,
        source_bank=dict(path=str(bank),seal=br['seal']),normal_features=value['normal_features']))
    return dict(status='original_assets_ready',normal_forward_calls=calls,reused_complete_grids=reused,existing_complete_grids=existing,
        original_Source_origin=br['metadata']['origin'],Source_fold_banks='new in prepare-training')


def synthesize(a):
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    from . import reuse
    c=ip.catalog(a);cfg=ip.config(a);rows={r['relative_path']:r for r in ip.train_source(c)+sum(c['supports'].values(),[])};items=[];summaries=[]
    with Data(Path(c['data_root'])) as data:
        for row in rows.values():
            x,r=st.get(a.cache_root/'foreground'/ip.key(row));mask=x['mask'];image=np.asarray(data.image(row))
            # Foreground preview uses only clean normal RGB and auxiliary foreground.
            preview=image.copy();fg=synth.area_resize(mask.astype(np.float32),image.shape[:2])>=.5
            preview[fg]=np.round(.6*preview[fg]+.4*np.array([0,255,0])).astype(np.uint8)
            path=a.run_root/'previews/foreground'/(ip.key(row)+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            picture=Image.fromarray(preview);picture.thumbnail((512,512));picture.save(path)
        for item in synth.manifest(c,cfg):
            p=a.cache_root/'synthetic_rgb'/item['item_id'];meta=p/'manifest.json'
            if not meta.exists():reuse.synthetic(a,item)
            if not meta.exists():
                row=rows[item['origin']];donor=rows[item['donor']];fg,fr=st.get(a.cache_root/'foreground'/ip.key(row))
                clean=np.asarray(data.image(row));other=np.asarray(data.image(donor))
                if fr['metadata']['diagnostics']['status']!='valid_auxiliary_not_GT':value=None;diag=dict(status='invalid_foreground',attempts=0)
                else:value,diag=synth.generate(clean,other,fg['mask'],row['relative_path'],item['variant'])
                p.mkdir(parents=True,exist_ok=True)
                if value is not None:
                    rgb=p/'image.png';Image.fromarray(value.pop('rgb')).save(rgb)
                    value['soft_output']=synth.area_resize(value.pop('soft'),(row['image_height']//4,row['image_width']//4))
                    value['change']=np.round(value['change']*255).astype(np.uint8)
                    st.put(p/'labels',value,dict(identity=ip.identity(a),item=item))
                    if item['variant']<4:
                        picture=Image.open(rgb).convert('RGB');picture.thumbnail((512,512));dest=a.run_root/'previews/synthetic'/(item['item_id']+'.png');dest.parent.mkdir(parents=True,exist_ok=True);picture.save(dest)
                    diag.update(rgb_sha256=st.sha(rgb),label_seal=st.sealed_json(p/'labels/manifest.json')['seal'])
                st.sealed_json(meta,dict(identity=ip.identity(a),item=item,**diag))
            r=st.sealed_json(meta)
            if r['identity']!=ip.identity(a) or r['item']!=item:raise ValueError('synthesis inputs changed')
            if r['status']=='valid':st.verify_file(p/'image.png',r['rgb_sha256'])
            items.append(dict(**item,status=r['status'],record_seal=r['seal']))
            summaries.append(dict(origin=item['origin'],domain=rows[item['origin']].get('domain','Source'),variant=item['variant'],kind=item['kind'],
                scale=('small','medium','large')[(item['variant']+st.seed(item['origin'])%3)%3],status=r['status'],reused='origin_seal' in r))
    st.sealed_json(a.run_root/'synthetic_manifest.json',dict(identity=ip.identity(a),items=items,
        valid=sum(r['status']=='valid' for r in items),invalid=sum(r['status']!='valid' for r in items)))
    from collections import Counter
    counts=Counter((r['domain'],r['kind'],r['scale'],r['status']) for r in summaries)
    st.write_csv(a.run_root/'synthetic_type_scale_counts.csv',[dict(domain=d,kind=k,scale=s,status=status,count=n) for (d,k,s,status),n in sorted(counts.items())])
    byorigin=[]
    for origin,row in rows.items():
        selected=[r for r in summaries if r['origin']==origin];valid=sum(r['status']=='valid' for r in selected)
        byorigin.append(dict(origin=origin,population=row.get('domain','Source'),planned=len(selected),valid=valid,normal_only=valid==0))
    st.write_csv(a.run_root/'synthetic_origin_summary.csv',byorigin);reuse.flush(a,'synthesize')
    return dict(status='fixed_synthetic_pool_recorded',slots=len(items),invalid=sum(r['status']!='valid' for r in items))


def existing_query(a,row):
    """Only proven full ORIGINAL Q; never recover vectors from distances or z."""
    file=a.run_root/'eligible_raw_providers.json'
    if not file.exists():return None
    for provider in st.sealed_json(file)['providers']:
        if provider['kind']=='reference_residual':
            p=Path(provider['cache'])/'queries'/row['domain']/row['sample_id']
            if not (p/'manifest.json').exists():continue
            x,r=st.get(p);m=r['metadata']
            if m['identity']!=provider['parent_identity'] or m['sample']!=row:raise ValueError('parent raw query identity differs')
            geometry=dict(grid=m['grid'],positions=x['positions']);q={l:x[f'q{l}'] for l in LAYERS}
            raw_layers({**{f'q{l}':q[l] for l in LAYERS},'positions':x['positions']},geometry)
            return q,geometry,dict(origin=str(p),seal=r['seal'])
        if provider['kind']=='correspondence':
            p=Path(provider['cache'])/'queries'/row['domain']/row['sample_id']
            if not (p/'manifest.json').exists():continue
            x,r=st.get(p,tuple(f'L{l}' for l in LAYERS));m=r['metadata']
            if m['identity']!=provider['parent_identity'] or m['sample']!=row:raise ValueError('parent original Q identity differs')
            geometry=m['geometry'];q={l:x[f'L{l}'] for l in LAYERS}
        elif provider['kind']=='raw_manifest':
            entry=provider['record'].get('queries',{}).get(row['domain']+'/'+row['sample_id'])
            if entry is None:continue
            x,r=st.get(entry['path'])
            if r['seal']!=entry['seal'] or r['metadata']['sample']!=row:raise ValueError('explicit original Q identity differs')
            geometry=dict(grid=r['metadata']['grid'],positions=x['positions']);q={l:x[f'q{l}'] for l in LAYERS}
        else:continue
        geometry=dict(geometry,positions=np.asarray(geometry['positions'],np.int64))
        raw_layers({**{f'q{l}':q[l] for l in LAYERS},'positions':geometry['positions']},geometry)
        return q,geometry,dict(origin=str(p) if provider['kind']=='correspondence' else entry['path'],seal=r['seal'])
    return None
