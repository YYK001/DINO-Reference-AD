"""Small metadata proofs for original feature providers, never payload rehashes."""
from pathlib import Path
from importlib.metadata import version
from . import inputs as ip
st=ip.st
POSTPROCESS=('evaluate','aupro','summarize','pack')


def small_record(a,path):
    """Read sealed authority metadata once per unchanged file within a stage."""
    path=Path(path).resolve();info=path.stat();stamp=(info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    cache=getattr(a,'_feature_authorities',{})
    if path not in cache or cache[path][0]!=stamp:cache[path]=(stamp,st.sealed_json(path))
    a._feature_authorities=cache
    return cache[path][1]


def native_extraction(a):
    """Same dependency record as the raw provider, without reading checkpoint bytes.

    The original host independently validates the pinned weight on actual load.
    Only small Python/config files are read; no Torch import or CUDA probe.
    """
    from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import CHECKPOINT_SHA256
    cfg=a.official_root/'config.json';info=cfg.stat();stamp=(str(a.official_root),str(a.dinov3_root),info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    if getattr(a,'_native_feature_stamp',None)==stamp:return a._native_feature_record
    result=dict(checkpoint_sha256=CHECKPOINT_SHA256,host_config=st.read_json(cfg),
        precision=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True),
        software={n:version(n) for n in ('numpy','scipy','Pillow','torch','torchvision')},
        code={rel:st.sha(ip.ROOT/rel) for rel in ('DINOv3/relation_reliability/superadd_host.py',
            'DINOv3/ShiftADD/aebad_s_p0/memory.py','external_baselines/superadd_external/memory_variants.py',
            'DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/runtime.py')},
        external_code={name+'/'+p.relative_to(root).as_posix():st.sha(p)
            for name,root in [('dinov3',a.dinov3_root),('host',a.official_root)] for p in sorted(root.rglob('*.py'))})
    a._native_feature_stamp=stamp;a._native_feature_record=result
    return result


def expected(a):
    frozen=a.run_root/'inputs.json'
    if frozen.exists():
        record=small_record(a,frozen)
        if 'feature_inputs' not in record:raise ValueError('run predates feature-provider binding; use a new run, never rewrite old identities')
        return record['feature_inputs']['extraction']
    if a.parent_run:
        return small_record(a,a.parent_run/'assets.json')['references']['extraction']
    # A complete offline export supplies its extraction record without requiring
    # the backbone code or checkpoint entity to remain mounted.
    descriptors=[getattr(a,'feature_manifest',None)]
    for root in (a.raw_cache,a.feature_root):
        if root:descriptors.extend(root/name for name in ('feature_provider.json','raw_assets.json','inputs.json'))
    for p in descriptors:
        if p is not None and p.exists():
            r=small_record(a,p)
            if 'extraction' in r:
                value=r['extraction']
                from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import CHECKPOINT_SHA256
                required=('checkpoint_sha256','host_config','precision','software','code','external_code')
                if not all(k in value for k in required) or value['checkpoint_sha256']!=CHECKPOINT_SHA256:raise ValueError('provider needs the complete original pinned extraction record')
                original=st.read_json(ip.ROOT/'external_baselines/SuperADD/config.json')
                precision=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True)
                if value['host_config']!=original or value['precision']!=precision:raise ValueError('provider preprocessing/precision configuration differs from the original extraction')
                return value
    return native_extraction(a)


def compatible(actual,wanted):
    if actual!=wanted:raise ValueError('feature extraction configuration differs (checkpoint/preprocessing/precision/software/code); no mixed-provider reuse')


def authorities(a):
    """Known run records or an explicit sealed feature export; no disk search."""
    result=[];seen=set()
    def add(path,kind):
        p=Path(path).resolve()
        if p in seen or not p.exists():return
        seen.add(p);result.append((p,kind,small_record(a,p)))
    explicit=getattr(a,'feature_manifest',None)
    if explicit:add(explicit,'export')
    for root in (a.raw_cache,a.feature_root):
        if root:
            for name in ('feature_provider.json','raw_assets.json'):add(root/name,'export')
            add(root/'inputs.json','run')
    if a.parent_run:
        assets=small_record(a,a.parent_run/'assets.json')
        add(a.parent_run/'assets.json','parent')
        provider=assets['references'].get('parent_run')
        if provider:
            add(Path(provider)/'raw_assets.json','export')
            add(Path(provider)/'inputs.json','run')
    return result


def proof(a,path,row,manifest):
    """Bind extraction and the exact bundle manifest to an authority."""
    wanted=expected(a);m=manifest['metadata'];path=Path(path).resolve()
    # An inline record is valid only if it declares the complete extraction
    # dependency record; checkpoint alone is not a preprocessing proof.
    if 'extraction' in m:
        compatible(m['extraction'],wanted)
        bound=[]
        for file,kind,r in authorities(a):
            explicit=getattr(a,'feature_manifest',None)
            local=any(root is not None and file.parent==root.resolve() and path.is_relative_to(root.resolve()) for root in (a.raw_cache,a.feature_root))
            if kind=='export' and ((explicit is not None and file==explicit.resolve()) or local):
                compatible(r.get('extraction'),wanted)
                ref=r.get('normal_features',{}).get(row['relative_path']) or r.get('records',{}).get(row['relative_path'])
                if ref and (Path(ref['path']).resolve()!=path or ref['seal']!=manifest['seal']):raise ValueError('inline feature differs from provider manifest commitment')
                bound.append(dict(path=str(file),seal=r['seal']))
        return dict(kind='inline',extraction_identity=st.digest(wanted),authority=bound)
    for file,kind,r in authorities(a):
        refs=[];actual=None
        if kind=='parent':
            actual=r['references']['extraction']
            refs=[r.get('normal_features',{}).get(row['relative_path']),
                  r.get('queries',{}).get(row.get('domain','')+'/'+row.get('sample_id',''))]
        elif kind=='export':
            actual=r.get('extraction')
            refs=[r.get('normal_features',{}).get(row['relative_path']),r.get('records',{}).get(row['relative_path'])]
        else:
            root=r.get('cache',r.get('cache_root'));actual=r.get('extraction')
            if root and actual and path.is_relative_to(Path(root).resolve()) and m.get('identity')==r.get('identity'):
                compatible(actual,wanted)
                return dict(kind='run',extraction_identity=st.digest(actual),authority=[dict(path=str(file),seal=r['seal'])])
        for ref in refs:
            if ref and Path(ref['path']).resolve()==path:
                if ref['seal']!=manifest['seal']:raise ValueError('feature manifest differs from provider commitment: '+str(path))
                compatible(actual,wanted)
                return dict(kind=kind,extraction_identity=st.digest(actual),authority=[dict(path=str(file),seal=r['seal'])])
    raise ValueError('feature has no matching extraction/provider provenance: '+str(path)+'; supply --feature-manifest with a sealed raw_assets/feature_provider export')


def binding(ref):
    if ref is None:return None
    return {k:ref[k] for k in ('path','seal','names','positions','grid','provider')}


def inventory(a,c):
    wanted=expected(a);refs={}
    rows=[x for x in c['source'] if x['calibration']]+c['supports']['background']+c['samples']
    for row in rows:refs[row['relative_path']]=binding(ip.feature_ref(a,row))
    return dict(format='original_features_bound_v1',extraction=wanted,references=refs)


def verify_saved(a,saved):
    if 'feature_inputs' not in saved:raise ValueError('run predates feature-provider binding; use a new run; old manifests remain unchanged')
    if getattr(a,'stage',None) in POSTPROCESS:return
    # Compare available providers to frozen per-image decisions, including None
    # (fresh extraction). Adding a newly appearing provider is not silent reuse.
    current=inventory(a,ip.catalog(a))
    if current!=saved['feature_inputs']:raise ValueError('feature provider extraction/manifest changed since freeze; use a new run')
