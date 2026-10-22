"""Read-only inherited identities and complete original support NPZ inventory."""
import contextlib
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_gap_soft_segment_p0 import inputs as parent
from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import PROTOCOL_ID
from .method import PROTOCOL,LAYERS
from .storage import sha,digest,read_json

DOMAINS=parent.DOMAINS


@contextlib.contextmanager
def feature_stream(location):
    """Portable file or archive::member, without extracting to C: or replacing inputs."""
    if '::' not in str(location):
        with Path(location).open('rb') as f:yield f
    else:
        file,member=str(location).split('::',1)
        if file.lower().endswith('.zip'):
            with zipfile.ZipFile(file) as z:
                with z.open(member) as f:yield io.BytesIO(f.read())
        else:
            with tarfile.open(file,'r:*') as t:
                f=t.extractfile(member)
                if f is None:raise ValueError('archive member is not a file')
                with f:yield f


def feature_header(location,expected):
    with feature_stream(location) as f:
        h=hashlib.sha256()
        for blob in iter(lambda:f.read(1<<20),b''):h.update(blob)
        if h.hexdigest()!=expected['feature_sha256']:raise ValueError('historical support SHA differs')
        f.seek(0)
        with np.load(f,allow_pickle=False) as z:
            if (str(z['protocol_id'].item())!=PROTOCOL_ID or str(z['relative_path'].item())!=expected['relative_path']
                or z['grid'].tolist()!=expected['grid']):raise ValueError('historical support protocol/path/grid differs')
        f.seek(0)
        with zipfile.ZipFile(f) as z:
            for l in LAYERS:
                with z.open(f'layer_{l}.npy') as stream:
                    version=np.lib.format.read_magic(stream)
                    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0 if version==(2,0) else None
                    if reader is None:raise ValueError('unsupported NPY header')
                    shape,_,dtype=reader(stream)
                    if shape!=(int(np.prod(expected['grid'])),1280) or dtype!=np.float32:raise ValueError('complete FP32 original features required')
    return dict(location=str(location),sha256=h.hexdigest(),header_verified=True,
                finiteness='deferred to explicit support loading')


def load_layer(records,locations,domain,layer):
    banks={}
    for row in records[domain]:
        key=(domain,row['support_index'])
        if key not in locations:raise FileNotFoundError(f'missing original support features: {key}')
        location=locations[key];feature_header(location,row)
        with feature_stream(location) as f:
            with np.load(f,allow_pickle=False) as z:bank=z[f'layer_{layer}'].copy()
        if bank.dtype!=np.float32 or bank.shape!=(int(np.prod(row['grid'])),1280) or not np.isfinite(bank).all():raise ValueError('invalid/nonfinite complete support features')
        banks[row['support_index']]=bank
    return banks


class Bundle:
    def __init__(self,paths=None,recovered_supports=None):
        self.paths=paths or parent.defaults();self.parent=parent.Bundle(self.paths)
        p=self.parent;self.protocol=PROTOCOL;self.samples=p.samples;self.supports=p.supports
        self.medians,self.scales,self.state=p.medians,p.scales,p.state
        self.bg,self.instances,self.references,self.maps,self.metrics=p.bg,p.instances,p.references,p.maps,p.metrics
        self.expected={d:[] for d in DOMAINS}
        for row in p.supports['background']:
            self.expected['background'].append(dict(row,domain='background'))
        supports=read_json(Path(self.paths['parent_run'])/'cache/cache_status.json')['supports']
        for d in DOMAINS[1:]:
            for sample in p.supports[d]:
                rows=[x for x in supports if (x['domain'],x['support_index'])==(d,sample['support_index'])]
                if len(rows)!=1:raise ValueError('one historical feature record per original support required')
                row=rows[0]
                if any(row[k]!=sample[k] for k in ('support_index','relative_path','image_sha256')):raise ValueError('support history mismatch')
                self.expected[d].append(dict(sample,domain=d,feature_sha256=row['sha256'],grid=row['grid'],historical_file=row['file']))
        # Original support identities are preserved; this tool has its own code identity.
        self.frozen=dict(protocol=PROTOCOL,parent=p.frozen,original_support_features=self.expected,
            code={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
            shared_geometry_sha256=sha(Path(parent.ROOT)/'DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py'),
            original_backend_sha256=sha(Path(parent.ROOT)/'DINOv3/relation_reliability/official_metrics.py'))
        self.protected_paths=list(p.paths.values())+list(p.bg['paths'].values())
        self.original_expected={d:[dict(r) for r in rows] for d,rows in self.expected.items()}
        if recovered_supports is not None:
            from .recovery import attach
            attach(self,Path(recovered_supports))
        self.identity=digest(self.frozen)

    def load(self,sample):
        # Read historical endpoints; do not generate PAIR or alter query retrieval.
        p=self.parent;d,sid=sample['domain'],sample['sample_id']
        if d=='background':
            with parent.bg.Archive(p.bg['paths']['raw_archive'],p.bg['raw_index']) as archive:
                raw,meta,segments=parent.bg.load_image(p.bg,archive,sample)
            from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.gates import scatter
            aligned=[]
            for l in LAYERS:
                x=segments[l];aligned.append(scatter(x['segment_distance'],x['positions'],tuple(meta['grid'])).ravel()[raw['positions']])
            raw['segment_distance']=np.stack(aligned)
            # Used only to audit explicitly recovered support features against
            # ORIGINAL saved endpoint lengths, not to change query prediction.
            raw['endpoint_length']=np.stack([scatter(segments[l]['endpoint_length'],segments[l]['positions'],tuple(meta['grid'])).ravel()[raw['positions']] for l in LAYERS])
            origin=dict(capture=p.bg['captures'][sid]['sha256'],segment={str(l):p.bg['segments'][sid,l]['sha256'] for l in LAYERS})
        else:
            record=p.cross.records[d,sid];path=p.cross.cache_root/parent.bg.safe(record['file'])
            raw,meta=parent.cross.decode(path.read_bytes(),p.cross,sample,record);origin=dict(capture=record['sha256'])
        if 'target_index' not in raw or raw['target_index'].shape!=raw['target_distance'].shape or raw['target_index'].dtype!=np.int64:
            raise ValueError('original eight-support nearest row indices required')
        for k,row in enumerate(self.expected[d]):
            if np.any(raw['target_index'][k]<0) or np.any(raw['target_index'][k]>=np.prod(row['grid'])):raise ValueError('endpoint row outside support bank')
        raw['union_distance']=np.minimum(raw['source_distance'],raw['target_distance'].min(0))
        return raw,meta,origin


def discover(roots):
    """Scan actual archive member names, never infer features from archive names."""
    candidates=[];inventory=[];seen=set()
    for root in map(Path,roots):
        if not root.exists():continue
        for f in ([root] if root.is_file() else root.rglob('*')):
            if not f.is_file() or f in seen:continue
            seen.add(f)
            if f.suffix=='.npz':candidates.append(str(f.resolve()))
            if (f!=root and not any(s in f.name.lower() for s in ('aebad','support'))) or not f.name.lower().endswith(('.tar','.tar.gz','.zip')):continue
            found=[]
            try:
                if f.suffix=='.zip':
                    with zipfile.ZipFile(f) as z:names=z.namelist()
                else:
                    with tarfile.open(f,'r:*') as z:names=[m.name for m in z if m.isfile()]
                for name in names:
                    p=Path(name);stem=p.stem
                    possible=(stem.isdigit() and 1<=int(stem)<=8) or 'support' in p.name.lower() or 'feature' in p.name.lower() or 'support_features' in p.parts
                    if name.endswith('.npz') and possible:
                        loc=str(f.resolve())+'::'+name;found.append(loc);candidates.append(loc)
                inventory.append(dict(path=str(f.resolve()),bytes=f.stat().st_size,candidate_support_NPZ_members=found,status='member_names_checked'))
            except (OSError,ValueError,tarfile.TarError,zipfile.BadZipFile) as exc:
                inventory.append(dict(path=str(f.resolve()),status='archive_unreadable',error=str(exc)))
    return sorted(set(candidates)),inventory


def locate(bundle,roots,domains,scan=False):
    expected=[r for d in domains for r in bundle.expected[d]];locations={};available=[];mismatches=[]
    if scan:candidates,archives=discover(roots)
    else:
        candidates=[];archives=[]
        for root in map(Path,roots):
            if root.is_file() and root.suffix=='.npz':candidates.append(str(root.resolve()))
            elif root.is_dir():candidates.extend(str(f.resolve()) for f in root.rglob('*.npz'))
    # Index each small header ONCE. Do not reopen every unrelated distance NPZ
    # once per support image when scanning a large results directory.
    by_relative={}
    for location in sorted(set(candidates)):
        try:
            with feature_stream(location) as f:
                with np.load(f,allow_pickle=False) as z:
                    if 'relative_path' not in z:continue
                    relative=str(z['relative_path'].item())
            by_relative.setdefault(relative,[]).append(location)
        except (OSError,ValueError,KeyError,EOFError,tarfile.TarError,zipfile.BadZipFile) as exc:
            mismatches.append(dict(location=location,reason=str(exc)))
    for row in expected:
        for location in by_relative.get(row['relative_path'],[]):
            try:
                proof=feature_header(location,row);key=(row['domain'],row['support_index'])
                locations[key]=location;available.append(dict(row,**proof));break
            except (OSError,ValueError,KeyError,EOFError,tarfile.TarError,zipfile.BadZipFile) as exc:
                mismatches.append(dict(domain=row['domain'],support_index=row['support_index'],location=location,reason=str(exc)))
    missing=[dict(row,required_fields='protocol_id;relative_path;grid;layer_7;layer_15;layer_23;layer_31',
        required_shape=f'{int(np.prod(row["grid"]))}x1280 float32 per layer',
        expected_uncompressed_feature_bytes=int(np.prod(row['grid']))*1280*4*4,
        status='missing_matching_complete_support_features') for row in expected if (row['domain'],row['support_index']) not in locations]
    return locations,available,missing,mismatches,archives


def preflight(bundle,roots,domains,scan=True):
    locations,available,missing,bad,archives=locate(bundle,roots,domains,scan=scan)
    raw_missing=[x for x in bundle.parent.missing() if x['domain'] in domains]
    examples=[]
    for d in domains:
        sample=next(s for s in bundle.samples if s['domain']==d)
        try:
            raw,meta,origin=bundle.load(sample)
            examples.append(dict(domain=d,sample_id=sample['sample_id'],status='SHA_and_complete_raw_endpoint_contract_verified',
                origin=origin,positions=int(len(raw['positions'])),target_distance_shape=list(raw['target_distance'].shape)))
        except (OSError,ValueError,KeyError) as exc:
            raw_missing.append(dict(domain=d,sample_id=sample['sample_id'],status='raw_example_rejected',reason=str(exc)))
    report=dict(status='ready_for_explicit_stages' if not missing and not raw_missing else 'missing_inputs',
        identity=bundle.identity,domains=list(domains),counts={d:PROTOCOL['populations'][d] for d in domains},
        expected_support_files=sum(len(bundle.expected[d]) for d in domains),verified_support_files=available,
        required_support_files=[r for d in domains for r in bundle.expected[d]],missing_inputs=missing+raw_missing,
        rejected_candidates=bad,archive_inventory=archives,
        historical_parent_identity=bundle.parent.identity,historical_score_audit=bundle.parent.cross.historical_audit,
        cached_query_distances='available except listed missing files; per-image SHA/endpoints checked when used',
        existing_parent_metadata_verified=True,stage_dependencies=dict(prepare_support='full original support features; NEW per-other-image k5 retrieval',
        support_loo='prepared support neighbors and original features; NEW support-only geometry',
        predict='features, prepared neighbor tables, cached query distances/indices; NEW lazy support relation checks; no Q needed',
        evaluate='frozen predictions, original GT and calibration; exact CPU pixels',
        evaluate_aupro='frozen maps/GT identity and original ADEval CUDA; no model'),
        raw_cache_examples=examples,raw_verification_scope='all path presence; one decoded SHA-verified example/domain; every used file independently SHA-verified at processing, not a full preflight raw decode',
        torch_imported='torch' in __import__('sys').modules,model_inference=False,feature_extraction=False,
        support_neighbor_computation=False,GPU_used=False,experiment_directory_created=False)
    if report['torch_imported']:raise RuntimeError('CPU preflight unexpectedly imported Torch')
    return report
