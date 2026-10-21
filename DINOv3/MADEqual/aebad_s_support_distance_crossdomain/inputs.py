"""Portable parent provenance; original cache identity never depends on new code."""
from __future__ import annotations
import hashlib
import io
import json
import tarfile
from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import inputs as old
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import audit
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic import inputs as base

ROOT=Path(__file__).resolve().parents[3]
DOMAINS=('illumination','view')
COUNTS={'illumination':(67,198),'view':(84,280)}
SUMMARY_SHA='5b458bfcbb1a7faea89529428d23dca4f7b8f76b7fcfadd35b6b76261a4eb681'
SUMMARY_NAME='aebad_segment_pair_compat_20261017_091539_501e80_EVAL_SUCCESS_SUMMARY_20261017_162654.tar.gz'
LAYERS=(7,15,23,31)
MAIN_FEATURES=('gap2','gap4','gap8')
sha,digest,safe,write_json,write_csv=base.sha,base.digest,base.safe,base.write_json,base.write_csv


class Parent:
    def __init__(self,path):
        self.path=Path(path).resolve()
        if sha(self.path)!=SUMMARY_SHA:raise ValueError('wrong/modified frozen two-domain summary')
        self.source_sha256=SUMMARY_SHA
        self.origin='historical_summary_archive'
        self.content={}
        with tarfile.open(self.path,'r|gz') as archive:
            seen=set()
            for m in archive:
                name=safe(m.name.rstrip('/'))
                if name in seen or not(m.isfile() or m.isdir()):raise ValueError('ambiguous summary member')
                seen.add(name)
                if m.isfile() and name.endswith(('.json','.csv','.npz','.md')):self.content[name]=archive.extractfile(m).read()
        self.validate()

    def validate(self):
        """Shared frozen-asset/manifest validation, also used by explicit reruns."""
        matches=[n for n in self.content if n.endswith('/source_provenance/cache/cache_status.json')]
        if len(matches)!=1:raise ValueError('ambiguous original cache provenance root')
        self.prefix=matches[0][:-len('source_provenance/cache/cache_status.json')]
        self.cache=self.json('source_provenance/cache/cache_status.json')
        self.frozen=self.json('source_provenance/cache/input_manifest.json')
        self.protocol=self.json('source_provenance/cache/protocol.json')
        self.identity=self.cache['identity']
        if (self.frozen['identity']!=self.identity or digest(self.frozen['frozen'])!=self.identity
                or self.frozen['frozen']['protocol']!=self.protocol or self.cache['status']!='capture_complete'
                or self.protocol['domains']!=list(DOMAINS) or self.protocol['layers']!=list(LAYERS)
                or self.protocol['seed']!=42 or self.protocol['score_atol']!=1e-5
                or self.protocol['historical_score_atol']!=1e-4 or self.protocol['distance_unit']!='euclidean_over_1280_once'):
            raise ValueError('original joint cache protocol/identity incomplete')
        catalog_path=Path(old.__file__).with_name('frozen_catalog.json')
        if sha(catalog_path)!=self.frozen['frozen']['catalog_sha256'] or sha(catalog_path)!=old.CATALOG_SHA:
            raise ValueError('frozen 629-image catalog differs')
        self.catalog=json.loads(catalog_path.read_bytes())
        for name,h in self.catalog['formal_sha256'].items():self.check('source_provenance/frozen_inputs/formal/'+name,h)
        self.check('source_provenance/frozen_inputs/image_calibration_state.json',self.catalog['image_state_sha256'])
        self.state=self.json('source_provenance/frozen_inputs/image_calibration_state.json')
        if digest({k:v for k,v in self.state.items() if k!='state_sha256'})!=self.state['state_sha256']:
            raise ValueError('invalid Source image state provenance')
        with np.load(io.BytesIO(self.bytes('source_provenance/frozen_inputs/formal/source_mad_state.npz')),allow_pickle=False) as f:
            self.medians,self.scales=f['medians'].copy(),f['scales'].copy()
        if (self.medians.shape!=(4,) or self.scales.shape!=(4,) or self.medians.dtype!=np.float32
                or self.scales.dtype!=np.float32 or not np.isfinite(self.medians).all()
                or not np.isfinite(self.scales).all() or np.any(self.scales<=0)):
            raise ValueError('invalid original Source patch MAD')
        cpu=self.json('cpu/status.json');manifest=self.json('cpu/analysis_manifest.json')
        self.check('cpu/analysis_manifest.json',cpu['analysis_manifest_sha256'])
        if cpu['identity']!=self.identity or cpu['images']!=629 or not cpu['joint_domains_complete'] or cpu['numerical_failed_checks']:
            raise ValueError('incomplete same-cache CPU reference')
        for name,h in manifest['files'].items():self.check('cpu/'+name,h)
        prediction=self.json('source_provenance/prediction/prediction_manifest.json')
        self.check('source_provenance/prediction/prediction_manifest.json',manifest['prediction_manifest_sha256'])
        if prediction['identity']!=self.identity or prediction['images']!=629 or prediction['GT_used'] or prediction['labels_used_by_method']:
            raise ValueError('unfrozen parent prediction')
        self.scores={(r['domain'],r['sample_id'],r['branch']):r for r in self.csv('cpu/per_image_scores.csv')}
        self.instance_rows=self.csv('cpu/defect_instances.csv')
        self.instances={(r['domain'],r['sample_id'],int(r['instance_id']),r['branch']):r for r in self.instance_rows}
        checks=self.csv('source_provenance/cache/capture_verification.csv')
        self.historical_audit=dict(parent_capture=audit(checks),parent_CPU=cpu['historical_score_audit'])
        # Repeated legacy checks are retained in their own audit, never counted as new independent tests.
        self.source_scores={}
        for r in checks:
            if r['check_kind']=='SOURCE_CURRENT':
                key=(r['domain'],r['sample_id']);v=float(r['actual_score'])
                if key in self.source_scores and self.source_scores[key]!=v:raise ValueError('conflicting same-cache Source scores')
                self.source_scores[key]=v
        self.records={}
        for r in self.cache['records']:
            key=(r['domain'],r['sample_id'])
            if key in self.records:raise ValueError('duplicate domain/sample capture')
            if safe(r['file'])!=f'raw_cache/{key[0]}/{key[1]}.npz':raise ValueError('unexpected original cache layout')
            self.records[key]=r
        expected=set()
        for domain in DOMAINS:
            d=self.catalog['domains'][domain];ss=d['samples'];supports=d['supports']
            if ((sum(s['label']==0 for s in ss),sum(s['label']==1 for s in ss))!=COUNTS[domain]
                    or len({s['sample_id'] for s in ss})!=len(ss) or [s['support_index'] for s in supports]!=list(range(1,9))
                    or len({s['relative_path'] for s in supports})!=8
                    or {s['relative_path'] for s in ss}&{s['relative_path'] for s in supports}):
                raise ValueError('original domain population/support order/exclusion differs')
            for s in ss:
                if s['domain']!=domain:raise ValueError('catalog domain mismatch')
                expected.add((domain,s['sample_id']))
                for b in ('UNION_K8_CH1','SEGMENT_K8_CH1'):
                    ref=self.scores[domain,s['sample_id'],b]
                    if int(ref['label'])!=s['label'] or ref['relative_path']!=s['relative_path']:raise ValueError('reference score scope changed')
        if set(self.records)!=expected or len(expected)!=629 or set(self.source_scores)!=expected:
            raise ValueError('complete original 629 raw/Source-score records required')
        self.run_config=self.json('source_provenance/run_config.json')

    def bytes(self,name):return self.content[self.prefix+safe(name)]
    def json(self,name):return json.loads(self.bytes(name))
    def csv(self,name):return base.read_csv(self.bytes(name))
    def check(self,name,h):
        if hashlib.sha256(self.bytes(name)).hexdigest()!=h:raise ValueError('parent metadata SHA mismatch: '+name)
    def required(self):
        return [dict(domain=d,sample_id=sid,relative_path=r['file'],sha256=r['sha256'],
            original_identity=self.identity,required_fields='positions;source_distance;target_distance[8,4,M];segment_distance')
            for (d,sid),r in self.records.items()]


def decode(blob,parent,sample,record):
    if hashlib.sha256(blob).hexdigest()!=record['sha256']:raise ValueError('raw SHA mismatch; original history cannot be re-enrolled')
    with np.load(io.BytesIO(blob),allow_pickle=False) as f:
        missing={'metadata','positions','source_distance','target_distance','segment_distance'}-set(f.files)
        if missing:raise ValueError('eight-support raw capture required; missing '+','.join(sorted(missing)))
        meta=json.loads(str(f['metadata'].item()));a={k:f[k].copy() for k in f.files if k!='metadata'}
    domain=sample['domain']
    expected=dict(format='same_call_target_support_loo_v1',tool=old.TOOL,identity=parent.identity,domain=domain,
        sample_id=sample['sample_id'],relative_path=sample['relative_path'],image_sha256=sample['image_sha256'],
        distance_unit='euclidean_over_1280_once')
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('original capture domain/sample/identity differs')
    supports=parent.catalog['domains'][domain]['supports'];fields=('support_index','relative_path','image_sha256')
    if [[s[k] for k in fields] for s in meta['supports']]!=[[s[k] for k in fields] for s in supports]:
        raise ValueError('original support identity/order differs')
    if 'layers' in meta and meta['layers']!=list(LAYERS):raise ValueError('raw layer order differs')
    grid=meta['grid'];p=a['positions'];n=int(np.prod(grid))
    if (len(grid)!=2 or min(grid)<3 or meta['output_shape']!=[sample['image_height']//4,sample['image_width']//4]
            or p.dtype!=np.int64 or p.shape!=(n,) or not np.array_equal(np.sort(p),np.arange(n))):
        raise ValueError('invalid geometry/physical positions')
    for key,shape in (('source_distance',(4,n)),('target_distance',(8,4,n)),('segment_distance',(4,n))):
        if a[key].dtype!=np.float32 or a[key].shape!=shape or not np.isfinite(a[key]).all() or np.any(a[key]<0):
            raise ValueError('invalid raw distances: '+key)
    u=np.minimum(a['source_distance'],a['target_distance'].min(0))
    if np.any(a['segment_distance']>u+1e-7+1e-6*np.abs(u)):raise ValueError('SEGMENT exceeds UNION')
    # Optional saved geometry is validated, never reconstructed from unsaved features.
    geom=('alpha','endpoint_length','endpoint_support','endpoint_index')
    if all(k in a for k in geom) and 'target_index' in a:
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import validate_geometry
        validate_geometry(a)
    return a,meta


class Data(old.Data):
    """Reuse byte/GT decoding; directories need only GT, not RGB or training."""
    def __init__(self,path):
        path=Path(path)
        if path.is_file():
            super().__init__(path);return
        for root in (path,path/'AeBAD_S',path/'AeBAD/AeBAD_S'):
            if (root/'ground_truth').is_dir():
                self.path,self.root,self.zip,self.names=path,root,None,set();return
        raise FileNotFoundError('missing AeBAD-S ground_truth root: '+str(path))


def verify_data(parent,path):
    from PIL import Image
    count=0;missing=[]
    with Data(path) as data:
        for domain in DOMAINS:
            for s in parent.catalog['domains'][domain]['samples']:
                if s['label']:
                    relative=s['relative_path'].replace('test/','ground_truth/',1)
                    try:
                        blob=data.bytes(relative,s['gt_sha256'])
                        with Image.open(io.BytesIO(blob)) as im:
                            if im.size!=(s['image_width'],s['image_height']):raise ValueError('GT geometry differs')
                        count+=1
                    except (OSError,KeyError,ValueError) as exc:
                        missing.append(dict(domain=domain,sample_id=s['sample_id'],relative_path=relative,
                            status='missing_or_modified_GT',reason=str(exc)))
    return count,missing
