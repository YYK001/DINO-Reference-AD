"""Portable, read-only parent adapters; no bank, checkpoint or feature dependency."""
from pathlib import Path
import json
import numpy as np
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0 import inputs as bg
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.gates import scatter
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import pair_gate, predict
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain import inputs as cross
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain.live_parent import LiveParent
from .method import PROTOCOL,BASES,LAYERS

ROOT=Path(__file__).resolve().parents[3]
sha,digest,read_json,write_json,write_csv=bg.sha,bg.digest,bg.json_file,bg.write_json,bg.write_csv
csv_file=lambda p:bg.read_csv(Path(p).read_bytes())
DOMAINS=('background','illumination','view')
OLD=dict(zip(BASES,('UNION_K8_CH1','SEGMENT_K8_CH1','SEGMENT_PAIR_GATE_K8_CH1')))
BACK={v:k for k,v in OLD.items()}


def defaults():
    out=ROOT/'DINOv3/outputs/mad_equal'
    return dict(parent_run=out/'aebad_crossdomain_local_resume_20261021_210119',
        background_cpu=out/'aebad_s_neighborhood_segment_local_20261016_134948_cpu',
        background_aupro=out/'aebad_s_neighborhood_segment_local_20261016_134948_aupro_20261016_171444',
        data=ROOT/'datasets/AeBAD.zip')


def verified_cpu(directory,identity,images):
    s=read_json(directory/'status.json');m=read_json(directory/'analysis_manifest.json')
    if (s['status']!='cpu_evaluation_complete_aupro_pending' or s['identity']!=identity or s['numerical_failed_checks'] or m['identity']!=identity
        or m['images']!=images or sha(directory/'analysis_manifest.json')!=s['analysis_manifest_sha256']):
        raise ValueError('incomplete/incompatible historical CPU reference')
    for f,h in m['files'].items():
        if sha(directory/bg.safe(f))!=h:raise ValueError('modified parent CPU file: '+f)
    return m


class Bundle:
    def __init__(self,paths):
        self.paths={k:Path(v).resolve() for k,v in paths.items()}
        self.cross=LiveParent(self.paths['parent_run'])
        p=bg.defaults(ROOT);p['data']=self.paths['data'];self.bg=bg.context(p)
        self.state=self.bg['state'];self.medians=self.bg['medians'];self.scales=self.bg['scales']
        if (self.state!=self.cross.state or not np.array_equal(self.medians,self.cross.medians)
            or not np.array_equal(self.scales,self.cross.scales)):
            raise ValueError('parents do not share the same frozen Source calibration')
        bm=verified_cpu(self.paths['background_cpu'],self.bg['identity'],297)
        self.samples=[];self.supports={'background':self.bg['capture']['supports']}
        for s in self.bg['samples']:
            s=dict(s,domain='background')
            s['image_sha256']=self.bg['capture']['image_sha256'][s['relative_path']]
            if s['label']:s['gt_sha256']=self.bg['capture']['gt_sha256'][s['relative_path'].replace('test/','ground_truth/',1)]
            self.samples.append(s)
        for d in cross.DOMAINS:
            self.samples.extend(self.cross.catalog['domains'][d]['samples'])
            self.supports[d]=self.cross.catalog['domains'][d]['supports']
        self.references={};self.maps={};self.instances={};self.metrics={};self.assets={}
        for d in DOMAINS:
            directory=self.paths['background_cpu'] if d=='background' else self.paths['parent_run']/'cpu'
            for table,target in (('per_image_scores.csv',self.references),('map_manifest.csv',self.maps)):
                for r in csv_file(directory/table):
                    domain=r.get('domain','background');b=BACK.get(r['branch'])
                    if domain==d and b:target[d,r['sample_id'],b]=r
            for r in csv_file(directory/'defect_instances.csv'):
                domain=r.get('domain','background');b=BACK.get(r['branch'])
                if domain==d and b:self.instances[d,r['sample_id'],int(r['instance_id']),b]=r
        # Every historical AUPRO table has a verified provenance chain to CPU.
        for group,directory in [('background',self.paths['background_aupro']),('cross',self.paths['parent_run']/'aupro')]:
            status=read_json(directory/'status.json')
            if not status['all_metrics_complete']:raise ValueError('parent AUPRO incomplete')
            if group=='background':
                if (status['identity']!=self.bg['identity'] or status['analyzed_images']!=297
                    or status['cpu_analysis_manifest_sha256']!=sha(self.paths['background_cpu']/'analysis_manifest.json')):
                    raise ValueError('background AUPRO parent mismatch')
                files=status['output_sha256']
            else:
                if status['identity']!=self.cross.identity or status['images']!=629:raise ValueError('cross AUPRO parent mismatch')
                manifest=read_json(directory/'result_manifest.json')
                if (sha(directory/'result_manifest.json')!=status['result_manifest_sha256']
                    or manifest['cpu_analysis_manifest_sha256']!=sha(self.paths['parent_run']/'cpu/analysis_manifest.json')):
                    raise ValueError('cross AUPRO manifest mismatch')
                files=manifest['files']
            for f,h in files.items():
                if sha(directory/bg.safe(f))!=h:raise ValueError('modified parent AUPRO file: '+f)
            self.assets[str(directory/'status.json')]=sha(directory/'status.json')
            self.assets[str(directory/'metrics.csv')]=sha(directory/'metrics.csv')
            for r in csv_file(directory/'metrics.csv'):
                d='background' if group=='background' else r['domain'];b=BACK.get(r['branch'])
                if d not in DOMAINS or not b or (group=='background' and r.get('normal_group')!='all85'):continue
                self.metrics[d,b]=dict(r,parent_metric_file=str(directory/'metrics.csv'))
        expected={(s['domain'],s['sample_id'],b) for s in self.samples for b in BASES}
        if len(self.samples)!=926 or set(self.references)!=expected or set(self.maps)!=expected:
            raise ValueError('original 926 image baseline maps/scores incomplete')
        if set(self.metrics)!={(d,b) for d in DOMAINS for b in BASES}:raise ValueError('baseline metrics missing')
        for d in DOMAINS:
            selected=[s for s in self.samples if s['domain']==d];supp=self.supports[d]
            if ([len(selected),sum(s['label']==0 for s in selected),sum(s['label']==1 for s in selected)]!=PROTOCOL['populations'][d]
                or [s['support_index'] for s in supp]!=list(range(1,9))
                or {s['relative_path'] for s in selected}&{s['relative_path'] for s in supp}):
                raise ValueError('population/support order/exclusion changed')
        dependencies=['aebad_s_support_distance_diagnostic/profiles.py','aebad_s_projection_layer_diagnostic/numerics.py',
            'aebad_s_projection_layer_diagnostic/metrics.py','aebad_s_ch1_scoring_inheritance/scoring.py',
            'aebad_s_segment_pair_crossdomain_p0/analysis.py','aebad_s_segment_pair_crossdomain_p0/method.py',
            'aebad_s_support_distance_crossdomain/spatial_audit.py','aebad_s_union_diagnostics/spatial.py',
            'aebad_s_neighborhood_segment_p0/inputs.py','aebad_s_neighborhood_segment_p0/analysis.py',
            'aebad_s_neighborhood_segment_p0/prediction.py','aebad_s_support_distance_crossdomain/inputs.py',
            'aebad_s_support_distance_crossdomain/live_parent.py']
        self.frozen=dict(protocol=PROTOCOL,samples=self.samples,supports=self.supports,Source_image_MAD=self.state,
            medians=self.medians.tolist(),scales=self.scales.tolist(),background_identity=self.bg['identity'],
            background_assets=self.bg['manifest']['assets'],background_CPU_manifest=sha(self.paths['background_cpu']/'analysis_manifest.json'),
            cross_parent_identity=self.cross.identity,cross_parent_metadata_sha256=self.cross.source_sha256,
            cross_records=self.cross.required(),historical_metrics=self.assets,
            code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
            shared_code={n:sha(ROOT/'DINOv3/MADEqual'/n) for n in dependencies})
        self.identity=digest(self.frozen)

    def load(self,sample):
        """Complete grids only, no GT read. Background storage orders may differ."""
        d,sid=sample['domain'],sample['sample_id']
        if d=='background':
            with bg.Archive(self.bg['paths']['raw_archive'],self.bg['raw_index']) as archive:
                raw,meta,segments=bg.load_image(self.bg,archive,sample)
            segment=[];pair=[]
            for j,l in enumerate(LAYERS):
                grid=tuple(meta['grid']);s=segments[l];p=raw['positions']
                v=scatter(s['segment_distance'],s['positions'],grid).ravel()[p]
                endpoints=scatter(s['endpoint_support'],s['positions'],grid).reshape(2,-1)[:,p]
                g,_,_=pair_gate(endpoints,p,grid)
                u=np.minimum(raw['source_distance'][j],raw['target_distance'][:,j].min(0))
                segment.append(v);pair.append(np.where(g.ravel()[p],v,u))
            raw['segment_distance']=np.stack(segment);raw['pair_distance']=np.stack(pair)
            origin=dict(capture=self.bg['captures'][sid]['sha256'],segment={str(l):self.bg['segments'][sid,l]['sha256'] for l in LAYERS})
        else:
            record=self.cross.records[d,sid]
            path=self.cross.cache_root/bg.safe(record['file'])
            raw,meta=cross.decode(path.read_bytes(),self.cross,sample,record)
            reference,_=predict(raw,tuple(meta['grid']));raw['pair_distance']=reference['pair_distance']
            origin=dict(capture=record['sha256'])
        raw['union_distance']=np.minimum(raw['source_distance'],raw['target_distance'].min(0))
        return raw,meta,origin

    def missing(self):
        result=[]
        for (d,sid),r in self.cross.records.items():
            p=self.cross.cache_root/bg.safe(r['file'])
            if not p.is_file():result.append(dict(domain=d,sample_id=sid,path=str(p),sha256=r['sha256']))
        return result


def initialize(out,bundle):
    out=Path(out).resolve()
    for p in (*bundle.paths.values(),*bundle.bg['paths'].values()):
        if out==p or out in p.parents or p in out.parents:raise ValueError('output overlaps protected input')
    if out.drive and out.drive.upper()!='G:':raise ValueError('local artifacts must remain on G:')
    manifest=dict(identity=bundle.identity,frozen=bundle.frozen,paths={k:str(v) for k,v in bundle.paths.items()})
    if out.exists():
        if read_json(out/'input_manifest.json')!=manifest:raise ValueError('existing run input/code identity differs; use a fresh output')
    else:
        out.mkdir(parents=True);write_json(out/'input_manifest.json',manifest);write_json(out/'protocol.json',PROTOCOL)
    return out
