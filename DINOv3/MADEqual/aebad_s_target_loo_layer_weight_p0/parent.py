"""Explicit domain parameterization of the validated parent metadata reader."""
from pathlib import Path
import numpy as np
from .protocol import LAYERS,U,S,COMBINED,COUNTS
from DINOv3.MADEqual.aebad_s_blind_context_p0.inputs import st,read_csv,header,positions


class Parent:
    """Domain-bound original SACP reader; no change to the old blind-context tool."""
    def __init__(self,a):
        self.domain=a.domain;self.count=COUNTS[self.domain][0]
        self.run=a.parent_run;self.cache=a.parent_cache
        self.inputs=st.sealed_json(self.run/'inputs.json');self.identity=self.inputs['identity']
        if any(self.inputs['config'].get(k)!=v for k,v in dict(layers=list(LAYERS),dimension=1280,Source_rows=100000,seed=42).items()):raise ValueError('frozen backbone/memory/seed differs')
        self.assets=st.sealed_json(self.run/'assets.json')
        if self.assets['seal']!=self.inputs['assets_seal']:raise ValueError('parent asset metadata differs')
        self.extraction=self.assets['references']['extraction']
        self.catalog_record=st.sealed_json(self.run/'catalog.json');self.catalog=self.catalog_record['catalog']
        if st.digest(self.catalog)!=self.inputs['catalog']:raise ValueError('parent catalog changed')
        source=self.catalog['source']
        if len(source)!=521 or sum(r['calibration'] for r in source)!=66:raise ValueError('original 521/66 split required')
        if [r['relative_path'] for r in source]!=sorted(r['relative_path'] for r in source) or any(r['source_index']!=i or r['calibration']!=(i%8==0) for i,r in enumerate(source)):raise ValueError('Source order/mod8 calibration differs')
        self.samples=[r for r in self.catalog['samples'] if r['domain']==self.domain];self.supports=self.catalog['supports'][self.domain]
        if any(r.get('domain')!=self.domain for r in self.supports):raise ValueError('support image belongs to another domain')
        if [len(self.samples),sum(r['label']==0 for r in self.samples),sum(r['label']==1 for r in self.samples)]!=list(COUNTS[self.domain]):raise ValueError('original per-domain evaluation population required')
        if [r['support_index'] for r in self.supports]!=list(range(1,9)) or any(r['label'] for r in self.supports):raise ValueError('ordered K8 normals required')
        if {r['relative_path'] for r in self.samples}&{r['relative_path'] for r in self.supports}:raise ValueError('support included in evaluation')
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import support_key
        all_normal=self.supports+[r for r in self.samples if r['label']==0]
        if [r['relative_path'] for r in self.supports]!=[r['relative_path'] for r in sorted(all_normal,key=lambda r:support_key(r['relative_path'],self.domain))[:8]]:raise ValueError('original seed42 support rule differs')
        state=st.read_json(self.run/'status.json')
        if state['identity']!=self.identity or state['status']!='all_five_metrics_complete' or state['pending'] or state['numerical_failed_checks']:raise ValueError('completed parent required')
        self.mad=st.sealed_json(self.run/'input_distance_mad.json')
        if self.mad['identity']!=self.identity:raise ValueError('parent MAD identity differs')
        if any(self.mad[k]!=self.assets['references']['mad'][k] for k in ('patch','image')):raise ValueError('Source bank provider and MAD differ')
        for v in (self.mad['patch'],self.mad['image']['peak'],self.mad['image']['local']):
            if len(v['medians'])!=4 or len(v['scales'])!=4 or not np.isfinite(v['medians']).all() or not np.isfinite(v['scales']).all() or min(v['scales'])<=0:raise ValueError('invalid frozen MAD')
        self.manifest=st.sealed_json(self.run/'predictions'/(self.domain+'.json'))
        if self.manifest['identity']!=self.identity or [(r['domain'],r['sample_id']) for r in self.manifest['records']]!=[(r['domain'],r['sample_id']) for r in self.samples]:raise ValueError('incomplete parent domain freeze')
        self.seals={r['sample_id']:r['seal'] for r in self.manifest['records']}
        self.cpu=st.sealed_json(self.run/'evaluation'/self.domain/'status.json')
        if self.cpu['identity']!=self.identity or self.cpu['prediction_seal']!=self.manifest['seal'] or self.cpu['images']!=self.count or self.cpu['numerical_failed_checks']:raise ValueError('parent CPU freeze differs')
        self._tables={};self.metrics=[r for r in read_csv(self.run/'metrics.csv') if r['domain']==self.domain and r['branch'] in (U,S,COMBINED)]
        if len(self.metrics)!=6:raise ValueError('six historical score/branch rows required')
        if {(r['branch'],r['scoring']) for r in self.metrics}!={(b,s) for b in (U,S,COMBINED) for s in ('CH1','CURRENT')}:raise ValueError('historical score rows differ')
        for r in self.metrics:
            if any(not np.isfinite(float(r[k])) or not 0<=float(r[k])<=1 for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')):raise ValueError('invalid historical metric')
        cpumetrics=self.run/'evaluation'/self.domain/'metrics.csv'
        if st.sha(cpumetrics)!=self.cpu['files']['metrics.csv']:raise ValueError('historical CPU metric table changed')
        lookup={(r['branch'],r['scoring']):r for r in read_csv(cpumetrics)}
        for r in self.metrics:
            for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP'):
                b=(S if k.startswith('image_') else U) if r['branch']==COMBINED else r['branch']
                if float(r[k])!=float(lookup[b,r['scoring']][k]):raise ValueError('root/historical CPU metrics differ')
            if r['branch']==COMBINED and float(r['AUPRO'])!=float(next(x for x in self.metrics if x['branch']==U and x['scoring']==r['scoring'])['AUPRO']):raise ValueError('fixed combination must use UNION AUPRO')
        for b in (U,S):
            v=st.read_json(self.run/'aupro'/self.domain/(b+'.json'))
            if v.get('seal')!=st.digest({k:x for k,x in v.items() if k!='seal'}) or v['identity']!=self.identity or v['verified_images']!=self.count or v['backend']!='original_ADEval_1.1.0_CUDA' or v['nstrips']!=200 or v['FPR_max']!=.3:raise ValueError('parent AUPRO definition differs')
            if any(float(r['AUPRO'])!=v['AUPRO'] for r in self.metrics if r['branch']==b):raise ValueError('parent AUPRO metric mismatch')

    def table(self,name):
        if name not in self._tables:
            p=self.run/'evaluation'/self.domain/name
            if st.sha(p)!=self.cpu['files'][name]:raise ValueError('modified parent small result table: '+name)
            self._tables[name]=read_csv(p)
        return self._tables[name]

    def prediction(self,s,load=True):
        if s['domain']!=self.domain:raise ValueError('sample belongs to another parent domain')
        path=self.cache/'predictions'/self.domain/s['sample_id'];m=st.sealed_json(path/'manifest.json')
        meta=m['metadata']
        if m['seal']!=self.seals[s['sample_id']] or meta['identity']!=self.identity or meta['sample']!=s or meta['calibration_seal']!=self.mad['seal']:raise ValueError('baseline prediction identity differs')
        n=int(np.prod(meta['grid']))
        for b in (U,S):header(path,b,(4,n),np.float32)
        header(path,'raw_union',(4,n),np.float32)
        header(path,'positions',(n,),np.int64)
        if meta['output_shape']!=[s['image_height']//4,s['image_width']//4]:raise ValueError('evaluation resize differs')
        if load:
            x,_=st.get(path,(U,S,'positions'));positions(x['positions'],n);return x,m
        return path
