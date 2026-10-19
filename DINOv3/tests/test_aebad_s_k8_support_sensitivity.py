"""Real CPU kernels. Torch/Exact/model boundaries are explicitly substituted."""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from DINOv3.MADEqual.aebad_s_k8_support_sensitivity_p0 import (
    inputs as ip, selection, scheduler, storage, capture, prediction, evaluation, reporting,
)
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import BRANCHES,U,S,P,predict,pair_gate
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score

ROOT=Path(__file__).resolve().parents[2]


def exact_numpy(query,bank):
    # Only the Torch cdist boundary is replaced, not selection/geometry/gates/maps.
    d=np.sqrt(np.sum((query[:,None].astype(float)-bank[None].astype(float))**2,axis=-1))
    index=d.argmin(1)
    return d[np.arange(len(query)),index].astype(np.float32),index.astype(np.int64)


def small_capture():
    rng=np.random.RandomState(81); query=rng.normal(size=(9,1280)).astype(np.float32)
    source=rng.normal(size=(5,1280)).astype(np.float32)
    features=[rng.normal(size=(i%7+2,1280)).astype(np.float32) for i in range(24)]
    features[1]=features[0].copy()  # true per-support distance tie
    d,idx,t,ti=scheduler.scan_layer(query,source,24,lambda i:features[i],exact_numpy)
    raw=dict(positions=np.array([8,4,0,7,1,5,3,6,2],np.int64),
        source_distance=np.stack([d]*4),source_index=np.stack([idx]*4),
        target_distance=np.stack([t]*4,axis=1),target_index=np.stack([ti]*4,axis=1))
    mapping={str(s):list(range(j*8,(j+1)*8)) for j,s in enumerate(selection.SEEDS)}
    raw,_=scheduler.attach_geometry(raw,mapping,lambda i,l:features[i])
    return raw,mapping,features,query,source


def fixture(root):
    root=Path(root); data=root/'data'; (data/'test/good/background').mkdir(parents=True)
    (data/'test/defect/background').mkdir(parents=True);(data/'ground_truth/defect/background').mkdir(parents=True)
    raw,mapping,features,query,source=small_capture()
    supports=[]
    for i in range(24):
        supports.append(dict(relative_path=f'test/good/background/s{i}.png',domain='background',sample_id=f'support_{i}',
                             label=0,image_sha256=str(i),image_height=32,image_width=32))
    plan=dict(groups={'background':{str(s):[dict(supports[i],support_seed=s,support_index=j+1)
                         for j,i in enumerate(mapping[str(s)])] for s in selection.SEEDS}},samples=[])
    ctx=dict(paths={'output_root':root/'output','data':data,'project':root,'formal_dir':root/'formal',
        'image_state':root/'state.json','memory':root/'memory','checkpoint':root/'checkpoint',
        'dinov3':root/'dinov3','official':root/'official'},identity='test_frozen_identity',plan=plan,
        medians=np.zeros(4,np.float32),scales=np.full(4,.1,np.float32),
        state={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')},
        assets={'image_state_sha256':'fixed_state'},ledger=storage.Ledger(root/'output','test_frozen_identity'))
    for i,label in enumerate((0,1)):
        name=f'test/{"defect" if label else "good"}/background/q{i}.png'
        Image.fromarray(np.zeros((32,32,3),np.uint8)).save(data/name)
        sample=dict(domain='background',sample_id=f'test_{i:04d}',relative_path=name,label=label,
                    image_height=32,image_width=32,image_sha256=storage.sha(data/name))
        if label:
            mask=np.zeros((32,32),np.uint8); mask[12:18,13:20]=255
            gt=name.replace('test/','ground_truth/',1); Image.fromarray(mask).save(data/gt)
            sample['gt_sha256']=storage.sha(data/gt)
        meta=dict(identity=ctx['identity'],domain='background',sample_id=sample['sample_id'],relative_path=name,
            image_sha256=sample['image_sha256'],grid=[3,3],output_shape=[8,8],mapping=mapping,
            supports=[dict(relative_path=s['relative_path'],image_sha256=s['image_sha256'],grid=[len(f),1])
                      for s,f in zip(supports,features)],distance_unit=ip.PROTOCOL['distance_unit'])
        sample['source_score']=statistics(raw['source_distance'],raw['positions'],(3,3),(8,8),ctx['medians'],ctx['scales'])[0]
        union=np.minimum(raw['source_distance'],raw['target_distance'][:8].min(0))
        sample['union_score']=statistics(union,raw['positions'],(3,3),(8,8),ctx['medians'],ctx['scales'])[0]
        plan['samples'].append(sample)
        scheduler.validate_shared(ctx,sample,raw,meta)
        checks=capture.audit_scores(ctx,sample,raw,meta)
        path='capture/'+capture.sample_key(sample)+'.npz'
        ctx['ledger'].commit('capture/'+capture.sample_key(sample),dict(file=path,checks=checks),
                            {path:storage.npz_bytes(raw,meta)})
    return ctx,raw


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths={k:Path(v).resolve() for k,v in ip.defaults(ROOT).items()}
        cls.ctx=ip.build_plan(cls.paths)

    def test_real_seed42_exact_formal_order(self):
        formal=storage.csv_file(self.paths['formal_dir']/'support_manifest.csv')
        for d in selection.DOMAINS:
            rows=sorted([r for r in formal if r['domain']==d and r['branch']=='target_matched'],key=lambda r:int(r['support_index']))
            self.assertEqual([r['relative_path'] for r in rows],
                             [r['relative_path'] for r in self.ctx['plan']['groups'][d]['42']])

    def test_crc_function_is_original_body_and_explicit_seed(self):
        import zlib
        for d in selection.DOMAINS:
            for seed in selection.SEEDS:
                for s in self.ctx['plan']['groups'][d][str(seed)]:
                    key=selection.original_key()(SimpleNamespace(path=ROOT/s['relative_path']),dataset_root=ROOT,domain=d,seed=seed)
                    self.assertEqual(key,(zlib.crc32(f"{seed}|{d}|{s['relative_path']}".encode())&0xffffffff,s['relative_path']))

    def test_all27_common_sets_and_exclusions(self):
        plan=self.ctx['plan']; self.assertEqual(len(plan['samples']),878)
        self.assertEqual(len(plan['excluded']),72)
        for d,total in zip(selection.DOMAINS,(281,249,348)):
            held={s['relative_path'] for s in plan['samples'] if s['domain']==d}
            self.assertEqual(len(held),total)
            for seed in selection.SEEDS:
                ss=plan['groups'][d][str(seed)]
                self.assertEqual(len(ss),8); self.assertFalse(held&{s['relative_path'] for s in ss})
                for _ in BRANCHES: self.assertEqual(len(held),total)
        self.assertTrue(all(r['intersection']==0 for r in plan['overlap']))

    def test_reordering_candidates_does_not_redraw_seed43_44(self):
        samples=self.ctx['plan']['samples']+self.ctx['plan']['excluded']
        # strip excluded-only annotations; identity/order is unaffected
        support=storage.csv_file(self.paths['formal_dir']/'support_manifest.csv')
        a=selection.select(list(reversed(samples)),support,root=ROOT)
        for d in selection.DOMAINS:
            for seed in selection.SEEDS:
                self.assertEqual([s['relative_path'] for s in a['groups'][d][str(seed)]],
                                 [s['relative_path'] for s in self.ctx['plan']['groups'][d][str(seed)]])

    def test_support_seed_does_not_change_source_or_calibration(self):
        self.assertEqual(ip.PROTOCOL['source_seed'],42)
        self.assertEqual(ip.PROTOCOL['Source_brightness'],[.8,1.2])
        self.assertEqual(self.ctx['assets']['memory_sha256'],'1b9e00c85fd6be35ba365d378b1da7fb2a673400319a256bb62408fe6e001717')

    def test_default_preflight_readonly_no_torch(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'downloads') as tmp:
            output=Path(tmp)/'not_created'
            result=subprocess.run([sys.executable,'-B','-m',ip.__package__+'.runner','preflight','--output-root',str(output)],
                                  cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertFalse(output.exists())
            report=json.loads(result.stdout); self.assertFalse(report['torch_imported'])
            self.assertEqual(report['combinations'],27)


class KernelTests(unittest.TestCase):
    def test_shared_search_count_and_group_legacy_equivalence_including_ties(self):
        raw,mapping,features,query,source=small_capture()
        calls=[]
        def exact(q,b): calls.append(id(b)); return exact_numpy(q,b)
        scheduler.scan_layer(query,source,24,lambda i:features[i],exact)
        self.assertEqual(len(calls),25)
        # Execute unchanged original eight-support scan with its Torch boundary
        # substituted. Do not execute any Torch imports from that source file.
        source_path=ROOT/'DINOv3/MADEqual/robustad_target_fewshot_p0/scoring.py'
        tree=ast.parse(source_path.read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_support_prefix_minima','_notify_nearest')]
        body=ast.parse('from __future__ import annotations').body+funcs
        class Array:
            def __init__(self,a):self.a=a
            def numpy(self):return self.a
        def legacy_exact(q,b):return tuple(Array(a) for a in exact_numpy(q,b))
        env=dict(np=np,PREFIXES=(1,2,4,8),_exact_top1_chunked=legacy_exact)
        exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source_path),'exec'),env)
        for seed in selection.SEEDS:
            group=scheduler.group_raw(raw,mapping,seed)
            old=env['_support_prefix_minima'](query,[features[i] for i in mapping[str(seed)]])
            np.testing.assert_array_equal(group['target_distance'][:,0].min(0),old[8])
            selected=np.argsort(group['target_distance'],axis=0,kind='stable')[:2]+1
            np.testing.assert_array_equal(group['endpoint_support'],selected.swapaxes(0,1))
            from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import add_geometry
            old_raw={k:group[k] for k in ('positions','source_distance','source_index','target_distance','target_index')}
            original,_=add_geometry(old_raw,{l:[features[i] for i in mapping[str(seed)]] for l in ip.LAYERS})
            new_pred,_=predict(group,(3,3)); old_pred,_=predict(original,(3,3))
            for k,v in new_pred.items():np.testing.assert_array_equal(v,old_pred[k])

    def test_query_extraction_called_once_with_original_flags(self):
        events=[]
        class Tensor:
            shape=(1,3,32,32)
            def to(self,**kw): events.append(('to',kw));return self
            def __getitem__(self,k): return self
        model=SimpleNamespace(preprocessing=lambda t:t)
        def extract(m,p,**kw):events.append(('extract',kw));return [SimpleNamespace(queries={})]
        result,shape=scheduler.extract_query(model,object(),'fake_device',lambda im:Tensor(),extract)
        self.assertEqual(shape,[32,32]);self.assertEqual(sum(r[0]=='extract' for r in events),1)
        self.assertEqual(events[-1][1],dict(device='fake_device',query_masks=[None],include_identity_descriptors=False))

    def test_positions_permutation_physical_gate_and_boundary(self):
        raw,mapping,*_=small_capture(); a=scheduler.group_raw(raw,mapping,42)
        pred,_=predict(a,(3,3)); perm=np.array([8,7,3,2,4,6,1,0,5])
        b={k:(v[...,perm] if v.shape[-1]==9 else v) for k,v in a.items()}
        other,_=predict(b,(3,3))
        np.testing.assert_array_equal(pred['pair_gate'],other['pair_gate'])
        np.testing.assert_array_equal(other['pair_distance'],pred['pair_distance'][:,perm])
        self.assertEqual(int(pred['neighbor_count'][0,0,0]),3)
        self.assertEqual(int(pred['neighbor_count'][0,1,1]),8)
        gate,_,n=pair_gate(np.array([[1],[2]],np.int8),np.array([0],np.int64),(1,1))
        self.assertFalse(gate.any());self.assertEqual(int(n.sum()),0)

    def test_geometry_different_support_sizes_and_local_ID_range(self):
        raw,mapping,*_=small_capture()
        for seed in selection.SEEDS:
            a=scheduler.group_raw(raw,mapping,seed)
            self.assertTrue(np.all((a['endpoint_support']>=1)&(a['endpoint_support']<=8)))
            self.assertEqual(a['target_index'].shape,(8,4,9));predict(a,(3,3))

    def test_original_containment_violation_is_rejected(self):
        raw,mapping,*_=small_capture(); a=scheduler.group_raw(raw,mapping,42)
        a['segment_distance']=a['source_distance']+1
        with self.assertRaises(ValueError):predict(a,(3,3))

    def test_history_only_seed42_and_signed_strict_vs_compatible(self):
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import historical_score
        rows=[];sample=dict(domain='background',sample_id='case')
        self.assertTrue(historical_score(rows,sample,'UNION_CURRENT',1-2e-5,1))
        self.assertFalse(rows[0]['strict_verified']);self.assertLess(rows[0]['signed_error'],0)
        self.assertFalse(historical_score(rows,sample,'UNION_CURRENT',1+.00011,1))

    def test_real_background12_segment_pair_CH1_regression(self):
        from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0 import inputs as old
        from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import CASES
        paths=old.defaults(ROOT)
        if not paths['raw_archive'].is_file():self.skipTest('real historical cache absent')
        pred=ROOT/'DINOv3/outputs/mad_equal/aebad_s_neighborhood_segment_local_20261016_134948_predict'
        cpu=ROOT/'DINOv3/outputs/mad_equal/aebad_s_neighborhood_segment_local_20261016_134948_cpu'
        records={r['sample_id']:r for r in storage.read_json(pred/'prediction_manifest.json')['records']}
        scores={(r['sample_id'],r['branch']):float(r['image_score']) for r in storage.csv_file(cpu/'per_image_scores.csv')}
        _,state,medians,scales=ip.metadata(ip.defaults(ROOT))
        ctx=dict(state=state,medians=medians,scales=scales,assets={'image_state_sha256':'same_frozen_state'})
        maximum=0.;patches=0
        with old.Archive(paths['raw_archive']) as archive:
            cap={r['sample_id']:r for r in archive.json(old.CROOT+'/status.json')['records']}
            seg={(r['sample_id'],r['layer']):r for r in archive.json(old.SROOT+'/predictions.json')['records']}
            for case in CASES:
                sid=case['sample_id'] if isinstance(case,dict) else case[0]
                r=cap[sid];raw,meta=archive.npz(old.CROOT+'/'+r['file'],r['sha256']);layers=[]
                for layer in ip.LAYERS:
                    r=seg[(sid,layer)];v,_=archive.npz(old.SROOT+'/'+r['file'],r['sha256'])
                    order=np.argsort(v['positions'])[raw['positions']]
                    layers.append({k:v[k][...,order] for k in ('segment_distance','endpoint_support','endpoint_index','alpha','endpoint_length')})
                for k in layers[0]:raw['seed42_'+k]=np.stack([v[k] for v in layers])
                # Correction is not an input to predict; this regression exercises
                # the saved real geometry and gates, not feature-based recomputation.
                raw['seed42_squared_correction']=np.zeros((4,2,len(raw['positions'])))
                actual,_=predict(scheduler.group_raw(raw,{'42':list(range(8))},42),tuple(meta['grid']))
                saved,_=storage.load_npz(pred/records[sid]['file'],records[sid]['sha256'])
                for key in ('pair_distance','pair_gate','pair_votes','neighbor_count'):
                    np.testing.assert_array_equal(actual[key],saved[key])
                for row in prediction.score(ctx,actual,meta):
                    error=abs(row['image_score']-scores[(sid,row['branch'])]);maximum=max(maximum,error)
                    self.assertLessEqual(error,1e-5)
                patches+=actual['pair_distance'].size
        print(f'REAL_BACKGROUND12: layer_patches={patches}; max_CH1_error={maximum:.9g}')


class StorageAndPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'downloads');self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_modified_history_rejected_without_reenrollment(self):
        l=storage.Ledger(self.root/'r','id');l.commit('one',{}, {'one.npz':b'original'})
        (l.root/'one.npz').write_bytes(b'modified')
        with self.assertRaises(ValueError):storage.Ledger(l.root,'id')

    def test_orphan_and_wrong_identity_rejected(self):
        l=storage.Ledger(self.root/'r','id')
        with self.assertRaises(ValueError):storage.Ledger(l.root,'new_id')
        (l.root/'orphan.npz').write_bytes(b'data')
        with self.assertRaises(ValueError):storage.Ledger(l.root,'id')

    def test_completed_units_never_overwritten(self):
        l=storage.Ledger(self.root/'r','id');l.commit('one',{}, {'file':b'data'})
        before=(l.root/'file').read_bytes()
        with self.assertRaises(ValueError):l.commit('one',{}, {'file':b'changed'})
        self.assertEqual(before,(l.root/'file').read_bytes())

    def test_prepare_resume_never_reselects(self):
        root=self.root/'frozen';root.mkdir()
        sentinel=object()
        with patch.object(ip,'load',return_value=sentinel),patch.object(ip,'build_plan',side_effect=AssertionError('redraw')):
            self.assertIs(ip.prepare({'output_root':root}),sentinel)

    def test_CPU_completed_combination_reused_without_sort(self):
        ctx,_=fixture(self.root);prediction.run(ctx,['background'],workers=1)
        for seed in selection.SEEDS:
            for branch in BRANCHES:
                key='cpu/'+evaluation.unit_name('background',seed,branch);name=key+'/result.json'
                # Valid scope fixture replaces an already completed computation;
                # resume must still validate its identity and image list.
                result=dict(identity=ctx['identity'],status='cpu_complete_AUPRO_pending',
                    metric=dict(domain='background',support_seed=seed,branch=branch,image_count=2,
                        image_AUROC=.5,image_AP=.5,pixel_AUROC=.5,pixel_AP=.5),
                    scores=[dict(sample_id=s['sample_id']) for s in ctx['plan']['samples']],
                    map_manifest=[dict(sample_id=s['sample_id']) for s in ctx['plan']['samples']])
                ctx['ledger'].commit(key,dict(file=name,status='cpu_complete_AUPRO_pending'),{name:storage.json_bytes(result)})
        with patch.object(evaluation,'_cpu_unit',side_effect=AssertionError('completed unit recomputed')):
            result=evaluation.run_cpu(ctx,['background'],workers=1,work_dir=self.root/'scratch')
        self.assertEqual(result['new_combinations'],0);self.assertEqual(result['reused_combinations'],9)

    def test_incomplete_prediction_cannot_evaluate(self):
        ctx,_=fixture(self.root)
        with self.assertRaises(ValueError):prediction.require_frozen(ctx,'background')
        with self.assertRaises(ValueError):evaluation.run_cpu(ctx,['background'],workers=1,work_dir=self.root/'scratch')

    def test_complete_capture_needs_no_model_or_memory(self):
        ctx,_=fixture(self.root)
        self.assertFalse(ctx['paths']['memory'].exists());self.assertFalse(ctx['paths']['checkpoint'].exists())
        result=capture.run(ctx,['background'])
        self.assertEqual(result['new_query_extractions'],0)
        self.assertEqual(result['complete_query_reuses'],2)
        self.assertFalse('torch' in sys.modules)

    def test_actual_cpu_prediction_and_resume_parallel_equivalence(self):
        ctx,_=fixture(self.root)
        prediction.run(ctx,['background'],workers=2)
        old={k:dict(v) for k,v in ctx['ledger'].state['records'].items()}
        result=prediction.run(ctx,['background'],workers=1)
        self.assertEqual(result['new_images'],0);self.assertEqual(old,ctx['ledger'].state['records'])
        for s in ctx['plan']['samples']:
            rows=[]
            for seed in selection.SEEDS:
                a,meta,sc=prediction.load(ctx,s,seed);self.assertEqual(len(sc),3);rows.extend(sc)
            self.assertEqual(len(rows),9)
        self.assertFalse('torch' in sys.modules)

    def test_cpu_exact_subset_and_gt_after_freeze_no_model(self):
        ctx,_=fixture(self.root);prediction.run(ctx,['background'],workers=2)
        # One genuine production CPU evaluation unit; no environment/model/GPU substitution.
        r=evaluation._cpu_unit(ctx,'background',42,U,1,1<<30,self.root/'scratch')
        self.assertEqual(r['metric']['image_count'],2);self.assertEqual(r['metric']['pixel_count'],128)
        self.assertIsNone(r['metric']['AUPRO']);self.assertEqual(len(r['map_manifest']),2)
        self.assertTrue(r['GT_used_only_after_predictions_frozen']);self.assertFalse('torch' in sys.modules)
        self.assertFalse(list((self.root/'scratch').iterdir()))

    def test_full_small_domain_cpu_summary_stays_partial_without_other_domains_and_AUPRO(self):
        ctx,_=fixture(self.root);prediction.run(ctx,['background'],workers=2)
        run=evaluation.run_cpu(ctx,['background'],workers=2,memory_budget=1<<30,work_dir=self.root/'scratch')
        self.assertEqual(run['new_combinations'],9)
        out=reporting.run(ctx)
        self.assertEqual(out['status'],'partial_evidence');self.assertFalse(out['all_requested_metrics_complete'])
        self.assertEqual(out['completed_CPU_combinations'],9);self.assertEqual(out['completed_AUPRO_combinations'],0)
        self.assertEqual(len(out['pending']),27)
        summary=Path(out['summary_directory'])
        self.assertEqual(len(storage.csv_file(summary/'per_image_scores.csv')),18)
        self.assertTrue((summary/'objective_report.md').is_file());self.assertFalse('torch' in sys.modules)

    def test_prepared_manifest_mutation_is_rejected(self):
        ctx,_=fixture(self.root);frozen=self.root/'new_output'
        plan=dict(ctx['plan'],excluded=[],candidates=[],overlap=[])
        # Only metadata/fixture supply is substituted; real prepare/ledger/load
        # write and validate the frozen snapshot, checksums and manifests.
        catalog={'memory_sha256':'memory'}
        assets=dict(memory_sha256='memory',checkpoint_sha256='checkpoint')
        paths=dict(ctx['paths'],output_root=frozen)
        for d in selection.DOMAINS:
            plan['groups'].setdefault(d,{str(s):[] for s in selection.SEEDS})
        planned=dict(plan=plan,assets=assets,state=ctx['state'],medians=ctx['medians'],scales=ctx['scales'])
        with patch.object(ip,'build_plan',return_value=planned),patch.object(ip,'inventory',return_value={}),\
             patch.object(ip,'metadata',return_value=(catalog,ctx['state'],ctx['medians'],ctx['scales'])):
            prepared=ip.prepare(paths)
            with patch.object(ip,'build_plan',side_effect=AssertionError('redraw during resume')):
                again=ip.prepare(paths);self.assertEqual(again['identity'],prepared['identity'])
            (frozen/'support_manifest.csv').write_text('modified')
            with self.assertRaises(ValueError):ip.prepare(paths)

    def test_scoring_backend_change_rejected(self):
        ctx,_=fixture(self.root);prediction.run(ctx,['background'],workers=1)
        real=prediction.score
        def drift(*args):
            rows=real(*args)
            for r in rows:r['image_score']+=1
            return rows
        with patch.object(prediction,'score',drift):
            with self.assertRaises(ValueError):prediction.load(ctx,ctx['plan']['samples'][0],42)

    def test_labels_not_in_method_signature_and_no_GT_in_prediction(self):
        import inspect
        self.assertNotIn('label',inspect.signature(predict).parameters)
        ctx,_=fixture(self.root)
        with patch.object(ip.Data,'gt',side_effect=AssertionError('GT used during prediction')):
            prediction.run(ctx,['background'],workers=2)


class SummaryTests(unittest.TestCase):
    def test_domain_equal_then_sample_std_and_delta_direction(self):
        rows=[]
        for di,d in enumerate(selection.DOMAINS):
            for seed in selection.SEEDS:
                for bi,b in enumerate(BRANCHES):
                    v=di*.1+(seed-42)*.01+bi*.003
                    rows.append(dict(domain=d,support_seed=seed,branch=b,image_count=(di+1)*100,
                                     **{k:v for k in reporting.METRICS}))
        metrics,comp,summary=reporting.aggregate(rows)
        m=next(r for r in metrics if r['domain']=='macro_three_domains' and r['support_seed']==42 and r['branch']==U)
        self.assertAlmostEqual(m['image_AP'],.1)
        c=next(r for r in comp if r['domain']=='macro_three_domains' and r['support_seed']==42 and r['branch']==P
               and r['reference']==U and r['metric']=='image_AP')
        self.assertAlmostEqual(c['delta'],.006)
        r=next(r for r in summary if r['domain']=='macro_three_domains' and r['branch']==U and r['reference'] is None and r['metric']=='image_AP')
        self.assertAlmostEqual(r['mean'],.11);self.assertAlmostEqual(r['sample_std'],.01)

    def test_partial_metrics_never_full_macro(self):
        _,_,rows=reporting.aggregate([dict(domain='background',support_seed=42,branch=U,image_AP=.8)])
        self.assertTrue(all(r['status']=='pending' for r in rows))

    def test_gt_small_groups_same_across_seeds_and_na_retained(self):
        rows=[]
        for seed in selection.SEEDS:
            for i in range(5):
                for b in BRANCHES:
                    rows.append(dict(domain='background',support_seed=seed,sample_id=str(i),instance_id=1,
                        branch=b,area_original_pixels=i+1,response_status='mapped_instance_disappeared',
                        margin=None,fixed_union_margin=None,instance_max=None,ring_max=None))
        tables=reporting.spatial_tables(rows,[])
        g=tables['small_instance_groups.csv']
        for seed in selection.SEEDS:self.assertEqual(sum(r['in_smallest_area_quartile'] for r in g if r['support_seed']==seed),2)
        self.assertTrue(all(r['not_computable']==r['total'] for r in tables['instance_margin_summary.csv']))


if __name__=='__main__':unittest.main()
