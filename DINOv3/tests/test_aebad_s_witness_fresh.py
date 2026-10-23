"""CPU-only production adapters; synthetic data are not a CUDA/model validation."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh, fresh_inputs as fi, fresh_runtime as rt
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_evaluation as ev, storage as st, prediction, method
from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import PROTOCOL_ID
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import FORMAT

ROOT=Path(__file__).resolve().parents[2]
TEMP=ROOT/'tmp'


def fixture(root):
    """Two tiny images/domain; only catalog population validation is replaced."""
    run=root/'run'; cache=root/'cache'; data=root/'images'
    run.mkdir();cache.mkdir();(data/'train/good').mkdir(parents=True)
    identity='synthetic-fresh-capture'
    samples=[];supports={};records=[]
    positions=np.array([8,0,4,1,5,2,6,3,7],np.int64)
    state=dict(peak=dict(medians=[0]*4,scales=[1]*4),local=dict(medians=[0]*4,scales=[1]*4))
    state['state_sha256']=st.digest(state)
    st.write_json(run/'image_calibration_state.json',state)
    st.seal(cache/'source_patch_mad.npz',identity,dict(medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32)),{})
    st.seal(cache/'source_bank.npz',identity,dict(bank_7=np.ones((2,3),np.float32)),{})
    for domain in fi.DOMAINS:
        supports[domain]=[]
        for i in range(1,9):
            row=dict(domain=domain,support_index=i,relative_path=f'test/good/{domain}/support{i}.png',
                     image_sha256=f'synthetic{i}',image_height=12,image_width=12,label=0)
            supports[domain].append(row)
            st.seal(cache/'supports'/domain/f'{i:03d}.npz',identity,
                dict(protocol_id=np.asarray(PROTOCOL_ID),relative_path=np.asarray(row['relative_path']),
                    grid=np.array([3,3],np.int32),**{f'layer_{l}':np.full((9,1280),i,np.float32) for l in method.LAYERS}),
                dict(grid=[3,3]),compressed=False)
        for j in range(2):
            sid=f'test_{j:04d}'
            rel=f'test/{"ablation" if j else "good"}/{domain}/q{j}.png'
            image=data/rel;image.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(12,12),(30+j*50,0,0)).save(image)
            row=dict(domain=domain,sample_id=sid,relative_path=rel,label=j,image_sha256=st.sha(image),image_height=12,image_width=12)
            if j:
                gt=data/rel.replace('test/','ground_truth/',1);gt.parent.mkdir(parents=True,exist_ok=True)
                mask=np.zeros((12,12),np.uint8);mask[4:8,4:8]=255;Image.fromarray(mask).save(gt)
                row['gt_sha256']=st.sha(gt)
            samples.append(row)
            base=np.tile(np.linspace(.2,.9,9,dtype=np.float32)+j,(4,1))
            target=np.stack([base+np.float32(.1*i) for i in range(8)])
            raw=dict(positions=positions,source_distance=base+np.float32(.4),source_index=np.zeros((4,9),np.int64),
                target_distance=target,target_index=np.zeros((8,4,9),np.int64),union_distance=base,
                segment_distance=base-np.float32(.05))
            meta=dict(format=FORMAT,domain=domain,sample_id=sid,relative_path=rel,image_sha256=row['image_sha256'],
                grid=[3,3],output_shape=[3,3],source_rows=[100000]*4,supports=[dict(r,grid=[3,3]) for r in supports[domain]])
            meta['scores']=rt.baseline_scores(raw,meta,np.zeros(4,np.float32),np.ones(4,np.float32),state)
            file=f'capture/{domain}/{sid}.npz';record=st.seal(cache/file,identity,raw,meta)
            records.append(dict(domain=domain,sample_id=sid,file=file,sha256=record['sha256'],
                record_sha256=st.sha((cache/file).with_suffix('.json')),scores=meta['scores']))
    c=dict(source=[],samples=samples,supports=supports,data_root=str(data))
    st.write_json(run/'fresh_catalog.json',c)
    st.write_json(run/'setup.json',dict(identity=identity,cache=str(cache),catalog_sha256=st.sha(run/'fresh_catalog.json')))
    st.write_json(run/'capture_manifest.json',dict(identity=identity,status='capture_complete',records=records))
    st.write_json(run/'capture_manifest.seal.json',dict(sha256=st.sha(run/'capture_manifest.json')))
    return run,cache,data


class Fresh(unittest.TestCase):
    def setUp(self):
        TEMP.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=TEMP)
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_defaults_and_single_device(self):
        a=fresh.parser().parse_args(['--run-root','x','--cache-root','y','--data','z','--checkpoint','w'])
        self.assertEqual(a.stage,'preflight');self.assertEqual(a.cpu_workers,8);self.assertEqual(a.device,'cuda:0')

    def test_train_order_and_split(self):
        source=[dict(relative_path=f'train/good/{i:04d}.png') for i in reversed(range(521))]
        test=[dict(relative_path=f'test/good/{d}/{i:03d}.png',domain=d,label=0) for d in fi.DOMAINS for i in range(12)]
        c=fi.select_catalog(source,test)
        self.assertEqual(len(c['source']),521)
        self.assertEqual(sum(r['calibration'] for r in c['source']),66)
        self.assertEqual([r['source_index'] for r in c['source'] if r['calibration']],list(range(0,521,8)))
        for d in fi.DOMAINS:
            expected=sorted((r['relative_path'] for r in test if r['domain']==d),key=lambda p:fi.support_key(p,d))[:8]
            self.assertEqual([r['relative_path'] for r in c['supports'][d]],expected)
            self.assertFalse(set(expected)&{r['relative_path'] for r in c['samples']})

    def test_selection_preserves_global_ids(self):
        tests=[dict(relative_path=f'test/good/{d}/{i:02d}.png',domain=d,label=0) for d in ('same',*fi.DOMAINS) for i in range(10)]
        c=fi.select_catalog([],tests[::-1]);full=sorted(tests,key=lambda r:r['relative_path'])
        for row in c['samples']:
            self.assertEqual(row['sample_id'],f'test_{[r["relative_path"] for r in full].index(row["relative_path"]):04d}')

    def test_preflight_does_not_import_torch_or_write(self):
        out=self.root/'never-created'
        code='import sys; from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh; fresh.main('+repr([
            '--run-root',str(out),'--cache-root',str(self.root/'cache'),'--data',str(self.root/'missing.zip'),
            '--checkpoint',str(self.root/'missing.pth')])+'); assert "torch" not in sys.modules'
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('"old_caches_required": false',r.stdout)
        self.assertFalse(out.exists())

    def test_native_adapter_no_historical_inputs(self):
        run,cache,data=fixture(self.root)
        with patch.object(fi,'validate_catalog'):
            b=fi.Bundle(run)
        self.assertEqual(len(b.samples),6)
        self.assertEqual(len(b.locations),24)
        a,meta,origin=b.load(b.samples[0])
        self.assertTrue(np.array_equal(a['union_distance'],np.minimum(a['source_distance'],a['target_distance'].min(0))))
        self.assertIn('fresh_capture_sha256',origin)
        self.assertEqual(meta['domain'],'background')

    def test_offline_bundle_does_not_need_bank_entity(self):
        run,cache,_=fixture(self.root)
        (cache/'source_bank.npz').unlink()
        with patch.object(fi,'validate_catalog'):b=fi.Bundle(run)
        self.assertEqual(len(b.locations),24)

    def test_modified_raw_rejected(self):
        run,cache,_=fixture(self.root)
        with patch.object(fi,'validate_catalog'):b=fi.Bundle(run)
        p=cache/b.records['background','test_0000']['file']
        with p.open('ab') as f:f.write(b'tampered')
        with self.assertRaisesRegex(ValueError,'modified'):b.load(b.samples[0])

    def test_modified_completed_manifest_rejected(self):
        run,_,_=fixture(self.root)
        p=run/'capture_manifest.json';s=st.read_json(p);s['records']=s['records'][:-1];st.write_json(p,s)
        with patch.object(fi,'validate_catalog'),self.assertRaisesRegex(ValueError,'sealed'):fi.Bundle(run)

    def test_duplicate_manifest_records_rejected(self):
        run,_,_=fixture(self.root)
        p=run/'capture_manifest.json';s=st.read_json(p);s['records'][1]=s['records'][0];st.write_json(p,s)
        st.write_json(run/'capture_manifest.seal.json',dict(sha256=st.sha(p)))
        with patch.object(fi,'validate_catalog'),self.assertRaisesRegex(ValueError,'coverage'):fi.Bundle(run)

    def test_real_freeze_cpu_evaluation_and_partial_summary(self):
        run,cache,_=fixture(self.root)
        with patch.object(fi,'validate_catalog'):b=fi.Bundle(run)
        out=st.initialize(run/'experiment',b)
        for sample in b.samples:
            raw,meta,origin=b.load(sample)
            for off,l in enumerate(method.LAYERS):
                gate=np.arange(9)%2==0
                distances,red=method.reduction(raw['union_distance'][off],raw['segment_distance'][off],gate)
                st.seal(prediction.layer_path(out,sample['domain'],l,sample['sample_id']),b.identity,
                    dict(positions=raw['positions'],gate=gate,**distances),
                    dict(domain=sample['domain'],layer=l,sample_id=sample['sample_id'],origin=origin,
                         grid=meta['grid'],output_shape=meta['output_shape'],counts=dict(accepted_positions=int(gate.sum())),reduction=red,neighbors='fixture'))
        with contextlib.redirect_stdout(io.StringIO()):
            for d in fi.DOMAINS:prediction.freeze_domain(b,out,d,workers=2)
            for d in fi.DOMAINS:ev.evaluate(b,out,d,2)
            result=ev.summarize(b,out)
        self.assertEqual(result['status'],'partial_evidence')
        self.assertTrue(any('AUPRO' in s for s in result['pending']))
        rows=ev.old_eval.ip.parent.csv_file(out/'evaluation/background/metrics.csv')
        self.assertEqual(len(rows),8)
        self.assertTrue(all(r['pixel_metric_origin']=='fresh_exact_all_pixels_CPU' for r in rows))
        self.assertEqual({r['scoring'] for r in rows},{'CH1','CURRENT'})
        self.assertEqual(len(ev.old_eval.ip.parent.csv_file(out/'evaluation/background/defect_instances.csv')),4)
        with contextlib.redirect_stdout(io.StringIO()):
            for d in fi.DOMAINS:self.assertEqual(ev.evaluate(b,out,d,2)['status'],'CPU_reused')

    def test_gt_not_read_without_global_predictions(self):
        run,_,_=fixture(self.root)
        with patch.object(fi,'validate_catalog'):b=fi.Bundle(run)
        with patch.object(fi.Data,'gt',side_effect=AssertionError('GT read before freeze')) as gt:
            with self.assertRaises(FileNotFoundError):ev.evaluate(b,run/'experiment','background',2)
            gt.assert_not_called()

    def test_source_reuse_does_not_construct_model(self):
        run,cache,_=fixture(self.root)
        with patch.object(rt,'host',side_effect=AssertionError('unexpected model')):
            self.assertEqual(rt.source(SimpleNamespace(run_root=run,cache_root=cache))['status'],'source_bank_reused')

    def test_complete_capture_resume_never_constructs_model(self):
        run,cache,_=fixture(self.root)
        with patch.object(fi,'validate_catalog'),patch.object(rt,'host',side_effect=AssertionError('unexpected model')):
            result=rt.capture(SimpleNamespace(run_root=run,cache_root=cache))
        self.assertEqual(result['query_forward_calls'],0)
        self.assertEqual(result['support_forward_calls'],0)
        self.assertEqual(result['reused_query_caches'],6)

    def test_original_support_header_and_actual_layer_loader(self):
        run,_,_=fixture(self.root)
        with patch.object(fi,'validate_catalog'):b=fi.Bundle(run)
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import inputs
        values=inputs.load_layer(b.expected,b.locations,'view',31)
        self.assertEqual(set(values),set(range(1,9)))
        np.testing.assert_array_equal(values[3],np.full((9,1280),3,np.float32))

    def test_real_same_call_nn_and_segment_on_torch_cpu(self):
        # Only feature extraction is substituted. Original observe_fewshot_pil,
        # exact NN, callback logging, positions, units and SEGMENT execute.
        import torch
        from DINOv3.MADEqual.robustad_target_fewshot_p0 import scoring
        from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import CaptureLog
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import add_geometry,scatter
        rng=np.random.default_rng(42)
        q={l:torch.from_numpy(rng.normal(size=(9,1280)).astype(np.float32)) for l in method.LAYERS}
        banks={l:[torch.from_numpy(rng.normal(size=(10,1280)).astype(np.float32)) for _ in range(8)] for l in method.LAYERS}
        source={l:torch.from_numpy(rng.normal(size=(15,1280)).astype(np.float32)) for l in method.LAYERS}
        positions=np.array([8,1,2,3,4,5,6,7,0],np.int64)
        pending=SimpleNamespace(grid=(3,3),positions=positions,queries=q)
        log=CaptureLog()
        with patch.object(scoring,'_extract_processed_batch',return_value=[pending]):
            observed=scoring.observe_fewshot_pil(SimpleNamespace(preprocessing=lambda x:x),Image.new('RGB',(12,12)),
                official_banks=source,source_support=None,target_support=banks,device=torch.device('cpu'),
                nearest_observer=log.nearest,geometry_observer=log.geometry)
        raw=log.arrays()
        np.testing.assert_array_equal(scatter(raw['source_distance'],positions,(3,3)),observed.official_raw)
        union=np.minimum(raw['source_distance'],raw['target_distance'].min(0))
        np.testing.assert_array_equal(scatter(union,positions,(3,3)),observed.target_raw[8])
        geometry,diagnostics=add_geometry(raw,{l:[t.numpy() for t in banks[l]] for l in method.LAYERS})
        self.assertEqual(geometry['segment_distance'].shape,(4,9))
        self.assertTrue(np.all(geometry['segment_distance']<=union+1e-7+1e-6*np.abs(union)))
        self.assertTrue(all(r['triangle_failed_count']==0 for r in diagnostics))

    def test_support_selection_matches_original_production_rule(self):
        from DINOv3.MADEqual.aebad_s_target_fewshot.support import stable_support_key
        base=self.root/'images';p=base/'test/good/view/IMG_0001.png'
        record=SimpleNamespace(path=str(p))
        self.assertEqual(fi.support_key('test/good/view/IMG_0001.png','view'),
            stable_support_key(record,dataset_root=base,domain='view'))

    def test_actual_source_only_calibration_from_66_cached_observations(self):
        run=self.root/'run';cache=self.root/'cache';identity='calibration-fixture'
        run.mkdir();cache.mkdir()
        rows=[dict(relative_path=f'train/good/{i:04d}.png',calibration=i%8==0,
            image_height=12,image_width=12) for i in range(521)]
        st.write_json(run/'fresh_catalog.json',dict(source=rows))
        st.write_json(run/'setup.json',dict(identity=identity))
        rng=np.random.default_rng(24);values=[]
        for i,row in enumerate((r for r in rows if r['calibration']),1):
            raw=rng.uniform(.1,2,(4,3,3)).astype(np.float32);values.append(raw)
            st.seal(cache/'calibration'/f'{i:03d}.npz',identity,dict(raw=raw),
                dict(relative_path=row['relative_path'],grid=[3,3],output_shape=[3,3]))
        args=SimpleNamespace(run_root=run,cache_root=cache)
        with patch.object(rt,'host',side_effect=AssertionError('model should not be loaded')),contextlib.redirect_stdout(io.StringIO()):
            result=rt.calibration(args)
            before=st.sha(run/'image_calibration_state.json')
            rt.calibration(args)
        self.assertEqual(result['images'],66)
        self.assertEqual(before,st.sha(run/'image_calibration_state.json'))
        from DINOv3.relation_reliability.calibration import fit_calibrators
        _,expected=fit_calibrators({l:np.concatenate([a[j].ravel() for a in values]) for j,l in enumerate(method.LAYERS)})
        actual,_=st.arrays(cache/'source_patch_mad.npz',identity)
        np.testing.assert_array_equal(actual['medians'],expected.medians)
        np.testing.assert_array_equal(actual['scales'],expected.scales)
        state=st.read_json(run/'image_calibration_state.json')
        self.assertEqual(state['origin'],'fresh_Source_only_66')
        self.assertEqual(len(ev.old_eval.ip.parent.csv_file(run/'calibration_per_image.csv')),66)
        state['peak']['medians'][0]+=1;state['state_sha256']=st.digest({k:v for k,v in state.items() if k!='state_sha256'})
        st.write_json(run/'image_calibration_state.json',state)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'image MAD changed'):
            rt.calibration(args)

    def test_all_orchestrator_stops_after_actual_stage_error(self):
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import autodl
        root=self.root/'run';cache=self.root/'cache';code=self.root/'code';code.mkdir()
        calls=[]
        def launch(label,cmd,output):
            calls.append(label)
            if label=='capture':raise RuntimeError('synthetic real stage failure')
        argv=['all','--run-root',str(root),'--cache-root',str(cache),'--data',str(self.root/'data.zip'),
            '--checkpoint',str(self.root/'weight.pth')]
        with patch.object(fi,'ROOT',code),patch.object(fresh,'configuration',return_value={'fixture':True}),\
             patch.object(autodl,'run',side_effect=launch),patch.object(autodl,'pack',return_value=root.parent/'review.tar.gz'),\
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            rc=fresh.main(argv)
        self.assertEqual(rc,2)
        self.assertEqual(calls[-1],'capture')
        self.assertNotIn('prepare-support',calls)
        self.assertEqual(Path(str(root)+'.exitcode').read_text().strip(),'2')
        self.assertFalse(st.read_json(root/'pipeline_status.json')['success'])


if __name__=='__main__':unittest.main()
