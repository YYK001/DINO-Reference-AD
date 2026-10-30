"""CPU tests with scalar response fixtures. No real model/retrieval/CUDA."""
import copy,io,json,sys,time,tempfile,unittest,subprocess
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_loo_weight_separability_audit import inputs as ip,numerics as nm,prediction as pred,evaluation as ev,analysis,reporting
from DINOv3.MADEqual.aebad_s_loo_weight_separability_audit.protocol import *
from DINOv3.MADEqual.aebad_s_loo_weight_separability_audit.runner import parser,normalize,pack


def fixture(root):
    a=normalize(parser().parse_args(['--output-root',str(root/'audit'),'--parent-run',str(root/'parent'),'--data',str(root/'GT')]))
    a._started=time.perf_counter();a.allow_over_12h=False;a.stage='predict'
    p=SimpleNamespace(root=root/'parent',identity='frozen-parent',inputs=dict(parent_cache=str(root/'old42'),source={'seal':'Source42'},mad={'seal':'MAD','patch':{}},extraction={'frozen':'original'},seal='parent-inputs'),settings={},weights={},records={},maps={},support_overlaps={d:0 for d in DOMAINS})
    samples={d:[dict(domain=d,sample_id=f'sample{i}',relative_path=f'test/{d}/{i}.png',label=i,image_height=32,image_width=40,image_sha256=str(i)) for i in (0,1)] for d in DOMAINS}
    groups={d:dict(common=samples[d],supports={str(s):[dict(relative_path=f'support/{d}/{s}/{i}') for i in range(8)] for s in SEEDS}) for d in DOMAINS}
    p.common=dict(seal='parent-common',domains=groups)
    w={'WS':np.full((4,3),.25).tolist(),'W42':np.tile(np.array([.1,.2,.3,.4])[:,None],(1,3)).tolist(),'W43':np.tile(np.array([.4,.3,.2,.1])[:,None],(1,3)).tolist()}
    for d in DOMAINS:
        for name,matrix in w.items():p.weights[d,name]=matrix
    origins={};rng=np.random.default_rng(11)
    for d in DOMAINS:
        for m in SEEDS:
            p.maps[d,m]={};weight_rows=[]
            for b,matrix in ((SOURCE,w['WS']),(TARGET,w[diagonal(m)])):
                for j,l in enumerate(LAYERS):
                    for k,stat in enumerate(('u','p','q')):weight_rows.append(dict(branch=b,layer=l,statistic=stat,Source_scale=1.,Target_LOO_scale=2.,mixed_scale=1.,weight=matrix[j][k]))
            p.settings[d,m]=dict(weight_rows=weight_rows,metrics=[])
            for row in samples[d]:
                z=rng.random((4,20),dtype=np.float32)+row['label'];pos=rng.permutation(20).astype(np.int64)
                path=root/f'original{m}'/d/row['sample_id']
                meta=dict(grid=[4,5],output_shape=[8,10])
                if m==42:meta.update(sample=row,calibration_seal='MAD')
                else:meta['dependencies']=dict(domain=d,support_seed=m,Source_seed=42,source='Source42',supports=groups[d]['supports']['43'],extraction=p.inputs['extraction'],patch={},image=row)
                r=ip.st.put(path,{U:z,'positions':pos},meta);origins[d,m,row['sample_id']]=dict(path=str(path),seal=r['seal'])
                images,sc,stat=nm.responses(z,pos,[4,5],[8,10],w)
                p.records[d,m,row['sample_id']]=dict(map_hashes={SOURCE:ev.backend.shared.array_sha(images['WS']),TARGET:ev.backend.shared.array_sha(images[diagonal(m)])},scores={SOURCE:sc['WS'],TARGET:sc[diagonal(m)]})
                original=np.zeros((32,40),bool)
                if row['label']:original[4:12,4:12]=True
                mask=ev.backend.shared.resized_mask(original,[8,10]).astype(np.uint8)
                p.maps[d,m][row['sample_id'],SOURCE]=dict(mask_uint8_sha256=ev.backend.shared.array_sha(mask))
            for name in ('WS',diagonal(m)):
                b=SOURCE if name=='WS' else TARGET
                for rule in READOUTS:
                    metric=ev.image_metrics([r['label'] for r in samples[d]],[p.records[d,m,r['sample_id']]['scores'][b][rule] for r in samples[d]])
                    p.settings[d,m]['metrics'].append(dict(domain=d,branch=b,scoring=rule,**metric,pixel_AUROC=.8,pixel_AP=.3,AUPRO=.5))
    p.samples=lambda d:samples[d];p.record=lambda d,m,row:p.records[d,m,row['sample_id']]
    p.original=lambda d,m,row:origins[d,m,row['sample_id']]
    p.known_metrics=lambda d,m,w:[r for r in p.settings[d,m]['metrics'] if r['branch']==(SOURCE if w=='WS' else TARGET)]
    a._parent=p;ip.freeze(a)
    class Data:
        def __init__(self,path):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def gt(self,row):
            result=np.zeros((32,40),bool)
            if row['label']:result[4:12,4:12]=True
            return result
    return a,p,Data


class AuditTests(unittest.TestCase):
    def test_default_preflight_and_all_CPU_imports_no_torch(self):
        code='import sys; from DINOv3.MADEqual.aebad_s_loo_weight_separability_audit import inputs,numerics,prediction,evaluation,analysis,reporting,smoke; from DINOv3.MADEqual.aebad_s_loo_weight_separability_audit.runner import parser; assert parser().parse_args([]).stage=="preflight"; assert "torch" not in sys.modules'
        r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)

    def test_crossing_changes_only_full_weight_matrix_and_positions(self):
        rng=np.random.default_rng(2);z=rng.random((4,20),dtype=np.float32);pos=rng.permutation(20);before=z.copy()
        w={'WS':np.full((4,3),.25),'W42':np.tile([.1,.2,.3,.4],(3,1)).T,'W43':np.tile([.4,.3,.2,.1],(3,1)).T}
        images,scores,stats=nm.responses(z,pos,[4,5],[8,10],w)
        for key in WEIGHTS:
            image,score=nm.kernel.weighted(z,pos,[4,5],[8,10],w[key]);np.testing.assert_array_equal(images[key],image);self.assertEqual(scores[key],score)
        order=rng.permutation(20);other=nm.responses(z[:,order],pos[order],[4,5],[8,10],w)
        for key in WEIGHTS:np.testing.assert_array_equal(images[key],other[0][key])
        lm=nm.per_layer_maps(z,pos,[4,5],[8,10]);np.testing.assert_array_equal(stats[:,1],lm.max(axis=(1,2)))
        _,p,q=nm.from_z(z,pos,[4,5],[8,10]);np.testing.assert_array_equal(stats[:,2],q);np.testing.assert_array_equal(z,before)

    def test_exact_pair_ties_and_pair_AUROC_independent_check(self):
        samples=[dict(label=x) for x in (0,0,1,1)];v=np.array([1.,1.,1.,np.nextafter(np.float32(1),np.float32(2))])
        scores=[{w:{READOUTS[0]:float(x)} for w in WEIGHTS} for x in v];stats=np.zeros((4,4,3));weights={w:np.full((4,3),.25) for w in WEIGHTS}
        rows,contrib,compact,checks=nm.pair_analysis(samples,scores,stats,weights)
        self.assertTrue(any(r.get('actual')==.75 for r in checks))
        c=compact['W42_minus_WS'];self.assertEqual(int((c['old_state']==0).sum()),2);self.assertEqual(int((c['old_state']==1).sum()),2)
        self.assertTrue(np.all(c['old_gap'][c['old_state']==1]>0))

    def test_pair_layer_contributions_reproduce_gap_and_no_mapmax_linearization(self):
        rng=np.random.default_rng(3);samples=[dict(label=x) for x in (0,0,1,1)];stats=rng.random((4,4,3)).astype(np.float32)
        weights={'WS':np.full((4,3),.25),'W42':np.tile([.1,.2,.3,.4],(3,1)).T,'W43':np.tile([.4,.3,.2,.1],(3,1)).T};scores=[]
        for x in stats:
            scores.append({w:{READOUTS[0]:float(.5*np.sum(np.asarray(mat,np.float32)[:,1]*x[:,1],dtype=np.float32)+.5*np.sum(np.asarray(mat,np.float32)[:,2]*x[:,2],dtype=np.float32)),READOUTS[1]:1e9} for w,mat in weights.items()})
        rows,parts,compact,checks=nm.pair_analysis(samples,scores,stats,weights)
        self.assertEqual(len(parts),3*3*4*2)
        for x in compact.values():self.assertLess(float(abs((x['new_gap']-x['old_gap'])-x['layer_pq_contributions'].sum(axis=(1,2))).max()),SCORE_ATOL)

    def test_normal_top1_same_WS_locations_and_deterministic_ties(self):
        source=np.zeros((5,10),np.float32);source.ravel()[:2]=10;new=source.copy();new.ravel()[0]=9;new.ravel()[-1]=100
        row=nm.normal_tail({'WS':source,'W42':new,'W43':source})[0]
        self.assertEqual((row['anchor_y'],row['anchor_x']),(0,0));self.assertEqual(row['top_pixels'],1);self.assertEqual(row['fixed_top1_mean_change'],-1);self.assertEqual(row['own_peak_change'],90);self.assertTrue(row['own_peak_moved'])

    def test_fixed_margin_separates_peak_movement_and_original_16px_ring(self):
        z=np.zeros((4,9),np.float32);z[0,0]=8;z[1,1]=6
        weights={'WS':np.full((4,3),.25),'W42':np.tile([.1,.6,.2,.1],(3,1)).T,'W43':np.tile([.4,.3,.2,.1],(3,1)).T}
        maps,sc,s=nm.responses(z,np.arange(9),[3,3],[3,3],weights);layers=nm.per_layer_maps(z,np.arange(9),[3,3],[3,3]);gt=np.zeros((3,3),bool);gt[0,:2]=True
        rows,checks=nm.instance_changes(maps,layers,gt,weights);r=next(r for r in rows if r['comparison']=='W42_minus_WS')
        self.assertEqual(r['fixed_margin_group'],'decreased');self.assertEqual(r['own_margin_group'],'improved');self.assertTrue(r['inside_peak_moved']);self.assertLess(abs(sum(r[f'contribution_L{l}'] for l in LAYERS)-r['fixed_margin_change']),SCORE_ATOL)
        self.assertEqual(r['area_original_pixels'],2);self.assertEqual(r['ring_radius_original_pixels'],16)

    def test_empty_instance_and_ring_remain_NA(self):
        images={w:np.zeros((2,2),np.float32) for w in WEIGHTS};layers=np.zeros((4,2,2),np.float32);weights={w:np.full((4,3),.25) for w in WEIGHTS}
        for gt in (np.eye(32,dtype=bool)[:1].repeat(32,axis=0),np.ones((32,32),bool)):
            # First narrow component can disappear under nearest-neighbor mapping.
            if not gt.all():gt[:]=False;gt[0,0]=True
            rows,_=nm.instance_changes(images,layers,gt,weights)
            self.assertEqual(len(rows),3);self.assertTrue(all(r['fixed_margin_group']=='NA' for r in rows));self.assertTrue(all(r['fixed_margin_change'] is None for r in rows))

    def test_no_GT_before_global_freeze_and_prediction_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,p,Data=fixture(Path(tmp));parent_bytes={x:x.read_bytes() for x in Path(tmp).glob('original*/**/manifest.json')}
            with self.assertRaisesRegex(ValueError,'ALL six'):pred.global_freeze(a)
            a.domains=['background'];a.memory_seeds=[42]
            with patch.object(ev,'GTData',side_effect=AssertionError('no GT in prediction')),redirect_stdout(io.StringIO()):self.assertEqual(pred.freeze(a)['status'],'pending')
            with self.assertRaisesRegex(ValueError,'ALL six'):analysis.analyze(a)
            a.domains=list(DOMAINS);a.memory_seeds=list(SEEDS)
            with redirect_stdout(io.StringIO()):r=pred.freeze(a)
            self.assertEqual(r['images'],12);before=(a.output_root/'responses_frozen.json').read_bytes()
            with patch.object(pred,'load_maps',side_effect=AssertionError('completed prediction must be reused')),redirect_stdout(io.StringIO()):pred.freeze(a)
            self.assertEqual(before,(a.output_root/'responses_frozen.json').read_bytes());self.assertTrue(all(x.read_bytes()==v for x,v in parent_bytes.items()))
            self.assertEqual(ip.signature(a,'view',42)['weight_seed'],43);self.assertEqual(ip.signature(a,'view',43)['weight_seed'],42)
            target=ip.record_path(a,'view',43,p.samples('view')[0]);changed=ip.st.sealed_json(target);changed['scores']['WS'][READOUTS[0]]+=1
            changed['seal']=ip.st.digest({k:v for k,v in changed.items() if k!='seal'});ip.st.write_json(target,changed)
            with self.assertRaisesRegex(ValueError,'modified'):reporting.summarize(a)

    def test_real_CPU_pipeline_six_cross_only_and_completed_AUPRO_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,p,Data=fixture(Path(tmp))
            with redirect_stdout(io.StringIO()):pred.freeze(a)
            with patch.object(ev,'GTData',Data),redirect_stdout(io.StringIO()),patch.object(ev.backend,'exact_pixel_result',wraps=ev.backend.exact_pixel_result) as pixel:
                ev.evaluate(a);self.assertEqual(pixel.call_count,6);ev.evaluate(a);self.assertEqual(pixel.call_count,6)
            with patch.object(analysis,'GTData',Data),redirect_stdout(io.StringIO()):analysis.analyze(a);analysis.analyze(a)
            status=reporting.summarize(a);self.assertEqual(status['status'],'pending');self.assertEqual(status['metric_rows'],36);self.assertEqual(status['new_AUPRO_complete'],0)
            for d in DOMAINS:
                for m in SEEDS:
                    path=ip.unit(a,d,m);maps=ip.parent_reporting.rows(path/'evaluation/map_manifest.csv');sig=ip.signature(a,d,m)
                    ip.st.sealed_json(path/'aupro.json',dict(**sig,branch=branch(m,crossed(m)),map_identity=ip.st.digest(maps),nstrips=200,FPR_max=.3,AUPRO=.55,verified_images=2,backend='original_ADEval_1.1.0_CUDA'))
            with patch.object(ev,'GTData',Data),patch.object(ev,'replay',side_effect=AssertionError('complete AUPRO must not recompute')):self.assertEqual(ev.aupro(a)['jobs'],6)
            status=reporting.summarize(a);self.assertFalse(status['pending']);self.assertEqual(status['status'],'weight_cross_and_separability_audit_complete')
            self.assertEqual(len(ip.parent_reporting.rows(a.output_root/'fixed_comparisons.csv')),270)
            self.assertNotIn('torch',sys.modules)

    def test_export_is_relocatable_and_original_manifests_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,p,Data=fixture(root);a.export_root=root/'export';original=p.original('view',43,p.samples('view')[0]);before=(Path(original['path'])/'manifest.json').read_bytes()
            result=ip.export_responses(a);self.assertEqual(len(result['records']),12)
            a.response_root=a.export_root
            ref=ip.inspect(a,'view',43,p.samples('view')[0]);self.assertTrue(Path(ref['path']).is_relative_to(a.export_root));self.assertEqual(before,(Path(original['path'])/'manifest.json').read_bytes())
            self.assertEqual(before,(Path(ref['path'])/'manifest.json').read_bytes());self.assertFalse(list(a.export_root.rglob('q7.npy')))

    def test_output_isolation_and_no_training_model_or_seed44_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):normalize(parser().parse_args(['--output-root',str(Path(tmp)/'parent/out'),'--parent-run',str(Path(tmp)/'parent')]))
        self.assertNotIn('checkpoint',[action.dest for action in parser()._actions]);self.assertNotIn(44,SEEDS)

    def test_smoke_is_fixed_and_does_not_read_GT_before_global_freeze(self):
        from DINOv3.MADEqual.aebad_s_loo_weight_separability_audit import smoke
        cases=[dict(label=1,id=0),dict(label=1,id=1),dict(label=0,id=2)]
        toy=SimpleNamespace(samples=lambda d:cases)
        self.assertEqual(smoke.selected(toy,'view'),cases[:2]);self.assertEqual(smoke.selected(toy,'view',True),[cases[2],cases[0]])
        with tempfile.TemporaryDirectory() as tmp:
            a,p,Data=fixture(Path(tmp));a.stage='smoke'
            with patch.object(smoke,'GTData',side_effect=AssertionError('no GT yet')):
                r=smoke.run(a);self.assertFalse(r['GT_used']);self.assertEqual(len(r['rows']),6)
                with self.assertRaisesRegex(ValueError,'ALL six'):smoke.run(a,with_gt=True)
            self.assertFalse((a.output_root/'responses_frozen.json').exists())

    def test_budget_pause_does_not_change_population_or_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,p,Data=fixture(Path(tmp));a._started=time.perf_counter()-44000;before=copy.deepcopy(p.weights)
            with self.assertRaisesRegex(RuntimeError,'12h'):pred.budget(a,5,10)
            self.assertEqual(before,p.weights);self.assertTrue((a.output_root/'budget_pause.json').exists())

    def test_bad_diagonal_response_is_rejected_before_any_GT(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,p,Data=fixture(Path(tmp));row=p.samples('background')[0];p.record('background',42,row)['map_hashes'][SOURCE]='wrong-diagonal'
            with patch.object(ev,'GTData',side_effect=AssertionError('no GT on invalid predictions')):
                with self.assertRaisesRegex(ValueError,'reproduction'):pred.freeze(a)

    def test_corrupt_small_completion_is_not_resealed_and_scratch_excluded(self):
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,p,Data=fixture(root)
            with redirect_stdout(io.StringIO()):pred.freeze(a)
            with patch.object(ev,'GTData',Data),redirect_stdout(io.StringIO()):ev.evaluate(a)
            path=ip.unit(a,'view',43)/'evaluation/metrics.csv';path.write_text('modified',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'modified'):ev.completed(a,'view',43,'evaluation')
            scratch=a.output_root/'pixel_scratch';scratch.mkdir();(scratch/'old.bin').write_bytes(b'tmp')
            archive=pack(a)
            with tarfile.open(archive['archive']) as t:self.assertFalse(any('scratch' in n or n.endswith('.bin') for n in t.getnames()))

    @unittest.skipUnless((ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433/inputs.json').exists(),'local parent review not shipped')
    def test_real_parent_inventory_902_common_no_bank_or_feature_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=normalize(parser().parse_args(['--output-root',str(Path(tmp)/'audit')]))
            r=ip.preflight(a);self.assertEqual(r['status'],'missing_inputs');self.assertEqual(r['unique_images'],902);self.assertEqual(r['memory_image_records'],1804)
            rows=ip.parent_reporting.rows(a.output_root/'missing_inputs.csv');self.assertEqual(sum(r['role']=='standardized_UNION_four_layers_and_positions' for r in rows),1804)
            self.assertFalse(any('bank' in r['role'] or 'checkpoint' in r['role'] or 'feature' in r['role'] for r in rows))


if __name__=='__main__':unittest.main()
