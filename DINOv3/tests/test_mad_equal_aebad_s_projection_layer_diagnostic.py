"""Independent small references and synthetic full-scope CPU integration only."""
from __future__ import annotations
import contextlib
import copy
import csv
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic import numerics as math,metrics,inputs,spatial,analysis,runner
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import interpolation_stencil,components,instance_regions


def reference_resize(a,shape):
    result=np.empty(shape,np.float32)
    for y in range(shape[0]):
        for x in range(shape[1]):result[y,x]=sum(float(a[yy,xx])*w for yy,xx,w in interpolation_stencil((y,x),a.shape,shape))
    return result


def fixture(root):
    """Synthetic 297 images, tiny complete pixel fields; no historical experiment."""
    cases=[dict(sample_id=s,relative_path=p,label=int(i>=10),image_height=32,image_width=48) for i,(s,p) in enumerate(inputs.replay.CASES)]
    extra=[dict(sample_id=f'synthetic_n_{i}',relative_path=f'test/good/background/n{i}.png',label=0,image_height=32,image_width=48) for i in range(75)]
    extra +=[dict(sample_id=f'synthetic_a_{i}',relative_path=f'test/defect/background/a{i}.png',label=1,image_height=32,image_width=48) for i in range(210)]
    samples=cases+extra;cache={};old_scores={'U':[],'P':[]};old_instances={'U':[],'P':[]};anchors=[]
    medians=np.array([1,2,3,4],np.float32);scales=np.array([.5,1,2,3],np.float32)
    p=np.array([3,0,5,1,4,2],np.int64);rng=np.random.RandomState(42);small=0
    dataset=root/'toy.zip'
    with zipfile.ZipFile(dataset,'w') as z:
        for ordinal,sample in enumerate(samples):
            sid=sample['sample_id'];d=(rng.uniform(3,6,(4,2,3))+sample['label']).astype(np.float32)
            pd=(d*np.array([.9,1.01,.96,.92],np.float32)[:,None,None]).astype(np.float32)
            maps={b:{l:reference_resize(((v-medians[:,None,None])/scales[:,None,None])[j],(8,12)) for j,l in enumerate(math.LAYERS)} for b,v in [('U',d),('P',pd)]}
            for b,v in [('U',d),('P',pd)]:maps[b]['FUSED']=reference_resize(((v-medians[:,None,None])/scales[:,None,None]).mean(0),(8,12))
            cache[sid]=dict(U=d.reshape(4,-1)[:,p].copy(),P=pd.reshape(4,-1)[:,p].copy(),positions=p.copy(),
                metadata=dict(grid=[2,3],output_shape=[8,12],geometry=dict(original_height=32,original_width=48,processed_height=32,processed_width=48,grid=[2,3])))
            for b in ('U','P'):
                y,x=np.unravel_index(np.argmax(maps[b]['FUSED']),(8,12))
                old_scores[b].append(dict(branch='UNION_K8' if b=='U' else 'PROJECTION_TARGET512_SOURCE',sample_id=sid,
                    relative_path=sample['relative_path'],label=str(sample['label']),image_score=str(float(maps[b]['FUSED'].max())),full_peak_y=str(y),full_peak_x=str(x)))
            if not sample['label']:continue
            mask=np.zeros((32,48),bool)
            tiny=small<3
            if tiny:mask[1,1]=True;small+=1
            else:mask[10:18,12:20]=True
            blob=io.BytesIO();Image.fromarray(mask.astype(np.uint8)*255).save(blob,format='PNG')
            z.writestr(inputs.PREFIX+sample['relative_path'].replace('test/','ground_truth/',1),blob.getvalue())
            labels,descs=components(mask)
            for desc in descs:
                inside,ring=instance_regions(labels==desc['instance_id'],mask,(8,12))
                state='mapped_instance_disappeared' if not inside.any() else ('empty_ring' if not ring.any() else 'available')
                for b in ('U','P'):
                    im=float(maps[b]['FUSED'][inside].max()) if inside.any() else None
                    rm=float(maps[b]['FUSED'][ring].max()) if ring.any() else None
                    old_instances[b].append(dict(branch='UNION_K8' if b=='U' else 'PROJECTION_TARGET512_SOURCE',sample_id=sid,**desc,
                        instance_max='' if im is None else str(im),ring_max='' if rm is None else str(rm),margin='' if im is None or rm is None else str(im-rm),
                        patch_coverage=str(int(math.adaptive_max(labels==desc['instance_id'],(2,3)).sum())),response_status=state,original_smallest_three_rule=str(tiny)))
                if state=='available':
                    for role,region in [('instance',inside),('ring',ring)]:
                        index=int(np.argmax(np.where(region,maps['U']['FUSED'],-np.inf)));y,x=divmod(index,12)
                        anchors.append(dict(branch='UNION_K8',reference='UNION_K8',sample_id=sid,instance_id=str(desc['instance_id']),region=role,
                            reference_peak_y=str(y),reference_peak_x=str(x),score_at_fixed_reference_peak=str(float(maps['U']['FUSED'][y,x]))))
    paths=dict(parent=root/'parent.bin',projection=root/'projection.bin',mad=root/'mad.npz',data=dataset,output=root/'output')
    paths['parent'].write_bytes(b'synthetic-parent');paths['projection'].write_bytes(b'synthetic-projection')
    np.savez(paths['mad'],medians=medians,scales=scales)
    context=dict(samples=samples,cache=cache,medians=medians,scales=scales,paths=paths,pixel_count=297*8*12,patch_count=297*6,
                 checksums={k:inputs.sha(v) for k,v in paths.items() if k!='output'})
    labels=[s['label'] for s in samples]
    for b,k in [('U','parent'),('P','projection')]:
        # CPU exact independent scikit-learn reference for this toy fixture.
        scores=[float(r['image_score']) for r in old_scores[b]]
        pixel_y=[];pixel_scores=[]
        with inputs.DatasetZip(dataset).archive as z:
            for sample in samples:
                c=cache[sample['sample_id']];d=c[b];physical=np.empty_like(d);physical[:,p]=d
                fused=reference_resize(((physical.reshape(4,2,3)-medians[:,None,None])/scales[:,None,None]).mean(0),(8,12))
                gt=np.zeros((8,12),np.uint8)
                if sample['label']:
                    with z.open(inputs.PREFIX+sample['relative_path'].replace('test/','ground_truth/',1)) as f:
                        with Image.open(f) as im:gt=np.asarray(im.resize((12,8),Image.Resampling.NEAREST))>0
                pixel_y.extend(gt.reshape(-1));pixel_scores.extend(fused.reshape(-1))
        name='UNION_K8' if b=='U' else 'PROJECTION_TARGET512_SOURCE'
        context[k]=dict(data={'evaluation/per_image_scores.csv':old_scores[b],'evaluation/defect_instances.csv':old_instances[b],
             'evaluation/metrics.csv':[dict(branch=name,image_AUROC=str(roc_auc_score(labels,scores)),image_AP=str(average_precision_score(labels,scores)),
                pixel_AUROC=str(roc_auc_score(pixel_y,pixel_scores)),pixel_AP=str(average_precision_score(pixel_y,pixel_scores)))]})
    context['parent']['data']['evaluation/fixed_reference_instance_peaks.csv']=anchors
    return context


class NumericTests(unittest.TestCase):
    def test_bilinear_matches_independent_half_pixel_stencils(self):
        rng=np.random.RandomState(7)
        for grid,out in [((2,3),(7,11)),((4,5),(2,3)),((1,1),(8,9)),((1,5),(3,2)),((3,4),(3,4))]:
            a=rng.normal(size=grid).astype(np.float32)
            np.testing.assert_allclose(math.bilinear(a,out),reference_resize(a,out),atol=2e-6,rtol=1e-6)

    def test_adaptive_max_has_overlapping_bins_not_nearest_resize(self):
        rng=np.random.RandomState(3)
        for shape,grid in [((7,9),(3,4)),((2,3),(5,7)),((8,8),(2,2))]:
            mask=rng.rand(*shape)>.8
            reference=np.array([[mask[y*shape[0]//grid[0]:((y+1)*shape[0]+grid[0]-1)//grid[0],x*shape[1]//grid[1]:((x+1)*shape[1]+grid[1]-1)//grid[1]].any() for x in range(grid[1])] for y in range(grid[0])])
            np.testing.assert_array_equal(math.adaptive_max(mask,grid),reference)
        mask=np.zeros((7,7),bool);mask[2,2]=True
        self.assertEqual(math.adaptive_max(mask,(3,3)).sum(),4)

    def test_positions_negative_MAD_and_fusion_score_rule(self):
        physical=np.array([[[1.,5],[3,2]],[[5,1],[2,3]],[[2,3],[5,1]],[[3,2],[1,5]]],np.float32)
        p=np.array([3,1,0,2],np.int64);raw=physical.reshape(4,4)[:,p]
        z,maps,error=math.reconstruct(raw,p,(2,2),(7,7),np.full(4,4,np.float32),np.ones(4,np.float32))
        np.testing.assert_array_equal(z,physical-4);self.assertLess(z.min(),0)
        self.assertLess(error,1e-5)
        self.assertNotAlmostEqual(float(maps['FUSED'].max()),np.mean([v.max() for k,v in maps.items() if k!='FUSED']))
        with self.assertRaises(ValueError):math.reconstruct(raw,np.array([0,0,1,2],np.int64),(2,2),(7,7),np.zeros(4,np.float32),np.ones(4,np.float32))

    def test_exact_metrics_random_ties_and_chunk_boundary_groups(self):
        rng=np.random.RandomState(42)
        for n in (4,19,1003):
            y=rng.randint(0,2,n).astype(np.uint8);y[:2]=[0,1];s=rng.randint(-3,5,n).astype(np.float32)
            for chunk in (1,3,29,10000):
                metric=metrics.exact_binary(y,s,memory_budget_bytes=1<<30,chunk_size=chunk)
                self.assertAlmostEqual(metric['AUROC'],roc_auc_score(y,s),places=13)
                self.assertAlmostEqual(metric['AP'],average_precision_score(y,s),places=13)

    def test_degenerate_metric_and_insufficient_budget_not_approximated(self):
        metric=metrics.exact_binary(np.zeros(5,np.uint8),np.ones(5,np.float32),memory_budget_bytes=1<<30)
        self.assertFalse(metric['valid']);self.assertIsNone(metric['AUROC']);self.assertIsNone(metric['AP'])
        all_positive=metrics.exact_binary(np.ones(5,np.uint8),np.arange(5,dtype=np.float32),memory_budget_bytes=1<<30)
        self.assertIsNone(all_positive['AUROC']);self.assertEqual(all_positive['AP'],1);self.assertTrue(all_positive['AP_valid'])
        with self.assertRaises(MemoryError):metrics.exact_binary(np.array([0,1]),np.array([0.,1.]),memory_budget_bytes=1)

    def test_fixed_position_decomposition_separates_both_peak_moves(self):
        u=np.array([[5.,2.,4.,1.]],np.float32);p=np.array([[3.,8.,2.,6.]],np.float32)
        um={l:u.copy() for l in math.LAYERS};pm={l:p.copy() for l in math.LAYERS};um['FUSED']=u;pm['FUSED']=p
        inside=np.array([[True,True,False,False]]);ring=~inside
        c,r=spatial.decompose(um,pm,((0,0),(0,2)),inside,ring)
        self.assertEqual(r['e_in'],5);self.assertEqual(r['e_ring'],4)
        self.assertEqual(r['delta_margin'],1);self.assertEqual(sum(c.values()),0);self.assertEqual(r['identity_error'],0)

    def test_default_missing_input_preflight_no_torch_no_output(self):
        self.assertNotIn('torch',sys.modules)
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            rc=runner.main(['--project-root',td,'--no-write'])
            self.assertEqual(rc,2);self.assertFalse((Path(td)/'DINOv3').exists())
        self.assertNotIn('torch',sys.modules)

    def test_raw_identity_and_physical_mapping_guard(self):
        sample=dict(sample_id='toy',relative_path='test/toy.png',image_height=8,image_width=8)
        raw=dict(metadata=dict(identity=inputs.PARENT_ID,sample_id='toy',relative_path=sample['relative_path'],image_sha256='image',distance_divisor=1280,
            grid=[2,2],output_shape=[2,2],geometry=dict(processed_height=8,processed_width=8,original_height=8,original_width=8)),
            grid=np.array([2,2],np.int32),processed_shape=np.array([8,8],np.int32),positions=np.array([3,1,0,2],np.int64),source_distance=np.ones((4,4),np.float32))
        inputs.validate_raw(raw,sample,inputs.PARENT_ID,'image')
        raw['positions'][0]=1
        with self.assertRaises(ValueError):inputs.validate_raw(raw,sample,inputs.PARENT_ID,'image')

    def test_pair_rejects_Source_modification_and_Target_union_confusion(self):
        sample=dict(sample_id='toy',relative_path='test/toy.png',image_height=8,image_width=8)
        meta=dict(identity=inputs.PARENT_ID,sample_id='toy',relative_path=sample['relative_path'],image_sha256='image',distance_divisor=1280,
            grid=[2,2],output_shape=[2,2],geometry=dict(processed_height=8,processed_width=8,original_height=8,original_width=8))
        raw=dict(metadata=meta,grid=np.array([2,2],np.int32),processed_shape=np.array([8,8],np.int32),positions=np.array([3,1,0,2],np.int64),source_distance=np.ones((4,4),np.float32)*2)
        p=copy.deepcopy(raw);p['metadata'].update(identity=inputs.PROJECTION_ID,parent_raw_sha256='original',prototype_ids={'7':'id'},OT_called=False)
        p.update(projection_distance=np.ones((4,4),np.float32),projection_union_distance=np.ones((4,4),np.float32))
        inputs.validate_pair(raw,p,sample,'image','original',{'7':'id'})
        p['source_distance'][0,0]+=1
        with self.assertRaises(ValueError):inputs.validate_pair(raw,p,sample,'image','original',{'7':'id'})
        p['source_distance']=raw['source_distance'].copy();p['projection_union_distance'][0,0]=3
        with self.assertRaises(ValueError):inputs.validate_pair(raw,p,sample,'image','original',{'7':'id'})

    def test_bounded_parallel_execution_keeps_manifest_order(self):
        import threading
        lock=threading.Lock();active=0;peak=0
        def fn(i):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            value=i*i
            with lock:active-=1
            return value
        self.assertEqual(list(analysis.bounded_images(range(50),fn,3)),[i*i for i in range(50)])
        self.assertLessEqual(peak,3)

    def test_pooled_patch_and_image_equal_are_not_conflated(self):
        arrays=np.r_[np.zeros(1),np.full(9,10.)]
        self.assertEqual(analysis.statistics(arrays)['mean'],9)
        self.assertEqual(np.mean([analysis.statistics(np.zeros(1))['mean'],analysis.statistics(np.full(9,10.))['mean']]),5)

    def test_incomplete_or_smoke_only_archive_cannot_supply_formal_scope(self):
        def blob(value):return json.dumps(value).encode()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'incomplete.tar.gz';frozen={};identity=inputs.digest(frozen)
            members={'inputs.json':blob(dict(identity=identity,frozen=frozen)),
                     'infer_status.json':blob(dict(identity=identity,status='predictions_frozen',images=12,records=[])),
                     'evaluation/status.json':blob(dict(status='analysis_complete')),
                     **{'evaluation/'+name+'.csv':b'branch,sample_id\n' for name in ('metrics','per_image_scores','defect_instances')}}
            members['infer_status.json']=blob(dict(json.loads(members['infer_status.json']),inputs_sha256=__import__('hashlib').sha256(members['inputs.json']).hexdigest()))
            with tarfile.open(path,'w:gz') as archive:
                for name,value in members.items():
                    info=tarfile.TarInfo(inputs.PARENT+'/'+name);info.size=len(value);archive.addfile(info,io.BytesIO(value))
            with patch.object(inputs,'PARENT_ID',identity),self.assertRaisesRegex(ValueError,'complete archive'):
                inputs.read_bundle(path,inputs.PARENT)

    def test_one_real_case_reconstruction_only_no_domain_evaluation(self):
        root=Path(__file__).resolve().parents[2];package=root/'downloads'/f'{inputs.PARENT}_results_20261014.tar.gz'
        if not package.exists():self.skipTest('approved local package absent')
        arrays=None;scores=None;sid='test_0023'
        with tarfile.open(package,'r|gz') as t:
            for item in t:
                if item.name==inputs.PARENT+f'/raw/{sid}.npz':
                    with np.load(io.BytesIO(t.extractfile(item).read()),allow_pickle=False) as f:
                        arrays={k:f[k].copy() for k in ('source_distance','target_full_distance','positions')};meta=json.loads(str(f['metadata'].item()))
                elif item.name==inputs.PARENT+'/evaluation/per_image_scores.csv':scores=list(csv.DictReader(io.StringIO(t.extractfile(item).read().decode())))
        mad=root/'server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1/source_mad_state.npz'
        with np.load(mad) as f:med,scale=f['medians'],f['scales']
        _,maps,error=math.reconstruct(np.minimum(arrays['source_distance'],arrays['target_full_distance']),arrays['positions'],tuple(meta['grid']),tuple(meta['output_shape']),med,scale)
        expected=float(next(r for r in scores if r['sample_id']==sid and r['branch']=='UNION_K8')['image_score'])
        self.assertLessEqual(abs(float(maps['FUSED'].max())-expected),1e-5);self.assertLessEqual(error,1e-5)
        self.assertNotIn('torch',sys.modules)


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=Path(cls.temp.name);cls.context=fixture(cls.base)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def run_context(self,name):
        context=dict(self.context,paths=dict(self.context['paths'],output=self.base/name))
        args=SimpleNamespace(memory_budget_gib=1,cpu_workers=2)
        return context,args

    def test_full_synthetic_scope_exact_metrics_and_contributions(self):
        c,a=self.run_context('complete')
        with contextlib.redirect_stdout(io.StringIO()):s=analysis.analyze(c,a)
        self.assertEqual(s['status'],'analysis_complete',s);self.assertEqual(s['exact_pixel_responses'],10);self.assertEqual(s['failed_checks'],0)
        out=c['paths']['output'];self.assertFalse((out/'scratch').exists())
        self.assertEqual(len(inputs.replay.read_csv(out/'layer_metrics.csv')),10)
        rows=inputs.replay.read_csv(out/'defect_instances.csv')
        self.assertEqual(len(rows),212*10);self.assertTrue(any(r['response_status']=='mapped_instance_disappeared' and r['margin']=='' for r in rows))
        contributions=inputs.replay.read_csv(out/'fixed_position_layer_contributions.csv')
        self.assertEqual(len(contributions),212*4)
        self.assertEqual({int(r['normal_count']) for r in inputs.replay.read_csv(out/'ranking_summary.csv')},{10,75,85})
        within=inputs.replay.read_csv(out/'within_image_pixel_metrics.csv');self.assertEqual(len(within),2120)
        self.assertTrue(any(r['valid']=='False' for r in within));self.assertNotIn('torch',sys.modules)

    def test_failed_historical_reconstruction_blocks_formal_metrics_and_conclusion(self):
        c,a=self.run_context('bad_baseline');c=dict(c,parent=copy.deepcopy(c['parent']))
        c['parent']['data']['evaluation/per_image_scores.csv'][0]['image_score']='100'
        with contextlib.redirect_stdout(io.StringIO()):s=analysis.analyze(c,a)
        self.assertEqual(s['status'],'partial_evidence');self.assertEqual(s['exact_pixel_responses'],0);self.assertGreater(s['failed_checks'],0)
        report=(c['paths']['output']/'diagnostic_report.md').read_text(encoding='utf8')
        self.assertIn('不输出确定性归因',report);self.assertTrue((c['paths']['output']/'scratch').exists())

    def test_output_overlap_and_nonempty_rejected(self):
        paths=dict(self.context['paths'],output=self.context['paths']['data'])
        with self.assertRaises(ValueError):inputs.guard(paths)
        out=self.base/'existing';out.mkdir();(out/'keep').write_text('old')
        with self.assertRaises(ValueError):inputs.guard(dict(self.context['paths'],output=out))

    def test_incomplete_state_never_complete(self):
        self.assertFalse(analysis.complete(dict(analyzed_images=296,exact_pixel_responses=10,reconstruction_passed=True,failed_checks=0,protected_inputs_unchanged=True),{}))

    def test_missing_GT_stays_partial_without_formal_pixel_metrics(self):
        c,a=self.run_context('missing_GT')
        with patch.object(inputs.DatasetZip,'mask',side_effect=FileNotFoundError('missing GT')),contextlib.redirect_stdout(io.StringIO()):s=analysis.analyze(c,a)
        self.assertEqual(s['status'],'partial_evidence');self.assertEqual(s['exact_pixel_responses'],0)
        self.assertIn('missing GT',s['error']);self.assertTrue((c['paths']['output']/'scratch').exists())


if __name__=='__main__':unittest.main()
