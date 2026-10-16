"""Math/cache/flow fixtures only. No real Torch, CUDA, model or feature extraction."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from DINOv3.MADEqual.aebad_s_target_local_segment import runner,inputs,analysis
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import (
    BRANCHES,DIMENSION,LAYERS,NumericalError,bounded_distances,geometry_layer,scatter,
)
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import save_raw,FORMAT
from DINOv3.MADEqual.aebad_s_union_support_loo.inputs import digest,TOOL as CAPTURE_TOOL
from DINOv3.tests.test_mad_equal_aebad_s_union_support_loo import actual_map_backend

ROOT=Path(__file__).resolve().parents[2]
FORMAL=ROOT/'server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1'
replay=inputs.replay


def vectors(query,endpoints,source=10):
    a,b=[np.linalg.norm(query-t,axis=1)/1280 for t in endpoints]
    c2=np.sum((endpoints[1]-endpoints[0])**2,axis=1)/(1280**2)
    swap=a>b
    return bounded_distances(np.minimum(a,b),np.maximum(a,b),c2,np.full(len(a),source)),swap


def full_protocol_fixture():
    """Actual original mask/evaluator AST, with AUPRO explicitly stubbed.

    This validates delegation/flow, never the real CUDA metric backend.
    """
    from sklearn.metrics import average_precision_score,roc_auc_score
    path=ROOT/'DINOv3/MADEqual/aebad_s_full_p0/runner.py'
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body
           if isinstance(n,ast.FunctionDef) and n.name in ('_load_mask','_evaluate')]
    env=dict(np=np,Image=Image,roc_auc_score=roc_auc_score,average_precision_score=average_precision_score,
             official_aupro_adeval=lambda masks,maps,nstrips: .5)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)
    return SimpleNamespace(_load_mask=env['_load_mask'],_evaluate=env['_evaluate'])


def frozen_fixture(root: Path):
    """Exactly 297 tiny synthetic captures with real membership/manifest only."""
    results=root/'formal';results.mkdir()
    for name in ('sample_scores.csv','support_manifest.csv','complete.json','memory_manifest.json','source_mad_state.npz'):
        shutil.copy2(FORMAL/name,results/name)
    score_rows=replay.read_csv(results/'sample_scores.csv')
    for row in score_rows:
        if row['domain']=='background' and row['method'] in replay.METHODS:
            row['image_score']=6. if row['method']==replay.METHODS[0] else .75
    replay.write_csv(results/'sample_scores.csv',score_rows)
    np.savez(results/'source_mad_state.npz',protocol_id=np.asarray(replay.PROTOCOL_ID),
             medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32))
    samples=[dict(sample_id=r['sample_id'],relative_path=r['relative_path'],label=int(r['label']),
                  image_height=8,image_width=8,source_score=6.,union_score=.75)
             for r in sorted(score_rows,key=lambda r:r['relative_path']) if r['domain']=='background' and r['method']==replay.METHODS[0]]
    assert len(samples)==297
    support_root=root/'features';support_root.mkdir()
    manifest=sorted((r for r in replay.read_csv(results/'support_manifest.csv')
                     if r['branch']=='target_matched' and r['domain']=='background'),key=lambda r:int(r['support_index']))
    supports=[];banks=[]
    for row in manifest:
        j=int(row['support_index'])
        bank=np.zeros((4,1280),np.float32);bank[:,0]=(j+np.arange(4)*.2)*1280
        path=support_root/f'{j:02d}.npz'
        np.savez(path,protocol_id=np.asarray(replay.PROTOCOL_ID),relative_path=np.asarray(row['relative_path']),
                 grid=np.array([2,2],np.int32),**{f'layer_{l}':bank for l in LAYERS})
        supports.append(dict(support_index=j,relative_path=row['relative_path'],grid=[2,2],geometry={'grid':[2,2]},
                             feature_sha256=replay.sha256(path),image_sha256='a'*64))
        banks.append(bank)
    data=root/'data';(data/'test').mkdir(parents=True);(data/'ground_truth').mkdir()
    sample_by_id={s['sample_id']:s for s in samples}
    cases=[dict(sample_by_id[sid]) for sid,_ in replay.CASES]
    for case in cases:
        path=data/case['relative_path'];path.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(8,8),'gray').save(path)
    image_hashes={s['relative_path']:('a'*64) for s in samples}
    image_hashes.update({s['relative_path']:replay.sha256(data/s['relative_path']) for s in cases})
    gt_hashes={}
    for s in samples:
        if s['label']:
            rel=inputs.gt_relative(s['relative_path']);path=data/rel;path.parent.mkdir(parents=True,exist_ok=True)
            mask=np.zeros((8,8),np.uint8);mask[:4,:4]=255;Image.fromarray(mask).save(path)
            gt_hashes[rel]=replay.sha256(path)
    frozen=dict(protocol_id=replay.PROTOCOL_ID,formal_sha256={name:replay.sha256(results/name) for name in
        ('sample_scores.csv','support_manifest.csv','complete.json','memory_manifest.json','source_mad_state.npz')},
        input_code_sha256={k:replay.sha256(ROOT/rel) for k,rel in (
            ('code_runner','DINOv3/MADEqual/aebad_s_target_fewshot/runner.py'),
            ('code_host','DINOv3/relation_reliability/superadd_host.py'),
            ('code_mad','DINOv3/relation_oracle_cache/extract.py'))},
        supports=[{k:s[k] for k in ('support_index','relative_path','feature_sha256','image_sha256')} for s in supports])
    identity=digest(frozen);captured=root/'capture';records=[]
    positions=np.array([2,0,3,1],np.int64)
    query=np.zeros((4,1280),np.float64);query[:,0]=np.array([.25,2.25,3.5,5.5])*1280
    target=[];indices=[]
    for bank in banks:
        distances=np.linalg.norm(query[:,None]-bank[None],axis=2)/1280
        idx=np.argmin(distances,axis=1)
        target.append(distances[np.arange(4),idx].astype(np.float32));indices.append(idx.astype(np.int64))
    arrays=dict(positions=positions,source_distance=np.full((4,4),6,np.float32),source_index=np.zeros((4,4),np.int64),
                target_distance=np.repeat(np.stack(target)[:,None],4,axis=1),
                target_index=np.repeat(np.stack(indices)[:,None],4,axis=1))
    for s in samples:
        meta=dict(format=FORMAT,identity=identity,sample_id=s['sample_id'],relative_path=s['relative_path'],
                  grid=[2,2],output_shape=[2,2],source_rows=[1]*4,supports=supports,image_sha256=image_hashes[s['relative_path']])
        path=captured/f"raw_cache/{s['sample_id']}.npz";save_raw(path,arrays,meta)
        records.append(dict(sample_id=s['sample_id'],file=f"raw_cache/{s['sample_id']}.npz",sha256=replay.sha256(path)))
    replay.atomic_json(captured/'inputs.json',dict(tool=CAPTURE_TOOL,scope='background',frozen=frozen,identity=identity,
        samples=samples,selected_cases=cases,supports=supports,image_sha256=image_hashes,gt_sha256=gt_hashes))
    replay.write_csv(captured/'capture_verification.csv',[dict(check_kind='synthetic_fixture',verified=True)])
    replay.atomic_json(captured/'status.json',dict(tool=CAPTURE_TOOL,scope='background',status='capture_verified',identity=identity,
        records=records,inputs_sha256=replay.sha256(captured/'inputs.json'),verification_sha256=replay.sha256(captured/'capture_verification.csv')))
    args=runner.build_parser().parse_args(['analyze','--project-root',str(ROOT),'--results-dir',str(results),
        '--capture-dir',str(captured),'--support-cache-dir',str(support_root),'--data',str(data),'--output-dir',str(root/'output')])
    return args,arrays,banks


class GeometryTests(unittest.TestCase):
    def test_known_vectors_match_direct_midpoint_and_segment_projection(self):
        rng=np.random.default_rng(42)
        q=rng.normal(size=(7,1280))*1280;t1=rng.normal(size=q.shape)*1280;t2=rng.normal(size=q.shape)*1280
        (values,diag),swap=vectors(q,(t1,t2),source=100)
        midpoint=np.linalg.norm(q-(t1+t2)/2,axis=1)/1280
        direction=t2-t1;alpha=np.clip(np.sum((q-t1)*direction,axis=1)/np.sum(direction**2,axis=1),0,1)
        segment=np.linalg.norm(q-(t1+alpha[:,None]*direction),axis=1)/1280
        union=np.minimum(np.linalg.norm(q-t1,axis=1),np.linalg.norm(q-t2,axis=1))/1280
        np.testing.assert_allclose(values['K8_MIDPOINT'],np.minimum(union,midpoint),rtol=1e-7)
        np.testing.assert_allclose(values['K8_SEGMENT'],np.minimum(union,segment),rtol=1e-7)
        self.assertEqual(diag['endpoint_identity_max_error'],0)

    def test_interior_and_outside_projection(self):
        values,diag=bounded_distances(np.array([.25,1]),np.array([.75,2]),np.array([1,1]),np.array([3,3]))
        np.testing.assert_allclose(values['alpha'],[.25,0]);np.testing.assert_allclose(values['K8_SEGMENT'],[0,1])

    def test_duplicate_endpoints_fall_back_to_union(self):
        v,d=bounded_distances(np.array([2]),np.array([2]),np.array([0]),np.array([1]))
        self.assertEqual(d['degenerate_count'],1)
        self.assertEqual(v['K8_SEGMENT'][0],1);self.assertEqual(v['K8_MIDPOINT'][0],1)

    def test_nearly_equal_endpoints_do_not_add_a_method_gate(self):
        v,d=bounded_distances(np.array([1]),np.array([1]),np.array([1e-20]),np.array([3]))
        self.assertEqual(d['degenerate_count'],0);self.assertTrue(np.isfinite(v['alpha']).all())
        self.assertEqual(v['alpha'][0],.5)

    def test_midpoint_keeps_source_and_all_original_endpoints(self):
        v,_=bounded_distances(np.array([1]),np.array([2]),np.array([1]),np.array([.1]))
        self.assertEqual(v['K8_MIDPOINT'][0],np.float32(.1))
        self.assertEqual(v['K8_SEGMENT'][0],np.float32(.1))

    def test_rounding_negative_square_is_bounded_and_recorded(self):
        v,d=bounded_distances(np.array([1]),np.array([1]),np.array([4.000004]),np.array([2]))
        self.assertEqual(v['K8_SEGMENT'][0],0);self.assertEqual(d['correction_count'],2)
        self.assertGreater(d['maximum_squared_correction'],0)

    def test_severe_incompatible_sides_fail(self):
        with self.assertRaises(NumericalError):bounded_distances(np.array([1]),np.array([1]),np.array([9]),np.array([2]))
        with self.assertRaises(NumericalError):bounded_distances(np.array([1]),np.array([2]),np.array([0]),np.array([2]))

    def test_nonfinite_negative_unordered_sides_rejected(self):
        for a,b,c2 in ((-1,2,1),(np.nan,2,1),(2,1,1),(1,2,-1)):
            with self.subTest(a=a):
                with self.assertRaises(ValueError):bounded_distances(np.array([a]),np.array([b]),np.array([c2]),np.array([3]))

    def test_stable_different_support_candidates_and_no_double_scaling(self):
        banks=[np.zeros((2,1280),np.float32) for _ in range(3)]
        banks[0][:,0]=[0,640];banks[1][:,0]=[1280,1920];banks[2][:,0]=[2560,3200]
        # Within support #1 two rows are nearby; only its captured row participates.
        target=np.array([[.25,.25],[.75,.25],[1.75,1.25]],np.float32)
        indices=np.array([[0,1],[0,0],[0,0]],np.int64)
        v,d=geometry_layer(target,indices,np.full(2,3,np.float32),banks,chunk=1)
        np.testing.assert_array_equal(v['endpoint_support'],[[1,1],[2,2]])
        np.testing.assert_array_equal(v['endpoint_index'],[[0,1],[0,0]])
        np.testing.assert_allclose(v['K8_UNION'],[.25,.25]);np.testing.assert_allclose(v['K8_SEGMENT'],[0,0])
        self.assertEqual(d['near_tie_count'],1)

    def test_chunk_bound_does_not_change_geometry(self):
        banks=[np.zeros((2,1280),np.float32),np.zeros((2,1280),np.float32)]
        banks[1][:,0]=1280
        t=np.array([[.1,.2,.3],[.9,.8,.7]],np.float32);idx=np.zeros((2,3),np.int64)
        one,_=geometry_layer(t,idx,np.ones(3,np.float32),banks,chunk=1)
        all_,_=geometry_layer(t,idx,np.ones(3,np.float32),banks,chunk=100)
        for key in one:np.testing.assert_array_equal(one[key],all_[key])

    def test_single_support_fallback_is_only_unit_case(self):
        v,d=geometry_layer(np.array([[.5]],np.float32),np.array([[0]],np.int64),np.array([1]),[np.zeros((1,1280),np.float32)])
        self.assertEqual(d['single_support_fallback_count'],1)
        self.assertEqual(v['K8_MIDPOINT'][0],.5);self.assertEqual(v['K8_SEGMENT'][0],.5)

    def test_feature_dimension_and_index_rejected(self):
        with self.assertRaises(ValueError):geometry_layer(np.ones((1,1)),np.ones((1,1),int),np.ones(1),[np.zeros((1,1280))])
        with self.assertRaises(ValueError):geometry_layer(np.ones((1,1)),np.zeros((1,1),int),np.ones(1),[np.zeros((1,1279))])

    def test_positions_scatter_is_nontrivial_and_rejects_duplicates(self):
        positions=np.array([2,0,3,1],np.int64);raw=np.tile(np.arange(4,dtype=np.float32),(4,1))
        np.testing.assert_array_equal(scatter(raw,positions,(2,2))[0],[[1,3],[0,2]])
        with self.assertRaises(ValueError):scatter(raw,np.array([0,0,2,3],np.int64),(2,2))

    def test_original_map_delegation_has_exactly_three_branches(self):
        backend,trace=actual_map_backend();a=dict(positions=np.array([2,0,3,1],np.int64))
        raws={b:np.full((4,4),i+1,np.float32) for i,b in enumerate(BRANCHES)}
        maps=analysis.maps_from_compact(raws,a,dict(grid=[2,2],output_shape=[2,2]),backend)
        self.assertEqual(tuple(maps),BRANCHES);self.assertEqual(trace,['layer_MAD_then_mean','interpolate_after_mean']*3)


class InputAndFlowTests(unittest.TestCase):
    def test_help_and_default_readonly_preflight_do_not_import_torch(self):
        self.assertNotIn('torch',sys.modules)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as ctx:runner.main(['--help'])
        self.assertEqual(ctx.exception.code,0)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'unused'
            with contextlib.redirect_stdout(io.StringIO()):
                rc=runner.main(['--no-write','--capture-dir',str(Path(td)/'missing'),'--results-dir',str(FORMAL),
                                '--data',str(Path(td)/'missing_data'),'--output-dir',str(out)])
            self.assertEqual(rc,2);self.assertFalse(out.exists())
        self.assertNotIn('torch',sys.modules)

    def test_preflight_requires_only_offline_inputs_and_target_only_data(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=frozen_fixture(Path(td));check=inputs.preflight(args)
            self.assertEqual(check['errors'],[]);self.assertEqual(len(check['samples']),297)
            self.assertFalse((Path(args.data)/'train/good').exists())
            self.assertFalse((Path(args.results_dir)/'official_memory.pt').exists())
            self.assertFalse(any('checkpoint' in r['kind'] or r['kind']=='memory' for r in check['required_files']))
            self.assertNotIn('torch',sys.modules)

    def test_original_hashes_and_support_order_remain_protected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=frozen_fixture(Path(td))
            path=Path(args.capture_dir)/'raw_cache/test_0000.npz';path.write_bytes(path.read_bytes()+b'changed')
            self.assertTrue(inputs.preflight(args)['errors'])
        with tempfile.TemporaryDirectory() as td:
            args,_,_=frozen_fixture(Path(td));path=Path(args.support_cache_dir)/'01.npz'
            path.write_bytes(path.read_bytes()+b'changed')
            self.assertTrue(inputs.preflight(args)['errors'])
        supports=[dict(support_index=j,relative_path='test/good/background/'+name) for j,name in enumerate(replay.TARGET_NAMES,1)]
        supports[0],supports[1]=supports[1],supports[0]
        with self.assertRaises(ValueError):inputs.support_manifest_matches(supports,FORMAL/'support_manifest.csv')

    def test_output_cannot_overwrite_capture_or_partial_results(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=frozen_fixture(Path(td));args.output_dir=args.capture_dir
            with self.assertRaises(ValueError):inputs.preflight(args)
            args.output_dir=str(Path(td)/'output');Path(args.output_dir).mkdir();(Path(args.output_dir)/'half.npz.tmp').write_bytes(b'partial')
            with self.assertRaises(ValueError):inputs.preflight(args)
            (Path(args.output_dir)/'half.npz.tmp').unlink()
            replay.atomic_json(Path(args.output_dir)/'preflight.json',dict(tool=CAPTURE_TOOL))
            with self.assertRaises(ValueError):inputs.preflight(args)

    def execute_fixture(self,root: Path,offset=0.,pixel_missing=False):
        args,_,_=frozen_fixture(root);check=inputs.preflight(args)
        self.assertEqual(check['errors'],[])
        backend,_=actual_map_backend();backend.mad.medians=np.zeros(4,np.float32);backend.mad.scales=np.ones(4,np.float32)
        if offset:
            original=backend.runner._method_map
            backend.runner._method_map=lambda *a,**kw: original(*a,**kw)+offset
        output=Path(args.output_dir);output.mkdir()
        evaluator=full_protocol_fixture();original_mask=evaluator._load_mask
        def mask(record,shape):
            state=json.loads((output/'status.json').read_text())
            self.assertTrue(state['predictions_frozen']);self.assertEqual(state['geometry_layer_images'],1188)
            return original_mask(record,shape)
        evaluator._load_mask=mask
        with patch.object(runner,'runtime_map_backend',return_value=backend),patch.object(runner,'runtime_full_protocol',return_value=evaluator),\
             patch.object(runner,'runtime_pixel_backend',side_effect=RuntimeError('synthetic unavailable CUDA') if pixel_missing else None),\
             patch('DINOv3.MADEqual.aebad_s_target_local_segment.visualization.plot_case'),contextlib.redirect_stdout(io.StringIO()):
            rc=runner.analyze(args,check)
        return rc,json.loads((output/'status.json').read_text()),output

    def test_full_scope_fixture_freezes_predictions_before_gt_and_delegates_evaluator(self):
        with tempfile.TemporaryDirectory() as td:
            rc,state,out=self.execute_fixture(Path(td))
            self.assertEqual(rc,0);self.assertEqual(state['status'],'analysis_complete')
            rows=replay.read_csv(out/'per_image_scores.csv')
            self.assertEqual(len(rows),891);self.assertEqual({r['branch'] for r in rows},set(BRANCHES))
            self.assertTrue(state['formal_baseline_reproduced']);self.assertTrue(state['pixel_metrics_complete'])
            self.assertEqual(len(json.loads((out/'predictions.json').read_text())['records']),1188)
            self.assertNotIn('torch',sys.modules)

    def test_consistent_map_offset_cannot_be_marked_complete(self):
        with tempfile.TemporaryDirectory() as td:
            rc,state,out=self.execute_fixture(Path(td),offset=1.)
            self.assertEqual(rc,3);self.assertEqual(state['status'],'partial_evidence')
            self.assertFalse(state['formal_baseline_reproduced']);self.assertEqual(state['numerical_failed_checks'],594)

    def test_missing_cuda_pixel_backend_is_partial_without_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            rc,state,out=self.execute_fixture(Path(td),pixel_missing=True)
            self.assertEqual(rc,3);self.assertEqual(state['status'],'partial_evidence')
            self.assertTrue(state['formal_baseline_reproduced']);self.assertFalse(state['pixel_metrics_complete'])
            self.assertTrue(all(r['AUPRO']=='' for r in replay.read_csv(out/'metrics.csv')))

    def test_mechanism_uses_only_gt_patches_and_k8_reference(self):
        raws={BRANCHES[0]:np.ones((4,4),np.float32)*3,BRANCHES[1]:np.array([[2,2,3,3]]*4,np.float32),
              BRANCHES[2]:np.array([[1,1,2,2]]*4,np.float32)}
        positions=np.array([2,0,3,1],np.int64);mask=np.array([[1,0],[0,0]],bool)
        rows=analysis.mechanism(dict(sample_id='q',label=1),raws,np.ones(4),positions,mask)
        self.assertTrue(all(r['selected_patch_count']==1 for r in rows))
        np.testing.assert_allclose([r['R_value'] for r in rows if r['branch']==BRANCHES[2]],[2]*4)
        self.assertTrue(all(r['reference']=='K8_UNION' for r in rows))


if __name__=='__main__':unittest.main()
