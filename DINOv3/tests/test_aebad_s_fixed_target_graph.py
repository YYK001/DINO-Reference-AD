"""Small actual CPU kernels; no real backbone, GPU, image extraction or experiment."""
import contextlib
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
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import graph,numerics,selection,protocol,inputs as ip,runner
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.neighbors import tables

ROOT=Path(__file__).resolve().parents[2]


class Geometry(unittest.TestCase):
    def test_all_segments_matches_independent_projection_and_swapped_endpoints(self):
        rng=np.random.default_rng(3);q=rng.normal(size=(11,7)).astype(np.float32)
        x=rng.normal(size=(13,7)).astype(np.float32);e=np.array([[0,1],[2,5],[4,8],[7,12]])
        masks=np.array([[1,1,1,1],[0,1,0,1]],bool);u=np.full(len(q),2,np.float32)
        actual,winner,_=numerics.scan_segments(q,x,e,masks,u,query_chunk=3,edge_chunk=2,reference_chunk=4)
        distance,alpha=numerics.direct_projection(q[:,None,:],x[e[:,0]][None],x[e[:,1]][None])
        for i,m in enumerate(masks):np.testing.assert_allclose(actual[i],distance[:,m].min(1)/1280,rtol=1e-7)
        swapped,_,_=numerics.scan_segments(q,x,e[:,::-1],masks,u,query_chunk=4,edge_chunk=1,reference_chunk=3)
        np.testing.assert_array_equal(actual,swapped)

    def test_winning_segment_endpoints_are_not_query_Top2(self):
        q=np.zeros((1,2),np.float32);x=np.array([[0,.5],[0,-.6],[-2,0],[2,0]],np.float32)
        self.assertEqual(np.argsort(np.linalg.norm(x,axis=1))[:2].tolist(),[0,1])
        distance,winner,_=numerics.scan_segments(q,x,[[2,3]],[[True]],np.array([.5/1280],np.float32))
        self.assertEqual(distance[0,0],0);self.assertEqual(winner[0,0],0)

    def test_interior_existing_discrete_point_has_no_fabricated_new_coverage(self):
        q=np.array([[0,0]],np.float32);x=np.array([[-1,0],[1,0],[0,0]],np.float32)
        distance,winner,_=numerics.scan_segments(q,x,[[0,1]],[[True]],np.zeros(1,np.float32))
        self.assertEqual(distance[0,0],0);self.assertEqual(winner[0,0],-1)

    def test_large_offset_cancellation_uses_direct_reference(self):
        x=np.array([[1e5,1e5],[1e5+4,1e5]],np.float32)
        q=np.array([[1e5+2,1e5+.015625]],np.float32)
        actual,_,_=numerics.scan_segments(q,x,[[0,1]],[[True]],np.array([1],np.float32))
        expected=numerics.direct_projection(q,x[0],x[1])[0]/1280
        np.testing.assert_allclose(actual[0],expected,atol=1e-12)

    def test_subset_and_Source_anchor(self):
        q=np.array([[0,0],[0,2]],np.float32);x=np.array([[-1,0],[1,0],[-1,1],[1,1]],np.float32)
        u=np.array([0,.0001],np.float32)
        d,_,_=numerics.scan_segments(q,x,[[0,1],[2,3]],[[1,1],[1,0],[0,0]],u)
        self.assertTrue(np.all(d[0]<=d[1]));self.assertTrue(np.all(d[1]<=u));np.testing.assert_array_equal(d[2],u)


class Graph(unittest.TestCase):
    def pairs(self,banks,k=100):return {key:(i,d) for key,(i,d) in tables(banks,k=min(k,min(map(len,banks.values()))),query_chunk=2,reference_chunk=2).items()}

    def test_global_cross_image_k_not_per_image_k(self):
        b={1:np.array([[0.,0.]]),2:np.array([[1.,0.]]),3:np.array([[10.,0.]])}
        p=graph.mutual_pool(b,self.pairs(b),max_k=1)
        self.assertEqual(p['edges'].tolist(),[[0,1]])
        self.assertEqual(p['rank'].tolist(),[1])

    def test_bidirectional_distance_reuse_matches_both_exhaustive_tables_including_ties(self):
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.neighbors import bidirectional
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.neighbors import topk
        rng=np.random.default_rng(6);q=rng.normal(size=(8,9)).astype(np.float32);r=rng.normal(size=(11,9)).astype(np.float32)
        r[3]=r[0];q[5]=q[1]
        one,two=bidirectional(q,r,k=5,query_chunk=3,reference_chunk=2)
        expected=[topk(q,r,5,query_chunk=4,reference_chunk=5),topk(r,q,5,query_chunk=2,reference_chunk=3)]
        for got,wanted in zip((one,two),expected):
            np.testing.assert_array_equal(got[0],wanted[0]);np.testing.assert_allclose(got[1],wanted[1],atol=1e-14)

    def test_fold_recomputes_neighbors_scales_and_quantiles(self):
        b={1:np.array([[0.,0.]]),2:np.array([[1.,0.]]),3:np.array([[3.,0.]])}
        pairs=self.pairs(b);full=graph.mutual_pool(b,pairs,max_k=1)
        folded=graph.mutual_pool({1:b[1],3:b[3]},pairs,max_k=1)
        self.assertEqual(full['edges'].tolist(),[[0,1]])
        self.assertEqual(folded['node_support'].tolist(),[1,3])
        self.assertEqual(folded['edges'].tolist(),[[0,1]])
        self.assertEqual(folded['length'].tolist(),[3.])
        np.testing.assert_array_equal(folded['scales'],[3,3])
        self.assertNotEqual(graph.length_limits(full)[10,.5],graph.length_limits(folded)[10,.5])

    def test_ties_follow_original_support_then_patch_row_and_zero_removed(self):
        b={1:np.array([[0.,0.],[0.,0.]]),2:np.array([[0.,0.],[1.,0.]]),3:np.array([[1.,0.],[2.,0.]])}
        p=graph.mutual_pool(b,self.pairs(b),max_k=2)
        self.assertGreater(p['zero_length_removed'],0);self.assertTrue((p['length']>0).all())
        self.assertTrue((p['node_support'][p['edges'][:,0]]!=p['node_support'][p['edges'][:,1]]).all())

    def test_complete_witness_finds_point_outside_endpoint_Top5(self):
        b={1:np.array([[-2.,0.]],np.float32),2:np.array([[2.,0.]],np.float32),
           3:np.array([[-2.,.01*i] for i in range(1,6)]+[[2.,.01*i] for i in range(1,6)]+[[0.,0.]],np.float32)}
        nodes,off,ids,rows=graph.layout(b)
        for endpoint in (b[1][0],b[2][0]):
            self.assertNotIn(10,np.argsort(np.linalg.norm(b[3]-endpoint,axis=1))[:5])
        p=dict(edges=np.array([[0,1]]),scales=np.ones(len(nodes)))
        w=graph.witness_pool(b,p,edge_chunk=1,reference_chunk=3)
        self.assertEqual(w['witness_row'][0],10);self.assertEqual(w['witness_support'][0],3)
        self.assertEqual(w['eta'][0],0);self.assertTrue((w['source_counts']==1).all())

    def test_heldout_and_endpoint_images_cannot_witness_or_supply_scales(self):
        b={1:np.array([[-2.,0.]]),2:np.array([[2.,0.]])}
        p=dict(edges=np.array([[0,1]]),scales=np.ones(2))
        w=graph.witness_pool(b,p)
        self.assertTrue(np.isinf(w['eta'][0]));self.assertTrue((w['source_counts']==0).all())

    def test_zero_scale_does_not_receive_epsilon(self):
        b={1:np.array([[-2.,0.]]),2:np.array([[2.,0.]]),3:np.array([[0.,0.]])}
        p=dict(edges=np.array([[0,1]]),scales=np.zeros(3))
        w=graph.witness_pool(b,p)
        self.assertTrue(np.isinf(w['eta'][0]));self.assertEqual(w['invalid_scale_points'][0],1)

    def test_graph_and_witness_invariant_to_blocking(self):
        rng=np.random.default_rng(18);b={c:rng.normal(size=(6,5)).astype(np.float32) for c in range(1,5)}
        p=graph.mutual_pool(b,self.pairs(b))
        one=graph.witness_pool(b,p,edge_chunk=1,reference_chunk=2)
        two=graph.witness_pool(b,p,edge_chunk=9,reference_chunk=5)
        for key in ('eta','witness_row','witness_support','source_counts','alpha_histogram'):
            np.testing.assert_allclose(one[key],two[key],atol=1e-12)


class Selection(unittest.TestCase):
    def fixture(self):
        scores={protocol.config_id(c):np.full(8,10.) for c in protocol.CONFIGS}
        costs={k:np.ones((8,2)) for k in scores}
        return scores,costs

    def test_no_normal_gain_selects_empty(self):
        s,c=self.fixture();r=selection.choose(s,np.full(8,10.),c,witness=True)
        self.assertTrue(r['empty_selected'])

    def test_top_two_objective_and_95pct_complexity_not_mean_or_test_scores(self):
        s,c=self.fixture();a,b=map(protocol.config_id,protocol.CONFIGS[:2])
        s[a]=np.full(8,8.);s[b]=np.full(8,8.09)
        c[a]=np.full((8,2),3.);c[b]=np.full((8,2),1.)
        r=selection.choose(s,np.full(8,10.),c,witness=True)
        self.assertEqual(r['selected']['config_id'],b)
        s[b]=np.array([0,0,0,0,0,0,10,10.])
        r=selection.choose(s,np.full(8,10.),c,witness=True)
        self.assertEqual(r['selected']['config_id'],a)

    def test_lambda_image_equal_not_patch_count_equal_and_beta_totals(self):
        u=[np.ones(1,np.float32),np.ones(9,np.float32)]
        g=[np.zeros(1,np.float32),np.zeros(9,np.float32)]
        w=[np.zeros(1,np.float32),np.ones(9,np.float32)]
        lam,audit=selection.matched_coefficient(u,w,g)
        self.assertEqual(lam,.5)
        normal,image,audit=selection.controls(u[1],g[1],w[1],lam)
        self.assertEqual(audit['beta_image'],0);np.testing.assert_array_equal(normal,np.full(9,.5))
        np.testing.assert_array_equal(image,u[1])

    def test_zero_denominator_and_invalid_subset(self):
        u=np.ones(3,np.float32);lam,_=selection.matched_coefficient([u],[u],[u]);self.assertEqual(lam,0)
        with self.assertRaises(ValueError):selection.controls(u,u,u*.5,0)


class Contracts(unittest.TestCase):
    def test_post_stage_storage_is_metadata_only_and_missing_roots_are_zero(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);run=root/'run';run.mkdir();(run/'small.bin').write_bytes(b'12345')
            names=('run_root','cache_root','graph_root','query_cache_root','prediction_root','evaluation_scratch')
            a=SimpleNamespace(**{name:run if name=='run_root' else root/name for name in names})
            with patch.object(ip.st,'sha',side_effect=AssertionError('content read for storage observation')):
                value=runner.storage_observation(a)
            self.assertEqual(value['roots'][0]['logical_retained_bytes'],5)
            self.assertTrue(all(r['logical_retained_bytes']==0 for r in value['roots'][1:]))
            self.assertFalse(value['capacity_admission_checks'])

    def test_graph_transfer_rates_use_matching_unfiltered_graph_and_keep_empty_NA(self):
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.reporting import graph_transfer
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            run=Path(temp);plain=(10,.5,None);witness=(10,.5,.5)
            ip.st.write_json(run/'selection/background.json',dict(main=dict(selected=dict(config=witness)),
                paired_config=plain,plain=dict(selected=dict(config=plain))))
            rows=[]
            for h in range(9):
                for cfg,count in ((plain,10),(witness,h if h else 7)):
                    rows.append(dict(domain='background',layer=7,fold=h,config=protocol.config_id(cfg),
                        edges=count,raw_total_length=float(count)))
            ip.st.write_csv(run/'graph_summary.csv',rows)
            result=graph_transfer(SimpleNamespace(run_root=run))
            row=next(r for r in result if r['branch']==protocol.W and r['layer']==7)
            self.assertAlmostEqual(row['seven_image_mean_witness_pass_rate'],.45)
            self.assertAlmostEqual(row['eight_image_witness_pass_rate'],.7)
            empty=next(r for r in result if r['branch']==protocol.W and r['layer']==15)
            self.assertIsNone(empty['eight_image_witness_pass_rate'])

    def test_partial_support_estimate_does_not_require_other_domains(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);run=root/'run';cache=root/'cache'
            ip.st.write_json(run/'fresh_catalog.json',dict(supports={'background':[dict(support_index=1),dict(support_index=2)]},
                samples=[dict(domain='background')]))
            for i in (1,2):
                p=cache/'supports/background'/f'{i:03d}.npz';p.parent.mkdir(parents=True,exist_ok=True)
                np.savez(p,grid=np.array([3,3]))
            result=ip.estimate(SimpleNamespace(run_root=run,cache_root=cache,domains=['background'],query_mode='stream',
                query_chunk=2,witness_edge_chunk=3,reference_chunk=4))
            self.assertEqual(result['execution_domains'],['background'])
            self.assertEqual(len(result['domains']),1)

    def test_reuse_accepts_own_source_manifest_and_rejects_changed_host(self):
        from importlib.metadata import version
        required=('DINOv3/relation_reliability/superadd_host.py',
            'external_baselines/superadd_external/memory_variants.py',
            'external_baselines/superadd_external/run.py',
            'DINOv3/MADEqual/aebad_s_support_witness_segment_p0/fresh_runtime.py',
            'DINOv3/ShiftADD/aebad_s_p0/memory.py')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);old=root/'old';run=root/'new'
            official=root/'SuperADD';dino=root/'dinov3'
            official.mkdir();dino.mkdir()
            ip.st.write_json(official/'config.json',ip.st.read_json(ROOT/'external_baselines/SuperADD/config.json'))
            (dino/'hubconf.py').write_text('# frozen source\n')
            catalog=dict(revision=protocol.REVISION,source=[dict(relative_path='train/001.png',image_sha256='x')])
            configuration=dict(code={p:ip.st.sha(ROOT/p) for p in required},
                host_config=ip.st.read_json(official/'config.json'),
                external_source_code={'dinov3/hubconf.py':ip.st.sha(dino/'hubconf.py')},
                software={p:version(p) for p in ('numpy','scipy','torch','torchvision')})
            ip.st.write_json(old/'fresh_catalog.json',catalog);ip.st.write_json(run/'fresh_catalog.json',catalog)
            ip.st.write_json(old/'setup.json',dict(configuration=configuration,cache=str(root/'old_cache'),identity='frozen'))
            a=SimpleNamespace(reuse_fresh_run=old,run_root=run,official_root=official,dinov3_root=dino)
            self.assertEqual(ip.reuse_context(a)[2],'frozen')
            (dino/'hubconf.py').write_text('# modified source\n')
            with self.assertRaisesRegex(ValueError,'external host/source code differs'):ip.reuse_context(a)

    def test_job_selectors_are_execution_only_and_round_trip_as_separate_arguments(self):
        p=runner.parser()
        a=runner.normalize(p.parse_args(['prepare-graphs','--run-root',str(ROOT/'fake_run'),
            '--cache-root',str(ROOT/'fake_cache'),'--data','missing.zip','--checkpoint','missing.pth',
            '--domains','view','--layers','7','31','--heldouts','1','8']))
        arguments=runner.common_args(a)
        copied=p.parse_args(['prepare-graphs',*arguments])
        self.assertEqual(copied.domains,['view']);self.assertEqual(copied.layers,[7,31]);self.assertEqual(copied.heldouts,[1,8])

    def test_default_preflight_no_torch_no_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            out=Path(temp)/'not_created';cache=Path(temp)/'cache'
            script="""import sys,json
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.runner import main
main(['--run-root',sys.argv[1],'--cache-root',sys.argv[2],'--data','missing.zip','--checkpoint','missing.pth'])
assert 'torch' not in sys.modules
"""
            p=subprocess.run([sys.executable,'-B','-c',script,str(out),str(cache)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertFalse(out.exists());self.assertFalse(cache.exists())

    def test_no_GT_or_query_inputs_to_graph_or_selection(self):
        import inspect
        for fn in (graph.mutual_pool,graph.witness_pool,selection.choose):
            self.assertFalse({'GT','labels','query','test_scores'}&set(inspect.signature(fn).parameters))

    def test_source_stage_dependency_binding_does_not_modify_old_globals(self):
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import runtime
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_runtime as old
        original=old.host
        def fn(a):return host(a)
        # Exercise actual binding using a small function, not a real model.
        with patch.object(old,'source',fn),patch.object(runtime,'host',lambda a:'bound'):
            self.assertEqual(runtime.original_stage(None,'source'),'bound')
        self.assertIs(old.host,original)

    def test_incomplete_prediction_and_AUPRO_cannot_complete(self):
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import evaluation
        with self.assertRaises(ValueError):evaluation.validate_aupro(dict(AUPRO=1.5),dict(identity='x'))


class TorchCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import torch
        except ImportError:raise unittest.SkipTest('Torch absent; production Torch kernels not validated')
        cls.torch=torch;cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):cls.torch.set_num_threads(cls.old_threads)

    def test_prepared_lengths_are_blocked_without_all_edge_directions(self):
        rng=np.random.default_rng(51);nodes=rng.normal(size=(10,1280)).astype(np.float32)
        edges=np.asarray([(i,j) for i in range(10) for j in range(i+1,10)])
        masks=np.ones((1,len(edges)),bool)
        prepared=numerics.PreparedSearch(nodes,edges,masks,'torch:cpu',edge_chunk=2)
        direction=nodes[edges[:,1]].astype(np.float64)-nodes[edges[:,0]].astype(np.float64)
        np.testing.assert_allclose(prepared.c2.numpy(),(direction*direction).sum(-1),atol=1e-12)
        self.assertFalse(hasattr(prepared,'directions'))
        q=nodes[:3]*.25;union=np.ones(3,np.float32)
        actual=numerics.scan_segments(q,nodes,edges,masks,union,device='torch:cpu',
            prepared=prepared,edge_chunk=3,query_chunk=2)
        expected=numerics.scan_segments(q,nodes,edges,masks,union,edge_chunk=4,query_chunk=1)
        np.testing.assert_array_equal(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])

    def test_actual_device_segment_kernel_on_CPU_matches_NumPy(self):
        rng=np.random.default_rng(15);nodes=rng.normal(size=(17,1280)).astype(np.float32)
        q=rng.normal(size=(7,1280)).astype(np.float32);edges=np.array([[0,1],[3,7],[5,9],[6,12]])
        masks=np.array([[1,1,1,1],[0,1,0,1],[0,0,0,0]],bool);u=np.full(7,1,np.float32)
        one=numerics.scan_segments(q,nodes,edges,masks,u,query_chunk=3,edge_chunk=2,reference_chunk=5)
        two=numerics.scan_segments(q,nodes,edges,masks,u,device='torch:cpu',query_chunk=2,edge_chunk=3,reference_chunk=6)
        np.testing.assert_array_equal(one[0],two[0]);np.testing.assert_array_equal(one[1],two[1])

    def test_bidirectional_Torch_hot_path_on_CPU_matches_direct_numpy(self):
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.neighbors import bidirectional
        rng=np.random.default_rng(22);q=rng.normal(size=(6,1280)).astype(np.float32);r=rng.normal(size=(9,1280)).astype(np.float32)
        r[2]=r[0]
        one=bidirectional(q,r,5,query_chunk=2,reference_chunk=3)
        two=bidirectional(q,r,5,device='torch:cpu',query_chunk=3,reference_chunk=2)
        for a,b in zip(one,two):
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(a[1],b[1],atol=1e-12)

    def test_actual_device_witness_kernel_on_CPU_matches_NumPy(self):
        rng=np.random.default_rng(14);b={c:rng.normal(size=(5,9)).astype(np.float32) for c in range(1,5)}
        pairs={k:v for k,v in tables(b,k=5).items()};pool=graph.mutual_pool(b,pairs)
        one=graph.witness_pool(b,pool,edge_chunk=7,reference_chunk=3)
        two=graph.witness_pool(b,pool,device='torch:cpu',edge_chunk=9,reference_chunk=4)
        for k in ('eta','witness_support','witness_row','source_counts','alpha_histogram'):
            np.testing.assert_allclose(one[k],two[k],atol=1e-12)

    def test_original_UNION_SEGMENT_paths_and_units(self):
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import runtime
        from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import _support_prefix_minima
        from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
        from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
        rng=np.random.default_rng(1);q=rng.normal(size=(9,1280)).astype(np.float32)
        b={i:rng.normal(size=(9,1280)).astype(np.float32) for i in range(1,9)}
        source=self.torch.tensor(rng.normal(size=(12,1280)).astype(np.float32))
        s,t,idx=runtime.original_nearest(q,source,b)
        expected=_support_prefix_minima(self.torch.tensor(q),[self.torch.tensor(v) for v in b.values()])[8]
        np.testing.assert_array_equal(t.min(0),expected)
        exact,_=_exact_top1_chunked(self.torch.tensor(q),source)
        np.testing.assert_array_equal(s,exact.numpy().astype(np.float32)/1280)
        geo,_=geometry_layer(t,idx,s,list(b.values()))
        np.testing.assert_array_equal(geo['K8_UNION'],np.minimum(s,expected))

    def test_seven_active_branches_share_original_discrete_and_new_graph_distances(self):
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import runtime,prediction
        q=np.zeros((3,1280),np.float32)
        banks={i:np.zeros((5,1280),np.float32) for i in range(1,9)}
        banks[1][:,0]=-2;banks[2][:,0]=2
        for i in range(3,9):banks[i][:,1]=.5+(i-3)
        nodes,off,ids,rows=graph.layout(banks)
        pool=dict(edges=np.array([[0,5]]),rank=np.array([1]),length=np.array([4.]),eta=np.array([.5]),
            witness_support=np.array([3]),witness_row=np.array([0]))
        limits={(k,c):4. for k in protocol.KS for c in protocol.QUANTILES}
        choice=dict(main=dict(selected=dict(config=[10,.5,.75])),paired_config=[10,.5,None],
            plain=dict(selected=dict(config=[10,.5,None])),lambda_by_layer=[.25]*4)
        a=SimpleNamespace(device='cpu',query_chunk=2,edge_chunk=2,reference_chunk=7)
        values,diagnostics,audit=prediction.layer_branches(a,'background',7,q,
            self.torch.full((11,1280),3,dtype=self.torch.float32),banks,pool,limits,choice)
        self.assertEqual(set(values),set(protocol.BRANCHES))
        self.assertTrue((values[protocol.W]==0).all());self.assertTrue((values[protocol.U]>0).all())
        self.assertTrue((diagnostics['winner_'+protocol.W]==0).all())
        np.testing.assert_array_equal(values[protocol.ID],values[protocol.W])
        np.testing.assert_allclose(values[protocol.ND],values[protocol.U]*.75)

    def test_prepared_graph_reuse_does_not_accumulate_query_state(self):
        q=np.array([[0,.1],[0,.2]],np.float32);nodes=np.array([[-1,0],[1,0]],np.float32)
        edges=np.array([[0,1]]);mask=np.array([[True]]);u=np.ones(2,np.float32)
        prepared=numerics.PreparedSearch(nodes,edges,mask,'torch:cpu')
        one=numerics.scan_segments(q,nodes,edges,mask,u,device='torch:cpu',prepared=prepared)
        two=numerics.scan_segments(q[::-1].copy(),nodes,edges,mask,u,device='torch:cpu',prepared=prepared)
        np.testing.assert_array_equal(one[0],two[0][:,::-1])


class Integration(unittest.TestCase):
    """Actual NPZ graph/LOO/select/predict/CPU metric/resume paths on tiny data.

    Only RGB backbone output and source bank loading are substituted; formal
    NN and geometry execute on Torch CPU. No ADEval/GPU is invoked.
    """
    def test_cached_pipeline_and_interrupted_resume(self):
        try:import torch
        except ImportError:raise unittest.SkipTest('Torch unavailable')
        from PIL import Image
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import runtime,prediction,evaluation,reporting
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.inputs import st
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as temp:
                root=Path(temp);run=root/'run';cache=root/'cache';data=root/'data'
                (data/'train/good').mkdir(parents=True)
                a=SimpleNamespace(run_root=run,cache_root=cache,graph_root=cache/'graph',prediction_root=cache/'prediction',
                    query_cache_root=cache/'Q',evaluation_scratch=run/'scratch',device='cpu',query_chunk=4,
                    reference_chunk=16,edge_chunk=32,witness_edge_chunk=8,cpu_workers=2,seed=42,
                    evaluation_scope='primary',query_mode='disk',reuse_fresh_run=None)
                st.write_json(run/'setup.json',dict(identity='fixture'))
                samples=[];supports={};rng=np.random.default_rng(19);queries={}
                for d in protocol.DOMAINS:
                    supports[d]=[]
                    for h in range(1,9):
                        values=np.zeros((9,1280),np.float32)
                        # The orchestration fixture exercises the legal all-zero
                        # edge/empty-selection case. Nonempty exhaustive graph
                        # geometry is tested independently above, without a
                        # 108-fold dense-graph benchmark in a lightweight suite.
                        relative=f'test/good/{d}/support_{h}.png'
                        row=dict(relative_path=relative,image_sha256='unused_support_raster',image_height=64,image_width=64,
                            label=0,domain=d,sample_id=f'support_{d}_{h}',support_index=h)
                        supports[d].append(row)
                        st.seal(cache/'supports'/d/f'{h:03d}.npz','fixture',dict(relative_path=np.asarray(relative),
                            grid=np.array([3,3],np.int32),**{f'layer_{l}':values.copy() for l in protocol.LAYERS}),dict(grid=[3,3]))
                    for y in (0,1):
                        relative=f'test/{"defect" if y else "good"}/{d}/query{y}.png'
                        p=data/relative;p.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(np.full((64,64,3),y*100,np.uint8)).save(p)
                        row=dict(relative_path=relative,image_sha256=st.sha(p),image_height=64,image_width=64,
                            label=y,domain=d,sample_id=f'query_{y}')
                        if y:
                            gt=data/relative.replace('test/','ground_truth/',1);gt.parent.mkdir(parents=True,exist_ok=True)
                            mask=np.zeros((64,64),np.uint8);mask[24:32,24:32]=255;Image.fromarray(mask).save(gt);row['gt_sha256']=st.sha(gt)
                        samples.append(row)
                st.write_json(run/'fresh_catalog.json',dict(source=[],data_root=str(data),samples=samples,supports=supports))
                st.seal(cache/'source_patch_mad.npz','fixture',dict(medians=np.zeros(4,np.float32),scales=np.full(4,.001,np.float32)),{})
                state=dict(peak=dict(medians=[0]*4,scales=[1]*4),local=dict(medians=[0]*4,scales=[1]*4))
                state['state_sha256']=st.digest(state);st.write_json(run/'image_calibration_state.json',state)
                for d in protocol.DOMAINS:
                    b=ip.load_banks(a,d,7)
                    for h in range(1,9):
                        refs={k:v for k,v in b.items() if k!=h}
                        src=torch.full((11,1280),3,dtype=torch.float32)
                        s,t,idx=runtime.original_nearest(b[h],src,refs)
                        st.seal(cache/'support_baseline'/d/f'{h:03d}.npz','fixture',dict(source=np.stack([s]*4),
                            target=np.stack([t]*4),indices=np.stack([idx]*4)),{})
                with contextlib.redirect_stdout(io.StringIO()):
                    runtime.prepare_graphs(a);runtime.support_loo(a);runtime.select(a)
                    # Resume uses exactly the completed tables/LOO jobs.
                    with patch.object(runtime,'scan_segments',side_effect=AssertionError('LOO recomputed')):
                        runtime.support_loo(a)
                source={l:torch.full((11,1280),3,dtype=torch.float32) for l in protocol.LAYERS}
                calls=[]
                def extract(model,image,device):
                    calls.append(1);q=np.zeros((9,1280),np.float32)
                    q[:,:3]=np.asarray(image)[0,0,0]/100+.1
                    return {f'layer_{l}':q.copy() for l in protocol.LAYERS},dict(grid=[3,3],positions=np.arange(9,dtype=np.int64),processed_shape=[48,48])
                with (patch.object(runtime,'precision'),patch.object(runtime,'host',return_value=(object(),'cpu')),
                      patch.object(runtime,'source_banks',return_value=source),patch.object(prediction,'extraction',side_effect=extract)):
                    with contextlib.redirect_stdout(io.StringIO()):prediction.freeze(a)
                self.assertEqual(len(calls),6)
                with (patch.object(runtime,'host',side_effect=AssertionError('reloaded host')),
                      patch.object(runtime,'precision',side_effect=AssertionError('touched CUDA')),
                      patch.object(runtime,'load_graph',side_effect=AssertionError('required unused graph')),
                      patch.object(ip,'load_banks',side_effect=AssertionError('required unused support features'))):
                    prediction.freeze(a)
                self.assertEqual(len(prediction.verified(a)),6)
                with contextlib.redirect_stdout(io.StringIO()):
                    for d in protocol.DOMAINS:evaluation.evaluate(a,d)
                with patch.object(evaluation,'sample_work',side_effect=AssertionError('CPU recomputed')):
                    for d in protocol.DOMAINS:evaluation.evaluate(a,d)
                result=reporting.summarize(a)
                self.assertFalse(result['all_metrics_complete']);self.assertEqual(len(result['pending']),21)
                # Modified intermediate/result must not be re-enrolled on resume.
                path=run/'evaluation/background'/f'{protocol.U}_exact_pixels.json'
                previous=path.read_text();path.write_text(previous.replace('"AUROC":', '"AUROC": 1.5, "old":',1))
                with self.assertRaises(ValueError):evaluation.restore(a,'background')
        finally:torch.set_num_threads(old_threads)


if __name__=='__main__':unittest.main()
