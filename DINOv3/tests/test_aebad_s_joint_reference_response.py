"""CPU-only fixtures; never initialize Torch/CUDA or read real test GT."""
import csv
import gc
import itertools
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import expit
from DINOv3.MADEqual.aebad_s_joint_reference_response_p0 import numerics as nm, inputs as ip, runtime as rt, prediction as pp, evaluation as ev, reporting as report, runner, smoke
from DINOv3.MADEqual.aebad_s_joint_reference_response_p0.protocol import *
from DINOv3.MADEqual.aebad_s_joint_reference_response_p0.inputs import st
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values, response
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.numerics import weighted


def direct(s, t, refs):
    all_prob = []
    for j, (rs, rtarget) in enumerate(refs):
        a = expit(rs.astype(np.float64)[None]-s.astype(np.float64).T[:, None]).prod(-1)
        b = expit(rtarget.astype(np.float64)[None]-t[j].astype(np.float64).T[:, None]).prod(-1)
        u = expit(np.minimum(rs, rtarget).astype(np.float64)[None]-np.minimum(s, t[j]).astype(np.float64).T[:, None]).prod(-1)
        all_prob.append(np.stack([(a*b).mean(1), a.mean(1)*b.mean(1), u.mean(1)]))
    return -np.log(np.mean(all_prob, axis=0))/np.asarray([8, 8, 4])[:, None]


class Mathematics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(13)
        self.s = self.rng.normal(size=(4, 11)).astype('float32')
        self.t = self.rng.normal(size=(8, 4, 11)).astype('float32')
        self.refs = [(self.rng.normal(size=(5, 4)).astype('float32'), self.rng.normal(size=(5, 4)).astype('float32')) for _ in range(8)]
        self.w = np.full(4, .25, np.float32)

    def test_stable_log_matches_direct_float64(self):
        result = nm.score_native(self.s, self.t, self.refs, self.w, query_chunk=3, reference_chunk=2)
        np.testing.assert_allclose(np.stack([result[k] for k in (JOINT, INDEPENDENT, UNION4)]), direct(self.s, self.t, self.refs), atol=4e-7, rtol=1e-6)

    def test_query_response_lowering_monotone_each_dimension(self):
        initial = nm.score_native(self.s, self.t, self.refs, self.w)
        for side in ('Source', 'Target'):
            for layer in range(4):
                s, t = self.s.copy(), self.t.copy()
                if side == 'Source':s[layer] -= .7
                else:t[:, layer] -= .7
                lower = nm.score_native(s, t, self.refs, self.w)
                for k in MAPS:self.assertTrue(np.all(lower[k] <= initial[k]+2e-7), (side, layer, k))

    def test_extreme_tails_not_epsilon_clipped(self):
        refs = [(np.zeros((3, 4), 'float32'), np.zeros((3, 4), 'float32'))]*8
        s = np.asarray([[100, 101, 1000]]*4, np.float32);t = np.repeat(s[None], 8, 0)
        x = nm.score_native(s, t, refs, self.w)
        for b in (JOINT, INDEPENDENT, UNION4):
            self.assertTrue(np.isfinite(x[b]).all());self.assertTrue(np.all(np.diff(x[b]) > 0))

    def test_lower_than_all_normals_is_not_outlier(self):
        refs = [(np.zeros((4, 4), 'float32'), np.zeros((4, 4), 'float32'))]*8
        result = nm.score_native(np.full((4, 2), -1000, 'float32'), np.full((8, 4, 2), -1000, 'float32'), refs, self.w)
        for b in (JOINT, INDEPENDENT, UNION4):np.testing.assert_array_equal(result[b], 0.)

    def test_analytic_independent_equals_all_block_permutations(self):
        rs, rtarget = self.refs[0][0][:3], self.refs[0][1][:3];query_s = self.s[:, :1];query_t = self.t[:, :, :1]
        a = expit(rs.astype(float)-query_s[:, 0]).prod(1)
        b = expit(rtarget.astype(float)-query_t[0, :, 0]).prod(1)
        perm = np.mean([np.mean(a*b[list(order)]) for order in itertools.permutations(range(3))])
        self.assertAlmostEqual(perm, a.mean()*b.mean(), places=15)
        # Real kernel, identical groups, each Target FOUR-vector block permuted together.
        result = nm.score_native(query_s, np.repeat(query_t[:1], 8, 0), [(rs, rtarget)]*8, self.w)
        self.assertAlmostEqual(float(result[INDEPENDENT][0]), -np.log(perm)/8, places=6)

    def test_probability_mean_before_log_and_equal_image_weight(self):
        refs = [(np.full((n, 4), float(j)-3, 'float32'), np.full((n, 4), float(j)-3, 'float32')) for j, n in enumerate((1, 2, 3, 4, 5, 6, 7, 8))]
        s = np.zeros((4, 1), 'float32');t = np.zeros((8, 4, 1), 'float32')
        value = nm.score_native(s, t, refs, self.w)[JOINT][0]
        probability = np.asarray([expit(float(j)-3)**8 for j in range(8)])
        self.assertAlmostEqual(value, -np.log(probability.mean())/8, places=6)
        self.assertGreater(abs(value-(-np.log(probability)/8).mean()), .1)
        self.assertGreater(abs(value-(-np.log(np.average(probability, weights=np.arange(1, 9))))/8), .01)

    def test_block_sizes_same_definition(self):
        first = nm.score_native(self.s, self.t, self.refs, self.w, query_chunk=1, reference_chunk=1)
        second = nm.score_native(self.s, self.t, self.refs, self.w, query_chunk=17, reference_chunk=100)
        for k in MAPS:np.testing.assert_allclose(first[k], second[k], atol=3e-7, rtol=1e-6)

    def test_joint_and_independent_can_differ(self):
        refs = [(np.asarray([[3]*4, [-3]*4], np.float32), np.asarray([[3]*4, [-3]*4], np.float32))]*8
        value = nm.score_native(np.zeros((4, 1), 'float32'), np.zeros((8, 4, 1), 'float32'), refs, self.w)
        self.assertLess(float(value[JOINT][0]), float(value[INDEPENDENT][0]))

    def test_cf7_capacity_native_mean_before_fusion(self):
        weights = np.asarray([.1, .2, .3, .4], np.float32)
        result = nm.score_native(self.s, self.t, self.refs, weights)
        expected = np.sum(np.minimum(self.s[None], self.t).mean(0, dtype=np.float32)*weights[:, None], 0, dtype=np.float32)
        np.testing.assert_array_equal(result[CAPACITY], expected)

    def test_heldout_is_entire_support(self):
        dt = np.ones((4, 8, 3), 'float32')*9
        for j in range(8):
            dt[:, j] = j
        seven = nm.seven_targets(dt)
        np.testing.assert_array_equal(seven[0], 1.)
        for j in range(1, 8):np.testing.assert_array_equal(seven[j], 0.)
        for j in range(8):self.assertEqual(heldout_indices(j), [i for i in range(8) if i != j])
        with self.assertRaises(ValueError):heldout_indices(8)

    def test_min_and_common_MAD_commute_without_second_division(self):
        ds = np.abs(self.s);dt = np.abs(self.t.transpose(1, 0, 2))
        patch = dict(medians=[.1, .2, .3, .4], scales=[.4, .3, .2, .1])
        s, t = nm.normalized(ds, dt, patch)
        for j in range(8):
            np.testing.assert_allclose(np.minimum(s, t[j]), z_values(np.minimum(ds, dt[:, heldout_indices(j)].min(1)), patch), atol=1e-7, rtol=1e-6)

    def test_physical_mapping_all_layers_and_targets(self):
        p = np.asarray([4, 0, 1, 3, 2, 5], np.int64);v = np.arange(24).reshape(4, 6).astype('float32')
        mapped = nm.physical(v, p, (2, 3))
        np.testing.assert_array_equal(mapped[:, p], v)
        permutation = np.asarray([5, 2, 0, 1, 4, 3])
        np.testing.assert_array_equal(nm.physical(v[:, permutation], p[permutation], (2, 3)), mapped)

    def test_uniform_deterministic_sampling_uses_no_values_or_labels(self):
        first = nm.sample_positions('test/good/a.png', (30, 30))
        self.assertEqual(len(first), 512);self.assertEqual(len(set(first)), 512)
        np.testing.assert_array_equal(first, nm.sample_positions('test/good/a.png', (30, 30)))
        self.assertFalse(np.array_equal(first, nm.sample_positions('test/good/b.png', (30, 30))))
        np.testing.assert_array_equal(nm.sample_positions('small', (3, 4)), np.arange(12))

    def test_reject_misaligned_or_nonfinite_inputs(self):
        with self.assertRaises(ValueError):nm.physical(np.ones((4, 6)), np.zeros(6, np.int64), (2, 3))
        refs = [(np.ones((3, 4), 'float32'), np.ones((2, 4), 'float32'))]*8
        with self.assertRaises(ValueError):nm.score_native(self.s, self.t, refs, self.w)


class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.root = Path(self.tmp.name)
        self.a = runner.normalize(runner.parser().parse_args(['--run-root', str(self.root/'run'), '--cache-root', str(self.root/'cache'), '--parent-run', str(self.root/'parent'), '--sacp-run', str(self.root/'sacp'), '--data', str(self.root/'data')]))
        self.a.run_root.mkdir();self.rng = np.random.default_rng(5);self.grid = [3, 3];self.pos = np.asarray([8, 2, 1, 4, 5, 7, 0, 6, 3], np.int64)
        patch_state = dict(medians=[.1]*4, scales=[.2]*4)
        self.p = SimpleNamespace(inputs=dict(source=dict(seal='same_Source', path=str(self.root/'Source')), extraction={'checkpoint': 'frozen', 'preprocess': 'original'}, mad=dict(patch=patch_state, image=dict(peak=dict(medians=[0]*4, scales=[1]*4), local=dict(medians=[0]*4, scales=[1]*4)))),
             identity='parent-id', common={'domains': {}, 'seal': 'common-seal'}, weights={}, records={}, maps={}, settings={})
        self.p.inputs['seal'] = 'parent-seal';self.p.samples = lambda d: self.p.common['domains'][d]['common']
        self.p.record = lambda d, seed, row: self.p.records[d, seed, row['sample_id']]
        self.a._joint_parent = self.p;self.baselines = {}
        for d in DOMAINS:
            samples = [dict(domain=d, sample_id='test_'+str(i), relative_path=f'test/{d}/{i}.png', image_sha256='img'+d+str(i), image_height=32, image_width=32, label=i) for i in (0, 1)]
            supports = {str(s): [dict(domain=d, relative_path=f'test/good/{d}/{s}_{j}.png', image_sha256=f's{s}_{j}', image_height=32, image_width=32, sample_id=f's{s}_{j}', label=0) for j in range(8)] for s in SEEDS}
            self.p.common['domains'][d] = dict(common=samples, supports=supports)
            self.p.weights[d, 'WS'] = np.full((4, 3), .25).tolist()
        ip.freeze(self.a)
        for d in DOMAINS:
            for seed in SEEDS:
                for j, row in enumerate(self.p.common['domains'][d]['supports'][str(seed)]):
                    ds = self.rng.uniform(.05, .8, (4, 9)).astype('float32');dt = self.rng.uniform(.05, .6, (4, 9)).astype('float32')
                    st.put(ip.normal_path(self.a, d, seed, j), dict(Source_distance=ds, Target7_distance=dt, positions=self.pos), dict(contract=ip.split_contract(self.a, d, seed, row, j), grid=self.grid))
                metric_rows = []
                for row in self.p.samples(d):
                    ds = self.rng.uniform(.1, .9, (4, 9)).astype('float32');dt = self.rng.uniform(.2, 1., (4, 8, 9)).astype('float32')
                    st.put(ip.query_path(self.a, d, seed, row), dict(Source_distance=ds, target_distance=dt, positions=self.pos),
                           dict(contract=ip.split_contract(self.a, d, seed, row), grid=self.grid, output_shape=[8, 8]))
                    zu = z_values(np.minimum(ds, dt.min(1)), patch_state);zs = zu-.15
                    self.baselines[d, seed, row['sample_id']] = {U: zu, S: zs, 'positions': self.pos}
                    maps = {};scores = {}
                    for b, value in ((U, zu), (S, zs)):
                        maps[b], scores[b] = response(value, self.pos, self.grid, [8, 8], self.p.inputs['mad']['image'])
                    maps[SOURCE], scores[SOURCE] = weighted(zu, self.pos, self.grid, [8, 8], self.p.weights[d, 'WS'])
                    self.p.records[d, seed, row['sample_id']] = dict(scores=scores, map_hashes={b: ev.backend.shared.array_sha(v) for b, v in maps.items()})
        # Rebuild lookup: test fixtures preserve identical GT across all settings.
        for d in DOMAINS:
            for seed in SEEDS:
                self.p.maps[d, seed] = {(r['sample_id'], U): dict(mask_uint8_sha256=ev.backend.shared.array_sha(ev.backend.shared.resized_mask(self.gt(r), [8, 8]).astype('uint8'))) for r in self.p.samples(d)}
                self.p.settings[d, seed] = dict(metrics=[dict(branch=b, scoring=rule, image_AUROC=.5, image_AP=.5, pixel_AUROC=.5, pixel_AP=.1, AUPRO=.2) for b, rules in ((U, ('CH1', 'CURRENT')), (S, ('CH1', 'CURRENT')), (SOURCE, ('PEAK_LOCAL_WEIGHT', 'WEIGHTED_MAP_MAX'))) for rule in rules])
                for metric in self.p.settings[d, seed]['metrics']:
                    ss=[self.p.record(d, seed, row)['scores'][metric['branch']][metric['scoring']] for row in self.p.samples(d)]
                    metric.update(ev.image_metrics([row['label'] for row in self.p.samples(d)],ss))
        self.original_patch = patch.object(ip, 'original_baseline', side_effect=self.baseline);self.original_patch.start()

    def baseline(self, a, d, seed, row, keys):
        return self.baselines[d, seed, row['sample_id']], dict(metadata=dict(grid=self.grid)), dict(path='fixture', seal='fixture')

    @staticmethod
    def gt(row):
        result = np.zeros((32, 32), np.uint8)
        if row['label']:result[8:20, 10:22] = 1
        return result

    def tearDown(self):
        self.original_patch.stop();gc.collect();self.tmp.cleanup()

    def test_shared_Source_scientific_identity_ignores_support_seed(self):
        row = self.p.samples('background')[0];contract = ip.source_contract(self.a, row)
        self.assertNotIn('support_seed', contract)
        self.assertNotEqual(contract, ip.source_contract(self.a, dict(row, image_sha256='changed')))
        x, r = ip.inspect_split(self.a, 'background', 43, row)
        st.put(ip.source_path(self.a, 'background', row), dict(Source_distance=x['Source_distance'], positions=x['positions']), dict(contract=contract, grid=self.grid))
        shared, _, origin = ip.inspect_source(self.a, 'background', row)
        np.testing.assert_array_equal(shared['Source_distance'], x['Source_distance']);self.assertTrue(origin['support_seed_independent'])

    def test_support_seed_cannot_change_cache_contract(self):
        row = self.p.samples('background')[0]
        self.assertNotEqual(ip.split_contract(self.a, 'background', 42, row), ip.split_contract(self.a, 'background', 43, row))
        x, r = ip.inspect_normal(self.a, 'view', 43, 2)
        self.assertEqual(len(r['metadata']['contract']['reference_images']), 7)
        heldout = self.p.common['domains']['view']['supports']['43'][2]
        self.assertNotIn(ip.image_contract(heldout), r['metadata']['contract']['reference_images'])

    def test_normal_pairing_uses_same_physical_indices(self):
        rt.build_references(self.a);refs = rt.normal_model(self.a, 'background', 42)
        x, r = ip.inspect_normal(self.a, 'background', 42, 0)
        expected = nm.physical(z_values(x['Source_distance'], self.p.inputs['mad']['patch']), self.pos, self.grid).T
        np.testing.assert_array_equal(refs[0][0], expected)

    def test_subset_prepare_does_not_truncate_complete_sampling_manifest(self):
        rt.build_references(self.a);path=self.a.run_root/'normal_sampling.csv';before=path.read_bytes()
        model=st.sealed_json(self.a.run_root/'normal_model.json')
        self.a.domains=['view'];self.a.support_seeds=[43]
        result=rt.build_references(self.a)
        self.assertEqual(result['status'],'requested_normal_model_subset_complete')
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(model,st.sealed_json(self.a.run_root/'normal_model.json'))

    def test_predict_resume_and_cross_domain_names(self):
        rt.build_references(self.a);first = pp.predict(self.a)
        self.assertEqual(first['new_images'], 12)
        frozen = st.sealed_json(self.a.run_root/'predictions_frozen.json')
        second = pp.predict(self.a)
        self.assertEqual(second['new_images'], 0);self.assertEqual(second['reused_images'], 12)
        self.assertEqual(frozen, st.sealed_json(self.a.run_root/'predictions_frozen.json'))
        self.assertNotEqual(ip.record_path(self.a, 'view', 42, self.p.samples('view')[0]), ip.record_path(self.a, 'background', 42, self.p.samples('background')[0]))

    def test_GT_refused_before_global_prediction_freeze(self):
        with patch.object(ev, 'GTData') as data:
            with self.assertRaises(FileNotFoundError):ev.evaluate(self.a)
            data.assert_not_called()

    def test_frozen_reference_checks_and_scores(self):
        row = self.p.samples('background')[0];x, r = ip.inspect_split(self.a, 'background', 42, row)
        checks, _ = pp.baseline_checks(self.a, 'background', 42, row, x, r['metadata'])
        self.assertEqual(len(checks), 4);self.assertTrue(all(r['verified'] for r in checks))
        wrong = dict(x, Source_distance=x['Source_distance']+2, target_distance=x['target_distance']+2)
        with self.assertRaises(ValueError):pp.baseline_checks(self.a, 'background', 42, row, wrong, r['metadata'])

    def test_CPU_exact_end_to_end_fixture_and_resume(self):
        rt.build_references(self.a);pp.predict(self.a)
        outer = self
        class Data:
            def __init__(self, path):pass
            def __enter__(self):return self
            def __exit__(self, *args):pass
            def gt(self, row):return outer.gt(row)
        with patch.object(ev, 'GTData', Data):
            result = ev.evaluate(self.a);self.assertEqual(result['jobs'], 24)
        with patch.object(ev, 'GTData') as data:
            self.assertEqual(ev.evaluate(self.a)['jobs'], 24);data.assert_not_called()
        path = ev.destination(self.a, 'background', 42, JOINT)/'metrics.csv'
        with path.open(encoding='utf-8', newline='') as f:values = list(csv.DictReader(f))
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0]['pixel_AP'], values[1]['pixel_AP'])
        self.assertFalse('torch' in sys.modules)
        result = report.summarize(self.a)
        self.assertEqual(result['status'], 'pending');self.assertEqual(result['complete_AUPRO_jobs'], 0)
        with (self.a.run_root/'metrics.csv').open(encoding='utf-8',newline='') as f:combined=list(csv.DictReader(f))
        self.assertEqual(len(combined), 84)
        self.assertTrue(all(r['AUPRO']=='' for r in combined if r['branch'] in MAPS))
        # An altered completed table cannot be enrolled as a successful resume.
        path.write_text(path.read_text(encoding='utf-8')+'\n', encoding='utf-8')
        with self.assertRaises(ValueError):ev.completed(self.a, 'background', 42, JOINT)

    def test_actual_original_References_excludes_support_image(self):
        original = rt.OriginalReferences.__new__(rt.OriginalReferences)
        original.source = {'L'+str(l):np.full((2, 1280), 99, np.float32) for l in LAYERS}
        original.parts = [{l:np.full((2, 1280), j, np.float32) for l in LAYERS} for j in range(8)]
        original.source_seal = 'S';original.support_seals = [str(j) for j in range(8)]
        class Search:
            def __init__(self):self.calls=[]
            def query(self, q):return q
            def exact(self, q, bank, key):
                self.calls.append(key)
                return np.full(len(q), float(bank[0, 0]), np.float32), np.zeros(len(q), np.int64)
        original.search = Search()
        for heldout in range(8):
            original.search.calls.clear();values = original.distances(original.parts[heldout], heldout=heldout)
            for l in LAYERS:self.assertEqual(values[l]['support_indices'], heldout_indices(heldout))
            self.assertEqual(len(original.search.calls), 32)
            self.assertFalse(any(k[0]=='Target' and k[1]==str(heldout) for k in original.search.calls))

    def test_target_supplement_uploads_each_query_layer_once(self):
        reference = rt.References.__new__(rt.References)
        reference.parts = [{l:np.zeros((3, 1280), np.float32) for l in LAYERS} for _ in range(8)]
        reference.support_seals = list(range(8))
        class Search:
            def __init__(self):self.uploads=0;self.calls=0
            def query(self, q):self.uploads+=1;return q
            def exact(self, q, bank, key):self.calls+=1;return np.ones(len(q), np.float32), None
        reference.search = Search();distance = reference.target_only(reference.parts[0])
        self.assertEqual(reference.search.uploads, 4);self.assertEqual(reference.search.calls, 32)
        self.assertEqual(distance.shape, (4, 8, 3))

    def test_changed_feature_extraction_rejected(self):
        row = self.p.samples('background')[0]
        self.a.sacp_run.mkdir()
        st.sealed_json(self.a.sacp_run/'assets.json', dict(references=dict(extraction=self.p.inputs['extraction'], source=self.p.inputs['source'])))
        path = self.a.cache_root/'normal_features'/ip.previous.layer_inputs.old.key(row)
        arrays = {'q'+str(l):np.ones((9, 1280), np.float32) for l in LAYERS};arrays['positions']=self.pos
        st.put(path, arrays, dict(image=row, grid=self.grid, extraction=dict(checkpoint='different', preprocess='original')))
        with self.assertRaises(ValueError):ip.feature_ref(self.a, row)

    def test_changed_native_manifest_rejected_on_resume(self):
        rt.build_references(self.a);pp.predict(self.a)
        row = self.p.samples('background')[0];path = ip.prediction_path(self.a, 'background', 42, row)/'manifest.json'
        value = st.read_json(path);value['metadata']['grid']=[1, 9]
        st.write_json(path, value)
        with self.assertRaises(ValueError):pp.predict(self.a)

    def test_partial_capture_resume_shares_one_forward_and_preserves_Source_origin(self):
        self.a.domains=['background'];self.a.supplement_retrieval=True
        row = self.p.samples('background')[0]
        sx, sr = ip.inspect_split(self.a, 'background', 43, row)
        ds=np.array(sx['Source_distance']);origin={'kind':'original43_source'}
        for value in sx.values():value._mmap.close()
        st.put(ip.source_path(self.a,'background',row),dict(Source_distance=ds,positions=self.pos),
               dict(contract=ip.source_contract(self.a,row),grid=self.grid,origin=origin,GT_used=False))
        for seed in SEEDS:
            path=ip.query_path(self.a,'background',seed,row)
            for file in path.iterdir():file.unlink()
        original = st.sealed_json(ip.source_path(self.a,'background',row)/'manifest.json')
        outer=self
        class Ref:
            def __init__(self,a,d,seed,search=None):self.search=SimpleNamespace(stats={})
            def target_only(self,q):return np.ones((4,8,9),np.float32)
        def features(a,r,persist=False):
            return {l:np.zeros((9,1280),np.float32) for l in LAYERS},dict(grid=outer.grid,positions=outer.pos),dict(kind='fixture_one_forward')
        with patch.object(rt,'References',Ref),patch.object(ip.audit,'inspect',side_effect=FileNotFoundError),patch.object(rt,'features',side_effect=features) as feature:
            result=rt.capture(self.a)
            self.assertEqual(result['new_query_settings'],2);self.assertEqual(feature.call_count,1)
        self.assertEqual(original,st.sealed_json(ip.source_path(self.a,'background',row)/'manifest.json'))
        # Remove only the second setting: next invocation consumes its sibling's
        # unchanged shared Source record, without trying to rewrite provenance.
        path=ip.query_path(self.a,'background',43,row)
        for file in path.iterdir():file.unlink()
        with patch.object(rt,'References',Ref),patch.object(ip.audit,'inspect',side_effect=FileNotFoundError),patch.object(rt,'features',side_effect=features):rt.capture(self.a)
        self.assertEqual(original,st.sealed_json(ip.source_path(self.a,'background',row)/'manifest.json'))

    def test_CPU_smoke_path_records_unknown_AUPRO_and_no_GT(self):
        rt.build_references(self.a)
        st.write_csv(self.a.run_root/'missing_inputs.csv',[])
        st.write_csv(self.a.run_root/'necessary_work.csv',[])
        with patch.object(ip,'preflight',return_value={'status':'fixture_only'}),patch.object(ev,'GTData') as gt:
            result=smoke.run(self.a)
            gt.assert_not_called()
        self.assertEqual(result['status'],'pending_unmeasured_costs')
        self.assertEqual(result['unknown_phases'],['AUPRO_jobs'])
        r=st.read_json(self.a.run_root/'smoke/smoke_report.json')
        self.assertEqual(len(r['rows']),12);self.assertFalse(r['formal_metrics'])
        self.assertFalse('torch' in sys.modules)

    def test_scratch_is_owned_and_pack_excludes_large_temp(self):
        sentinel = self.a.evaluation_scratch/'background/seed42'/JOINT/'other.bin'
        sentinel.parent.mkdir(parents=True);sentinel.write_bytes(b'old')
        with self.assertRaises(RuntimeError):
            with ev.scratch(self.a, 'background', 42, JOINT, 11):raise RuntimeError('fixture allocation interruption')
        self.assertEqual(sentinel.read_bytes(), b'old')
        tmp = self.a.run_root/'pixel_scratch';tmp.mkdir();(tmp/'large.bin').write_bytes(b'not a report')
        packed = runner.pack(self.a)
        import tarfile
        with tarfile.open(packed['archive']) as t:self.assertFalse(any('pixel_scratch' in x for x in t.getnames()))

    def test_scratch_parent_overlap_rejected(self):
        args = runner.parser().parse_args(['--run-root', str(self.root/'new'), '--evaluation-scratch', str(self.root/'parent/scratch'), '--parent-run', str(self.root/'parent')])
        with self.assertRaises(ValueError):runner.normalize(args)


class Planning(unittest.TestCase):
    def test_retrieval_timing_separates_upload_from_per_query_cost(self):
        reference=SimpleNamespace(parts=[{l:np.zeros((1,2),np.float32) for l in LAYERS}],
             source={'L'+str(l):np.zeros((1,2),np.float32) for l in LAYERS},source_seal='S',
             load_source=lambda:None,target_only=lambda q:None,
             search=SimpleNamespace(query=lambda x:x,exact=lambda *args:(None,None),stats={}))
        with patch.object(smoke.time,'perf_counter',side_effect=[0,1,2,12,13,33,34,36,37,40]):
            result=smoke.retrieval_timing(reference)
        self.assertEqual(result['Source_four_layers_seconds'],2)
        self.assertEqual(result['Source_initial_upload_seconds'],8)
        self.assertEqual(result['Target_eight_supports_seconds'],3)
        self.assertEqual(result['Target_group_initial_upload_seconds'],17)
        estimated=smoke.estimate(dict(Source_retrieval=100,Source_resident_startup=1),
                                dict(Source_retrieval=2,Source_resident_startup=8))
        self.assertEqual(estimated['remaining_seconds'],208)

    def test_default_preflight_and_fixed_counts(self):
        a = runner.parser().parse_args([])
        self.assertEqual(a.stage, 'preflight');self.assertFalse(a.extract_missing);self.assertFalse(a.supplement_retrieval)
        self.assertEqual(work_plan()['new_map_jobs'], 24);self.assertEqual(sum(c[0] for c in COUNTS.values()), 902)
        self.assertEqual(CONFIG['bandwidth'], 1.)

    def test_unknown_estimate_not_zero_or_launch_permission(self):
        e = smoke.estimate(dict(query_forwards=902, exact_CPU_jobs=24), dict(exact_CPU_jobs=10))
        self.assertIsNone(e['remaining_seconds']);self.assertEqual(e['unknown_phases'], ['query_forwards'])
        e = smoke.estimate(dict(query_forwards=0, exact_CPU_jobs=24), dict(exact_CPU_jobs=10))
        self.assertEqual(e['remaining_seconds'], 240)

    def test_budget_unknown_and_over_12h_cannot_launch(self):
        from DINOv3.MADEqual.aebad_s_joint_reference_response_p0.budget import launch_check
        with tempfile.TemporaryDirectory() as root:
            a=SimpleNamespace(run_root=Path(root),stage='all',allow_over_12h=False,_started=time.perf_counter())
            path=Path(root)/'time_estimate.json'
            st.write_json(path,dict(status='unknown',remaining_seconds=None))
            with self.assertRaises(RuntimeError):launch_check(a)
            st.write_json(path,dict(scope='fixed_normal_asset_smoke',phases={'models':{'seconds':50000}}))
            with self.assertRaises(RuntimeError):launch_check(a)
            a.allow_over_12h=True;launch_check(a)
            # A CPU pipeline need not benchmark AUPRO; its predict subprocess
            # must not reintroduce an unrequested GPU evaluation dependency.
            a.allow_over_12h=False;a.stage='all-cpu'
            st.write_json(path,dict(scope='fixed_normal_asset_smoke',phases={'four_native_maps':{'seconds':10},'native_IO':{'seconds':1},'AUPRO_jobs':{'seconds':None}}))
            launch_check(a);a.stage='predict';launch_check(a)

    def test_macro_equal_domain_then_seed_and_pending_not_zero(self):
        def row(v):return [dict(branch=JOINT, scoring='MAP_MAX', **{k:v for k in METRICS})]
        result = report.aggregate([row(.2), row(.8), row(.5)], 'macro', 42)
        self.assertEqual(result[0]['AUPRO'], .5)
        self.assertIsNone(report.aggregate([row(.5), row(None)], 'pending', 43)[0]['AUPRO'])

    def test_parent_readers_validate_real_902_metadata_without_Torch(self):
        root = ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433'
        if not root.exists():self.skipTest('local historical review not shipped in release')
        p = ip.audit.Parent(root)
        self.assertEqual(sum(len(p.samples(d)) for d in DOMAINS), 902)
        for d in DOMAINS:
            supports = {r['relative_path'] for seed in SEEDS for r in p.common['domains'][d]['supports'][str(seed)]}
            self.assertFalse(supports & {r['relative_path'] for r in p.samples(d)})
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':unittest.main()
