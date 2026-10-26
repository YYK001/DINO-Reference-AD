"""CPU-only profile, immutable reuse, sparse mappings and requested completion tests."""
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import protocol as pr,inputs as ip,storage as st,runner,synthesis,tiles,reuse,reporting,training_data,training
from DINOv3.tests.test_aebad_s_reference_residual_segmentation import Temporary,catalog


class Profiles(Temporary):
    def test_default_and_full_plans_and_requested_branches(self):
        a=runner.parser().parse_args([]);self.assertEqual(a.profile,'compact_v1')
        cfg=pr.resolve();p=pr.job_plan(cfg)
        self.assertEqual((p['source_jobs'],p['target_jobs'],p['planned_updates'],p['planned_synthetic_slots']),(1,3,13000,1294))
        self.assertEqual(p['branches'],['UNION_K8','SEGMENT_K8','REF_ST_SEED42']);self.assertEqual(len(p['localization_jobs']),9)
        self.assertEqual(p['evaluation_images'],926);self.assertEqual(p['ensembles'],[]);self.assertEqual(p['primary'],'REF_ST_SEED42')
        self.assertTrue(all(r['status']=='not_requested' for r in p['not_requested']))
        full=pr.job_plan(pr.resolve('full_v1'))
        self.assertEqual((full['source_jobs'],full['target_jobs'],full['planned_updates'],full['planned_synthetic_slots']),(6,27,336000,5944))
        self.assertEqual(len(full['localization_jobs']),42)
    def test_compact_stratification_and_old_variant_donors(self):
        c=catalog();compact=synthesis.manifest(c,pr.resolve());full=synthesis.manifest(c,pr.resolve('full_v1'))
        self.assertEqual(len(compact),1294);index={(r['origin'],r['variant']):r for r in full}
        counts={'texture':0,'scratch':0};scales=set()
        for row in ip.train_source(c):
            selected=[r for r in compact if r['origin']==row['relative_path']]
            self.assertEqual(len(selected),2);self.assertEqual(sum(r['kind']=='poisson' for r in selected),1)
            other=next(r['kind'] for r in selected if r['kind']!='poisson');counts[other]+=1
        self.assertLessEqual(abs(counts['texture']-counts['scratch']),1)
        for row in sum(c['supports'].values(),[]):
            chosen=[r for r in compact if r['origin']==row['relative_path']]
            self.assertEqual([sum(r['kind']==kind for r in chosen) for kind in ('poisson','texture','scratch')],[8,4,4])
            bins=[sum((r['variant']+st.seed(r['origin'])%3)%3==s for r in chosen) for s in range(3)]
            self.assertLessEqual(max(bins)-min(bins),1)
        for r in compact:
            self.assertEqual(r,index[r['origin'],r['variant']]);scales.add((r['variant']+st.seed(r['origin'])%3)%3)
        self.assertEqual(scales,{0,1,2});self.assertEqual(reuse.rules(pr.resolve()),reuse.rules(pr.resolve('full_v1')))
    def test_old_full_directory_rejects_default_and_keeps_frozen_bytes(self):
        a=self.args();a.profile='compact_v1';path=a.run_root/'inputs.json'
        known=st.read_json(Path(pr.__file__).with_name('legacy_full_v1_compatibility.json'))['code']
        st.sealed_json(path,dict(config=pr.CONFIG,identity='old',code={'DINOv3/MADEqual/aebad_s_reference_residual_segmentation/'+k:v for k,v in known.items()}));before=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'explicit --profile full_v1'):ip.config(a)
        a.profile='full_v1';self.assertEqual(ip.config(a)['training_nn_layout'],'complete_grid_v1')
        self.assertEqual(path.read_bytes(),before);self.assertFalse((a.run_root/'resolved_config.json').exists())
    def test_resolved_profile_and_job_plan_seals_are_authoritative(self):
        a=self.args();a.profile='compact_v1';cfg=pr.resolve()
        st.sealed_json(a.run_root/'resolved_config.json',dict(config=cfg));st.sealed_json(a.run_root/'job_plan.json',dict(plan=pr.job_plan(cfg)))
        self.assertEqual(ip.plan(a)['planned_updates'],13000)
        a.profile='full_v1'
        with self.assertRaisesRegex(ValueError,'frozen profile differs'):ip.plan(a)
        a.profile='compact_v1';x=st.read_json(a.run_root/'job_plan.json');x['plan']['planned_updates']=1;st.write_json(a.run_root/'job_plan.json',x)
        with self.assertRaisesRegex(ValueError,'modified record'):ip.plan(a)
    def test_actual_freeze_writes_single_profile_plan_and_offline_validation(self):
        from importlib.metadata import version
        a=self.args();a.profile='compact_v1';a.stage='evaluate'
        f=st.read_json(Path(pr.__file__).with_name('frozen_origins.json'))
        cross=st.read_json(ip.ROOT/'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/frozen_catalog.json')['domains']
        lookup={('background',r['sample_id']):r for r in f['background_samples']}
        for d in pr.DOMAINS[1:]:lookup.update({(d,r['sample_id']):r for r in cross[d]['samples']})
        c=dict(source=[dict(r,source_index=i,calibration=i%8==0,label=0) for i,r in enumerate(f['source'])],
            supports={d:[dict(r,label=0,domain=d) for r in (f['background_supports'] if d=='background' else cross[d]['supports'])] for d in pr.DOMAINS},
            samples=[dict(lookup[r['domain'],r['sample_id']],**r) for r in f['evaluation_GT']],data_root='metadata-only')
        frozen=ip.freeze(a,c,dict(software={n:version(n) for n in ('numpy','scipy','Pillow')}))
        self.assertEqual(ip.plan(a)['planned_updates'],13000)
        self.assertEqual(frozen['resolved_config_seal'],st.sealed_json(a.run_root/'resolved_config.json')['seal'])
        self.assertEqual(frozen['job_plan_seal'],st.sealed_json(a.run_root/'job_plan.json')['seal'])
        runner.validate_runtime_inputs(a)
        flags=runner.common(a);self.assertEqual(flags[flags.index('--profile')+1],'compact_v1')
        inventory=ip.work_inventory(a);self.assertEqual(inventory['normal_feature_forwards_remaining'],545)
        self.assertEqual(inventory['full_grid_calibration_prediction_NN_remaining'],198+926)
    def test_schedule_uses_new_total_not_old_checkpoint_cutoff(self):
        self.assertEqual(int(1000*pr.CONFIG['warmup_fraction']),50)
        self.assertAlmostEqual(training.rate(49,1000),3e-4)
        self.assertAlmostEqual(training.rate(999,1000),pr.CONFIG['minimum_lr'])
        self.assertNotEqual(training.rate(999,1000),training.rate(999,2000))
    def test_estimation_tracks_profile_jobs_and_completed_work(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.smoke import estimate
        head=dict(train_seconds_per_update=.1)
        for profile,updates in (('compact_v1',13000),('full_v1',336000)):
            r=estimate(head,{},pr.resolve(profile));self.assertAlmostEqual(r['components_hours']['head_compute_transfer_lower_bound'],updates*.1/3600)
            self.assertIsNone(r['total_hours']);self.assertIsNone(r['stage_hours']['synthesis_and_training_cache'])
        jobs=pr.job_plan(pr.resolve())['training_jobs'];done=[(j['phase'],j['family'],j['seed']) for j in jobs]
        r=estimate(head,{},inventory=dict(completed_training_jobs=done,normal_feature_forwards_remaining=0,synthetic_forwards_remaining=0))
        self.assertEqual(r['components_hours']['head_compute_transfer_lower_bound'],0)
        self.assertEqual(r['components_hours']['synthetic_RGB_forwards'],0)
    def test_compact_only_builds_requested_training_dataset_and_jobs(self):
        a=self.args();a.profile='compact_v1';calls=[]
        with patch.object(training_data,'Dataset') as dataset,patch.object(training,'job',side_effect=lambda *args,**kw:calls.append(args[1:4])):
            training.train(a,source=True);training.train(a)
        self.assertEqual(len(calls),4);self.assertTrue(all(f=='REF_ST' and s==42 for _,f,s in calls))
        self.assertTrue(all(call.args[2] for call in dataset.call_args_list))


class DataAndReuse(Temporary):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
    def test_sparse_exact_positions_match_full_grid_and_permuted_compact_order(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.retrieval import exact_with_audit
        rng=np.random.default_rng(8);grid=[20,20];positions=rng.permutation(400);q=rng.normal(size=(400,1280)).astype(np.float32);bank=rng.normal(size=(19,1280)).astype(np.float32)
        plans=[dict(bounds=[2,3,8,9]),dict(bounds=[5,6,11,12])];needed=tiles.physical_union(plans,grid)
        self.assertLess(len(needed),72);self.assertEqual(len(needed),len(set(needed)))
        rows=tiles.rows_for(positions,needed,grid);old,idx,_=exact_with_audit(q,bank,'cpu');d,i,_=exact_with_audit(q[rows],bank,'cpu')
        np.testing.assert_allclose(d,old[rows],atol=1e-7,rtol=1e-6);np.testing.assert_array_equal(i,idx[rows])
        with self.assertRaisesRegex(ValueError,'absent'):tiles.rows_for(needed,np.array([399]),grid)
    def test_actual_sparse_preparation_and_Dataset_use_explicit_NN_row_mapping(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import retrieval
        a=self.args();a.profile='compact_v1';c=catalog();c['source']=c['source'][:25];rng=np.random.default_rng(11)
        row=c['supports']['background'][0];st.sealed_json(a.run_root/'catalog.json',dict(catalog=c));st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'))
        bank={f'L{l}':rng.normal(size=(11,1280)).astype(np.float32) for l in pr.LAYERS};st.put(a.cache_root/'source_original',bank,{})
        for origin in sum(c['supports'].values(),[]):
            st.put(ip.normal_path(a,origin),{**{f'q{l}':rng.normal(size=(4,1280)).astype(np.float32) for l in pr.LAYERS},'positions':np.arange(4)},dict(grid=[2,2]))
        # Constructed grids deliberately differ in size; no RGB/backbone is involved.
        grid=[40,40];positions=rng.permutation(1600);q={l:rng.normal(size=(1600,1280)).astype(np.float32) for l in pr.LAYERS};soft=np.zeros((16,16),np.float32);soft[2,2]=1
        item=dict(origin=row['relative_path'],variant=3,item_id='sparse-training',row=row,status='valid');refs=retrieval.References(a)
        record=training_data.prepare_one(a,item,q,dict(grid=grid,positions=positions),soft,refs,catalogue=c)
        self.assertLess(record['NN_patch_count'],1600);needed=np.array(record['NN_physical_positions']);take=tiles.rows_for(positions,needed,grid)
        sx,_=st.get(ip.item_path(a,item)/'source_nn');old,idx,_=retrieval.exact_with_audit(q[7],bank['L7'],'cpu')
        np.testing.assert_allclose(sx['d7'],old[take],atol=1e-7,rtol=1e-6);np.testing.assert_array_equal(sx['i7'],idx[take])
        st.put(a.cache_root/'input_normalization',{**{f'mu{l}':np.zeros(1280,np.float32) for l in pr.LAYERS},**{f'sigma{l}':np.asarray(1.,np.float32) for l in pr.LAYERS}}, {})
        st.sealed_json(a.run_root/'input_distance_mad.json',dict(patch=dict(medians=[0]*4,scales=[1]*4)))
        records=[dict(item=item,path=str(ip.item_path(a,item)),seal=record['seal'])]
        # Other population entries are metadata placeholders, never loaded in this mapping test.
        for i,origin in enumerate(ip.train_source(c)+c['supports']['background']):
            for variant in ((-1,2) if i==0 else (-1,)):
                other=dict(origin=origin['relative_path'],variant=variant,item_id=f'placeholder_{i}_{variant}',row=origin,status='normal' if variant==-1 else 'valid')
                records.append(dict(item=other,path='unused',seal='unused'))
        st.sealed_json(a.run_root/'training_manifest.json',dict(identity='fixture',records=records))
        dataset=training_data.Dataset(a,'background',True);tile=next(t for t in record['tiles'] if t['positive']);sample=dataset.load((item,tile))
        y,x,yy,xx=tile['bounds'];ids=(np.arange(y,yy)[:,None]*grid[1]+np.arange(x,xx)).ravel();rows=tiles.rows_for(positions,ids,grid)
        expected=q[7][rows].reshape(yy-y,xx-x,1280).transpose(2,0,1)
        np.testing.assert_array_equal(sample['values'][0][0,:1280].numpy(),expected)
        expected=(q[7][rows]-bank['L7'][idx[rows]]).reshape(yy-y,xx-x,1280).transpose(2,0,1)
        np.testing.assert_array_equal(sample['values'][0][0,1280:2560].numpy(),expected)
        del sx,refs,dataset
    def test_individual_normal_only_origins_stay_in_balanced_sampling(self):
        a=self.args();a.profile='compact_v1';c=catalog();c['source']=c['source'][:3];c['supports']={d:[] for d in pr.DOMAINS}
        st.sealed_json(a.run_root/'catalog.json',dict(catalog=c));st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'))
        st.put(a.cache_root/'input_normalization',{**{f'mu{l}':np.zeros(1280,np.float32) for l in pr.LAYERS},**{f'sigma{l}':np.asarray(1.,np.float32) for l in pr.LAYERS}}, {})
        records=[];tile=dict(tile=0,bounds=[0,0,2,2],center=[0,0,2,2],positive=True)
        for row in ip.train_source(c):
            for variant in ((-1,2) if row['source_index']==1 else (-1,)):
                item=dict(origin=row['relative_path'],variant=variant,item_id=f"{row['source_index']}_{variant}",row=row,status='normal' if variant==-1 else 'valid')
                p=ip.item_path(a,item);r=st.sealed_json(p/'complete.json',dict(tiles=[tile]));records.append(dict(item=item,path=str(p),seal=r['seal']))
        st.sealed_json(a.run_root/'training_manifest.json',dict(identity='fixture',records=records))
        d=training_data.Dataset(a,'source',False);seen=set()
        for update in range(30):
            choices=d.select(42,update);self.assertEqual([x[0]['variant']==-1 for x in choices],[True,False,True,False])
            seen.update(x[0]['origin'] for x in choices if x[0]['variant']==-1)
            self.assertTrue(all(x[0]['row']['source_index']==1 for x in choices if x[0]['variant']!=-1))
        self.assertEqual(len(seen),2);self.assertEqual(len(d.normal_only),1)
        d.by_origin[c['source'][1]['relative_path']]['synthetic']=[]
        # Population validation is at dataset construction, not hidden regeneration.
        for r in records:
            if r['item']['variant']!=-1:r['item']['origin']='not_in_population'
        # A wholly empty effective population is also blocked before feature preparation.
        st.sealed_json(a.run_root/'synthetic_manifest.json',dict(identity='fixture',items=[]))
        with self.assertRaisesRegex(ValueError,'population has no effective'):training_data.all_items(a)
    def parent_fixture(self):
        parent=self.args('parent');parent.profile='full_v1';current=self.args('compact');current.profile='compact_v1';c=catalog();c['source']=c['source'][:25]
        for a,identity in ((parent,'parent'),(current,'current')):
            st.sealed_json(a.run_root/'inputs.json',dict(identity=identity));st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
        provider=dict(kind='reference_residual',run=parent.run_root,cache=parent.cache_root,identity='parent',catalog=c,
            inputs=dict(config=pr.resolve('full_v1'),artifact_rules=reuse.rules(pr.resolve('full_v1'))))
        return parent,current,c,provider
    def test_reuse_selected_original_RGB_and_labels_without_rewriting_parent(self):
        parent,current,c,provider=self.parent_fixture();row=c['supports']['background'][0]
        item=next(r for r in synthesis.manifest(c,pr.resolve()) if r['origin']==row['relative_path'] and r['kind']=='scratch')
        fg=np.ones((8,8),bool)
        for a in (parent,current):st.put(a.cache_root/'foreground'/ip.key(row),dict(mask=fg),{})
        rng=np.random.default_rng(4);rgb=rng.integers(0,256,(64,64,3),dtype=np.uint8);value,diag=synthesis.generate(rgb,rgb[::-1],fg,item['origin'],item['variant']);self.assertIsNotNone(value)
        from PIL import Image
        p=parent.cache_root/'synthetic_rgb'/item['item_id'];p.mkdir(parents=True);Image.fromarray(value.pop('rgb')).save(p/'image.png')
        value['soft_output']=synthesis.area_resize(value.pop('soft'),(16,16));value['change']=np.round(value['change']*255).astype(np.uint8)
        labels=st.put(p/'labels',value,dict(identity='parent',item=item));record=st.sealed_json(p/'manifest.json',dict(identity='parent',item=item,**diag,rgb_sha256=st.sha(p/'image.png'),label_seal=labels['seal']))
        before=st.sha(p/'manifest.json')
        with patch.object(reuse,'sources',return_value=[provider]):self.assertTrue(reuse.synthetic(current,item))
        dest=current.cache_root/'synthetic_rgb'/item['item_id'];r=st.sealed_json(dest/'manifest.json')
        self.assertEqual(r['origin_seal'],record['seal']);self.assertEqual(st.sha(dest/'image.png'),st.sha(p/'image.png'));self.assertEqual(st.sha(p/'manifest.json'),before)
        self.assertEqual(reuse.content(dest/'labels'),reuse.content(p/'labels'))
        class InterruptedNN:
            @property
            def search(self):raise RuntimeError('fixture interrupt before NN')
        query={l:rng.normal(size=(400,1280)).astype(np.float32) for l in pr.LAYERS};positions=rng.permutation(400)
        entry=dict(item,row=row,status='valid',record_seal=record['seal'])
        with self.assertRaisesRegex(RuntimeError,'fixture interrupt'):
            training_data.prepare_one(parent,entry,query,dict(grid=[20,20],positions=positions),value['soft_output'],InterruptedNN(),catalogue=c)
        self.assertFalse((ip.item_path(parent,entry)/'complete.json').exists())
        with patch.object(reuse,'sources',return_value=[provider]):cached=reuse.tile_query(current,dict(entry,record_seal=r['seal']))
        self.assertIsNotNone(cached);q,g,soft=cached;take=tiles.rows_for(positions,g['positions'],[20,20])
        for l in pr.LAYERS:np.testing.assert_array_equal(q[l],query[l][take])
        np.testing.assert_array_equal(soft,value['soft_output'])
        with self.assertRaisesRegex(RuntimeError,'fixture interrupt'):
            training_data.prepare_one(current,dict(entry,record_seal=r['seal']),q,g,soft,InterruptedNN(),catalogue=c)
        linked=st.sealed_json(ip.item_path(current,entry)/'tiles/0/manifest.json')
        self.assertEqual(linked['metadata']['origin_seal'],st.sealed_json(ip.item_path(parent,entry)/'tiles/0/manifest.json')['seal'])
        original=(p/'image.png').read_bytes();(p/'image.png').write_bytes(original+b'modified')
        with patch.object(reuse,'sources',return_value=[provider]):
            with self.assertRaisesRegex(ValueError,'modified array/file'):reuse.synthetic(current,item)
        with patch.object(reuse,'sources',return_value=[provider]),patch.dict(pr.CONFIG,{'synthesis_seed':123}):
            self.assertFalse(reuse.compatible_rules(provider,pr.resolve(),'synthesis'))
    def test_parent_fold_reuse_ignores_training_budget_but_rejects_changed_sampler(self):
        parent,current,c,provider=self.parent_fixture();expected=dict(identity='current',kind='training_only_opposite_fold',fold='A',random_seed=1,
            origins=['train/one'],origin_indices=[1],sampler='original',budget=7,augmentation='original')
        old=dict(expected,identity='parent',candidate_ledger=[dict(stop=7)])
        p=parent.cache_root/'source_folds/fold_A';st.put(p,dict(L7=np.ones((7,1280),np.float32)),old);before=st.sha(p/'manifest.json')
        with patch.object(reuse,'sources',return_value=[provider]):self.assertTrue(reuse.fold(current,'fold_A',expected))
        self.assertEqual(st.sha(p/'manifest.json'),before)
        with patch.object(reuse,'sources',return_value=[provider]):self.assertFalse(reuse.fold(current,'fold_A',dict(expected,budget=8)))
    def test_complete_compact_status_does_not_wait_for_controls_or_ensemble(self):
        a=self.args();a.profile='compact_v1';st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'));plan=ip.plan(a)
        for job in plan['training_jobs']:
            p=training.path(a,job['phase'],job['family'],job['seed']);p.mkdir(parents=True);(p/'final.pt').write_bytes(b'fixture checkpoint, no training')
            digest=st.sha(p/'final.pt');st.sealed_json(p/'latest.json',dict(sha256=digest,file='final.pt'))
            st.sealed_json(p/'complete.json',dict(inputs=dict(identity='fixture'),updates=job['updates'],checkpoint_sha256=digest,parameter_count=1,checkpoint_bytes=1,seconds=1))
        for d in pr.DOMAINS:
            p=a.run_root/'evaluation'/d;rows=[]
            for b in plan['branches']:
                for rule in ('CH1','CURRENT'):rows.append(dict(domain=d,branch=b,scoring=rule,image_AUROC=.7,image_AP=.7,pixel_AUROC=.7,pixel_AP=.7,AUPRO=''))
                st.sealed_json(a.run_root/'aupro'/d/(b+'.json'),dict(identity='fixture',domain=d,branch=b,map_identity=st.digest([]),nstrips=200,FPR_max=.3,AUPRO=.7,backend='original_ADEval_1.1.0_CUDA'))
            for name,values in (('metrics.csv',rows),('map_manifest.csv',[]),('defect_instances.csv',[]),('numerical_verification.csv',[])):st.write_csv(p/name,values)
        # This fixture verifies reporting/completion only; no substitute AUPRO execution.
        with patch.object(reporting.ev,'restored',return_value=True):status=reporting.summarize(a)
        self.assertEqual(status['status'],'all_five_metrics_complete');self.assertEqual(status['pending'],[])
        self.assertEqual((status['training_jobs'],status['metric_rows'],status['aupro_jobs']['requested']),(4,32,9))
        self.assertTrue(status['not_requested']);rows=reporting.ev.read_csv(a.run_root/'training_repeat_summary.csv')
        self.assertTrue(rows);self.assertTrue(all(r['sample_std']=='NA' for r in rows));self.assertEqual(status['primary'],'REF_ST_SEED42')
    def test_old_long_schedule_checkpoint_not_accepted_as_new_short_schedule(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.model import Head
        torch=self.torch
        class Tiny:
            use_reference=True;manifest_seal='same-pool'
            def select(self,*args,**kwargs):return [0]
            def load(self,_):return dict(values=[torch.ones(1,15,4,4) for _ in pr.LAYERS],soft=np.zeros((4,4),np.float32),origin=[0,0],grid=[4,4],shape=[4,4],center=[0,0,4,4])
        a=self.args();st.sealed_json(a.run_root/'inputs.json',dict(identity='same-identity-to-isolate-schedule'))
        training.job(a,'source','REF_ST',42,dataset=Tiny(),head=Head(4,8),total=4,stop_after=2)
        a.profile='compact_v1'
        with self.assertRaisesRegex(ValueError,'checkpoint|inputs|identity'):
            training.job(a,'source','REF_ST',42,dataset=Tiny(),head=Head(4,8),total=2)
        self.assertFalse((training.path(a,'source','REF_ST',42)/'complete.json').exists())
    def test_import_complete_parent_NN_by_physical_rows_and_raw_feature_reuse(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import retrieval,runtime
        parent=self.args('legacy');current=self.args('compact');current.profile='compact_v1';c=catalog();c['source']=c['source'][:25]
        known=st.read_json(Path(pr.__file__).with_name('legacy_full_v1_compatibility.json'))['code']
        inputs=dict(identity='parent',config=copy.deepcopy(pr.CONFIG),code={'DINOv3/MADEqual/aebad_s_reference_residual_segmentation/'+k:v for k,v in known.items()})
        st.sealed_json(parent.run_root/'inputs.json',inputs);st.sealed_json(current.run_root/'inputs.json',dict(identity='current'))
        for a in (parent,current):st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
        rng=np.random.default_rng(17);positions=np.array([2,0,3,1],np.int64);seals={}
        for row in sum(c['supports'].values(),[]):
            rec=st.put(ip.normal_path(parent,row),{**{f'q{l}':rng.normal(size=(4,1280)).astype(np.float32) for l in pr.LAYERS},'positions':positions},
                dict(identity='parent',image=row,kind='original_complete_grid',grid=[2,2],output_shape=[16,16]));seals[row['relative_path']]=dict(seal=rec['seal'])
        br=st.put(parent.cache_root/'source_original',{f'L{l}':rng.normal(size=(7,1280)).astype(np.float32) for l in pr.LAYERS},dict(identity='parent',kind='original_source'))
        st.sealed_json(parent.run_root/'prepared.json',dict(source_seal=br['seal'],normal_features=seals))
        provider=dict(kind='reference_residual',run=parent.run_root,cache=parent.cache_root,identity='parent',inputs=inputs,catalog=c)
        st.import_bundle(parent.cache_root/'source_original',current.cache_root/'source_original',{f'L{l}':f'L{l}' for l in pr.LAYERS},dict(identity='current'))
        for row in sum(c['supports'].values(),[]):self.assertTrue(runtime.import_normal(current,row,[provider]))
        row=c['supports']['background'][0];item=dict(origin=row['relative_path'],variant=-1,item_id='normal-k7',row=row,status='normal')
        q={l:np.array(st.get(ip.normal_path(parent,row),(f'q{l}',))[0][f'q{l}']) for l in pr.LAYERS};refs=retrieval.References(parent)
        old=training_data.prepare_one(parent,item,q,dict(grid=[2,2],positions=positions),np.zeros((16,16),np.float32),refs,catalogue=c)
        oldpath=ip.item_path(parent,item);before=st.sha(oldpath/'complete.json');del refs
        refs=retrieval.References(current)
        with patch.object(reuse,'sources',return_value=[provider]):new=reuse.training_item(current,item,refs,c,{})
        self.assertIsNotNone(new);self.assertEqual(st.sha(oldpath/'complete.json'),before)
        x,_=st.get(ip.item_path(current,item)/'source_nn');oldx,_=st.get(oldpath/'source_nn');take=tiles.rows_for(positions,x['positions'],[2,2])
        np.testing.assert_array_equal(x['i7'],oldx['i7'][take]);np.testing.assert_array_equal(x['d7'],oldx['d7'][take])
        self.assertEqual(new['origin_seal'],old['seal']);self.assertEqual(new['NN_layout'],'physical_tile_union_v1')
        del x,oldx,refs
        # Hash rejection exercises the actual import path, not just the bundle reader.
        with (oldpath/'source_nn/d7.npy').open('wb') as f:np.save(f,np.full(4,9,np.float32))
        refs=retrieval.References(current)
        with patch.object(reuse,'sources',return_value=[provider]):
            with self.assertRaisesRegex(ValueError,'modified array'):reuse.training_item(current,item,refs,c,{})
        del refs
