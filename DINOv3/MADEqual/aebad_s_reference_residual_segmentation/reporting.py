"""Actual metrics only; fixed primary, controls, repeats and explicit pending state."""
import numpy as np
import os
from pathlib import Path
from types import FunctionType
from . import storage as st,inputs as ip,evaluation as ev,training
from .protocol import DOMAINS,U,S,COMBINED,member,ensemble,job_plan,resolve
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import reporting as previous

METRICS=previous.METRICS
combine=previous.combine


def storage_usage(root,scope):
    """One final metadata walk, no payload reads, no follow into linked parents."""
    root=Path(root);groups={};seen=set()
    for current,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(current)/d).is_symlink()]
        for name in files:
            p=Path(current)/name;relative=p.relative_to(root);category=relative.parts[0] if len(relative.parts)>1 else 'root_files'
            row=groups.setdefault(category,dict(scope=scope,category=category,files=0,logical_regular_bytes=0,unique_inode_allocated_bytes=0,
                hardlinked_logical_bytes=0,symlinks_not_followed=0))
            row['files']+=1
            if p.is_symlink():row['symlinks_not_followed']+=1;continue
            info=p.stat();row['logical_regular_bytes']+=info.st_size
            if info.st_nlink>1:row['hardlinked_logical_bytes']+=info.st_size
            key=(info.st_dev,info.st_ino)
            if key not in seen:
                seen.add(key);row['unique_inode_allocated_bytes']+=getattr(info,'st_blocks',0)*512
    return list(groups.values())


def comparisons(rows,plan=None):
    plan=job_plan(resolve()) if plan is None else plan
    index={(r['domain'],r['scoring'],r['branch']):r for r in rows};out=[]
    for row in rows:
        if row['branch'] not in (*plan['members'],*plan['ensembles']):continue
        refs=[U,S,COMBINED]
        if row['branch']==plan['primary']:refs.extend((ensemble('NOREF_ST'),ensemble('REF_SOURCE_QUERY')))
        elif row['branch'].startswith('REF_ST_SEED'):
            seed=int(row['branch'].split('SEED')[-1]);refs.extend((member('NOREF_ST',seed),member('REF_SOURCE_QUERY',seed)))
        for ref in refs:
            base=index.get((row['domain'],row['scoring'],ref))
            if base is None:continue
            for k in METRICS:
                delta=100*(row[k]-base[k]) if row[k] is not None and base[k] is not None else None
                out.append(dict(domain=row['domain'],scoring=row['scoring'],branch=row['branch'],reference=ref,metric=k,
                    delta=delta,units='percentage_points',direction='branch_minus_reference',status='complete' if delta is not None else 'pending'))
    return out


def training_completion(a):
    pending=[];jobs=[]
    for job in ip.plan(a)['training_jobs']:
        phase,family,seed=(job[k] for k in ('phase','family','seed'));p=training.path(a,phase,family,seed)/'complete.json'
        if not p.exists():pending.append(f'{phase}/{family}/seed{seed}: training');continue
        r=st.sealed_json(p);latest=st.sealed_json(p.parent/'latest.json')
        if (r['inputs']['identity']!=ip.identity(a) or r['updates']!=job['updates']
                or r['checkpoint_sha256']!=latest['sha256'] or st.sha(p.parent/latest['file'])!=latest['sha256']):raise ValueError('incomplete/modified final checkpoint')
        jobs.append(dict(phase=phase,family=family,seed=seed,**{k:r[k] for k in ('updates','parameter_count','checkpoint_bytes','seconds')}))
    return pending,jobs


def summarize(a):
    rows=[];instances=[];pending=[];failed=0
    cfg=ip.config(a);plan=ip.plan(a);branches=plan['branches'];primary=plan['primary'];seeds=cfg['training_seeds'];families=cfg['active_families']
    aupro_counts=dict(requested=len(plan['localization_jobs']),reused=0,newly_computed=0)
    for d in DOMAINS:
        if not ev.restored(a,d):pending.append(d+': CPU');continue
        local=ev.read_csv(a.run_root/'evaluation'/d/'metrics.csv')
        if {(r['branch'],r['scoring']) for r in local}!={(b,s) for b in branches for s in ('CH1','CURRENT')}:raise ValueError('CPU metric rows incomplete')
        instances.extend(ev.read_csv(a.run_root/'evaluation'/d/'defect_instances.csv'))
        failed+=sum(r['verified'].lower()!='true' for r in ev.read_csv(a.run_root/'evaluation'/d/'numerical_verification.csv'))
        for r in local:
            for k in METRICS:r[k]=float(r[k]) if r[k] not in ('',None) else None
            p=a.run_root/'aupro'/d/(r['branch']+'.json')
            if p.exists():
                from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
                maps=[v for v in ev.read_csv(a.run_root/'evaluation'/d/'map_manifest.csv') if v['branch']==r['branch']]
                record=st.read_json(p);validate_aupro(record,dict(identity=ip.identity(a),domain=d,branch=r['branch'],map_identity=st.digest(maps),nstrips=200,FPR_max=.3))
                r['AUPRO']=record['AUPRO'];r['AUPRO_status']='complete_original_ADEval';r['AUPRO_origin']=record.get('origin','this_run')
                if r['scoring']=='CH1':aupro_counts['reused' if record.get('origin')=='exact_parent_map_GT_population' else 'newly_computed']+=1
            else:pending.append(d+'/'+r['branch']+': AUPRO')
        rows.extend(local)
    if len(rows)==len(DOMAINS)*len(branches)*2:
        for b in branches:
            for rule in ('CH1','CURRENT'):
                v=[r for r in rows if r['branch']==b and r['scoring']==rule]
                rows.append(dict(domain='macro_3domains',branch=b,scoring=rule,aggregation='three_domains_equal',
                    **{k:float(np.mean([r[k] for r in v])) if all(r[k] is not None for r in v) else None for k in METRICS}))
    rows=combine(rows);st.write_csv(a.run_root/'metrics.csv',rows);delta=comparisons(rows,plan);st.write_csv(a.run_root/'metric_comparisons.csv',delta)
    repeats=[]
    for d in (*DOMAINS,'macro_3domains'):
        for rule in ('CH1','CURRENT'):
            for family in families:
                v=[r for r in rows if (r['domain'],r['scoring'])==(d,rule) and r['branch'] in [member(family,s) for s in seeds]]
                if len(v)!=len(seeds):continue
                for k in METRICS:
                    valid=all(r[k] is not None for r in v);x=[r[k] for r in v]
                    repeats.append(dict(domain=d,scoring=rule,family=family,metric=k,training_seeds=','.join(map(str,seeds)),
                        mean=float(np.mean(x)) if valid else None,sample_std='NA' if len(seeds)==1 else float(np.std(x,ddof=1)) if valid else None,
                        minimum=min(x) if valid else None,maximum=max(x) if valid else None,
                        interpretation='single_seed_std_NA_no_stability_claim' if len(seeds)==1 else 'training repeats on identical support/synthesis pool; not ensemble metric'))
    st.write_csv(a.run_root/'training_repeat_summary.csv',repeats)
    summary=[]
    if instances:
        fn=previous.margins;detail,summary=FunctionType(fn.__code__,dict(fn.__globals__,BRANCHES=branches,DOMAINS=DOMAINS,U=U),fn.__name__)(instances)
        st.write_csv(a.run_root/'defect_instances.csv',instances);st.write_csv(a.run_root/'margin_changes.csv',detail);st.write_csv(a.run_root/'margin_summary.csv',summary)
    tp,jobs=training_completion(a);pending.extend(tp);st.write_csv(a.run_root/'training_jobs.csv',jobs)
    if failed:pending.append('numerical_failed_checks='+str(failed))
    value=dict(status='all_five_metrics_complete' if not pending and len(rows)==plan['expected_metric_rows'] and len(jobs)==len(plan['training_jobs']) else 'pending',identity=ip.identity(a),
        profile=cfg['profile'],primary=primary,metric_rows=len(rows),training_jobs=len(jobs),pending=sorted(set(pending)),numerical_failed_checks=failed,
        not_requested=plan['not_requested'],aupro_jobs=aupro_counts,
        raw_NN_distances_changed=False,training_repeats_are_not_support_repeats=True)
    st.write_json(a.run_root/'status.json',value)
    st.write_csv(a.run_root/'storage_usage.csv',storage_usage(a.cache_root,'cache')+storage_usage(a.run_root,'run'))
    report=['# 真实Target-normal支持下的参照残差分割学习','',
        '这是借鉴参照残差与合成监督的新工程候选，不是ResAD/MemSeg/NSA官方复现。三域曾用于方法开发，不是盲测。',
        f"配置{cfg['profile']}，主候选{primary}；CH1为主表，CURRENT为补充。训练种子{seeds}，表中为%；差值为候选减参照的百分点。",'',
        '|域|分支|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['scoring']=='CH1':report.append('|'+r['domain']+'|'+r['branch']+'|'+'|'.join('pending' if r[k] is None else f'{100*r[k]:.5f}' for k in METRICS)+'|')
    report+=['','## 固定比较与实际取舍','']
    for d in (*DOMAINS,'macro_3domains'):
        for ref in (U,S,COMBINED,*[b for b in (ensemble('NOREF_ST'),ensemble('REF_SOURCE_QUERY')) if b in branches]):
            v=[r for r in delta if (r['domain'],r['scoring'],r['branch'],r['reference'])==(d,'CH1',primary,ref)]
            if len(v)!=5 or any(r['delta'] is None for r in v):report.append(f'- {d}/{ref}：pending，不作完整优势判断。');continue
            report.append(f'- {d}/{ref}：'+', '.join(f'{r["metric"]} {r["delta"]:+.5f}' for r in v)+'；'+
                ('五项均改善，仍需结合实例分布；单种子不作稳定性判断。' if all(r['delta']>0 for r in v) else '未形成五项一致提高，不能省略下降项。'))
    for r in summary:
        if r['branch']==primary and r['group']=='all':report.append(f'- {r["domain"]}/{r["definition"]}：相对UNION受损{r["damaged"]}，NA={r["NA"]}。')
    report+=['','## 边界与资源','',
        '新logit在域K8条件下用66张未训练Source图独立校准；原距离校准保持Source-only。独立尺度间margin差值仅为辅助描述，不等于信息损失或部署漏检。',
        '原始NN距离没有变化，因此不把新响应变化称为原始特征距离的R_N/R_A/G。',
        '未请求控制分支/集成标为not_requested，不是pending；其机制问题本轮不作结论。',
        '启用集成时逐层平均成员z，另用image MAD；单种子标准差为NA，不宣称跨种子稳定。',
        'SEGMENT_SCORE_UNION_MAP只组合既定图像/定位任务结果。共享定位图的CH1/CURRENT只计算一次像素评价。',
        '参数量、checkpoint字节和训练累计时间见training_jobs.csv；阶段耗时、峰值RAM/显存和新增数组字节见costs/。',
        '输入复用与重复工作见reuse_audit.json、prepared.json、synthetic_manifest.json、training_manifest.json。',
        '当前预算下结果不等于否定所有更充分训练；不根据测试结果自动延长训练、改参数、挑checkpoint、挑种子或按域切换分支。','',str(value)]
    (a.run_root/'objective_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8');return value
