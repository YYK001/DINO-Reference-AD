"""Shift/category-balanced summaries, ARD and review packaging."""
from __future__ import annotations
import json
import tarfile
import time
from pathlib import Path
import numpy as np
from . import inputs as ip,evaluation as ev,prediction as pr,calibration


def hierarchical(rows,expected):
    """Never substitute equal weighting over 11 shifts for two-category mean."""
    result=list(rows);categories={};metrics=ev.METRICS
    for category,shifts in expected.items():
        for branch in ip.BRANCHES:
            selected=[r for r in rows if r['category']==category and r['branch']==branch and r['shift'] in shifts]
            if len(selected)!=len(shifts) or {r['shift'] for r in selected}!=set(shifts):continue
            macro=dict(category=category,shift='macro',branch=branch,scope='category_shift_equal_macro',shifts=len(shifts))
            for k in metrics:macro[k]=float(np.mean([r[k] for r in selected])) if all(r.get(k) is not None for r in selected) else None
            categories[(category,branch)]=macro;result.append(macro)
    if set(expected)==set(ip.CATEGORIES):
        for branch in ip.BRANCHES:
            selected=[categories[(c,branch)] for c in ip.CATEGORIES if (c,branch) in categories]
            if len(selected)!=2:continue
            result.append(dict(category='macro_MetalParts_PCB',shift='macro',branch=branch,scope='category_equal_macro',
                **{k:float(np.mean([r[k] for r in selected])) if all(r.get(k) is not None for r in selected) else None for k in metrics}))
    return result


def ard_rows(targets,sources):
    refs={r['category']:r for r in sources if r['shift']=='source' and r['branch']==pr.SOURCE};rows=[]
    for r in targets:
        source=refs.get(r['category'])
        for metric in ev.METRICS:
            value=r.get(metric);baseline=source.get(metric) if source else None
            rows.append(dict(category=r['category'],shift=r['shift'],branch=r['branch'],metric=metric,
                target_metric=value,source_metric=baseline,ARD=min(0.,value-baseline) if value is not None and baseline is not None else None,
                status='available' if value is not None and baseline is not None else 'pending_Source_or_metric',
                Source_reference='same_category_SOURCE_CH1_source_evaluation_without_Target_support',
                adaptation='fixed_shift_K8_Target_normal',units='fraction'))
    return rows


def ard_macros(rows,expected):
    out=list(rows);cat_rows={}
    for category,shifts in expected.items():
        for branch in ip.BRANCHES:
            for metric in ev.METRICS:
                rr=[r for r in rows if r['category']==category and r['branch']==branch and r['metric']==metric]
                if len(rr)!=len(shifts) or {r['shift'] for r in rr}!=set(shifts):continue
                valid=all(r['ARD'] is not None for r in rr)
                r=dict(category=category,shift='macro',branch=branch,metric=metric,ARD=float(np.mean([r['ARD'] for r in rr])) if valid else None,
                    scope='mean_shiftwise_clipped_ARD',status='available' if valid else 'pending_Source_or_metric')
                out.append(r);cat_rows[(category,branch,metric)]=r
    if set(expected)==set(ip.CATEGORIES):
        for branch in ip.BRANCHES:
            for metric in ev.METRICS:
                rr=[cat_rows[(c,branch,metric)] for c in ip.CATEGORIES if (c,branch,metric) in cat_rows]
                if len(rr)!=2:continue
                valid=all(r['ARD'] is not None for r in rr)
                out.append(dict(category='macro_MetalParts_PCB',shift='macro',branch=branch,metric=metric,
                    ARD=float(np.mean([r['ARD'] for r in rr])) if valid else None,scope='category_equal_macro',status='available' if valid else 'pending_Source_or_metric'))
    return out


def summarize(contexts,tag,out):
    out=Path(out)
    if out.exists():raise FileExistsError('summary must be fresh')
    out.mkdir(parents=True);targets=[];sources=[];pending=[];parents=[];diagnostics={};cache_index=[]
    expected={c:tuple(ip.COUNTS[c]) for c in ip.CATEGORIES}
    for ctx in contexts:
        category=ctx['category'];state=ip.ledger(ctx)
        for r in state['records']:cache_index.append(dict(r,bytes=(ctx['root']/r['file']).stat().st_size,root=str(ctx['root'])))
        for group in ('source',*ctx['cat']['shifts']):
            selected=None
            for stage in ('aupro','cpu'):
                candidate=ev.directory(ctx,group,tag,stage)
                if (candidate/'status.json').is_file():
                    selected,_=ev.verified(ctx,group,tag,stage);break
            if selected is None:
                pending.append(dict(category=category,shift=group,reason='missing_completed_evaluation'));continue
            rr=ev.numeric_rows(selected/'metrics.csv')
            for r in rr:
                if any(r.get(k) is None for k in ev.METRICS):pending.append(dict(category=category,shift=group,reason='AUPRO_or_metric_pending'))
            (sources if group=='source' else targets).extend(rr)
            parents.append(dict(category=category,shift=group,directory=str(selected),manifest_sha256=ip.sha(selected/'manifest.json')))
            cpu=ev.directory(ctx,group,tag,'cpu')
            for name in ('per_image_scores.csv','ranking_summary.csv','mechanism_summary.csv','mechanism_per_image.csv','defect_instances.csv',
                         'small_instance_groups.csv','margin_repair_per_instance.csv','margin_repair_summary.csv','historical_baseline_comparison.csv','cases.csv'):
                if (cpu/name).is_file():diagnostics.setdefault(name,[]).extend(ip.read_csv(cpu/name))
    metrics=hierarchical(targets,expected);ard=ard_macros(ard_rows(targets,sources),expected)
    complete=(len(targets)==33 and len(sources)==2 and not pending and len(contexts)==2
              and all(r['ARD'] is not None for r in ard))
    comparisons=[]
    for r in metrics:
        for ref in (ip.BRANCHES[0],ip.BRANCHES[1]):
            if (r['branch'],ref) not in ((ip.BRANCHES[1],ip.BRANCHES[0]),(ip.BRANCHES[2],ip.BRANCHES[0]),(ip.BRANCHES[2],ip.BRANCHES[1])):continue
            other=next((s for s in metrics if s['category']==r['category'] and s['shift']==r['shift'] and s['branch']==ref),None)
            if other:
                for k in ev.METRICS:comparisons.append(dict(category=r['category'],shift=r['shift'],branch=r['branch'],reference=ref,metric=k,
                    delta=r[k]-other[k] if r[k] is not None and other[k] is not None else None,direction='branch_minus_reference',units='fraction'))
    for name,rows in dict(diagnostics,**{'metrics.csv':metrics,'source_reference_metrics.csv':sources,'ARD.csv':ard,
        'metric_comparisons.csv':comparisons,'pending.csv':pending,'cache_index.csv':cache_index}).items():ip.write_csv(out/name,rows)
    ip.write_json(out/'inputs.json',dict(protocol=ip.PROTOCOL,catalog_sha256=ip.CATALOG_SHA,
        categories={c['category']:c['identity'] for c in contexts},parents=parents,evaluation_tag=tag,
        Source_bank_provenance={c['category']:c['frozen'].get('bank_provenance','original_tree_first_enrollment') for c in contexts}))
    volume=[]
    for ctx in contexts:
        for sub in ('raw_cache','support_features','source_calibration_raw','source_evaluation_raw','predictions'):
            files=list((ctx['root']/sub).rglob('*.npz'))
            volume.append(dict(category=ctx['category'],kind=sub,root=str(ctx['root']/sub),files=len(files),bytes=sum(p.stat().st_size for p in files),preserved=True))
    ip.write_json(out/'large_cache_inventory.json',volume)
    lines=['# RobustAD MetalParts / PCB 冻结 SEGMENT / PAIR 验证','',
        '状态：'+('全部要求指标及ARD完成。' if complete else '部分结果；缺项见pending.csv，ARD可能待Source参照或AUPRO。'),
        '固定K8 Target-normal，Source参照为各类别独立SOURCE_CH1测试集；23张校准图不进入Source测试集。',
        '逐shift等权形成类别macro，再对MetalParts/PCB等权；不混入PiledBags或AeBAD。',
        '历史分数兼容阈值1e-4，严格1e-5结果另存；兼容通过不代表历史距离逐元素复现。','',
        '| category | shift | branch | I-AUROC | I-AP | P-AUROC | P-AP | AUPRO |','|---|---|---|---:|---:|---:|---:|---:|']
    for ctx in contexts:
        if 'rebuilt_source_root' in ctx['paths']:
            lines.insert(1,ctx['category']+'：使用独立重建Source库；原构库入口/seed42/每层100K。旧bank字节身份不可证明；历史校准与分数检查另存，不沿用旧距离缓存。')
    for r in metrics:lines.append('| '+' | '.join([r['category'],r['shift'],r['branch'],*['NA' if r.get(k) is None else f'{r[k]:.9f}' for k in ev.METRICS]])+' |')
    lines+=['','指标单位fraction。各shift的改善与下降均保留；比较顺序为SEGMENT对UNION、PAIR对SEGMENT、PAIR对UNION。',
        'macro不能替代逐shift结果。实例自身峰值和固定UNION峰值分别计数；修复与新增损失并报，映射消失/空环带保留NA。',
        'ARD先对每shift计算min(0,Target−Source)，再按上述层级平均。Source只用原库，Target三分支使用对应固定K8支持。',
        '结论只对应当前支持集与配置，不宣称超过SOTA或缺陷保持已经解决。']
    (out/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    ip.write_json(out/'status.json',dict(tool=ip.TOOL,status='all_requested_metrics_complete' if complete else 'partial_results_ARD_or_metrics_pending',
        all_requested_metrics_complete=complete,target_images=sum(int(r['images']) for r in targets if r['branch']==ip.BRANCHES[0]),
        target_shift_rows=len(targets),Source_reference_rows=len(sources),pending=pending,
        files={p.name:ip.sha(p) for p in out.iterdir() if p.is_file()}))
    return out


def package(contexts,summary,output,raw_category=None):
    summary=Path(summary);output=Path(output)
    if output.exists() or output.with_suffix(output.suffix+'.tmp').exists():raise FileExistsError('archive output must be fresh')
    status=ip.read_json(summary/'status.json')
    for name,h in status['files'].items():
        if ip.sha(summary/ip.safe(name))!=h:raise ValueError('review summary changed')
    snapshots=ip.read_json(summary/'inputs.json')
    if snapshots['categories']!={c['category']:c['identity'] for c in contexts}:raise ValueError('package context/summary identity mismatch')
    tag=snapshots['evaluation_tag']
    by_category={c['category']:c for c in contexts}
    for parent in snapshots['parents']:
        ctx=by_category[parent['category']];stage=Path(parent['directory']).name
        directory,_=ev.verified(ctx,parent['shift'],tag,stage)
        if directory.resolve()!=Path(parent['directory']).resolve() or ip.sha(directory/'manifest.json')!=parent['manifest_sha256']:
            raise ValueError('packaging parent evaluation changed since summary')
    for ctx in contexts:
        ip.ledger(ctx)
    selected={}
    def add(path,name):
        path=Path(path)
        if path.is_file():selected[name]=path
    for path in summary.rglob('*'):add(path,'summary/'+path.relative_to(summary).as_posix())
    add(Path(__file__).with_name('frozen_catalog.json'),'frozen_inputs/frozen_catalog.json')
    for ctx in contexts:
        cat=ctx['category'];root=ctx['root']
        if raw_category is None:
            if 'rebuilt_source_root' in ctx['paths']:
                from . import rebuild
                rebuilt=rebuild.directory(ctx['paths'],cat)
                rebuild.verify(ctx['paths'],cat)
                for name in ('rebuild_status.json','rebuild_plan.json','rebuild_failure.json','memory/selected_candidate_indices.npz'):
                    add(rebuilt/name,'frozen_inputs/'+cat+'/reconstructed_Source/'+name)
            for name in ip.FORMAL_FILES:add(ctx['paths']['formal_'+cat.lower()]/name,'frozen_inputs/'+cat+'/fewshot/'+name)
            for name in ('complete.json','memory/memory_provenance.json','calibration/calibration_state.npz'):
                add(ctx['source']/name,'frozen_inputs/'+cat+'/Source/'+name)
        if raw_category is None:
            for path in root.rglob('*'):
                if path.is_file() and path.suffix.lower() in ('.csv','.json','.md','.log','.png'):
                    if 'evaluations' in path.relative_to(root).parts and not path.is_relative_to(root/'evaluations'/tag):continue
                    add(path,cat+'/'+path.relative_to(root).as_posix())
        elif cat==raw_category:
            for sub in ('raw_cache','support_features','source_calibration_raw','source_evaluation_raw','predictions'):
                for path in (root/sub).rglob('*'):add(path,cat+'/'+path.relative_to(root).as_posix())
            for name in ('inputs.json','protocol.json','cache_status.json','image_calibration_state.json','calibration_status.json'):add(root/name,cat+'/'+name)
    if raw_category is not None and raw_category not in {c['category'] for c in contexts}:raise ValueError('raw category outside scope')
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix(output.suffix+'.tmp')
    with tarfile.open(temporary,'w' if raw_category else 'w:gz',**({} if raw_category else {'compresslevel':1})) as tar:
        for name,path in sorted(selected.items()):tar.add(path,arcname=name,recursive=False)
    temporary.replace(output)
    return dict(archive=str(output),bytes=output.stat().st_size,kind='original_cache_separate' if raw_category else 'review_tables_calibration_cases_logs',
                original_caches_deleted=False,all_requested_metrics_complete=status['all_requested_metrics_complete'])
