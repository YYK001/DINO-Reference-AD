"""Pending-aware seven-way comparisons; no best-test-branch selection."""
from collections import defaultdict
import numpy as np
from .inputs import st
from . import inputs as ip,evaluation as ev
from .protocol import BRANCHES,DOMAINS,U,S,G0,W,GL,ND,ID


def graph_transfer(a):
    path=a.run_root/'graph_summary.csv'
    if not path.exists():return []
    rows=ev.read_csv(path);result=[]
    from .protocol import LAYERS,config_id
    for d in DOMAINS:
        selection_path=a.run_root/'selection'/f'{d}.json'
        if not selection_path.exists():continue
        choice=st.read_json(selection_path)
        configs=[(W,choice['main']['selected']['config'] if choice['main']['selected'] else None),
            (G0,choice['paired_config']),
            (GL,choice['plain']['selected']['config'] if choice['plain']['selected'] else None)]
        for b,cfg in configs:
            for l in LAYERS:
                records=[r for r in rows if r['domain']==d and int(r['layer'])==l and cfg is not None and r['config']==config_id(tuple(cfg))]
                fold=[r for r in records if int(r['fold'])!=0];final=[r for r in records if int(r['fold'])==0]
                plain_records=[r for r in rows if r['domain']==d and int(r['layer'])==l and cfg is not None
                    and r['config']==config_id((cfg[0],cfg[1],None))]
                denominators={int(r['fold']):int(r['edges']) for r in plain_records}
                fold_rates=[int(r['edges'])/denominators[int(r['fold'])] for r in fold
                    if denominators.get(int(r['fold']),0)>0]
                result.append(dict(domain=d,branch=b,layer=l,config=config_id(tuple(cfg)) if cfg else 'EMPTY',
                    seven_image_folds=len(fold) if cfg else 8,
                    seven_image_mean_edges=float(np.mean([float(r['edges']) for r in fold])) if fold else 0,
                    eight_image_edges=int(final[0]['edges']) if final else 0,
                    seven_image_mean_total_raw_length=float(np.mean([float(r['raw_total_length']) for r in fold])) if fold else 0,
                    eight_image_total_raw_length=float(final[0]['raw_total_length']) if final else 0,
                    seven_image_mean_witness_pass_rate=float(np.mean(fold_rates)) if cfg and cfg[2] is not None and fold_rates else None,
                    seven_image_witness_rate_computable_folds=len(fold_rates) if cfg and cfg[2] is not None else 0,
                    eight_image_witness_pass_rate=int(final[0]['edges'])/denominators[0]
                        if cfg and cfg[2] is not None and final and denominators.get(0,0)>0 else None,
                    pass_rate_denominator='same k/length unfiltered G0; empty denominator is NA',
                    note='fold equal means; eight-image graph recomputed, no fixed growth assumption'))
    return result


def margins(instances):
    groups=defaultdict(dict)
    for r in instances:groups[r['domain'],r['sample_id'],r['instance_id']][r['branch']]=r
    records=[]
    for (d,sid,iid),branches in groups.items():
        if set(branches)!=set(BRANCHES):raise ValueError('incomplete seven-way instance alignment')
        for definition in ('margin','fixed_union_margin'):
            for b in BRANCHES:
                def value(key):
                    v=branches[key][definition]
                    return None if v in ('',None) else float(v)
                u,s,c=value(U),value(S),value(b);valid=all(v is not None for v in (u,s,c))
                row=dict(domain=d,sample_id=sid,instance_id=iid,branch=b,definition=definition,
                    small_group=branches[b]['small_group'],response_status=branches[b]['response_status'],
                    computable=valid,UNION=u,SEGMENT=s,candidate=c)
                if valid:
                    loss=s<u-1e-6
                    row.update(change_vs_UNION=c-u,change_vs_SEGMENT=c-s,damaged_vs_UNION=c<u-1e-6,
                        old_loss_improved=loss and c>s+1e-6,new_loss=not loss and c<u-1e-6,
                        old_loss_worsened=loss and c<s-1e-6,fully_repaired=loss and c>=u-1e-6,
                        partially_repaired=loss and c>s+1e-6 and c<u-1e-6)
                records.append(row)
    summary=[]
    for d in DOMAINS:
        for b in BRANCHES:
            for definition in ('margin','fixed_union_margin'):
                for group in ('all','small_original_group'):
                    rows=[r for r in records if (r['domain'],r['branch'],r['definition'])==(d,b,definition)
                          and (group=='all' or str(r['small_group']).lower()=='true')]
                    if not rows:continue
                    valid=[r for r in rows if r['computable']]
                    summary.append(dict(domain=d,branch=b,definition=definition,group=group,instances=len(rows),
                        computable=len(valid),NA=len(rows)-len(valid),
                        **{k:sum(r[k] for r in valid) for k in ('damaged_vs_UNION','old_loss_improved','new_loss',
                            'old_loss_worsened','fully_repaired','partially_repaired')},
                        mean_change_vs_UNION=float(np.mean([r['change_vs_UNION'] for r in valid])) if valid else None))
    return records,summary


def summarize(a):
    metrics=[];tables=defaultdict(list);pending=[];complete_domains=[]
    for d in DOMAINS:
        if not (a.run_root/'evaluation'/d/'status.json').exists():pending.append(d+': CPU');continue
        if not ev.restore(a,d):pending.append(d+': CPU');continue
        complete_domains.append(d)
        local=ev.read_csv(a.run_root/'evaluation'/d/'metrics.csv')
        for r in local:
            for k in ev.METRICS:r[k]=float(r[k]) if r[k] not in ('',None) else None
            path=a.run_root/'aupro'/d/(r['branch']+'.json')
            if path.exists():
                record=st.read_json(path)
                rows=[v for v in ev.read_csv(a.run_root/'evaluation'/d/'map_manifest.csv') if v['branch']==r['branch']]
                ev.validate_aupro(record,dict(identity=ip.identity(a),domain=d,branch=r['branch'],
                    map_identity=st.digest(rows),nstrips=200,FPR_max=.3))
                if record['verified_images']!=int(r['image_count']):raise ValueError('AUPRO image count differs')
                r.update(AUPRO=record['AUPRO'],AUPRO_status='complete_original_ADEval')
            else:pending.append(f'{d}: AUPRO {r["branch"]}')
        if len(local)!=14:raise ValueError('incomplete seven-branch CH1/CURRENT table')
        metrics.extend(local)
        for f in ('per_image_scores','numerical_verification','defect_instances','response_changes',
                  'coverage_audit','image_peak_decomposition','score_locations','ranking_summary','normal_tail'):
            tables[f].extend(ev.read_csv(a.run_root/'evaluation'/d/(f+'.csv')))
    if len(complete_domains)==3:
        for b in BRANCHES:
            for scoring in ('CH1','CURRENT'):
                rows=[r for r in metrics if (r['branch'],r['scoring'])==(b,scoring)]
                metrics.append(dict(domain='macro_3domains',branch=b,scoring=scoring,seed=a.seed,evaluation_scope=a.evaluation_scope,
                    aggregation='three_domains_equal',**{k:float(np.mean([r[k] for r in rows]))
                    if all(r[k] is not None for r in rows) else None for k in ev.METRICS}))
    lookup={(r['domain'],r['scoring'],r['branch']):r for r in metrics};comparisons=[]
    for d in (*complete_domains,*(['macro_3domains'] if len(complete_domains)==3 else [])):
        for scoring in ('CH1','CURRENT'):
            for ref in (U,S,G0,GL,ND,ID):
                for k in ev.METRICS:
                    c=lookup[d,scoring,W][k];r=lookup[d,scoring,ref][k]
                    comparisons.append(dict(domain=d,scoring=scoring,branch=W,reference=ref,metric=k,
                        value=c,reference_value=r,delta=c-r if c is not None and r is not None else None,
                        delta_percentage_points=100*(c-r) if c is not None and r is not None else None,
                        direction='WITNESS_GRAPH minus reference',seed=a.seed,evaluation_scope=a.evaluation_scope))
    mechanism=[]
    for d in complete_domains:
        for b in BRANCHES:
            values={}
            for group,name in (('normal_all_pixels','R_N'),('anomaly_GT_pixels','R_A')):
                rows=[float(r['mean_signed_response_reduction']) for r in tables['response_changes']
                      if (r['domain'],r['branch'],r['group'])==(d,b,group) and r['mean_signed_response_reduction'] not in ('',None)]
                values[name]=float(np.mean(rows)) if rows else None
            mechanism.append(dict(domain=d,branch=b,reference=U,aggregation='image_equal_within_group',
                units='fused_Source_patch_MAD_response',**values,G=values['R_N']-values['R_A']
                if all(v is not None for v in values.values()) else None))
    transfers,summary=margins(tables['defect_instances'])
    for d in DOMAINS:
        if not (a.run_root/'selection'/f'{d}.json').exists():pending.append(d+': selection')
    failed=sum(str(r['verified']).lower() not in ('true','1') for r in tables['numerical_verification'])
    expected=len(ip.catalog(a)['samples'])
    all_complete=not pending and len(metrics)==56 and failed==0
    output=dict(metrics=metrics,metric_comparisons=comparisons,mechanism_summary=mechanism,graph_transfer=graph_transfer(a),
        margin_transitions=transfers,margin_transition_summary=summary,**tables)
    for f,rows in output.items():st.write_csv(a.run_root/(f+'.csv'),rows)
    status=dict(status='all_requested_metrics_complete' if all_complete else 'partial_evidence',
        all_metrics_complete=all_complete,pending=sorted(set(pending)),identity=ip.identity(a),
        images=expected,CPU_domains_complete=complete_domains,numerical_failed_checks=failed,
        seed=a.seed,evaluation_scope=a.evaluation_scope,query_mode=a.query_mode)
    st.write_json(a.run_root/'status.json',status)
    lines=['# 固定Target局部图与第三方正常佐证：冻结实验','',
        '候选图是否新增覆盖、佐证是否超过普通图与缩距控制、检测与定位是否存在取舍，须分别阅读下列表格。',
        '仅正常整图留一选择配置；最高两图CH1均值J、95%最佳降幅复杂度折中。95%不是置信度。',
        '七图折重新计算全局Top-k、尺度和长度分位；正式八图重新构图。完整Source与Target点始终保留。',
        '三域曾用于研究，不是盲测。八支持留一不能认证低FPR；margin下降不等于部署漏检。','',
        f'Seed={a.seed}；评价范围={a.evaluation_scope}；Q模式={a.query_mode}；未与其他Source/MAD历史指标拼接。','',
        '|域|评分|分支|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in metrics:
        lines.append('|'+ '|'.join([r['domain'],r['scoring'],r['branch'],*[
            'pending' if r[k] is None else f'{100*r[k]:.5f}' for k in ev.METRICS]])+'|')
    lines+=['','## 问题A：候选图有没有新增覆盖','',
        '读取coverage_audit.csv的正常高响应/缺陷内部/周边有效新增覆盖、原始降距和MAD响应；'
        '图的端点覆盖率不等于查询新增覆盖。image_peak_decomposition.csv分离峰值移动、固定UNION峰值和CH1两部分。',
        '', '## 问题B：佐证是否超过普通图与缩距控制','',
        '读取WITNESS相对LOCAL_GRAPH_PAIRED、独立正常选择LOCAL_GRAPH_LOO、固定NORMAL_MATCHED_DAMP和逐图IMAGE_MATCHED_DAMP的差值。'
        'lambda只由正常留一拟合；beta是当前查询缩距总量控制。若空图/全零通过导致返回UNION，应报告退化，不能称为缺陷保护。',
        '', '## 问题C：检测与定位是否存在取舍','',
        '结合所有指标、两种margin口径、完全/部分恢复、新增/恶化损失与NA；不因平均指标正向省略局部损失。',
        '',str(status)]
    for d in complete_domains:
        lines += ['',f'### {d}：本轮实际结果']
        for b in (G0,W,GL):
            local=[r for r in tables['coverage_audit'] if r['domain']==d and r['branch']==b and r['group']=='normal_high_response_top1pct']
            fraction=[int(r['effective_new_coverage'])/int(r['patches']) for r in local if int(r['patches'])]
            lines.append(f'- {b}：正常高响应组有效新增覆盖率（图像×层等权）='
                +(f'{100*np.mean(fraction):.5f}%。' if fraction else 'NA。'))
        for ref in (U,G0,GL,ND,ID):
            diff=[r for r in comparisons if (r['domain'],r['scoring'],r['reference'])==(d,'CH1',ref)]
            lines.append('- WITNESS_GRAPH − '+ref+'（百分点）：'+', '.join(
                r['metric']+'='+('pending' if r['delta_percentage_points'] is None else f'{r["delta_percentage_points"]:+.5f}') for r in diff))
        for definition in ('margin','fixed_union_margin'):
            local=[r for r in summary if (r['domain'],r['branch'],r['definition'],r['group'])==(d,W,definition,'all')]
            if local:
                r=local[0];lines.append(f'- {definition}：相对UNION受损{r["damaged_vs_UNION"]}/{r["computable"]}个可计算实例，NA={r["NA"]}；'
                    f'旧损失完全恢复{r["fully_repaired"]}、部分恢复{r["partially_repaired"]}、新增损失{r["new_loss"]}。')
        transfer=output['graph_transfer']
        final=[r for r in transfer if r['domain']==d and r['branch']==W]
        if final and all(r['eight_image_edges']==0 for r in final):
            lines.append('本域最终WITNESS图全为空，WITNESS_GRAPH退化为UNION；不能记作成功保护缺陷，不自动放宽阈值。')
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    st.write_json(a.run_root/'delivery_manifest.json',dict(identity=ip.identity(a),
        files={f+'.csv':st.sha(a.run_root/(f+'.csv')) for f in output}))
    return status
