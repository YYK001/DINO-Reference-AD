"""Pending-aware domain tables, fixed comparisons and explicit loss transfers."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from . import storage as st,inputs as ip,prediction,evaluation
from .method import BASES,NEW,BRANCHES,U,S,W,C,LAYERS

METRICS=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')
def num(v):return None if v in (None,'') else float(v)


def margin_transitions(instances):
    aligned=defaultdict(dict)
    for r in instances:aligned[r['domain'],r['sample_id'],r['instance_id']][r['branch']]=r
    result=[]
    for (d,sid,iid),rows in aligned.items():
        if set(rows)!=set(BRANCHES):raise ValueError('incomplete four-way instance alignment')
        for definition in ('margin','fixed_union_margin'):
            u,s=num(rows[U][definition]),num(rows[S][definition])
            for b in NEW:
                v=num(rows[b][definition]);valid=all(x is not None for x in (u,s,v));row=dict(domain=d,sample_id=sid,instance_id=iid,
                    branch=b,definition=definition,small_group=rows[b]['small_group'],response_status=rows[b]['response_status'],
                    computable=valid,UNION=u,SEGMENT=s,candidate=v)
                if valid:
                    loss=s<u-1e-6;row.update(segment_loss=loss,candidate_change_vs_UNION=v-u,candidate_change_vs_SEGMENT=v-s,
                        damaged_vs_UNION=v<u-1e-6,A_old_loss_improved=loss and v>s+1e-6,
                        B_new_loss=not loss and v<u-1e-6,C_old_loss_worsened=loss and v<s-1e-6,
                        old_loss_fully_repaired=loss and v>=u-1e-6,old_loss_partially_repaired=loss and v>s+1e-6 and v<u-1e-6)
                result.append(row)
    summary=[]
    for d in sorted({v['domain'] for v in result}):
        for b in NEW:
            for definition in ('margin','fixed_union_margin'):
                for group in ('all','original_small_group'):
                    local=[v for v in result if (v['domain'],v['branch'],v['definition'])==(d,b,definition) and (group=='all' or str(v['small_group']).lower()=='true')]
                    valid=[v for v in local if v['computable']];row=dict(domain=d,branch=b,definition=definition,group=group,instances=len(local),computable=len(valid),NA=len(local)-len(valid))
                    for k in ('segment_loss','damaged_vs_UNION','A_old_loss_improved','B_new_loss','C_old_loss_worsened','old_loss_fully_repaired','old_loss_partially_repaired'):row[k]=sum(v[k] for v in valid)
                    for key in ('candidate_change_vs_UNION','candidate_change_vs_SEGMENT'):
                        values=[v[key] for v in valid];row['mean_'+key]=float(np.mean(values)) if values else None
                    summary.append(row)
    return result,summary


def summarize(bundle,out):
    out=st.initialize(out,bundle);metrics=[];instances=[];scores=[];checks=[];responses=[];reductions=[];connections=[];ranks=[];tails=[];loo=[];pending=[];complete_domains=[]
    for d in ip.DOMAINS:
        path=out/'evaluation'/d
        if not (path/'status.json').exists():pending.append(d+': CPU evaluation');continue
        prediction.verified(bundle,out,d)
        if evaluation.restore(bundle,out,d) is None:pending.append(d+': CPU evaluation');continue
        complete_domains.append(d);local=ip.parent.csv_file(path/'metrics.csv')
        for r in local:
            for k in METRICS:r[k]=num(r[k])
            if r['branch'] in NEW:
                file=out/'aupro'/d/(r['branch']+'.json')
                if file.exists():
                    a=st.read_json(file);maps=[v for v in ip.parent.csv_file(path/'map_manifest.csv') if v['branch']==r['branch']]
                    if (a['identity']!=bundle.identity or a['map_identity']!=st.digest(maps) or a['nstrips']!=200 or a['FPR_max']!=.3
                        or a['result_sha256']!=st.digest({k:v for k,v in a.items() if k!='result_sha256'}) or not np.isfinite(a['AUPRO']) or not 0<=a['AUPRO']<=1):raise ValueError('AUPRO identity/value differs')
                    r.update(AUPRO=a['AUPRO'],AUPRO_status='new_original_ADEval_CUDA')
                elif d+': AUPRO '+r['branch'] not in pending:pending.append(d+': AUPRO '+r['branch'])
        metrics.extend(local)
        for f,dest in [('defect_instances.csv',instances),('per_image_scores.csv',scores),('numerical_verification.csv',checks),
                       ('response_changes.csv',responses),('ranking_summary.csv',ranks),('normal_tail.csv',tails)]:dest.extend(ip.parent.csv_file(path/f))
        for f,dest in [('reduction_summary.csv',reductions),('connection_summary.csv',connections)]:dest.extend(ip.parent.csv_file(out/'prediction_tables'/d/f))
    for d in ip.DOMAINS:
        for l in LAYERS:
            path=out/'support_loo'/d/f'L{l}.json'
            if not path.exists():pending.append(f'{d}/L{l}: TARGET_ONLY_LOO_COVERAGE');continue
            saved=st.read_json(path)
            from .supports import load_tables,validate_loo
            _,_,neighbors=load_tables(bundle,out,d,l)
            validate_loo(saved,bundle.identity,d,l,neighbors)
            loo.extend(saved['rows'])
    if len(complete_domains)==3:
        for b in BRANCHES:
            for scoring in ('CH1','CURRENT'):
                local=[v for v in metrics if v['branch']==b and v['scoring']==scoring]
                metrics.append(dict(domain='macro_3domains',branch=b,scoring=scoring,aggregation='three_domains_equal',
                    **{k:float(np.mean([v[k] for v in local])) if all(v[k] is not None for v in local) else None for k in METRICS}))
    lookup={(v['domain'],v['scoring'],v['branch']):v for v in metrics};comparisons=[]
    for d in (*complete_domains,*(['macro_3domains'] if len(complete_domains)==3 else [])):
        for scoring in ('CH1','CURRENT'):
            for ref in (U,S,C):
                for k in METRICS:
                    x,y=lookup[d,scoring,W][k],lookup[d,scoring,ref][k]
                    comparisons.append(dict(domain=d,scoring=scoring,branch=W,reference=ref,metric=k,value=x,reference_value=y,
                        delta=x-y if x is not None and y is not None else None,delta_percentage_points=100*(x-y) if x is not None and y is not None else None,
                        direction='candidate_minus_reference',status='available' if x is not None and y is not None else 'pending_AUPRO'))
    mechanism=[]
    for d in complete_domains:
        for b in BRANCHES:
            values={}
            for group,name in [('normal_all_pixels','R_N'),('anomaly_GT_pixels','R_A'),('anomaly_normal_pixels','R_anomaly_normal')]:
                a=[float(v['mean_signed_response_reduction']) for v in responses if (v['domain'],v['branch'],v['group'])==(d,b,group) and v['mean_signed_response_reduction']!='']
                values[name]=float(np.mean(a)) if a else None
            mechanism.append(dict(domain=d,branch=b,reference=U,units='fused_Source_patch_MAD_response',aggregation='image_equal_within_group',**values,
                G=values['R_N']-values['R_A'] if values['R_N'] is not None and values['R_A'] is not None else None))
    transfers,transfer_summary=margin_transitions(instances)
    tables={'metrics.csv':metrics,'metric_comparisons.csv':comparisons,'per_image_scores.csv':scores,'numerical_verification.csv':checks,
        'reduction_summary.csv':reductions,'connection_summary.csv':connections,'response_changes.csv':responses,'mechanism_summary.csv':mechanism,
        'defect_instances.csv':instances,'margin_transitions.csv':transfers,'margin_transition_summary.csv':transfer_summary,
        'ranking_summary.csv':ranks,'normal_tail.csv':tails,'support_loo_metrics.csv':loo}
    for f,rows in tables.items():st.write_csv(out/f,rows)
    complete=len(complete_domains)==3 and not pending
    state=dict(identity=bundle.identity,status='all_requested_stages_complete' if complete else 'partial_evidence',all_metrics_complete=len(complete_domains)==3 and all(r['AUPRO'] is not None for r in metrics),
        domains_CPU_complete=complete_domains,images=sum(len(prediction.selected(bundle,d)) for d in complete_domains),pending=pending,
        numerical_failed_checks=sum(str(v['verified']).lower() not in ('true','1') for v in checks),model_inference=False,feature_extraction=False)
    st.write_json(out/'status.json',state);report(out,metrics,comparisons,transfer_summary,state)
    if 'recovered_supports' in bundle.frozen:
        recovery=bundle.frozen['recovered_supports']['manifest']
        with (out/'objective_report.md').open('a',encoding='utf-8') as f:
            f.write('\n## 支持恢复来源与复现边界\n\n'
                '本次使用显式重新提取并独立封存的原24张正常支持特征，不把新NPZ冒充原历史字节文件。'
                f'恢复身份：`{recovery["recovery_identity"]}`。'
                '全部926张、四层原缓存的SEGMENT距离与端点长度在原atol=1e-7/rtol=1e-6下通过兼容核查；'
                '原历史特征SHA保留，新文件SHA另记。本次查询仍使用历史距离与索引。'
                '由于原完整Q未保存，不能证明新支持银行对原Q的八支持Top1检索逐元素完全等价；'
                '本轮结果应在这一明确复现边界下解释。详见input_manifest.json中的recovered_supports。\n')
    st.write_json(out/'delivery_manifest.json',dict(identity=bundle.identity,files={f:st.sha(out/f) for f in [*tables,'status.json','objective_report.md']}))
    return state


def report(out,metrics,comparisons,margins,state):
    lines=['# 跨支持图佐证的局部参照扩展：固定 P0 实验','',
        '研究问题：第三张正常支持图对原Top2端点连线内部的有限候选佐证，能否改进连续参照的使用？这是支持端连接筛选，不是全局线段Exact检索。',
        '固定k_w=5、alpha∈[0.25,0.75]、rho≤0.5、至少一个第三方支持来源；三域同一配置，无测试GT调参。',
        '连接仅按正常支持定义，按需缓存。未物化完整支持图结构；不保证整条线段正常、物理同源或缺陷保持，也不主张互近邻、图或留一机制本身首创。',
        '完整Source和K8离散参照始终保留，拒绝回完整UNION。CH1为主，CURRENT为补充，两种评分共享像素图。三个域均参与过开发，不是盲测。','',
        '## 指标（百分比）','', '| 域 | 评分 | 分支 | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AUPRO |','|---|---|---|---:|---:|---:|---:|---:|']
    for v in metrics:lines.append('| '+' | '.join(str(v[k]) for k in ('domain','scoring','branch'))+' | '+' | '.join('pending' if v.get(k) is None else f'{100*v[k]:.4f}' for k in METRICS)+' |')
    lines+=['','## 主候选相对固定参照（百分点）','', '| 域 | 评分 | 参照 | 指标 | 变化 |','|---|---|---|---|---:|']
    for v in comparisons:lines.append('| '+' | '.join(v[k] for k in ('domain','scoring','reference','metric'))+' | '+('pending' if v['delta'] is None else f'{v["delta_percentage_points"]:+.5f}')+' |')
    lines+=['','## 实例损失与均匀缩距对照','',
        'margin下降超过1e-6判为受损。A包含部分恢复，必须与完全恢复、新增损失B、进一步恶化C同时阅读。受损数不是固定阈值漏检数。自身峰值与固定UNION峰值分开；小实例沿原分组，NA保留。',
        '| 域 | 分支 | 位置定义 | 组 | 有效 | NA | 受损 | 恢复A | 新增B | 恶化C | 完全恢复 | 部分恢复 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in margins:lines.append('| '+' | '.join(str(v[k]) for k in ('domain','branch','definition','group','computable','NA','damaged_vs_UNION','A_old_loss_improved','B_new_loss','C_old_loss_worsened','old_loss_fully_repaired','old_loss_partially_repaired'))+' |')
    lines+=['','## 留一与机制解释边界','',
        'support_loo_metrics.csv为TARGET_ONLY_LOO_COVERAGE：整张留出图不作为端点或佐证，且不加载Source。逐图逐层报告覆盖分布，不将该正常覆盖改善当作正式检测收益。',
        'connection_summary.csv同时保留唯一连接接受与query使用位置接受计数；connection缓存可追溯端点、第三图候选、alpha、rho和原因。缩距量匹配只在每图每层原生距离成立，不代表地图、图像分数或margin匹配。',
        'mechanism_summary.csv的R_N/R_A/G为融合Source MAD像素响应、组内图像等权；均值不保证逐实例保持。正常尾部与排序见normal_tail.csv/ranking_summary.csv。',
        'UNION和SEGMENT历史像素指标须在地图/GT SHA、分数和实例核查后才引用，来源列保留。只完成CPU时新增AUPRO仍pending；原200阈值、FPR≤0.3的ADEval另行显式执行。',
        '若仅减少缩距而接近UNION，或不优于同等缩距的均匀对照，不能据此声称佐证筛选有效。必须逐域逐指标结合实例损失判断，不能根据一次正向指标自动追加规则。','',
        '## 完成状态','',json_text(state),'',
        '离线资源与耗时见cost.json，分别列支持检索、连接判定、留一、缓存处理、地图评价；这些不是端到端模型推理延迟。未完成项不得填零或替代。']
    (Path(out)/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def json_text(value):
    import json
    return '```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```'
