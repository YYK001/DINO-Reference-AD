"""Combine separately run domains only under the same joint freeze."""
from pathlib import Path
from . import inputs as ip,analysis
from .verification import audit
from .aupro import metric_rows,_numeric,verified_result
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import protect_dependencies


def run(ctx,directories,output):
    metrics=[];rankings=[];mechanisms=[];repairs=[];parents=[];seen=set();complete=True;checks=[];historical=[]
    for directory in directories:
        root=Path(directory).resolve();status=ip.read_json(root/'status.json')
        if status['status']=='all_requested_metrics_complete':
            root,m=verified_result(ctx,root);cpu=Path(m['cpu_directory']);checksum=ip.sha(root/'result_manifest.json')
        elif status['status']=='cpu_evaluation_complete_aupro_pending':
            # CPU analysis manifest stores an absolute parent solely to locate it;
            # identity uses hashes, allowing explicit paths at normal CLI stages.
            m=ip.read_json(root/'analysis_manifest.json')
            root,m=analysis.verified_analysis(ctx,m['prediction_directory'],root);cpu=root;checksum=ip.sha(root/'analysis_manifest.json');complete=False
        else:raise ValueError('only verified complete CPU/AUPRO per-domain results may be summarized')
        domains=set(m['domains'])
        if seen&domains:raise ValueError('duplicate domain results; no selecting a best run')
        seen|=domains
        metrics.extend(r for r in metric_rows(root/'metrics.csv') if r['domain'] in domains)
        rankings.extend(ip.csv_file(cpu/'ranking_summary.csv'))
        mechanisms.extend(_numeric(ip.csv_file(cpu/'mechanism_summary.csv')))
        repairs.extend(_numeric(ip.csv_file(cpu/'margin_repair_summary.csv')))
        checks.extend(ip.csv_file(cpu/'numerical_verification.csv'))
        historical.extend(ip.csv_file(cpu/'historical_baseline_comparison.csv'))
        parents.append(dict(directory=str(root),sha256=checksum,domains=sorted(domains)))
    if seen!=set(ip.DOMAINS):raise ValueError('both complete validation domains required; background is excluded')
    if ip.context(ctx['paths'])['identity']!=ctx['identity']:raise ValueError('joint freeze changed during summary')
    protect_dependencies(output,*[Path(r['directory']) for r in parents]);out=ip.new_output(ctx,output,'summarize')
    metrics=analysis.append_macro(metrics)
    for name,rows in (('metrics.csv',metrics),('metric_comparisons.csv',analysis.metric_comparisons(metrics)),
                      ('ranking_summary.csv',rankings),('mechanism_summary.csv',mechanisms),('margin_repair_summary.csv',repairs)):
        ip.write_csv(out/name,rows)
    analysis.write_report(out,metrics,rankings,mechanisms,repairs,ip.DOMAINS)
    ip.write_json(out/'historical_score_audit.json',audit(checks))
    ip.write_csv(out/'historical_baseline_comparison.csv',historical)
    ip.write_csv(out/'background_development_reference.csv',[dict(r,domain='background',role='development_only_excluded_from_macro') for r in ctx['catalog']['background_reference']['rows']])
    ip.write_json(out/'status.json',dict(tool=ip.TOOL,status='two_domain_summary_complete' if complete else 'two_domain_CPU_summary_AUPRO_pending',
        identity=ctx['identity'],domains=list(ip.DOMAINS),all_metrics_complete=complete,parents=parents,historical_score_audit=audit(checks),
        output_sha256={p.name:ip.sha(p) for p in out.iterdir() if p.is_file() and p.name!='status.json'}))
    return out
