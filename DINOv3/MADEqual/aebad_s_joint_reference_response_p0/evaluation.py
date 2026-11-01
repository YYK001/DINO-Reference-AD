"""24 independent exact pixel/AUPRO jobs, shared by the two image readouts."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import time
import numpy as np
from . import inputs as ip, prediction as pred
from .inputs import st, GTData
from .protocol import *
from .budget import observe
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as backend
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro


def destination(a, d, seed, branch): return ip.unit(a, d, seed)/'evaluation'/branch


def completed(a, d, seed, branch):
    path = destination(a, d, seed, branch)/'status.json'
    if not path.exists():return None
    r = st.sealed_json(path)
    if any(r.get(k) != v for k, v in ip.signature(a, d, seed).items()) or r['branch'] != branch or r['prediction_seal'] != pred.global_freeze(a)['seal'] or r['images'] != len(ip.parent(a).samples(d)):
        raise ValueError('CPU result input identity differs')
    for name, expected in r['files'].items():
        if st.sha(path.parent/name) != expected:raise ValueError('modified small CPU result: '+name)
    return r


@contextmanager
def scratch(a, d, seed, branch, count):
    root = a.evaluation_scratch/d/f'seed{seed}'/branch;root.mkdir(parents=True, exist_ok=True)
    arrays = []
    with tempfile.TemporaryDirectory(prefix='joint_', dir=root) as tmp:
        try:
            for name, dtype in (('labels', np.uint8), ('scores', np.float32)):
                arrays.append(np.memmap(Path(tmp)/(name+'.bin'), mode='w+', dtype=dtype, shape=(count,)))
            yield arrays
        finally:
            for x in arrays:x._mmap.close()


def mask(a, data, d, seed, row, shape):
    value = backend.shared.resized_mask(data.gt(row), shape).astype(np.uint8)
    old = ip.parent(a).maps[d, seed][row['sample_id'], U]
    if backend.shared.array_sha(value) != old['mask_uint8_sha256']:
        raise ValueError('parent common GT/resize identity differs')
    return value


def evaluate(a):
    frozen = pred.global_freeze(a);p = ip.parent(a);jobs = []
    with ExitStack() as stack:
        data = None
        for d in a.domains:
            for seed in a.support_seeds:
                for b in MAPS:
                    if completed(a, d, seed, b):
                        jobs.append(dict(domain=d, support_seed=seed, branch=b, reused=True));continue
                    if data is None:data = stack.enter_context(GTData(a.data))
                    samples = p.samples(d);n = sum((s['image_height']//4)*(s['image_width']//4) for s in samples)
                    out = destination(a, d, seed, b);records = [];scores = [];offset = 0
                    with scratch(a, d, seed, b, n) as (labels, values):
                        for i, row in enumerate(samples, 1):
                            images, sc, meta = pred.replay(a, d, seed, row, only=b);image = images[b]
                            gt = mask(a, data, d, seed, row, meta['output_shape'])
                            if image.shape != gt.shape or not np.isfinite(image).all():raise ValueError('invalid evaluation map')
                            values[offset:offset+gt.size] = image.ravel();labels[offset:offset+gt.size] = gt.ravel();offset += gt.size
                            records.append(dict(sample_id=row['sample_id'], branch=b, output_height=gt.shape[0], output_width=gt.shape[1],
                                                map_float32_sha256=backend.shared.array_sha(image), mask_uint8_sha256=backend.shared.array_sha(gt)))
                            scores.append(dict(sample_id=row['sample_id'], label=row['label'], **sc[b]))
                            if i % 50 == 0 or i == len(samples):print(f'[joint exact CPU] {d}/seed{seed}/{b} {i}/{len(samples)}', flush=True)
                            observe(a)
                        if offset != n:raise ValueError('incomplete exact pixel population')
                        pos = int(labels.sum(dtype=np.int64));counts = dict(pixels=n, positive_pixels=pos, negative_pixels=n-pos)
                        pixel = backend.exact_pixel_result(out/'exact_pixels.json', ip.signature(a, d, seed)['identity'], st.digest(records), labels, values,
                                  domain=d, branch=b, prediction_sha=frozen['seal'], counts=counts)
                        metrics = [dict(**ip.signature(a, d, seed), branch=b, scoring=rule, image_reference=S if rule == 'SEGMENT_CH1' else b,
                                   image_count=len(samples), **image_metrics([r['label'] for r in scores], [r[rule] for r in scores]),
                                   pixel_AUROC=pixel['AUROC'], pixel_AP=pixel['AP'], AUPRO=None,
                                   image_metrics_inherited=rule == 'SEGMENT_CH1') for rule in READOUTS]
                    st.write_csv(out/'metrics.csv', metrics);st.write_csv(out/'map_manifest.csv', records);st.write_csv(out/'per_image_scores.csv', scores)
                    files = ('metrics.csv', 'map_manifest.csv', 'per_image_scores.csv', 'exact_pixels.json', 'exact_pixels.seal.json')
                    jobs.append(st.sealed_json(out/'status.json', dict(**ip.signature(a, d, seed), branch=b, images=len(samples), prediction_seal=frozen['seal'],
                                       files={name: st.sha(out/name) for name in files}, status='CPU_complete_AUPRO_pending')))
    return dict(status='requested_CPU_complete', jobs=len(jobs))


def au_expected(a, d, seed, branch):
    records = ip.audit.parent_reporting.rows(destination(a, d, seed, branch)/'map_manifest.csv')
    return dict(**ip.signature(a, d, seed), branch=branch, map_identity=st.digest(records), nstrips=200, FPR_max=.3), records


def aupro(a):
    pred.global_freeze(a);p = ip.parent(a);done = []
    with ExitStack() as stack:
        data = None
        for d in a.domains:
            for seed in a.support_seeds:
                for b in MAPS:
                    if not completed(a, d, seed, b):raise ValueError('exact CPU completion required before AUPRO')
                    expected, records = au_expected(a, d, seed, b);dest = ip.unit(a, d, seed)/'aupro'/(b+'.json')
                    if dest.exists():
                        r = st.read_json(dest);validate_aupro(r, expected)
                    else:
                        from importlib.metadata import version
                        if version('adeval') != '1.1.0':raise ValueError('original ADEval1.1.0 required; no dependency modification')
                        import torch
                        from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
                        if data is None:data = stack.enter_context(GTData(a.data))
                        images = [];masks = [];tick = time.perf_counter()
                        for row, ref in zip(p.samples(d), records):
                            mm, sc, meta = pred.replay(a, d, seed, row, only=b);gt = mask(a, data, d, seed, row, meta['output_shape'])
                            backend.shared.verify_map_record(ref, mm[b], gt);images.append(mm[b]);masks.append(gt)
                        torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
                        r = dict(expected, AUPRO=float(official_aupro_adeval(masks, images, nstrips=200)), verified_images=len(images),
                                 backend='original_ADEval_1.1.0_CUDA', seconds=time.perf_counter()-tick,
                                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                        r['seal'] = st.digest(r);validate_aupro(r, expected);st.write_json(dest, r)
                    if r['verified_images'] != len(p.samples(d)):raise ValueError('partial AUPRO list')
                    for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):setattr(a, key, max(getattr(a, key, 0), r[key]))
                    done.append(r);observe(a)
    return dict(status='requested_AUPRO_complete', jobs=len(done))


def cases(a):
    """Fixed first normal/anomaly per setting; no difficult-case search."""
    pred.global_freeze(a)
    status = a.run_root/'cases_status.json'
    if status.exists():
        saved = st.sealed_json(status)
        if saved['prediction_seal'] != pred.global_freeze(a)['seal'] or saved['records'] != 48:
            raise ValueError('fixed case identity differs')
        for name, sha in saved['files'].items():
            if st.sha(a.run_root/name) != sha:raise ValueError('fixed case artifact changed')
        return dict(status='fixed_cases_reused', case_records=48)
    from matplotlib import pyplot as plt
    records = []
    with GTData(a.data) as data:
        for d in a.domains:
            for seed in a.support_seeds:
                samples = ip.parent(a).samples(d)
                for label in (0, 1):
                    row = next(s for s in samples if s['label'] == label);images, _, meta = pred.replay(a, d, seed, row)
                    gt = mask(a, data, d, seed, row, meta['output_shape'])
                    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout='constrained')
                    for ax, (b, image) in zip(axes, images.items()):
                        ax.imshow(image, cmap='magma')
                        if gt.any() and not gt.all():ax.contour(gt, levels=[.5], colors='cyan', linewidths=.4)
                        ax.set_title(b, fontsize=7);ax.axis('off')
                        records.append(dict(domain=d, support_seed=seed, sample_id=row['sample_id'], label=label, branch=b,
                                            minimum=float(image.min()), median=float(np.median(image)), p99=float(np.quantile(image, .99)),
                                            maximum=float(image.max()), image_mean=float(image.mean(dtype=np.float64)),
                                            selection='first_image_of_each_label_in_frozen_common_order_no_metric_search'))
                    path = ip.unit(a, d, seed)/'cases';path.mkdir(parents=True, exist_ok=True)
                    fig.suptitle(d+'/seed'+str(seed)+'/'+row['sample_id']);fig.savefig(path/(row['sample_id']+'.png'), dpi=100);plt.close(fig)
    st.write_csv(a.run_root/'fixed_case_response_distributions.csv', records)
    files = [a.run_root/'fixed_case_response_distributions.csv']+[p for d in a.domains for seed in a.support_seeds for p in (ip.unit(a, d, seed)/'cases').glob('*.png')]
    if len(records) == 48:
        st.sealed_json(a.run_root/'cases_status.json', dict(prediction_seal=pred.global_freeze(a)['seal'],
                       records=len(records), files={p.relative_to(a.run_root).as_posix():st.sha(p) for p in files}))
    return dict(status='fixed_cases_complete', case_records=len(records))
