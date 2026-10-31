"""Summarize frozen endpoints with conditional-group and seed+group uncertainty."""
import json
from pathlib import Path
import numpy as np

REV=Path(__file__).resolve().parent

def compare(a,b):
    # Rows are independently trained models; columns are the SAME test episodes.
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    assert a.shape[1]==b.shape[1]==64
    rng=np.random.default_rng(20261030)
    groups=rng.integers(0,64,(10000,64))
    delta=b.mean(0)-a.mean(0)
    conditional=delta[groups].mean(1)
    sa=rng.integers(0,len(a),(10000,len(a)))
    sb=rng.integers(0,len(b),(10000,len(b)))
    dd=b[sb].mean(1)-a[sa].mean(1)
    crossed=np.take_along_axis(dd,groups,axis=1).mean(1)
    return dict(mean_delta=float(delta.mean()),relative_change=float(delta.mean()/a.mean()) if abs(a.mean())>1e-12 else None,conditional_paired_group_ci95=np.percentile(conditional,[2.5,97.5]).tolist(),seed_and_paired_group_ci95=np.percentile(crossed,[2.5,97.5]).tolist())

def summarize_matrices(families):
    stats={}
    for arm,arr in families.items():
        a=np.asarray(arr,dtype=float)
        stats[arm]=dict(mean=float(a.mean()),seed_std=float(a.mean(1).std(ddof=1)),seed_means=a.mean(1).tolist(),n_seeds=len(a),n_test_groups=a.shape[1])
    deltas={f'{b}_minus_{a}':compare(families[a],families[b]) for a,b in [('state','paired'),('state','unpaired'),('unpaired','paired')]}
    return dict(families=stats,comparisons=deltas)

def main():
    cfg=json.loads((REV/'config.json').read_text())
    root=Path(cfg['root'])
    reports={}
    physical={}
    for family,tags in cfg['families'].items():
        reports[family]=[];physical[family]=[]
        for tag in tags:
            path=REV/'latent'/f'{tag}.json'
            if not path.exists(): path=root/'reports'/f'{tag}_counterfactual_group_disjoint.json'
            reports[family].append(json.loads(path.read_text()))
            p=json.loads((REV/'physical'/f'{tag}.json').read_text())
            assert not p['smoke']
            physical[family].append(p)
    results=dict(protocol=str(REV/'PROTOCOL.md'),warning='Three seeds per family are few. Seed+group bootstrap intervals are sensitivity estimates, not a substitute for more training replications. All new results are reported irrespective of direction.',latent={},physical={})
    for variant in cfg['variants']:
        results['latent'][variant]={}
        for metric in ['teacher_mse','teacher_cosine_error','rollout_mse']:
            fam={arm:[r['variants'][variant]['group_'+metric] for r in rr] for arm,rr in reports.items()}
            results['latent'][variant][metric]=summarize_matrices(fam)
        results['physical'][variant]={}
        for arm,rr in physical.items():
            for p in rr:
                assert p['variants'][variant]['group_ids']==list(range(192,256))
                assert p['variants'][variant]['source_episodes']==physical['state'][0]['variants'][variant]['source_episodes']
        for metric in physical['state'][0]['variants'][variant]['metrics']:
            fam={arm:[r['variants'][variant]['metrics'][metric]['per_group'] for r in rr] for arm,rr in physical.items()}
            results['physical'][variant][metric]=summarize_matrices(fam)
            results['physical'][variant][metric]['per_horizon_seed_mean']={arm:np.mean([r['variants'][variant]['metrics'][metric]['per_horizon'] for r in rr],axis=0).tolist() for arm,rr in physical.items()}
    (REV/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    variant=cfg['primary_variant']
    rows=['# Minimal revision experiments','',f'Primary held-out composition: {variant}. All arms: 384 window records, 10,000 updates, 3 training seeds.','', '| Endpoint | State | Paired | Unpaired |','|---|---:|---:|---:|']
    endpoints=[('latent','rollout_mse'),('physical','rollout_block_euclidean_m'),('physical','rollout_effector_euclidean_m'),('physical','observed_block_euclidean_m'),('physical','true_state_persistence_block_euclidean_m')]
    for section,key in endpoints:
        s=results[section][variant][key]
        vals=[f"{s['families'][a]['mean']:.6f} +/- {s['families'][a]['seed_std']:.6f}" for a in ['state','paired','unpaired']]
        rows.append('| '+key+' | '+' | '.join(vals)+' |')
    rows+=['','Differences below use a two-way bootstrap of training seeds and paired test episodes; conditional group-only intervals are also saved in summary.json.','']
    for section,key in endpoints[:3]:
        rows.append('## '+key)
        for comparison,s in results[section][variant][key]['comparisons'].items():
            rows.append(f"- {comparison}: {s['mean_delta']:.6f}; seed+group 95% CI {s['seed_and_paired_group_ci95']}")
        rows.append('')
    rows+=['Collection-policy limitation: unpaired data cover more distinct source starts than paired data; this is not an isolated loss-level pairing test.','The original state pool has 192 records but 191 unique source starts. New unpaired data preserve that pool and keep the duplicate within one physics condition.','Physical results use common nominal train-only linear decoders; they are decoded predictions, not successful control.','No result-dependent choice of seeds, checkpoints, or test episodes was made.']
    (REV/'summary.md').write_text('\n'.join(rows)+'\n')
    print('\n'.join(rows),flush=True)

if __name__=='__main__':main()
