"""CPU-only verification of archived outputs; no training or external data access."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / 'revision_minimal_20261030'


def numeric_equal(actual, expected, path='', tol=1e-10):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        for key, val in expected.items():
            assert key in actual, (path, key)
            numeric_equal(actual[key], val, path+'/'+key, tol)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, val in enumerate(expected):
            numeric_equal(actual[i], val, path+f'/{i}', tol)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, rtol=tol, atol=tol), (path, actual, expected)
    elif expected is None:
        assert actual is None, path
    # Historical path strings and explanatory prose are not numerical endpoints.


def main():
    checks = []
    manifest_path = ROOT/'MANIFEST_SHA256.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
        checks.append(f'Archive SHA256 verified for {len(manifest)} files')

    cfg = json.loads((REV/'config.json').read_text())
    spec = importlib.util.spec_from_file_location('original_summary', REV/'summarize.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # All nine latent reports are supplied locally, so historical fallback paths
    # in the unchanged original summarizer are never needed.
    for tags in cfg['families'].values():
        for tag in tags:
            assert (REV/'latent'/f'{tag}.json').is_file(), tag
    with contextlib.redirect_stdout(io.StringIO()):
        mod.main()
    actual = json.loads((REV/'summary.json').read_text())
    expected = json.loads((REV/'summary_reference.json').read_text())
    for section in ['latent', 'physical']:
        numeric_equal(actual[section], expected[section], section)
    checks.append('All main summary values, seed SDs, and bootstrap intervals match')

    replay = {v: np.load(ROOT/'data'/f'cube_{v}_states.npz', allow_pickle=False)
              for v in cfg['variants']}
    nominal = replay['nominal']
    eps, starts = nominal['source_episode'], nominal['source_start']
    assert len(set(eps[:192])) == 128 and len(set(eps[192:])) == 64
    assert not set(eps[:192]) & set(eps[192:])
    assert len(set(zip(eps[:192].tolist(), starts[:192].tolist()))) == 191
    assert eps[45] == eps[173] == 4022 and starts[45] == starts[173] == 97
    assignment = json.loads((REV/'data_audit.json').read_text())['group_ids_by_variant']
    ids = [g for gs in assignment.values() for g in gs]
    assert sorted(ids) == list(range(192))
    source_sets = [set((int(eps[g]), int(starts[g])) for g in gs)
                   for gs in assignment.values()]
    for i in range(3):
        for j in range(i):
            assert not source_sets[i] & source_sets[j]
    for v, data in replay.items():
        assert data['action'].shape == (256,100,5)
        for key in ['source_episode','source_start','initial_qpos','initial_qvel','action']:
            np.testing.assert_array_equal(data[key], nominal[key])
        for obj in ['block','effector']:
            positions = data[obj+'_pos'][192:]
            persistence = np.linalg.norm(positions[:,2:3]-positions[:,3:],axis=-1)
            for tags in cfg['families'].values():
                for tag in tags:
                    report = json.loads((REV/'physical'/f'{tag}.json').read_text())
                    assert not report['smoke']
                    vr = report['variants'][v]
                    assert vr['group_ids'] == list(range(192,256))
                    np.testing.assert_array_equal(vr['source_episodes'],eps[192:])
                    key = f'true_state_persistence_{obj}_euclidean_m'
                    np.testing.assert_allclose(persistence,vr['metrics'][key]['per_group_horizon'],rtol=1e-6,atol=1e-7)
                    for metric in vr['metrics'].values():
                        arr=np.asarray(metric['per_group_horizon'])
                        assert arr.shape == (64,18) and np.isfinite(arr).all()
                        np.testing.assert_allclose(arr.mean(1),metric['per_group'],rtol=1e-10,atol=1e-10)
                        np.testing.assert_allclose(arr.mean(0),metric['per_horizon'],rtol=1e-10,atol=1e-10)
                        np.testing.assert_allclose(arr.mean(),metric['mean'],rtol=1e-10,atol=1e-10)
    checks += ['Strict source split and unpaired assignment verified',
               'All per-group/horizon aggregates verified',
               'True-state persistence independently recomputed from exported positions']

    frozen = json.loads((REV/'source_sha256.json').read_text())
    matched = 0
    for oldpath, digest in frozen.items():
        name=Path(oldpath).name
        candidates=[ROOT/'original_code'/name, ROOT/'lewm_snapshot'/name, REV/name]
        available=[p for p in candidates if p.exists()]
        if available:
            assert any(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p in available), oldpath
            matched += 1
    assert matched == len(frozen), ('Missing original hash targets',matched,len(frozen))
    checks.append(f'All {matched} launch-time source hashes verified')
    for tag in sum(cfg['families'].values(),[]):
        with np.load(REV/'physical'/f'{tag}_probe.npz',allow_pickle=False) as d:
            assert d['w'].shape==(192,6) and d['xm'].shape==(192,)
        m=json.loads((ROOT/'training_metadata'/f'{tag}.json').read_text())
        assert m['train'][-1]['step']==10000
    checks.append('Nine decoder parameter files and final training steps verified')

    report={'status':'passed','checks':checks,
            'summary_tolerance':{'rtol':1e-10,'atol':1e-10},
            'position_recompute_tolerance':{'rtol':1e-6,'atol':1e-7},
            'scope':'Numerical reaggregation and exported-state audit only; no retraining or visual inference.'}
    (ROOT/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
