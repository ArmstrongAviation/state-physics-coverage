"""Prepare the frozen d11 independent-physics collection baseline; never touch d8/d10."""
import hashlib
import json
from pathlib import Path
import h5py
import hdf5plugin  # noqa
import numpy as np

ROOT = Path('/data/chenyizi/lewm-data-study')
REV = ROOT / 'revision_minimal_20261030'
DATA = ROOT / 'counterfactual_v3_episode_disjoint'
VARIANTS = ['nominal', 'mass_high', 'delay_2']

def main():
    REV.mkdir(exist_ok=True)
    out = ROOT / 'cache/cf_v3_unpaired_d11'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    handles = {v: h5py.File(DATA / f'cube_{v}.h5') for v in VARIANTS}
    ref = handles['nominal']
    for v, f in handles.items():
        for key in ['source_episode', 'source_start', 'initial_qpos', 'initial_qvel', 'action']:
            assert np.array_equal(f[key][:], ref[key][:]), (v, key)
    episodes = ref['source_episode'][:]
    starts = ref['source_start'][:]
    assert not set(episodes[:192]) & set(episodes[192:256])
    identities = list(zip(episodes[:192].tolist(), starts[:192].tolist()))
    duplicates = {str(p):[i for i,x in enumerate(identities) if x==p] for p in set(identities) if identities.count(p)>1}
    assert len(set(identities)) == 191 and list(duplicates.values()) == [[45,173]], duplicates
    assert len(set(episodes[192:256])) == 64
    # One fixed random assignment, independent of outcome metrics; same across model seeds.
    rng = np.random.default_rng(20261030)
    groups = rng.permutation(192).reshape(3, 64)
    # Prevent the already-existing duplicate start from creating a cross-physics pair.
    row_a,col_a = np.argwhere(groups==45)[0]
    row_b,col_b = np.argwhere(groups==173)[0]
    if row_a != row_b:
        swap_col = next(i for i,g in enumerate(groups[row_a]) if g != 45)
        groups[row_a,swap_col],groups[row_b,col_b] = groups[row_b,col_b],groups[row_a,swap_col]
    entries = [(v, int(g), s) for v, gs in zip(VARIANTS, groups) for g in gs for s in (0, 10)]
    order = np.random.default_rng(2026).permutation(len(entries))
    entries = [entries[i] for i in order]
    assert len(entries) == len(set(entries)) == 384
    out.mkdir()
    pix = np.lib.format.open_memmap(out/'pixels.npy', mode='w+', dtype='uint8', shape=(384,10,224,224,3))
    act = np.lib.format.open_memmap(out/'actions.npy', mode='w+', dtype='float32', shape=(384,10,25))
    for i, (v,g,s) in enumerate(entries):
        f = handles[v]
        pix[i] = f['pixels'][g,s:s+10]
        raw = f['action'][g,s*5:(s+10)*5]
        assert raw.shape == (50,5) and np.isfinite(raw).all()
        act[i] = raw.reshape(10,25)
    pix.flush(); act.flush()
    np.save(out/'labels.npy', np.array([v for v,g,s in entries], dtype='U64'))
    np.save(out/'source_group.npy', np.array([(g,s) for v,g,s in entries], dtype='int64'))
    np.savez_compressed(out/'manifest.npz', train_specs=np.column_stack([np.arange(384),np.zeros(384,dtype='int64')]), probe_tr=np.array([],dtype='int64'), probe_val=np.array([],dtype='int64'), variants=np.array(VARIANTS))
    meta = dict(windows=384, unique_windows=384, unique_groups=192, source_episodes=len(set(episodes[:192])), assignment_seed=20261030, window_shuffle_seed=2026, starts=[0,10], label_counts={v:128 for v in VARIANTS}, group_ids_by_variant={v:gs.tolist() for v,gs in zip(VARIANTS,groups)}, source_episode_overlap_with_test=0, cross_variant_shared_groups=0, data_source=str(DATA), interpretation='Fixed-budget collection-policy comparison: paired 64 shared groups versus independently assigned 192 groups. Not a pure pairing effect at fixed unique-state diversity; no pairing-specific loss is used.')
    meta.pop('unique_windows')
    meta.pop('unique_groups')
    meta.update(group_records=192,unique_source_starts=191,duplicate_source_start_groups=duplicates,unique_variant_source_window_starts=len(set((v,int(episodes[g]),int(starts[g])+s*5) for v,g,s in entries)),duplicate_assignment_rule='The two pre-existing duplicate source-start records are assigned to the same physics condition, swapping one assignment deterministically if needed; all variant counts remain equal.')
    source_sets=[{identities[int(g)] for g in gs} for gs in groups]
    assert all(not source_sets[i]&source_sets[j] for i in range(3) for j in range(i))
    meta['cross_variant_shared_source_starts']=0
    meta['distribution_audit'] = {}
    for v,gs in zip(VARIANTS,groups):
        gs = np.sort(gs)
        meta['distribution_audit'][v] = dict(source_episodes=len(set(episodes[gs])), nominal_contact_mean=float(ref['gripper_contact'][gs].mean()), initial_block_mean=ref['block_pos'][gs,0].mean(0).tolist(), raw_action_mean=ref['action'][gs].mean((0,1)).tolist())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (REV/'data_audit.json').write_text(json.dumps(meta,indent=2)+'\n')
    state = ['d8_cf_state_s0','d8_cf_state_s1','d10_cf_state_s3']
    paired = ['d8_cf_mechanism_s0','d8_cf_mechanism_s1','d8_cf_mechanism_s2']
    base = json.loads((ROOT/'exp5b_runs'/state[0]/'metrics.json').read_text())['args']
    audits = {}
    for tag in state+paired:
        a = json.loads((ROOT/'exp5b_runs'/tag/'metrics.json').read_text())['args']
        diff = {k:v for k,v in a.items() if v != base.get(k)}
        assert set(diff) <= {'seed','tag','cache_dir','window_manifest'}, (tag,diff)
        assert (ROOT/'exp5b_runs'/tag/'weights_final.pt').stat().st_size > 0
        audits[tag] = diff
    cfg = dict(root=str(ROOT), revision=str(REV), data_dir=str(DATA), cache=str(out), families={'state':state,'paired':paired,'unpaired':[f'd11_cf_unpaired_s{s}' for s in range(3)]}, training_args=base, reusable_args_differences=audits, test_group_start=192,test_group_end=256, variants=VARIANTS+['mass_high_delay_2'], primary_variant='mass_high_delay_2', primary_physical_metric='rollout block mean Euclidean position error (meters), averaged over 18 future frames', calibration='Common nominal groups 0:192, frames 0:20; episode-disjoint calibration validation; never test groups or composed physics.', bootstrap='Report both conditional paired-group CI (seed means fixed) and crossed bootstrap resampling model seeds independently within arms and shared test groups. Three seeds remain a small sample.')
    (REV/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for f in handles.values(): f.close()
    print(json.dumps(meta,indent=2),flush=True)

if __name__ == '__main__': main()
