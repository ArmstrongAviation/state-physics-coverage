"""Frozen checkpoint evaluation with common train-only linear state decoders."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import h5py
import hdf5plugin  # noqa
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/chenyizi/lewm-data-study')
sys.path.insert(0, str(ROOT))
import train_data_arm as t

def ridge_fit(x, y, alpha):
    x,y = np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    xm,xs = x.mean(0),np.maximum(x.std(0),1e-6)
    ym,ys = y.mean(0),np.maximum(y.std(0),1e-6)
    xx,yy = (x-xm)/xs,(y-ym)/ys
    w = np.linalg.solve(xx.T@xx/len(xx)+alpha*np.eye(x.shape[1]),xx.T@yy/len(xx))
    return dict(xm=xm,xs=xs,ym=ym,ys=ys,w=w)

def decode(x, model):
    return (((x-model['xm'])/model['xs'])@model['w'])*model['ys']+model['ym']

def fit_probe(z,y,episodes):
    unique = np.unique(episodes)
    perm = np.random.default_rng(20261030).permutation(unique)
    nv = max(1,int(np.ceil(len(unique)*0.2)))
    val_eps = perm[:nv]
    mask = np.repeat(np.isin(episodes,val_eps),z.shape[1])
    x,yy = z.reshape(-1,z.shape[-1]),y.reshape(-1,6)
    scores=[]
    grid=[1e-5,1e-4,1e-3,1e-2,1e-1,1.0,10.0]
    for alpha in grid:
        m=ridge_fit(x[~mask],yy[~mask],alpha)
        score=float(np.mean(((decode(x[mask],m)-yy[mask])/m['ys'])**2))
        scores.append(score)
    alpha=grid[int(np.argmin(scores))]
    model=ridge_fit(x,yy,alpha)
    info=dict(alpha=alpha, alpha_grid=grid, validation_normalized_mse=scores, validation_source_episodes=val_eps.tolist(), fit_source_episodes=np.setdiff1d(unique,val_eps).tolist(), refit_source_episodes=unique.tolist(), calibration_frames=len(x), normalization='Features and targets standardized using fit rows only during selection, all calibration rows during refit; objective mean squared loss + alpha * squared coefficient norm.')
    return model,info

@torch.inference_mode()
def encode_file(jepa,path,start,end,device,frames=None,batch_groups=4):
    zs,ys,raws,eps,contact=[],[],[],[],[]
    with h5py.File(path) as f:
        for first in range(start,end,batch_groups):
            last=min(end,first+batch_groups)
            pix=np.asarray(f['pixels'][first:last],dtype=np.uint8)
            if frames is not None: pix=pix[:,:frames]
            pixels,b,n=t.to_gpu_normalized(pix,device)
            cls=jepa.encoder(pixels,interpolate_pos_encoding=True).last_hidden_state[:,0]
            z=jepa.projector(cls).reshape(b,n,t.EMB_DIM).cpu().numpy()
            zs.append(z)
            ys.append(np.concatenate([f['block_pos'][first:last,:n],f['effector_pos'][first:last,:n]],axis=-1))
            raws.append(f['action'][first:last])
            eps.append(f['source_episode'][first:last])
            contact.append(f['gripper_contact'][first:last,:n,0])
    return tuple(np.concatenate(a,axis=0) for a in (zs,ys,raws,eps,contact))

@torch.inference_mode()
def latent_predictions(jepa,z,raw,device,batch=8):
    teacher,rollout=[],[]
    for first in range(0,len(z),batch):
        zz=torch.from_numpy(z[first:first+batch]).to(device)
        rr=torch.from_numpy(raw[first:first+batch]).to(device)
        aa=jepa.action_encoder(rr.reshape(len(zz),20,25))
        hist=zz[:,:3].clone()
        tt,pp=[],[]
        for ti in range(2,z.shape[1]-1):
            actions=aa[:,ti-2:ti+1]
            tt.append(jepa.predict(zz[:,ti-2:ti+1],actions)[:,-1])
            pred=jepa.predict(hist,actions)[:,-1]
            pp.append(pred)
            hist=torch.cat([hist[:,1:],pred[:,None]],dim=1)
        teacher.append(torch.stack(tt,1).cpu().numpy())
        rollout.append(torch.stack(pp,1).cpu().numpy())
    return np.concatenate(teacher),np.concatenate(rollout)

def metric_array(out,name,a):
    a=np.asarray(a,dtype=np.float64)
    assert a.ndim==2 and np.isfinite(a).all(),name
    out[name]=dict(mean=float(a.mean()),per_horizon=a.mean(0).tolist(),per_group=a.mean(1).tolist(),per_group_horizon=a.tolist())

def cosine_error(a,b):
    return 1.0-(a*b).sum(-1)/(np.maximum(np.linalg.norm(a,axis=-1),1e-8)*np.maximum(np.linalg.norm(b,axis=-1),1e-8))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag',required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    rev=ROOT/'revision_minimal_20261030'
    cfg=json.loads((rev/'config.json').read_text())
    data=Path(cfg['data_dir'])
    output=rev/('smoke' if args.smoke else 'physical')
    output.mkdir(exist_ok=True)
    out=output/f'{args.tag}.json'
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device=torch.device(args.device)
    checkpoint=ROOT/'exp5b_runs'/args.tag/'weights_final.pt'
    jepa=t.build_jepa_random(device)
    jepa.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    jepa.eval().requires_grad_(False)
    calibration_end=8 if args.smoke else 192
    test_end=194 if args.smoke else 256
    z,y,_,eps,_=encode_file(jepa,data/'cube_nominal.h5',0,calibration_end,device,frames=20)
    model,probe_info=fit_probe(z,y,eps)
    np.savez_compressed(output/f'{args.tag}_probe.npz',**model)
    info=dict(tag=args.tag,checkpoint=str(checkpoint),smoke=args.smoke,decoder=probe_info,calibration=dict(variant='nominal',groups=[0,calibration_end],frames=[0,20],shared_across_models=True),evaluation=dict(group_start=192,group_end=test_end,history_frames=3,raw_actions_per_latent_step=5,horizon_latent_steps=18,horizon_raw_action_steps=90,position_units='meters',aggregation='Equal average over test groups and future frames; Euclidean norm for position error, mean over coordinates for MSE.'),variants={})
    variants=['mass_high_delay_2'] if args.smoke else cfg['variants']
    for variant in variants:
        zz,yy,raw,test_eps,contact=encode_file(jepa,data/f'cube_{variant}.h5',192,test_end,device)
        assert not set(eps)&set(test_eps)
        pred_tf,pred_roll=latent_predictions(jepa,zz,raw,device)
        true_z=zz[:,3:]
        true_y=yy[:,3:]
        metrics={}
        for label,pred in [('teacher',pred_tf),('rollout',pred_roll)]:
            metric_array(metrics,label+'_latent_mse',np.mean((pred-true_z)**2,axis=-1))
            metric_array(metrics,label+'_latent_cosine_error',cosine_error(pred,true_z))
        metric_array(metrics,'rollout_latent_train_standardized_mse',np.mean(((pred_roll-true_z)/model['xs'])**2,axis=-1))
        decoded={'observed':decode(true_z,model),'teacher':decode(pred_tf,model),'rollout':decode(pred_roll,model),'decoded_persistence':np.broadcast_to(decode(zz[:,2:3],model),true_y.shape),'true_state_persistence':np.broadcast_to(yy[:,2:3],true_y.shape)}
        for label,pred in decoded.items():
            for obj,sl in [('block',slice(0,3)),('effector',slice(3,6))]:
                diff=pred[...,sl]-true_y[...,sl]
                metric_array(metrics,f'{label}_{obj}_euclidean_m',np.linalg.norm(diff,axis=-1))
                metric_array(metrics,f'{label}_{obj}_coordinate_mse_m2',np.mean(diff**2,axis=-1))
        info['variants'][variant]=dict(source_episodes=test_eps.tolist(),group_ids=list(range(192,test_end)),contact_flags=contact[:,3:].astype(int).tolist(),metrics=metrics)
        # Match the old evaluator numerically before accepting new measurements.
        legacy=ROOT/'reports'/f'{args.tag}_counterfactual_group_disjoint.json'
        if not legacy.exists(): legacy=rev/'latent'/f'{args.tag}.json'
        if legacy.exists() and not args.smoke:
            old=json.loads(legacy.read_text())['variants'][variant]
            checks={}
            for oldkey,newkey in [('group_teacher_mse','teacher_latent_mse'),('group_rollout_mse','rollout_latent_mse'),('group_teacher_cosine_error','teacher_latent_cosine_error')]:
                delta=float(np.max(np.abs(np.array(old[oldkey])-np.array(metrics[newkey]['per_group']))))
                checks[oldkey]=delta
                if delta>2e-4: raise ValueError(f'Legacy mismatch {args.tag} {variant} {oldkey}: {delta}')
            info['variants'][variant]['legacy_max_absolute_difference']=checks
        print(args.tag,variant,'rollout_block_m',metrics['rollout_block_euclidean_m']['mean'],flush=True)
    tmp=out.with_suffix('.tmp')
    tmp.write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    tmp.replace(out)
    print('COMPLETE',out,flush=True)

if __name__=='__main__': main()
