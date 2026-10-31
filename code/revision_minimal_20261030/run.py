"""Durable experiment supervisor; separate files/status per job, no old runs overwritten."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

REV=Path(__file__).resolve().parent
ROOT=REV.parent
PY='/data/chenyizi/le-wm/.venv/bin/python'

def status(name,**values):
    d=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**values)
    p=REV/'status'/f'{name}.json'
    p.parent.mkdir(exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def execute(cmd,env,log):
    with open(log,'x') as f:
        print('RUN',json.dumps(cmd),flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def worker(args):
    cfg=json.loads((REV/'config.json').read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=f'{ROOT}:{ROOT}/src:/data/chenyizi/le-wm',PYTHONUNBUFFERED='1')
    name=args.worker
    try:
        if name=='existing':
            tags=cfg['families']['state']+cfg['families']['paired']
        else:
            seed=int(name)
            tag=f'd11_cf_unpaired_s{seed}'
            tags=[tag]
            if (ROOT/'exp5b_runs'/tag).exists(): raise FileExistsError(f'Run already exists: {tag}')
            status(name,stage='training',tag=tag,gpu=args.gpu)
            cmd=[PY,str(ROOT/'train_data_arm.py'),'--steps','10000','--batch-size','32','--num-workers','0','--eval-every','100000','--straight-every','100000','--save-every','10000','--log-every','1000','--skip-initial-eval','--skip-final-eval','--no-cereg','--seed',str(seed),'--window-manifest',cfg['cache']+'/manifest.npz','--cache-dir',cfg['cache'],'--tag',tag,'--device','cuda:0']
            execute(cmd,env,REV/'logs'/f'train_{tag}.log')
            metrics=json.loads((ROOT/'exp5b_runs'/tag/'metrics.json').read_text())
            assert metrics['train'][-1]['step']==10000
            for k,v in cfg['training_args'].items():
                if k not in {'seed','tag','cache_dir','window_manifest'}:
                    assert metrics['args'][k]==v,(k,metrics['args'][k],v)
            status(name,stage='latent_evaluation',tag=tag,gpu=args.gpu)
            cmd=[PY,str(ROOT/'evaluate_counterfactual.py'),'--checkpoint',str(ROOT/'exp5b_runs'/tag/'weights_final.pt'),'--files',*[str(Path(cfg['data_dir'])/f'cube_{v}.h5') for v in cfg['variants']],'--device','cuda:0','--group-start','192','--group-end','256','--out',str(REV/'latent'/f'{tag}.json')]
            execute(cmd,env,REV/'logs'/f'latent_{tag}.log')
        for tag in tags:
            status(name,stage='physical_evaluation',tag=tag,gpu=args.gpu)
            execute([PY,str(REV/'evaluate_physical.py'),'--tag',tag],env,REV/'logs'/f'physical_{tag}.log')
        status(name,stage='complete',gpu=args.gpu,tags=tags)
    except Exception:
        status(name,stage='failed',gpu=args.gpu,error=traceback.format_exc())
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker');ap.add_argument('--gpu',type=int);args=ap.parse_args()
    if args.worker is not None: return worker(args)
    for d in ['logs','latent','physical','status']:(REV/d).mkdir(exist_ok=True)
    if (REV/'status/supervisor.json').exists(): raise FileExistsError('Supervisor already launched; inspect status instead of rerunning.')
    # Snapshot source identities before any training starts.
    paths=[ROOT/'train_data_arm.py',ROOT/'evaluate_counterfactual.py',Path('/data/chenyizi/le-wm/jepa.py'),Path('/data/chenyizi/le-wm/module.py'),*REV.glob('*.py'),REV/'config.json',REV/'data_audit.json',REV/'PROTOCOL.md']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (REV/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    status('supervisor',stage='running',jobs={})
    children=[]
    for name,gpu in [('0',0),('1',2),('2',3),('existing',4)]:
        f=open(REV/'logs'/f'worker_{name}.log','x')
        p=subprocess.Popen([PY,str(REV/'run.py'),'--worker',name,'--gpu',str(gpu)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        children.append((name,p,f))
    status('supervisor',stage='running',jobs={n:p.pid for n,p,f in children})
    codes={n:p.wait() for n,p,f in children}
    for n,p,f in children:f.close()
    if any(codes.values()):
        status('supervisor',stage='failed',exit_codes=codes)
        raise RuntimeError(codes)
    try:
        execute([PY,str(REV/'summarize.py')],dict(os.environ,OPENBLAS_NUM_THREADS='4'),REV/'logs/summary.log')
        status('supervisor',stage='complete',exit_codes=codes,summary=str(REV/'summary.md'))
    except Exception:
        status('supervisor',stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
