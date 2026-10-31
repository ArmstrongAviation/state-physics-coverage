"""Regenerate manuscript tables and vector figures from saved experimental reports.
Requires numpy, matplotlib, and reportlab. No training or test-set selection is performed.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results';TAB=ROOT/'tables';FIG=ROOT/'figures'
for p in [TAB,FIG]:p.mkdir(exist_ok=True)
s=json.loads((RES/'main_summary.json').read_text())
variant='mass_high_delay_2'
arms=['state','paired','unpaired']
colors={'state':'#336990','paired':'#bd6035','unpaired':'#52816b'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.facecolor':'white'})
def pm(d,n=4,scale=1,stdkey='seed_std'):
 return f"${d['mean']*scale:.{n}f}\\pm{d[stdkey]*scale:.{n}f}$"
def ci(d,n=4,scale=1,key='seed_and_paired_group_ci95'):
 lo,hi=d[key]
 return f"${d['mean_delta']*scale:+.{n}f}$ $[{lo*scale:+.{n}f},{hi*scale:+.{n}f}]$"
def table(name,caption,label,columns,header,rows,wide=True,note=None,size='small'):
 env='table*' if wide else 'table'
 out=[f'\\begin{{{env}}}[t]','\\centering',f'\\caption{{{caption}}}',f'\\label{{{label}}}',f'\\{size}',r'\setlength{\tabcolsep}{5pt}',r'\renewcommand{\arraystretch}{1.12}',f'\\begin{{tabular}}{{{columns}}}',r'\toprule',header+r' \\',r'\midrule']
 out += [r+r' \\' for r in rows]
 out += [r'\bottomrule',r'\end{tabular}']
 if note:out += [r'\par\vspace{3pt}',r'\begin{minipage}{\linewidth}\footnotesize '+note+r'\end{minipage}']
 out += [f'\\end{{{env}}}']
 (TAB/f'{name}.tex').write_text('\n'.join(out)+'\n')

table('data_design','Principal training policies. N: nominal; M: mass and inertia multiplied by four; D: two-raw-action-step delay. All use 384 window records and 10,000 updates. Distinct source keys count episode/start pairs; distinct window identities also include condition and raw window start.','tab:data','lrrrrrr','Policy & Source records & Source keys & Episodes & N/M/D windows & Window records & Window identities',[
 'State (S) & 192 & 191 & 128 & 384/0/0 & 384 & 382',
 'Paired (P) & 64 & 64 & 64 & 128/128/128 & 384 & 384',
 'Unpaired (U) & 192 & 191 & 128 & 128/128/128 & 384 & 382'],size='footnotesize')
table('training','Shared model and training settings for all principal policies. Models are initialized randomly; only the data policy and random seed vary.','tab:training',r'p{0.44\columnwidth}p{0.43\columnwidth}','Setting & Value',[
 r'Image encoder & ViT-tiny, 12 layers, patch size 14',
 r'Image / latent dimensions & $224\times224$ RGB / 192',
 'Predictor & 6 layers, 16 heads, head width 64',
 'Predictor MLP / dropout & 2,048 / 0.1',
 'History / training window & 3 / 10 latent frames',
 'Raw actions per latent step & 5 (25 action coordinates)',
 'Optimizer / batch size & AdamW / 32',
 r'Learning rate / weight decay & $5\times10^{-5}$ / $10^{-3}$',
 'Updates / warmup & 10,000 / 2,000',
 'Schedule / precision & Linear warmup, cosine / BF16',
 r'SIGReg weight / projections & 0.09 / 1,024',
 'Gradient-norm clipping & 1.0',
 'Model seeds per main policy & 3'],wide=False,size='footnotesize')
metrics=[('latent','teacher_mse','Teacher-forced MSE'),('latent','teacher_cosine_error','Teacher-forced cosine error'),('latent','rollout_mse','Raw rollout MSE'),('physical','rollout_latent_train_standardized_mse','Standardized rollout MSE'),('physical','rollout_latent_cosine_error','Rollout cosine error')]
rows=[]
for sec,k,label in metrics:
 d=s[sec][variant][k]['families'];rows.append(label+' & '+' & '.join(pm(d[a]) for a in arms))
table('latent_main','Latent metrics on held-out MD for the strict source-episode split. Entries are mean $\\pm$ sample standard deviation over three training seeds. Errors average 64 test groups and 18 future frames; lower is better. Standardization uses model-specific coordinate deviations from the common nominal training calibration set.','tab:latent','lrrr','Metric & State (S) & Paired (P) & Unpaired (U)',rows)
contrast_keys=[('latent','rollout_mse','Raw rollout MSE',1,4),('physical','rollout_latent_train_standardized_mse','Standardized rollout MSE',1,4),('physical','rollout_latent_cosine_error','Rollout cosine error',1,4),('physical','rollout_block_euclidean_m','Block position (cm)',100,2),('physical','rollout_effector_euclidean_m','End-effector position (cm)',100,2)]
rows=[]
for sec,k,label,scale,n in contrast_keys:
 d=s[sec][variant][k]['comparisons'];rows.append(label+' & '+' & '.join(ci(d[c],n,scale) for c in ['paired_minus_state','paired_minus_unpaired','unpaired_minus_state']))
table('contrasts','Policy differences on held-out MD with 95\\% percentile intervals resampling training seeds independently within policies and test groups jointly (10,000 replicates). Negative values favor the first policy in the contrast. Intervals are descriptive and unadjusted across endpoints.','tab:contrasts','lccc',r'Endpoint & P $-$ S & P $-$ U & U $-$ S',rows,size='footnotesize')
rows=[]
for v,label in [('nominal','N'),('mass_high','M'),('delay_2','D'),('mass_high_delay_2','MD')]:
 d=s['latent'][v]['rollout_mse'];rows.append(label+' & '+' & '.join(pm(d['families'][a]) for a in arms)+' & '+ci(d['comparisons']['paired_minus_unpaired']))
table('variant_results','Raw rollout MSE across the four strict test conditions (same 64 episodes). Each policy has three training seeds. The final column gives the paired-minus-unpaired difference with a seed-and-group 95\\% interval. N, M, and D are seen training conditions for physics-diverse policies; MD is a held-out composition.','tab:variants','lrrrc',r'Condition & State (S) & Paired (P) & Unpaired (U) & P $-$ U',rows,size='footnotesize')
rows=[]
for mode,label in [('observed','Observed future latent'),('teacher','Teacher-forced prediction'),('rollout','Free-running prediction'),('decoded_persistence','Decoded-state persistence'),('true_state_persistence','True-state persistence')]:
 vals=[]
 for obj in ['block','effector']:
  d=s['physical'][variant][f'{mode}_{obj}_euclidean_m']['families'];vals += [pm(d[a],2,100) for a in arms]
 rows.append(label+' & '+' & '.join(vals))
table('physical_main','Physical position errors on strict held-out MD, in centimeters (mean $\\pm$ sample seed standard deviation; three models per policy). All entries use 64 test groups and the same 18 future frames. Observed-latent decoding measures readout error without dynamics prediction. True-state persistence has privileged initial simulator position and is only a reference.','tab:physical','lrrrrrr',r'Prediction / reference & \multicolumn{3}{c}{Block position} & \multicolumn{3}{c}{End-effector position} \\ & S & P & U & S & P & U',rows,size='footnotesize')

def crossed(a,b):
 rng=np.random.default_rng(20261030);groups=rng.integers(0,64,(10000,64))
 ai=rng.integers(0,len(a),(10000,len(a)));bi=rng.integers(0,len(b),(10000,len(b)))
 dd=b[bi].mean(1)-a[ai].mean(1)
 draws=np.take_along_axis(dd,groups,axis=1).mean(1)
 return {'mean_delta':float(b.mean()-a.mean()),'seed_and_paired_group_ci95':np.percentile(draws,[2.5,97.5]).tolist()}
def raw(tag):return json.loads((RES/f'{tag}_counterfactual_group_disjoint.json').read_text())['variants'][variant]['group_rollout_mse']
a=np.array([raw(t) for t in ['d8_cf_state_s0','d8_cf_state_s1','d10_cf_state_s3']]);factor={};rows=[]
for label,tag in [('Nominal + M','mass'),('Nominal + D','delay')]:
 b=np.array([raw(f'd10_cf_{tag}_s{i}') for i in range(2)]);d=crossed(a,b);factor[tag]={'mean':float(b.mean()),'seed_std':float(b.mean(1).std(ddof=1)),**d}
 rows.append(label+' & 96 & 2 & '+pm(factor[tag])+' & '+ci(d))
table('factor','Exploratory strict-split single-factor ablation on MD. Factor policies use 96 source groups and 384 window records each, with two training seeds. The common state reference has three seeds and raw rollout MSE $1.4101\\pm0.0092$. Differences are factor minus state, with seed-and-group 95\\% intervals recomputed from saved per-group outputs. Standardized and physical-decoder results were not measured for these factor checkpoints.','tab:factor','lrrrc',r'Factor policy & Source groups & Seeds & Raw rollout MSE & Difference [95\% interval]',rows,size='footnotesize')
(RES/'factor_seed_group_summary.json').write_text(json.dumps(factor,indent=2)+'\n')
old=json.loads((RES/'d2_sampling_multiseed.json').read_text());rows=[]
for arm,label in [('uniform','Uniform'),('contact','High contact'),('free','Low contact'),('phase','Balanced tails')]:
 d=old[arm];rows.append(label+f" & {d['n']} & "+' & '.join(pm(d[k],3,stdkey='std') for k in ['contact_mse','contact_p95','block_probe','gripper_probe']))
table('sampling','Earlier temporal-sampling diagnostic: 2,000 windows from 200 training episodes, 20,000 updates, and shared disjoint probe episodes. Values are seed means $\\pm$ sample standard deviations. Contact MSE and its p95 are teacher-forced latent errors; probes are mean Pearson correlations. Probe fitting differs from the principal physical evaluation.','tab:sampling','lrrrrr','Sampling policy & Seeds & Contact MSE & Contact p95 & Block probe & Gripper probe',rows,size='footnotesize')
old=json.loads((RES/'d6_scaling_summary.json').read_text());rows=[]
for pct in ['25','50','100']:
 d=old['coverage'][pct];f=d['families'];delta=old['deltas'][pct]['rollout_mse']
 rows.append(str(d['windows'])+' & '+str(f['state']['seeds'])+'/'+str(f['mechanism']['seeds'])+' & '+pm(f['state']['rollout_mse'])+' & '+pm(f['mechanism']['rollout_mse'])+' & '+ci(delta,key='paired_group_ci95'))
table('scaling','Earlier budget scan on a group-disjoint but source-episode-overlapping dataset. Each run uses 10,000 updates. Values are raw rollout MSE on MD; intervals average model seeds before resampling groups and do not adjust for shared source episodes. These exploratory results are not pooled with the strict-split study.','tab:scaling','rrr rc',r'Windows & S/P seeds & State (S) & Paired (P) & P $-$ S [group-only 95\% interval]',rows,size='footnotesize')

# Figure 1 retains the original visual framework with the audited three-policy design.
import runpy
runpy.run_path(str(ROOT/'scripts'/'build_framework.py'))

fig,axes=plt.subplots(2,2,figsize=(7.05,3.45))
for ax,(sec,k,label,scale,n) in zip(axes.flat,contrast_keys[:4]):
 d=s[sec][variant][k]['comparisons']
 for yy,(key,lab) in enumerate([('paired_minus_state','P - S'),('paired_minus_unpaired','P - U'),('unpaired_minus_state','U - S')]):
  v=d[key];mid=v['mean_delta']*scale;lo,hi=np.array(v['seed_and_paired_group_ci95'])*scale
  ax.errorbar(mid,yy,xerr=[[mid-lo],[hi-mid]],fmt='o',ms=4,capsize=3,color='#305f7d',lw=1.1)
 ax.axvline(0,color='#888',ls='--',lw=.7);ax.set_yticks(range(3),['P - S','P - U','U - S']);ax.invert_yaxis();ax.set_ylim(2.6,-.6);ax.set_title(label,loc='left');ax.grid(axis='x',alpha=.15)
 ax.set_xlabel('Mean difference'+(' (cm)' if scale==100 else ''))
fig.tight_layout(pad=1.0,h_pad=1.2,w_pad=2.0);fig.savefig(FIG/'metric_contrasts.pdf');fig.savefig(FIG/'metric_contrasts.png',dpi=200);plt.close(fig)

curves=json.loads((RES/'horizon_plot_data.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(7.05,2.55));x=np.arange(1,19)
for ax,obj,title in zip(axes,['block','effector'],['Block position','End-effector position']):
 for arm,label in [('state','State'),('paired','Paired'),('unpaired','Unpaired')]:
  mean=np.array(curves[arm][obj]['mean'])*100;sd=np.array(curves[arm][obj]['seed_sd'])*100
  ax.plot(x,mean,color=colors[arm],label=label,lw=1.4);ax.fill_between(x,mean-sd,mean+sd,color=colors[arm],alpha=.12,lw=0)
 ax.plot(x,np.array(curves['reference'][obj])*100,'--',color='#555',lw=1,label='True-state persistence')
 ax.set_xlabel('Forecast horizon (latent steps)');ax.set_ylabel('Mean position error (cm)');ax.set_title(title,loc='left');ax.set_xlim(1,18);ax.set_xticks([1,3,6,9,12,15,18]);ax.grid(alpha=.15)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.01))
fig.tight_layout(rect=[0,.10,1,1],pad=.8,w_pad=1.6);fig.savefig(FIG/'physical_horizon.pdf');fig.savefig(FIG/'physical_horizon.png',dpi=200);plt.close(fig)
print('Generated 9 tables and 3 figures. Factor intervals:',json.dumps(factor))
