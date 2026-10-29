import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from common import load_inputs,read_jsonl,DATA,parse_answer
from selector import features
R=Path('runs');R.mkdir(exist_ok=True);SEED=20261029

def dataset(split,mode):
 b,a=load_inputs(split);x,m,r,ans,g=features(b,a['embeddings'],mode);gold=read_jsonl(DATA/f'gsm8k_{split}.jsonl');y=np.zeros(m.shape,bool)
 for i,(row,t) in enumerate(zip(b,gold)):
  assert row['id']==t['id'] and row['question']==t['question']
  for j,group in enumerate(g[i]):y[i,j]=ans[i][group[0]] is not None and ans[i][group[0]]==parse_answer(t['answer'])
 assert y.any(axis=1).all()
 hard=np.array([len(set(v for v in row if v is not None))>=2 for row in ans])
 return x,m,r,y,hard,g

def fit(x,m,y,ids,reg):
 a=x[ids][m[ids]];mean=a.mean(0);scale=a.std(0);scale[scale<1e-8]=1
 xx=(x[ids]-mean)/scale;flat=xx.reshape(-1,x.shape[-1]);mm=m[ids];yy=y[ids]
 def obj(w):
  s=(flat@w).reshape(-1,4);s[~mm]=-1e30;s-=s.max(1,keepdims=True);p=np.exp(s);p/=p.sum(1,keepdims=True);z=(p*yy).sum(1);q=p*yy/np.maximum(z[:,None],1e-300)
  return -np.log(np.maximum(z,1e-300)).mean()+reg*(w@w),flat.T@(p-q).ravel()/len(ids)+2*reg*w
 z=minimize(obj,np.zeros(x.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-9})
 return mean,scale,z.x,{'success':bool(z.success),'iterations':int(z.nit),'message':str(z.message)}
def choose(x,m,z):
 s=((x-z[0])/z[1])@z[2];s[~m]=-1e30;return s.argmax(1)
def main():
 start=time.perf_counter();rng=np.random.default_rng(SEED);x,m,r,y,h,g=dataset('train','E1');folds=np.zeros(len(x),int)
 strata={}
 for i in range(len(x)):strata.setdefault((int(y[i].sum()),bool(h[i])),[]).append(i)
 for members in strata.values():
  for j,i in enumerate(rng.permutation(members)):folds[i]=j%5
 np.save(R/'folds.npy',folds);records=[];correct={};choices={}
 for mode in ('E1','G0','E2','E3','E4','A1','A2'):
  tick=time.perf_counter();x,m,r,y,h,g=dataset('train',mode);vx,vm,vr,vy,vh,vg=dataset('validation',mode);trials=[]
  for reg in (1e-5,1e-4,1e-3,1e-2):
   good=np.zeros(len(x),bool);opts=[]
   for f in range(5):
    ids=np.flatnonzero(folds!=f);hold=np.flatnonzero(folds==f);z=fit(x,m,y,ids,reg);c=choose(x[hold],m[hold],z);good[hold]=y[hold,c];opts.append(z[3])
   trials.append({'reg':reg,'oof':float(good.mean()),'oof_hard':float(good[h].mean()),'optimizers':opts});print(mode,reg,trials[-1]['oof_hard'],flush=True)
  best=max(trials,key=lambda t:(t['oof_hard'],t['reg']));z=fit(x,m,y,np.arange(len(x)),best['reg']);c=choose(vx,vm,z);good=vy[np.arange(len(vx)),c];correct[mode]=good;choices[mode]=[int(vr[i,j]) for i,j in enumerate(c)]
  folder=R/mode;folder.mkdir(exist_ok=True);np.savez(folder/'weights.npz',mean=z[0],scale=z[1],weight=z[2]);(folder/'config.json').write_text(json.dumps({'mode':mode,'reg':best['reg'],'seed':SEED}))
  records.append({'mode':mode,'reg':best['reg'],'validation':float(good.mean()),'validation_hard':float(good[vh].mean()),'trials':trials,'seconds':time.perf_counter()-tick,'optimizer':z[3]});(R/'experiments.json').write_text(json.dumps(records,indent=2));print('VALIDATION',mode,good.mean(),good[vh].mean(),flush=True)
 vx,vm,vr,vy,vh,vg=dataset('validation','G0');c=np.array([max(range(len(gs)),key=lambda j:len(gs[j])) for gs in vg]);correct['E0']=vy[np.arange(len(vx)),c];choices['E0']=[int(vr[i,j]) for i,j in enumerate(c)];records.insert(0,{'mode':'E0','validation':float(correct['E0'].mean()),'validation_hard':float(correct['E0'][vh].mean())})
 comparisons={}
 for left,right in [('E0','E2'),('G0','E2'),('E2','E3'),('E3','E4'),('E4','A1'),('E4','A2')]:
  d=correct[right].astype(int)-correct[left].astype(int);dh=d[vh];boot=[rng.choice(dh,len(dh),replace=True).mean() for _ in range(5000)];comparisons[right+'-'+left]={'corrected':int((d>0).sum()),'introduced':int((d<0).sum()),'hard_ci95':np.quantile(boot,[.025,.975]).tolist()}
 summary={'seed':SEED,'validation_n':len(vx),'hard_n':int(vh.sum()),'records':records,'comparisons':comparisons,'seconds':time.perf_counter()-start,'test_used':False};(R/'summary.json').write_text(json.dumps(summary,indent=2));(R/'validation_choices.json').write_text(json.dumps(choices));print('COMPLETE',flush=True)
if __name__=='__main__':main()
