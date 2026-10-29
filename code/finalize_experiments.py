import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import hashlib,json,platform,subprocess,time
from pathlib import Path
import numpy as np
from common import load_inputs,read_jsonl,DATA,parse_answer
from selector import features
R=Path('runs');summary=json.loads((R/'summary.json').read_text()); bank,arrays=load_inputs('validation');_,_,_,answers,groups=features(bank,arrays['embeddings'],'E4')
from selector import trace_features
traces=[trace_features(c['text'],r['question'],parse_answer(c['text'])) for r in bank for c in r['candidates']]
coverage=sum(t[9]>0 for t in traces)/len(traces)
choices=json.loads((R/'validation_choices.json').read_text());gold=read_jsonl(DATA/'gsm8k_validation.jsonl');cases=[]
for i,row in enumerate(bank):
 a,b=choices['E3'][i],choices['E4'][i]
 if a!=b:
  cases.append({'id':row['id'],'question':row['question'],'reference':gold[i]['answer'],'E3_index':a,'E4_index':b,'E3_text':row['candidates'][a]['text'],'E4_text':row['candidates'][b]['text']})
(R/'arithmetic_disagreements.json').write_text(json.dumps(cases,indent=2))
freeze={'mode':'E3','reason':'Predeclared preference for simpler configuration when gains are small: E4 improves validation by only one question; E3 has stronger train OOF hard accuracy. A1 gains three validation questions but is exploratory and retains implicit vote interactions.','test_consulted':False,'assets':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'E3').iterdir()},'arithmetic_candidate_coverage':coverage}
(R/'frozen_selection.json').write_text(json.dumps(freeze,indent=2))
measure={}
for split in ('validation','test'):
 tick=time.perf_counter()
 subprocess.run([os.sys.executable,'predict.py','--method','selector.py','--checkpoint','runs/E3','--split',split,'--output',f'runs/{split}-predictions.json'],check=True)
 subprocess.run([os.sys.executable,'score_predictions.py','--split',split,'--predictions',f'runs/{split}-predictions.json','--output',f'runs/{split}-score.json'],check=True)
 measure[split]=time.perf_counter()-tick
measure.update({'platform':platform.platform(),'processor':platform.processor(),'asset_bytes':sum(p.stat().st_size for p in (R/'E3').iterdir()),'threads':4})
(R/'resources.json').write_text(json.dumps(measure,indent=2));print(json.dumps(measure))
