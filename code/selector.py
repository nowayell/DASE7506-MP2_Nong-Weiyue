"""Answer-group features. No reference files are read here."""
import ast
import operator
import re
from fractions import Fraction
import numpy as np
from common import parse_answer

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
def arithmetic(expr):
    def visit(n):
        if isinstance(n, ast.Expression): return visit(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float): return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(n.op, ast.USub) else 1)*visit(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](visit(n.left), visit(n.right))
        raise ValueError('unsupported')
    if len(expr)>160: raise ValueError('too long')
    return visit(ast.parse(expr.strip(), mode='eval'))

def trace_features(text, question, answer):
    nums=re.findall(r'[-+]?\d+(?:\.\d+)?',text)
    qnums=set(re.findall(r'\d+(?:\.\d+)?',question))
    value=float(Fraction(answer)) if answer is not None else 0.
    basic=[np.log1p(len(text)),np.log1p(len(nums)),text.count('='),text.count('\n'),answer is None,value<0,value==int(value),np.log1p(abs(value)),len(qnums & set(nums))/max(1,len(qnums))]
    checked=wrong=0
    # Conservative: only complete lines containing a single numeric equality.
    for line in text.splitlines():
        line=line.strip().replace('×','*').replace('÷','/').replace('−','-')
        if re.fullmatch(r'[\d\s.,()+*/=\-]+',line) and line.count('=')==1:
            left,right=line.replace(',','').split('=')
            try:
                a,b=arithmetic(left),arithmetic(right)
                checked+=1; wrong+=int(a!=b)
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError): pass
    return basic+[checked,wrong,wrong/max(1,checked),checked==0]

BLOCKS={'E1':[], 'G0':[], 'E2':['votes'], 'E3':['votes','basic','similarity'], 'E4':['votes','basic','similarity','arithmetic'], 'A1':['basic','similarity','arithmetic'], 'A2':['votes','similarity','arithmetic']}
def features(bank, embeddings, mode):
    n=len(bank); x=np.zeros((n,4,1046),dtype=np.float64); mask=np.zeros((n,4),bool); representatives=np.zeros((n,4),int)
    answers=[]; groups_all=[]
    for i,row in enumerate(bank):
        ans=[parse_answer(c['text']) for c in row['candidates']]; answers.append(ans)
        groups=[]
        for j,a in enumerate(ans):
            found=next((g for g in groups if a is not None and ans[g[0]]==a),None)
            if found is None: groups.append([j])
            else: found.append(j)
        if mode=='E1': groups=[[j] for j in range(4)]
        groups_all.append(groups)
        h=embeddings[i].astype(float); hnorm=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12); sim=hnorm@hnorm.T
        tf=np.array([trace_features(c['text'],row['question'],a) for c,a in zip(row['candidates'],ans)])
        for k,g in enumerate(groups):
            mask[i,k]=True; representatives[i,k]=g[0]; x[i,k,:1024]=h[g].mean(axis=0)
            if 'votes' in BLOCKS[mode]: x[i,k,1024:1026]=[len(g),len(g)/4]
            if 'basic' in BLOCKS[mode]: x[i,k,1026:1035]=tf[g,:9].mean(axis=0)
            if 'similarity' in BLOCKS[mode]:
                other=[j for j in range(4) if j not in g]
                within=[sim[a,b] for a in g for b in g if a!=b]
                x[i,k,1035:1042]=[sim[g].mean(),np.mean(within) if within else 0,sim[np.ix_(g,other)].mean() if other else 0,bool(within),bool(other),len(g)*len(groups),len(g)**2]
            if 'arithmetic' in BLOCKS[mode]: x[i,k,1042:1046]=tf[g,9:].mean(axis=0)
    return x,mask,representatives,answers,groups_all

def predict(bank, arrays, checkpoint):
    from pathlib import Path
    import json
    checkpoint=Path(checkpoint)
    config=json.loads((checkpoint/'config.json').read_text())
    with np.load(checkpoint/'weights.npz') as z:
        x,mask,reps,_,_=features(bank,arrays['embeddings'],config['mode'])
        scores=((x-z['mean'])/z['scale'])@z['weight']; scores[~mask]=-1e30
    return [int(reps[i,k]) for i,k in enumerate(scores.argmax(axis=1))]
