"""Sparse GLMs, clustered sandwich, Rubin pooling and constrained categorical FCS."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy import sparse,linalg,special,stats
from threadpoolctl import threadpool_limits
threadpool_limits(1)
R=Path(__file__).resolve().parents[1]
SOURCE=R/'input/Study2_AnalysisPlan'
TARGETS=['urban','light','weather','surface','road_layout','junction','age_group_1','age_group_2','recorded_sex_1','recorded_sex_2']
LEVELS={'exposure':['two_cars','car_hgv','car_motorcycle','other_known','unknown'],'urban':['urban','rural'],'light':['daylight','dark_lit','dark_other'],'weather':['fine','rain','other'],'surface':['dry','wet','adverse'],'road_layout':['single','dual','roundabout','other'],'junction':['no_junction','junction'],'road_class':['minor','a_road','motorway_am'],'speed_group':['30','20','40','50','60','70'],'collision_year':['2021','2022','2023','2024'],'parked_count':['0','1','2']}
for slot in [1,2]:
 LEVELS[f'age_group_{slot}']=['adult','young','older','parked']
 LEVELS[f'recorded_sex_{slot}']=['male','female','parked']
NUM=['weekend','hour_sin','hour_cos','month_sin','month_cos']
CONTEXT=['collision_year','road_class','urban','speed_group','road_layout','junction','light','weather','surface','parked_count']
EXPOSURE_NAMES=['exposure=car_hgv','exposure=car_motorcycle','exposure=other_known']

def load_data():
 d=pd.read_csv(SOURCE/'data/analysis_observed_inputs.csv',dtype={v:str for v in ['collision_index','collision_year','police_force','speed_group','parked_count','vehicle_type_1','vehicle_type_2']})
 assert len(d)==50735 and d.collision_index.is_unique and d.multiple_ksi.sum()==4119
 return d

def dummy(values,levels,name,drop=True):
 codes=pd.Categorical(values,categories=levels).codes;assert (codes>=0).all(),name
 start=int(drop);ok=codes>=start
 out=sparse.csr_matrix((np.ones(ok.sum()),(np.flatnonzero(ok),codes[ok]-start)),shape=(len(codes),len(levels)-start))
 return out,[f'{name}={v}' for v in levels[start:]]

def model_matrix(d,stage='M2',missing_category=False,vehicle_age=False):
 n=len(d);parts=[];names=[]
 def add(v,levels,drop=True):
  vals=d[v]
  if missing_category:vals=vals.fillna('missing');levels=list(levels)+(['missing'] if vals.eq('missing').any() else [])
  a,b=dummy(vals,levels,v,drop);parts.append(a);names.extend(b)
 if stage=='M0':parts.append(sparse.csr_matrix(np.ones((n,1))));names.append('intercept')
 add('exposure',LEVELS['exposure'])
 if stage!='M0':
  add('police_force',sorted(d.police_force.unique(),key=int),False)
  for v in CONTEXT:add(v,LEVELS[v])
  parts.append(sparse.csr_matrix(d[NUM].to_numpy(float)));names.extend(NUM)
 if stage=='M2':
  age=d[['age_group_1','age_group_2']];sex=d[['recorded_sex_1','recorded_sex_2']]
  if missing_category:
   for value,label in [('young','any_young'),('older','any_older')]:
    vals=np.where((age==value).any(axis=1),'1',np.where(age.isna().any(axis=1),'missing','0'))
    a,b=dummy(vals,['0','1','missing'],label);parts.append(a);names.extend(b)
   vals=np.where(sex.isna().any(axis=1),'missing',(sex=='female').sum(axis=1).astype(str))
   a,b=dummy(vals,['0','1','2','missing'],'female_count');parts.append(a);names.extend(b)
  else:
   assert age.notna().all().all() and sex.notna().all().all()
   parts.append(sparse.csr_matrix(np.column_stack([(age=='young').any(axis=1),(age=='older').any(axis=1)]).astype(float)));names.extend(['any_young','any_older'])
   a,b=dummy((sex=='female').sum(axis=1).astype(str),['0','1','2'],'female_count');parts.append(a);names.extend(b)
 if vehicle_age:
  vals=d[['age_of_vehicle_1','age_of_vehicle_2']].to_numpy(float);assert np.isfinite(vals).all() and (vals>=0).all() and (vals<=100).all()
  parts.append(sparse.csr_matrix(np.column_stack([vals.min(axis=1)/10,vals.max(axis=1)/10])));names.extend(['min_vehicle_age_decades','max_vehicle_age_decades'])
 X=sparse.hstack(parts,format='csr');keep=np.asarray(X.power(2).sum(axis=0)).ravel()>1e-14
 return X[:,keep],np.asarray(names)[keep].tolist()

def glm_fit(X,y,groups,names,family='binomial',start=None,maxit=60):
 y=np.asarray(y,float);n,p=X.shape
 rank=np.linalg.matrix_rank((X.T@X).toarray(),tol=1e-7)
 if rank<p:raise RuntimeError(f'Design rank {rank} < {p}')
 b=np.zeros(p) if start is None else np.array(start,float).copy()
 if start is None:
  init=np.log((y.sum()+.5)/(n-y.sum()+.5)) if family=='binomial' else np.log(y.mean()+1e-8)
  for j,name in enumerate(names):
   if name=='intercept' or name.startswith('police_force='):b[j]=init
 def value(beta):
  eta=np.asarray(X@beta).ravel()
  return float((np.logaddexp(0,eta)-y*eta).sum()) if family=='binomial' else float((np.exp(np.clip(eta,-100,100))-y*eta).sum())
 converged=False
 for it in range(maxit):
  eta=np.asarray(X@b).ravel();mu=special.expit(eta) if family=='binomial' else np.exp(np.clip(eta,-100,100));w=mu*(1-mu) if family=='binomial' else mu
  H=(X.T@X.multiply(w[:,None])).toarray();score=np.asarray(X.T@(y-mu)).ravel()
  try:step=linalg.cho_solve(linalg.cho_factor(H,lower=True,check_finite=False),score,check_finite=False)
  except linalg.LinAlgError:raise RuntimeError('Singular information / possible separation')
  old=value(b);scale=1.
  while scale>1e-8:
   trial=b+scale*step
   if value(trial)<=old+1e-8:break
   scale*=.5
  b=trial
  if np.max(np.abs(score))<1e-6 or (np.max(np.abs(scale*step))<1e-8 and np.max(np.abs(score))<1e-4):converged=True;break
  if np.max(np.abs(b))>35:raise RuntimeError('Diverging coefficient / possible separation')
 if not converged:raise RuntimeError(f'GLM did not converge: {np.max(np.abs(score))}')
 eta=np.asarray(X@b).ravel();mu=special.expit(eta) if family=='binomial' else np.exp(eta);w=mu*(1-mu) if family=='binomial' else mu
 H=(X.T@X.multiply(w[:,None])).toarray();bread=linalg.inv(H,check_finite=False)
 labels,gi=np.unique(np.asarray(groups),return_inverse=True);G=len(labels)
 D=sparse.csr_matrix((np.ones(n),(gi,np.arange(n))),shape=(G,n));scores=(D@X.multiply((y-mu)[:,None])).toarray()
 cov=bread@(scores.T@scores)@bread*(G/(G-1))*((n-1)/(n-p))
 return dict(beta=b,cov=cov,bread=bread,n=n,events=float(y.sum()),p=p,G=G,iterations=it+1,converged=True,max_abs_score=float(np.max(np.abs(X.T@(y-mu)))),names=names,mu=mu)

def extract(fit):
 idx=[fit['names'].index(x) for x in EXPOSURE_NAMES];return fit['beta'][idx],fit['cov'][np.ix_(idx,idx)]

def pool(Q,U,dfcom=43):
 Q=np.asarray(Q);U=np.asarray(U);m=len(Q);q=Q.mean(0);ubar=U.mean(0);B=np.cov(Q,rowvar=False,ddof=1) if m>1 else np.zeros_like(ubar)
 T=ubar+(1+1/m)*B;se=np.sqrt(np.diag(T));lam=(1+1/m)*np.diag(B)/np.diag(T)
 if m==1:df=np.full(len(q),float(dfcom))
 else:
  old=(m-1)/np.maximum(lam,1e-12)**2;obs=((dfcom+1)/(dfcom+3))*dfcom*(1-lam);df=np.minimum(dfcom,1/(1/old+1/np.maximum(obs,1e-12)))
 crit=stats.t.ppf(.975,df);p=2*stats.t.sf(np.abs(q/se),df);order=np.argsort(p);adj=np.empty(len(p));adj[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
 return pd.DataFrame(dict(contrast=EXPOSURE_NAMES,beta=q,se=se,ratio=np.exp(q),lower=np.exp(q-crit*se),upper=np.exp(q+crit*se),df=df,p=p,p_holm=adj,fraction_variance_missing=lam,mcse=np.sqrt(np.diag(B)/m),mcse_over_se=np.sqrt(np.diag(B)/m)/se,m=m)),T

class FCS:
 def __init__(self,d):
  self.d=d;self.n=len(d);self.codes={};self.missing={}
  for v in TARGETS:
   c=pd.Categorical(d[v],categories=LEVELS[v]).codes.astype(np.int16);self.codes[v]=c;self.missing[v]=c<0
  parts=[sparse.csr_matrix(np.ones((self.n,1)))];names=['intercept']
  for v in ['exposure','collision_year','police_force','road_class','speed_group','parked_count','vehicle_type_1','vehicle_type_2']:
   lev=LEVELS[v] if v in LEVELS else sorted(d[v].dropna().unique().tolist(),key=int);a,b=dummy(d[v],lev,v);parts.append(a);names.extend(b)
  parts.append(sparse.csr_matrix(d[NUM+['multiple_ksi']].to_numpy(float)));names.extend(NUM+['multiple_ksi'])
  self.fixed=sparse.hstack(parts,format='csr');self.fixed_names=names;self.keep={};self.start={};self.cached_info={}
 def initialize(self,rng):
  c={v:x.copy() for v,x in self.codes.items()}
  for v in TARGETS:
   freq=np.bincount(c[v][self.train(v)],minlength=len(LEVELS[v])).astype(float);freq/=freq.sum();c[v][self.missing[v]]=rng.choice(len(freq),self.missing[v].sum(),p=freq)
  return c
 def train(self,v):
  valid=~self.missing[v]
  if v.startswith(('age_group_','recorded_sex_')):valid &= ~self.d['parked_'+v[-1]].to_numpy(bool)
  return valid
 def design(self,c,target):
  parts=[self.fixed];names=self.fixed_names.copy()
  for v in TARGETS:
   if v==target:continue
   codes=c[v];k=len(LEVELS[v]);ok=codes>0;parts.append(sparse.csr_matrix((np.ones(ok.sum()),(np.flatnonzero(ok),codes[ok]-1)),shape=(self.n,k-1)));names.extend([f'{v}={s}' for s in LEVELS[v][1:]])
  X=sparse.hstack(parts,format='csr')
  if target not in self.keep:
   Xt=X[self.train(target)];G=(Xt.T@Xt).toarray();_,RR,piv=linalg.qr(G,pivoting=True,mode='economic');rank=int((np.abs(np.diag(RR))>np.abs(RR[0,0])*1e-10).sum());self.keep[target]=np.sort(piv[:rank])
  assert self.keep[target][0]==0,'Augmentation requires retained intercept first'
  return X[:,self.keep[target]],np.asarray(names)[self.keep[target]].tolist()
 def update(self,c,v,rng,delta=None):
  tr=self.train(v);mis=self.missing[v];X,names=self.design(c,v);Xt=X[tr];Xm=X[mis];y=c[v][tr];K=len(LEVELS[v])-int(v.startswith(('age_group_','recorded_sex_')));assert y.max()<K;p=X.shape[1]
  raw=Xt[:,1:];mean=np.asarray(raw.mean(0)).ravel();var=np.asarray(raw.power(2).mean(0)).ravel()-mean**2;sd=np.sqrt(np.maximum(var,0)*len(y)/max(1,len(y)-1));lo=raw.min(axis=0).toarray().ravel();hi=raw.max(axis=0).toarray().ravel()
  nr=2*(p-1)*K;A=np.tile(np.r_[1.,mean],(nr,1));ay=[];row=0
  for j in range(p-1):
   for cat in range(K):
    for sign in [.5,-.5]:A[row,j+1]=np.clip(mean[j]+sign*sd[j],lo[j],hi[j]);ay.append(cat);row+=1
  ay=np.asarray(ay);aw=p/nr;Y=np.eye(K)[y];YA=np.eye(K)[ay]
  def fg(theta):
   B=theta.reshape(K-1,p);eta=np.column_stack([np.zeros(len(y)),Xt@B.T]);ea=np.column_stack([np.zeros(nr),A@B.T]);z=special.logsumexp(eta,axis=1);za=special.logsumexp(ea,axis=1);prob=np.exp(eta-z[:,None]);pa=np.exp(ea-za[:,None]);loss=float((z-eta[np.arange(len(y)),y]).sum()+aw*(za-ea[np.arange(nr),ay]).sum());grad=(Xt.T@(prob[:,1:]-Y[:,1:])).T+aw*(A.T@(pa[:,1:]-YA[:,1:])).T;return loss,np.asarray(grad).ravel()
  B=self.start.get(v,np.zeros((K-1)*p)).reshape(K-1,p).copy()
  if v not in self.start:counts=np.bincount(y,minlength=K)+.5;B[:,0]=np.log(counts[1:]/counts[0])
  def info(B):
   e=np.column_stack([np.zeros(len(y)),Xt@B.T]);ea=np.column_stack([np.zeros(nr),A@B.T]);pr=special.softmax(e,axis=1);pa=special.softmax(ea,axis=1);H=np.empty(((K-1)*p,(K-1)*p))
   for j in range(1,K):
    for k in range(j,K):
     w=pr[:,j]*((j==k)-pr[:,k]);wa=pa[:,j]*((j==k)-pa[:,k]);block=(Xt.T@Xt.multiply(w[:,None])).toarray()+aw*(A.T@(wa[:,None]*A));H[(j-1)*p:j*p,(k-1)*p:k*p]=block
     if k!=j:H[(k-1)*p:k*p,(j-1)*p:j*p]=block.T
   return (H+H.T)/2
  grad=fg(B.ravel())[1];H=self.cached_info.get(v)
  for iteration in range(45):
   if np.max(np.abs(grad))<.0001:break
   if H is None or iteration%8==7:H=info(B)
   step=linalg.solve(H,grad,assume_a='pos');old=fg(B.ravel())[0];scale=1.
   while scale>1e-8 and fg(B.ravel()-scale*step)[0]>old+1e-8:scale*=.5
   B=(B.ravel()-scale*step).reshape(K-1,p);grad=fg(B.ravel())[1]
  H=info(B)
  for refine in range(20):
   if np.max(np.abs(grad))<.0001:break
   step=linalg.solve(H,grad,assume_a='pos');old=fg(B.ravel())[0];scale=1.
   while scale>1e-9 and fg(B.ravel()-scale*step)[0]>old+1e-8:scale*=.5
   B=(B.ravel()-scale*step).reshape(K-1,p);grad=fg(B.ravel())[1];H=info(B)
  if np.max(np.abs(grad))>.001:raise RuntimeError(f'FCS optimization failed {v}, gradient {np.max(np.abs(grad))}')
  self.start[v]=B.ravel().copy();self.cached_info[v]=H
  if getattr(self,'capture_validation',False):self.validation_data=(Xt,y,A,ay,aw,B,H,fg,info,Xm)
  L=linalg.cholesky(H,lower=True,check_finite=False);draw=B.ravel()+linalg.solve_triangular(L.T,rng.normal(size=(K-1)*p),lower=False,check_finite=False);probs=special.softmax(np.column_stack([np.zeros(mis.sum()),Xm@draw.reshape(K-1,p).T]),axis=1)
  if delta is not None:probs*=np.asarray(delta)[None,:];probs/=probs.sum(1,keepdims=True)
  new=(rng.random(mis.sum())[:,None]>np.cumsum(probs,axis=1)).sum(1);c[v][mis]=np.minimum(new,K-1)
  return dict(target=v,gradient=float(np.max(np.abs(grad))),opt_iterations=iteration+1,parameters=(K-1)*p,observed=len(y)),probs

def complete_frame(base,c):
 out=base.copy()
 for v in TARGETS:out[v]=np.asarray(LEVELS[v],object)[c[v]]
 return out
