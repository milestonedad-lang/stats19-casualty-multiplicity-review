"""Actual fits for primary, prespecified sensitivities and supported displays."""
from engine import *
import argparse,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed

def scalar_pool(values,variances,dfcom):
 v=np.asarray(values);m=len(v);q=v.mean();u=np.mean(variances);b=np.var(v,ddof=1) if m>1 else 0;t=u+(1+1/m)*b;lam=(1+1/m)*b/t if t>0 else 0
 df=dfcom if m==1 else min(dfcom,1/(1/max((m-1)/max(lam,1e-12)**2,1e-12)+1/max(((dfcom+1)/(dfcom+3))*dfcom*(1-lam),1e-12)))
 se=np.sqrt(t);return dict(estimate=q,se=se,lower=q-stats.t.ppf(.975,df)*se,upper=q+stats.t.ppf(.975,df)*se,df=df,m=m)

def perform(d,stage='M2',family='binomial',vehicle_age=False,missing_category=False,start_parameters=None):
 X,names=model_matrix(d,stage,missing_category=missing_category,vehicle_age=vehicle_age)
 start=None if start_parameters is None else np.asarray([start_parameters.get(n,0.) for n in names]);fit=glm_fit(X,d.multiple_ksi if family=='binomial' else d.additional_ksi,d.police_force,names,family=family,start=start);q,u=extract(fit)
 return dict(q=q.tolist(),u=u.tolist(),n=fit['n'],events=fit['events'],G=fit['G'],p=fit['p'],iterations=fit['iterations'],score=fit['max_abs_score']),fit,X

def standardize(base,completed,fit,X):
 speed=base.speed_group.map({'20':'low','30':'low','40':'middle','50':'middle','60':'high','70':'high'});strata=base.road_class.astype(str)+'|'+base.urban.astype(str)+'|'+speed;counts=pd.crosstab(strata,base.exposure);out=[];expidx=[i for i,n in enumerate(fit['names']) if n.startswith('exposure=')]
 eta=np.asarray(X@fit['beta']).ravel()-np.asarray(X[:,expidx]@fit['beta'][expidx]).ravel()
 for group,name in zip(['car_hgv','car_motorcycle','other_known'],EXPOSURE_NAMES):
  supported=counts.index[(counts.two_cars>=20)&(counts[group]>=20)];mask=strata.isin(supported).to_numpy()&base.urban.notna().to_numpy();n=int(mask.sum());assert n>0;idx=fit['names'].index(name);A=X[mask];e=eta[mask];p0=special.expit(e);p1=special.expit(e+fit['beta'][idx]);gradients=[]
  for prob,which in [(p0,None),(p1,idx)]:
   weight=prob*(1-prob);g=np.asarray(A.T@weight).ravel()/n;g[expidx]=0
   if which is not None:g[which]=weight.mean()
   gradients.append(g)
  g0,g1=gradients;gd=g1-g0;out.append(dict(contrast=name,n=n,strata=len(supported),reference=float(p0.mean()),comparison=float(p1.mean()),difference=float((p1-p0).mean()),var_reference=float(g0@fit['cov']@g0),var_comparison=float(g1@fit['cov']@g1),var_difference=float(gd@fit['cov']@gd),G=fit['G']))
 return out

def analyze_chain(ident,force=False):
 source=R/'data/imputations'/f'chain_{ident:03d}.npz';sourcehash=hashlib.sha256(source.read_bytes()).hexdigest();output=R/'results/chain_fits'/f'fits_{ident:03d}.json'
 if not force and output.exists():
  try:
   if json.loads(output.read_text()).get('input_imputation_sha256')==sourcehash:return dict(chain=ident,status='existing')
  except json.JSONDecodeError:pass
 base=load_data();z=np.load(source);c={v:z[v] for v in TARGETS};d=complete_frame(base,c);results={};failures={};start_parameters=None
 def attempt(label,t,stage='M2',family='binomial',vehicle_age=False):
  try:r,f,X=perform(t,stage,family,vehicle_age,start_parameters=start_parameters if family=='binomial' else None);results[label]=r;return f,X
  except Exception as ex:failures[label]=str(ex);return None,None
 primary,X=attempt('M2_primary',d)
 if primary is None:raise RuntimeError('Primary model failure: '+str(failures))
 start_parameters=dict(zip(primary['names'],primary['beta']));attempt('M1_context',d,'M1');attempt('Concordant_IBR',d[d.concordant_ibr]);attempt('All_casualty_IBR',d[d.all_casualty_ibr]);attempt('Exclude_flag_discordance',d[~d.any_flag_discordance]);attempt('Exclude_2024',d[d.collision_year.ne('2024')]);attempt('Exclude_Metropolitan_2024',d[~(d.police_force.eq('1')&d.collision_year.eq('2024'))]);attempt('Known_vehicle_types',d[d.exposure.ne('unknown')])
 agevalid=d[['age_of_vehicle_1','age_of_vehicle_2']].apply(pd.to_numeric).apply(lambda x:x.between(0,100)).all(axis=1);attempt('Known_vehicle_age_adjusted',d[agevalid],vehicle_age=True);attempt('Additional_KSI_Poisson',d,family='poisson')
 for scenario,factor,kind,index in [('MNAR_age_half',.5,'age',1),('MNAR_age_double',2.,'age',2),('MNAR_sex_half',.5,'sex',3),('MNAR_sex_double',2.,'sex',4)]:
  changed={v:x.copy() for v,x in c.items()};rng=np.random.default_rng(np.random.SeedSequence([20261104,ident,index]))
  for v in TARGETS:
   if (kind=='age' and v.startswith('age_group_')) or (kind=='sex' and v.startswith('recorded_sex_')):
    mask=base[v].isna().to_numpy();pr=z['prob_'+v].copy()
    if kind=='age':pr[:,1:]*=factor
    else:pr[:,1]*=factor
    pr/=pr.sum(1,keepdims=True);draw=(rng.random(mask.sum())[:,None]>np.cumsum(pr,axis=1)).sum(1);changed[v][mask]=np.minimum(draw,pr.shape[1]-1)
  attempt(scenario,complete_frame(base,changed))
 for group in sorted(d.police_force.unique(),key=int):
  label='LOO_'+group;mask=d.police_force.ne(group).to_numpy();xs=X[mask];keep=np.asarray(xs.power(2).sum(axis=0)).ravel()>1e-14;xs=xs[:,keep];names=np.asarray(primary['names'])[keep].tolist()
  try:
   f=glm_fit(xs,d.multiple_ksi.to_numpy()[mask],d.police_force.to_numpy()[mask],names,start=primary['beta'][keep]);q,u=extract(f);results[label]=dict(q=q.tolist(),u=u.tolist(),n=f['n'],events=f['events'],G=f['G'],p=f['p'],iterations=f['iterations'],score=f['max_abs_score'])
  except Exception as ex:failures[label]=str(ex)
 payload=dict(chain=ident,input_imputation_sha256=sourcehash,iterations=int(z['iterations']),models=results,failures=failures,standardization=standardize(base,d,primary,X),primary_full=dict(beta=primary['beta'].tolist(),cov=primary['cov'].tolist(),names=primary['names']))
 temp=output.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(json.dumps(payload));temp.replace(output);return dict(chain=ident,status='completed',successful_models=len(results),failures=failures)

def fixed_models():
 d=load_data();res={}
 for label,t,stage,missing in [('M0_unadjusted',d,'M0',False),('M2_complete_case',d[d.extended_complete],'M2',False),('M1_complete_context',d[d.context_complete],'M1',False),('M2_missing_category',d,'M2',True)]:
  try:r,_,_=perform(t,stage,missing_category=missing);res[label]=r
  except Exception as e:res[label]={'error':str(e)}
 (R/'results/fixed_models.json').write_text(json.dumps(res,indent=2))

def summarize(chains=50,minimum_iterations=60):
 rows=[]
 for i in range(1,chains+1):
  r=json.loads((R/'results/chain_fits'/f'fits_{i:03d}.json').read_text());assert r['iterations']>=minimum_iterations;assert hashlib.sha256((R/'data/imputations'/f'chain_{i:03d}.npz').read_bytes()).hexdigest()==r['input_imputation_sha256'];rows.append(r)
 assert len(set(r['iterations'] for r in rows))==1
 labels=sorted(set.union(*[set(r['models'])|set(r['failures']) for r in rows]));tables=[];failure=[]
 for label in labels:
  good=[r['models'][label] for r in rows if label in r['models']]
  if len(good)!=len(rows):failure.append(dict(model=label,successful=len(good),total=len(rows),reason='No selective pooling; at least one imputation failed'));continue
  out,_=pool([r['q'] for r in good],[r['u'] for r in good],min(r['G'] for r in good)-1);out['model']=label;out['n']=good[0]['n'];out['outcome_sum']=good[0]['events'];out['groups']=good[0]['G'];out['parameters']=good[0]['p'];tables.append(out)
 for label,r in json.loads((R/'results/fixed_models.json').read_text()).items():
  if 'error' in r:failure.append(dict(model=label,successful=0,total=1,reason=r['error']));continue
  out,_=pool([r['q']],[r['u']],r['G']-1);out['model']=label;out['n']=r['n'];out['outcome_sum']=r['events'];out['groups']=r['G'];out['parameters']=r['p'];tables.append(out)
 allresults=pd.concat(tables,ignore_index=True);allresults['effect_measure']=np.where(allresults.model.eq('Additional_KSI_Poisson'),'conditional_extra_KSI_mean_ratio','odds_ratio');allresults.to_csv(R/'results/model_estimates.csv',index=False);allresults[allresults.model.isin(['M0_unadjusted','M1_context','M2_primary'])].to_csv(R/'results/main_model_estimates.csv',index=False);pd.DataFrame(failure,columns=['model','successful','total','reason']).to_csv(R/'results/failed_models.csv',index=False)
 riskrows=[]
 for contrast in EXPOSURE_NAMES:
  rr=[next(x for x in r['standardization'] if x['contrast']==contrast) for r in rows]
  for measure in ['reference','comparison','difference']:riskrows.append(dict(contrast=contrast,measure=measure,n=rr[0]['n'],support_strata=rr[0]['strata'],**scalar_pool([r[measure] for r in rr],[r['var_'+measure] for r in rr],min(r['G'] for r in rr)-1)))
 pd.DataFrame(riskrows).to_csv(R/'results/supported_standardization.csv',index=False)
 main=allresults[allresults.model.eq('M2_primary')];loo=allresults[allresults.model.str.startswith('LOO_')];influence=[]
 for contrast in EXPOSURE_NAMES:
  m=main[main.contrast.eq(contrast)].iloc[0];v=loo[loo.contrast.eq(contrast)]
  influence.append(dict(contrast=contrast,min_OR=float(v.ratio.min()),max_OR=float(v.ratio.max()),largest_log_OR_shift=float(np.max(np.abs(v.beta-m.beta))),most_influential_group=v.iloc[np.argmax(np.abs(v.beta-m.beta))].model,successful_deletions=len(v)))
 pd.DataFrame(influence).to_csv(R/'results/leave_one_group_out_summary.csv',index=False);summary=dict(imputations=len(rows),iterations=rows[0]['iterations'],primary=main.to_dict('records'),failed_models=failure,max_primary_mcse_over_se=float(main.mcse_over_se.max()),all_primary_fits_converged=True,maximum_primary_score=max(r['models']['M2_primary']['score'] for r in rows))
 (R/'results/analysis_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--chains',type=int,default=50);p.add_argument('--minimum-iterations',type=int,default=60);p.add_argument('--summarize-only',action='store_true');p.add_argument('--force',action='store_true');a=p.parse_args()
 if not a.summarize_only:
  for i in range(1,a.chains+1):
   with np.load(R/'data/imputations'/f'chain_{i:03d}.npz') as z:assert int(z['iterations'])>=a.minimum_iterations
  fixed_models()
  with ProcessPoolExecutor(max_workers=a.workers) as pool:
   for future in as_completed([pool.submit(analyze_chain,i,a.force) for i in range(1,a.chains+1)]):print(json.dumps(future.result()),flush=True)
 summarize(a.chains,a.minimum_iterations)
