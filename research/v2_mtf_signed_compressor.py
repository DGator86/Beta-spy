from __future__ import annotations
import json, hashlib, math
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

DATA=Path('/mnt/data/v2_mtf_dataset.csv')
OUT=Path('/mnt/data')

CONFIG={
    'version':'V2.1-MTF-SIGNED-VALIDATION-COMPRESSOR',
    'horizons':[5,15,30],
    'big_move_threshold_bps':{'5':4.5,'15':7.5,'30':10.5},
    'base_horizon_weights':{'5':0.25,'15':0.50,'30':0.25},
    'warmup_sessions':8,
    'magnitude_model':{'type':'HistGradientBoostingClassifier','max_depth':2,'max_iter':80,'learning_rate':0.05,'l2_regularization':4.0,'min_samples_leaf':30,'random_state':7},
    'direction_model':{'type':'LogisticRegression','C':0.08,'class_weight':'balanced','max_iter':800},
    'abs_move_model':{'type':'HistGradientBoostingRegressor','max_depth':2,'max_iter':80,'learning_rate':0.05,'l2_regularization':4.0,'min_samples_leaf':30,'random_state':11},
    'validator':{'decay':0.97,'direction_alignment_decay':0.95,'initial_brier':0.25,'min_matured_for_full_trust':40},
    'compression':{'direction_big_weighting':True,'shrink_to_neutral':True},
    'decision':{
        'min_validated_big_prob':0.55,
        'min_validated_direction_edge':0.12,
        'min_composite_trust':0.25,
        'min_agreement':0.35,
        'quiet_max_validated_big_prob':0.30,
        'one_directional_signal_per_day':True,
    },
    'development_end':'2026-08-17',
    'postlock_start':'2026-08-18',
    'postlock_end':'2026-08-25',
    'notes':'Architecture and thresholds frozen before postlock V2 scoring; postlock dates were already observed under V1, so this is an untuned causal replay, not a pristine blind test.'
}
CONFIG_JSON=json.dumps(CONFIG,sort_keys=True,separators=(',',':'))
CONFIG_HASH=hashlib.sha256(CONFIG_JSON.encode()).hexdigest()
(OUT/'v2_1_mtf_config_LOCKED.json').write_text(json.dumps({**CONFIG,'sha256':CONFIG_HASH},indent=2))

D=pd.read_csv(DATA)
D['timestamp']=pd.to_datetime(D['timestamp'],utc=True)
D=D.sort_values('timestamp').reset_index(drop=True)
label_cols={c for c in D.columns if c.startswith('y_')}
exclude={'date','timestamp','entry'}|label_cols
features=[c for c in D.columns if c not in exclude]
Xall=D[features].replace([np.inf,-np.inf],np.nan).fillna(0.0)
dates=list(D['date'].drop_duplicates())

@dataclass
class TrustState:
    mag_brier: float=0.25
    dir_brier: float=0.25
    abs_mae_ratio: float=1.0
    n_mag: int=0
    n_dir: int=0
    n_abs: int=0
    dir_alignment: float=0.0

    def update(self,p_big:float,p_up:float,p_abs:float,y_bps:float,thr:float,decay:float=0.97):
        y_big=float(abs(y_bps)>=thr)
        mb=(p_big-y_big)**2
        self.mag_brier=decay*self.mag_brier+(1-decay)*mb
        self.n_mag+=1
        if y_big:
            y_up=float(y_bps>0)
            db=(p_up-y_up)**2
            self.dir_brier=decay*self.dir_brier+(1-decay)*db
            edge=2*p_up-1
            if abs(edge)>=0.02:
                hit=(1.0 if edge*y_bps>0 else -1.0)
                da=CONFIG['validator']['direction_alignment_decay']
                self.dir_alignment=da*self.dir_alignment+(1-da)*hit
            self.n_dir+=1
        scale=max(abs(y_bps),thr,1.0)
        ratio=min(abs(p_abs-abs(y_bps))/scale,3.0)
        self.abs_mae_ratio=decay*self.abs_mae_ratio+(1-decay)*ratio
        self.n_abs+=1

    def scores(self):
        mag_skill=max(0.0,1-self.mag_brier/0.25)
        dir_skill=max(0.0,1-self.dir_brier/0.25)
        abs_skill=max(0.0,1-self.abs_mae_ratio)
        mag_ramp=min(1.0, math.sqrt(self.n_mag/40.0)) if self.n_mag else 0.0
        dir_ramp=min(1.0, math.sqrt(self.n_dir/20.0)) if self.n_dir else 0.0
        mag=mag_skill*mag_ramp
        direction=dir_skill*dir_ramp
        overall=0.60*mag+0.25*direction+0.15*abs_skill
        align_ramp=min(1.0, math.sqrt(self.n_dir/20.0)) if self.n_dir else 0.0
        signed_align=self.dir_alignment*align_ramp
        return mag,direction,abs_skill,max(0.0,min(1.0,overall)),signed_align

class HorizonModels:
    def __init__(self,h): self.h=h; self.mag=None; self.dir=None; self.abs=None
    def fit(self,X,y_bps,thr):
        y_bps=np.asarray(y_bps,float); valid=np.isfinite(y_bps); X=X.loc[valid]; y=y_bps[valid]
        ymag=(np.abs(y)>=thr).astype(int)
        self.mag=HistGradientBoostingClassifier(max_depth=2,max_iter=80,learning_rate=.05,l2_regularization=4,min_samples_leaf=30,random_state=7)
        self.mag.fit(X,ymag)
        big=np.abs(y)>=thr
        self.dir=None
        if big.sum()>=30 and len(np.unique((y[big]>0).astype(int)))==2:
            self.dir=make_pipeline(StandardScaler(),LogisticRegression(C=.08,max_iter=800,class_weight='balanced'))
            self.dir.fit(X.loc[big],(y[big]>0).astype(int))
        self.abs=HistGradientBoostingRegressor(max_depth=2,max_iter=80,learning_rate=.05,l2_regularization=4,min_samples_leaf=30,random_state=11)
        self.abs.fit(X,np.abs(y))
    def predict(self,X):
        pbig=self.mag.predict_proba(X)[:,1] if self.mag is not None else np.full(len(X),.5)
        pup=self.dir.predict_proba(X)[:,1] if self.dir is not None else np.full(len(X),.5)
        pabs=np.maximum(0.0,self.abs.predict(X)) if self.abs is not None else np.zeros(len(X))
        return pbig,pup,pabs

def agreement_score(edges,weights):
    if not edges: return 0.0
    w=np.asarray(weights,float); e=np.asarray(edges,float)
    if w.sum()<=0:return 0.0
    return float(abs(np.sum(w*np.sign(e)))/np.sum(w))

records=[]
trust={h:TrustState() for h in CONFIG['horizons']}
pending=deque()
models={}
for day_i,date in enumerate(dates):
    train_mask=D['date'].isin(dates[:day_i])
    day_mask=(D['date']==date)
    if day_i>=CONFIG['warmup_sessions']:
        models={}
        for h in CONFIG['horizons']:
            hm=HorizonModels(h); thr=CONFIG['big_move_threshold_bps'][str(h)]
            hm.fit(Xall.loc[train_mask],D.loc[train_mask,f'y_bps{h}'],thr); models[h]=hm
    day_indices=np.flatnonzero(day_mask.to_numpy())
    if day_i<CONFIG['warmup_sessions']:
        continue
    day_X=Xall.loc[day_mask]
    preds={h:models[h].predict(day_X) for h in CONFIG['horizons']}
    for local_j,idx in enumerate(day_indices):
        row=D.loc[idx]; ts=row['timestamp']
        while pending and pending[0]['maturity']<=ts:
            item=pending.popleft(); h=item['h']; y=D.loc[item['idx'],f'y_bps{h}']
            if np.isfinite(y):
                trust[h].update(item['pbig'],item['pup'],item['pabs'],float(y),CONFIG['big_move_threshold_bps'][str(h)],CONFIG['validator']['decay'])
        heads={}; weighted=[]; edges=[]; aw=[]
        for h in CONFIG['horizons']:
            pbig=float(preds[h][0][local_j]); pup=float(preds[h][1][local_j]); pabs=float(preds[h][2][local_j])
            mag_t,dir_t,abs_t,overall,signed_align=trust[h].scores()
            base=CONFIG['base_horizon_weights'][str(h)]
            w=base*(0.35+0.65*overall)
            edge=2*pup-1
            validated_head_edge=edge*signed_align
            heads[h]={'pbig':pbig,'pup':pup,'pabs':pabs,'mag_trust':mag_t,'dir_trust':dir_t,'trust':overall,'signed_alignment':signed_align,'w':w,'edge':edge,'validated_head_edge':validated_head_edge}
            weighted.append((w,pbig,validated_head_edge,pabs,overall)); edges.append(validated_head_edge); aw.append(w*pbig)
        sw=sum(x[0] for x in weighted)
        pbig_raw=sum(w*p for w,p,e,a,t in weighted)/sw
        sd=sum(w*p for w,p,e,a,t in weighted)
        edge_raw=(sum(w*p*e for w,p,e,a,t in weighted)/sd) if sd>0 else 0.0
        abs_raw=sum(w*a for w,p,e,a,t in weighted)/sw
        trust_comp=sum(w*t for w,p,e,a,t in weighted)/sw
        agree=agreement_score(edges,aw)
        pbig_valid=0.5+(pbig_raw-0.5)*trust_comp
        edge_valid=edge_raw*(0.5+0.5*agree)
        p_up_trade=max(0.0,min(1.0,pbig_valid*(1+edge_valid)/2))
        p_dn_trade=max(0.0,min(1.0,pbig_valid*(1-edge_valid)/2))
        p_neutral=max(0.0,1-pbig_valid)
        decision='NO_TRADE'; direction=0
        dc=CONFIG['decision']
        if trust_comp>=dc['min_composite_trust']:
            if pbig_valid<=dc['quiet_max_validated_big_prob']:
                decision='QUIET_CANDIDATE'
            elif pbig_valid>=dc['min_validated_big_prob'] and abs(edge_valid)>=dc['min_validated_direction_edge'] and agree>=dc['min_agreement']:
                direction=1 if edge_valid>0 else -1; decision='DIRECTIONAL_UP' if direction>0 else 'DIRECTIONAL_DOWN'
            elif pbig_valid>=dc['min_validated_big_prob']:
                decision='EXPANSION_UNCERTAIN'
        rec={'date':date,'timestamp':ts.isoformat(),'entry':row['entry'],'phase':'POSTLOCK' if date>='2026-08-18' else 'DEVELOPMENT',
             'pbig_raw':pbig_raw,'edge_raw':edge_raw,'expected_abs_bps':abs_raw,'composite_trust':trust_comp,'agreement':agree,
             'pbig_valid':pbig_valid,'edge_valid':edge_valid,'p_up_trade':p_up_trade,'p_down_trade':p_dn_trade,'p_neutral':p_neutral,
             'decision':decision,'direction':direction,'config_sha256':CONFIG_HASH}
        for h in CONFIG['horizons']:
            for k,v in heads[h].items(): rec[f'h{h}_{k}']=v
            y=row[f'y_bps{h}']; rec[f'y_bps{h}']=y
        records.append(rec)
        for h in CONFIG['horizons']:
            y=row[f'y_bps{h}']
            if np.isfinite(y):
                pending.append({'maturity':ts+pd.Timedelta(minutes=h),'idx':idx,'h':h,'pbig':heads[h]['pbig'],'pup':heads[h]['pup'],'pabs':heads[h]['pabs']})

R=pd.DataFrame(records)
R.to_csv(OUT/'v2_1_mtf_all_predictions.csv',index=False)

def summarize(z,name):
    out={'name':name,'rows':len(z)}
    if not len(z): return out
    ymag=(z.y_bps15.abs()>=7.5).astype(int)
    if ymag.nunique()>1:
        out['auc_big15']=roc_auc_score(ymag,z.pbig_valid)
    out['brier_big15']=brier_score_loss(ymag,z.pbig_valid)
    d=z[z.direction!=0].copy(); out['directional_signals']=len(d)
    if len(d):
        signed=d.direction*d.y_bps15; out['direction_accuracy']=float((signed>0).mean()); out['signed_bps_sum']=float(signed.sum()); out['signed_bps_mean']=float(signed.mean()); out['big_move_precision']=float((d.y_bps15.abs()>=7.5).mean())
    exp=z[z.decision=='EXPANSION_UNCERTAIN']; out['expansion_uncertain']=len(exp)
    if len(exp): out['expansion_big_precision']=float((exp.y_bps15.abs()>=7.5).mean()); out['expansion_avg_abs_bps']=float(exp.y_bps15.abs().mean())
    q=z[z.decision=='QUIET_CANDIDATE']; out['quiet_candidates']=len(q)
    if len(q): out['quiet_precision']=float((q.y_bps15.abs()<7.5).mean()); out['quiet_avg_abs_bps']=float(q.y_bps15.abs().mean())
    return out

summaries=[]
for phase,z in R.groupby('phase'): summaries.append(summarize(z,phase))
summaries.append(summarize(R,'ALL_OOS'))
S=pd.DataFrame(summaries); S.to_csv(OUT/'v2_1_mtf_summary.csv',index=False)

first=[]
for date,z in R.groupby('date',sort=False):
    q=z[z.direction!=0]
    if len(q): first.append(q.iloc[0])
F=pd.DataFrame(first)
if len(F):
    F['signed_bps']=F.direction*F.y_bps15
F.to_csv(OUT/'v2_1_mtf_first_cross_trades.csv',index=False)

R['trust_bin']=pd.qcut(R.composite_trust,4,duplicates='drop')
B=R.groupby(['phase','trust_bin'],observed=True).agg(n=('y_bps15','size'),trust=('composite_trust','mean'),big_p=('pbig_valid','mean'),actual_big=('y_bps15',lambda s:(s.abs()>=7.5).mean()),avg_abs=('y_bps15',lambda s:s.abs().mean())).reset_index()
B.to_csv(OUT/'v2_1_mtf_trust_bins.csv',index=False)

print('CONFIG_HASH',CONFIG_HASH)
print(S.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nFIRST CROSS')
if len(F):
    print(F[['date','timestamp','decision','pbig_valid','edge_valid','composite_trust','agreement','y_bps15','signed_bps']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
print('\nTRUST BINS')
print(B.to_string(index=False))
