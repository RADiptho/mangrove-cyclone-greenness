"""Stage 02. Two-harmonic baseline per event-cell (NDVI, EVI; relaxed and strict QA; main and placebo windows),
monthly anomalies, QA summaries. Stores the number of valid pre-landfall observations for every baseline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
import pandas as pd, numpy as np, time
t0=time.time()
obs=pd.read_pickle(W+"obs.pkl"); meta=pd.read_pickle(W+"meta.pkl")
P=len(meta)
obs["rel_m"]=np.floor(obs["rel_day"].astype(float)/30.4375).astype(np.int16)
obs=obs[(obs.rel_m>=-36)&(obs.rel_m<=24)].reset_index(drop=True)
w=2*np.pi*obs["day"].to_numpy(float)/365.2422
X=np.column_stack([np.ones(len(obs)),np.sin(w),np.cos(w),np.sin(2*w),np.cos(2*w),obs["sensor"].to_numpy(float)])
K=X.shape[1]
def fit(mask_fit, y):
    pid=obs["pid"].to_numpy()
    m=mask_fit & np.isfinite(y)
    XtX=np.zeros((P,K,K)); Xty=np.zeros((P,K))
    for a in range(K):
        Xty[:,a]=np.bincount(pid[m],weights=X[m,a]*y[m],minlength=P)
        for b in range(a,K):
            v=np.bincount(pid[m],weights=X[m,a]*X[m,b],minlength=P); XtX[:,a,b]=v; XtX[:,b,a]=v
    n=np.bincount(pid[m],minlength=P)
    XtX+=np.eye(K)[None]*1e-6
    beta=np.linalg.solve(XtX,Xty[...,None])[...,0]
    pred=(X*beta[pid]).sum(1)
    res=y-pred
    rss=np.bincount(pid[m],weights=res[m]**2,minlength=P)
    sd=np.sqrt(rss/np.maximum(n-K,1))
    return beta,pred,n,sd
out={}
for idx in ("ndvi","evi"):
    y=obs[idx].to_numpy(float)
    for qa in ("vrelax","vstrict"):
        valid=obs[qa].to_numpy()
        yy=np.where(valid,y,np.nan)
        for base,pre_mask in (("main",obs["rel_m"].to_numpy()<0),("placebo12",obs["rel_m"].to_numpy()<-12)):
            beta,pred,n,sd=fit(pre_mask,yy)
            an=yy-pred
            rm=obs["rel_m"].to_numpy()+36
            ok=np.isfinite(an)
            key=obs["pid"].to_numpy()[ok]*61+rm[ok]
            s=np.bincount(key,weights=an[ok],minlength=P*61); c=np.bincount(key,minlength=P*61)
            M=np.where(c>0,s/np.maximum(c,1),np.nan).reshape(P,61).astype(np.float32)
            out[(idx,qa,base)]={"M":M,"cnt":c.reshape(P,61).astype(np.int16),"beta":beta.astype(np.float32),"npre":n,"sdpre":sd.astype(np.float32)}
            print(idx,qa,base,"done",f"{time.time()-t0:.0f}s","median npre",np.median(n),flush=True)
pd.to_pickle(out,W+"anom.pkl")
for idx in ("ndvi","evi"):
    y=np.where(obs["vrelax"].to_numpy(),obs[idx].to_numpy(float),np.nan); ok=np.isfinite(y)
    key=obs["pid"].to_numpy()[ok]*61+(obs["rel_m"].to_numpy()[ok]+36)
    s=np.bincount(key,weights=y[ok],minlength=P*61); c=np.bincount(key,minlength=P*61)
    pd.to_pickle(np.where(c>0,s/np.maximum(c,1),np.nan).reshape(P,61).astype(np.float32),W+f"level_{idx}.pkl")
obs["period"]=np.where(obs.rel_m<0,"pre","post")
mm=meta[["pid","event_id","event_name","exposure_class"]]
q=obs.merge(mm,on="pid")
g=q.groupby(["event_id","event_name","exposure_class","sensor","period"])
qa=pd.DataFrame({"n_obs":g.size(),"n_fill":g["fill"].sum(),"n_good":g["sqa"].apply(lambda s:(s==0).sum()),
                 "n_marginal":g["sqa"].apply(lambda s:(s==1).sum()),"n_snow_ice":g["sqa"].apply(lambda s:(s==2).sum()),
                 "n_cloudy":g["sqa"].apply(lambda s:(s==3).sum()),"n_valid_strict":g["vstrict"].sum(),"n_valid_relaxed":g["vrelax"].sum(),
                 "n_cells":g["pid"].nunique()}).reset_index()
qa["sensor"]=qa["sensor"].map({0:"MOD13Q1 (Terra)",1:"MYD13Q1 (Aqua)"})
qa["frac_valid_relaxed"]=qa.n_valid_relaxed/qa.n_obs; qa["frac_valid_strict"]=qa.n_valid_strict/qa.n_obs
qa.to_pickle(W+"qa_summary.pkl")
q["cal_m"]=pd.to_datetime(q["day"],unit="D").dt.month
cov=q.groupby(["event_name","cal_m"])["vrelax"].mean().reset_index(); cov.to_pickle(W+"qa_calmonth.pkl")
print("DONE",time.time()-t0,flush=True)
