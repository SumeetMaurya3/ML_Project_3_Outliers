#!/usr/bin/env python3
"""Nested validation and CatBoost tuning for flight-price regression.

Assumes R^2 is the Kaggle metric (the competition metric was not supplied).
Five outer binned-stratified folds estimate generalization. Within every outer
training split, candidates are compared on a separate stratified tuning holdout;
then selected LightGBM/CatBoost/XGBoost models produce 3-fold inner OOF predictions
used to optimize a nonnegative, sum-to-one ensemble. The untouched outer fold is
used exactly once to evaluate that entire procedure.

Input files expected alongside this script: train(3).csv, test(3).csv,
submission(2).csv.
"""
from __future__ import annotations
import json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings('ignore', category=FutureWarning)
SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
OUT = Path(__file__).resolve().parent
TRAIN_PATH = OUT / 'train(3).csv'
TEST_PATH = OUT / 'test(3).csv'
SAMPLE_PATH = OUT / 'submission(2).csv'
MODEL_NAMES = ['LightGBM', 'CatBoost', 'XGBoost']

CAT_COLS = ['airline','source_city','departure_time','stops','arrival_time','destination_city',
            'flight_class','route','city_pair','route_stops','route_class','airline_route',
            'route_departure','airline_class']
NUM_COLS = ['duration','days_left','stop_count','is_business','duration_log','days_left_log',
            'duration_per_leg','duration_x_stops']

LGB_CONFIGS = [
    dict(n_estimators=1350, learning_rate=0.040, num_leaves=31, max_depth=-1,
         min_child_samples=25, subsample=0.85, subsample_freq=1,
         colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=5.0),
    dict(n_estimators=1650, learning_rate=0.030, num_leaves=47, max_depth=-1,
         min_child_samples=30, subsample=0.85, subsample_freq=1,
         colsample_bytree=0.85, reg_alpha=0.3, reg_lambda=8.0),
]
# Expanded search across tree depth, learning rate, leaf regularization and random strength.
CAT_CONFIGS = [
    dict(iterations=1000, depth=5, learning_rate=0.050, l2_leaf_reg=5.0, random_strength=0.5),
    dict(iterations=1100, depth=6, learning_rate=0.050, l2_leaf_reg=5.0, random_strength=0.5),
    dict(iterations=1200, depth=7, learning_rate=0.040, l2_leaf_reg=7.0, random_strength=0.7),
    dict(iterations=1200, depth=8, learning_rate=0.030, l2_leaf_reg=8.0, random_strength=1.0),
    dict(iterations=1200, depth=6, learning_rate=0.030, l2_leaf_reg=10.0, random_strength=0.5),
    dict(iterations=1100, depth=7, learning_rate=0.050, l2_leaf_reg=10.0, random_strength=1.0),
    dict(iterations=900, depth=5, learning_rate=0.070, l2_leaf_reg=8.0, random_strength=0.5),
    dict(iterations=1100, depth=8, learning_rate=0.040, l2_leaf_reg=12.0, random_strength=1.0),
]
XGB_CONFIGS = [
    dict(n_estimators=1200, max_depth=5, learning_rate=0.040, min_child_weight=8,
         subsample=0.90, colsample_bytree=0.90, reg_alpha=0.1, reg_lambda=10.0),
    dict(n_estimators=1450, max_depth=6, learning_rate=0.030, min_child_weight=6,
         subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=8.0),
]
CONFIGS = {'LightGBM': LGB_CONFIGS, 'CatBoost': CAT_CONFIGS, 'XGBoost': XGB_CONFIGS}


def log(s): print(f'[{time.strftime("%H:%M:%S")}] {s}', flush=True)
def get_r2(y,p): return float(r2_score(np.asarray(y), np.asarray(p)))
def get_rmse(y,p): return float(np.sqrt(mean_squared_error(np.asarray(y), np.asarray(p))))


def build_features(df):
    x = df.copy().drop(columns=[c for c in ['id','flight','price'] if c in df.columns], errors='ignore')
    base_cat = ['airline','source_city','departure_time','stops','arrival_time','destination_city','flight_class']
    for c in base_cat:
        if c not in x: x[c] = '__MISSING__'
        x[c] = x[c].fillna('__MISSING__').astype(str)
    src, dst = x['source_city'], x['destination_city']
    x['route'] = src + '__TO__' + dst
    s, d = src.to_numpy(), dst.to_numpy()
    lo, hi = np.where(s <= d, s, d), np.where(s <= d, d, s)
    x['city_pair'] = pd.Series(lo, index=x.index) + '__PAIR__' + pd.Series(hi, index=x.index)
    x['route_stops'] = x['route'] + '__' + x['stops']
    x['route_class'] = x['route'] + '__' + x['flight_class']
    x['airline_route'] = x['airline'] + '__' + x['route']
    x['route_departure'] = x['route'] + '__' + x['departure_time']
    x['airline_class'] = x['airline'] + '__' + x['flight_class']
    x['stop_count'] = x['stops'].map({'zero':0.0,'one':1.0,'two_or_more':2.0}).fillna(1.0)
    x['duration'] = pd.to_numeric(x.get('duration', np.nan), errors='coerce')
    x['days_left'] = pd.to_numeric(x.get('days_left', np.nan), errors='coerce')
    x['is_business'] = (x['flight_class'].str.lower() == 'business').astype(int)
    x['duration_log'] = np.log1p(x['duration'].clip(lower=0))
    x['days_left_log'] = np.log1p(x['days_left'].clip(lower=0))
    x['duration_per_leg'] = x['duration']/(1+x['stop_count'])
    x['duration_x_stops'] = x['duration']*(1+x['stop_count'])
    x[NUM_COLS] = x[NUM_COLS].replace([np.inf,-np.inf],np.nan)
    for c in CAT_COLS:
        if c not in x: x[c] = '__MISSING__'
        x[c] = x[c].fillna('__MISSING__').astype(str)
    return x[CAT_COLS+NUM_COLS]


def quantile_bins(y, q=10):
    s = pd.Series(np.asarray(y))
    b = pd.qcut(s, q=q, labels=False, duplicates='drop')
    if b.isna().any() or b.nunique() < 2:
        b = pd.cut(s, bins=min(q, max(2, int(s.nunique()))), labels=False, duplicates='drop')
    return np.asarray(b).astype(int)


def make_folds(X,y,n_splits,seed):
    bins = quantile_bins(y)
    return list(StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(np.zeros(len(y)),bins))


def split_tuning_holdout(y, seed):
    bins = quantile_bins(y)
    ss = StratifiedShuffleSplit(n_splits=1,test_size=0.20,random_state=seed)
    return next(ss.split(np.zeros(len(y)),bins))


def prep_lgb(Xtr, Xva, Xte=None):
    tr, va = Xtr.copy(), Xva.copy()
    te = Xte.copy() if Xte is not None else None
    for c in CAT_COLS:
        series = [tr[c].astype(str), va[c].astype(str)]
        if te is not None: series.append(te[c].astype(str))
        cats = pd.Index(pd.concat(series, ignore_index=True).unique())
        tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
        va[c] = pd.Categorical(va[c].astype(str), categories=cats)
        if te is not None: te[c] = pd.Categorical(te[c].astype(str), categories=cats)
    return tr, va, te


def make_lgb(cfg):
    return LGBMRegressor(objective='regression',random_state=SEED,n_jobs=4,verbosity=-1,**cfg)
def make_cat(cfg):
    return CatBoostRegressor(loss_function='RMSE',eval_metric='RMSE',random_seed=SEED,
                             thread_count=4,allow_writing_files=False,verbose=False,**cfg)
def make_xgb(cfg):
    return XGBRegressor(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',
                        random_state=SEED,n_jobs=4,verbosity=0,early_stopping_rounds=60,**cfg)


def make_xgb_encoder():
    try: ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
    except TypeError: ohe = OneHotEncoder(handle_unknown='ignore',sparse=True,dtype=np.float32)
    return ColumnTransformer([('cat',ohe,CAT_COLS),('num','passthrough',NUM_COLS)],remainder='drop',sparse_threshold=1.0)


def fit_eval(name,cfg,Xtr,ytr,Xva,yva):
    if name == 'LightGBM':
        a,b,_ = prep_lgb(Xtr,Xva)
        m = make_lgb(cfg)
        m.fit(a,ytr,categorical_feature=CAT_COLS,eval_set=[(b,yva)],
              callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(0)])
        p = m.predict(b,num_iteration=m.best_iteration_)
        it = int(m.best_iteration_ or cfg['n_estimators'])
        enc = None
    elif name == 'CatBoost':
        m = make_cat(cfg)
        m.fit(Xtr,ytr,cat_features=CAT_COLS,eval_set=(Xva,yva),early_stopping_rounds=75,
              use_best_model=True,verbose=False)
        p = m.predict(Xva)
        bi = m.get_best_iteration()
        it = int(bi+1 if bi is not None and bi >= 0 else cfg['iterations'])
        enc = None
    else:
        enc = make_xgb_encoder()
        atr = enc.fit_transform(Xtr); ava = enc.transform(Xva)
        m = make_xgb(cfg)
        m.fit(atr,ytr,eval_set=[(ava,yva)],verbose=False)
        p = m.predict(ava)
        bi = getattr(m,'best_iteration',None)
        it = int(bi+1 if bi is not None else cfg['n_estimators'])
    return np.asarray(p,dtype=float),it


def fit_final(name,cfg,niter,X,y,Xtest):
    niter = max(50,int(niter))
    if name == 'LightGBM':
        a,te,_ = prep_lgb(X,Xtest)
        params=dict(cfg);params['n_estimators']=niter
        m=LGBMRegressor(objective='regression',random_state=SEED,n_jobs=4,verbosity=-1,**params)
        m.fit(a,y,categorical_feature=CAT_COLS)
        return {'model':m,'encoder':None,'params':params},np.asarray(m.predict(te),float)
    if name == 'CatBoost':
        params=dict(cfg);params['iterations']=niter
        m=CatBoostRegressor(loss_function='RMSE',eval_metric='RMSE',random_seed=SEED,
                            thread_count=4,allow_writing_files=False,verbose=False,**params)
        m.fit(X,y,cat_features=CAT_COLS,verbose=False)
        return {'model':m,'encoder':None,'params':params},np.asarray(m.predict(Xtest),float)
    params=dict(cfg);params['n_estimators']=niter
    enc=make_xgb_encoder(); atr=enc.fit_transform(X); ate=enc.transform(Xtest)
    m=XGBRegressor(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',
                   random_state=SEED,n_jobs=4,verbosity=0,**params)
    m.fit(atr,y,verbose=False)
    return {'model':m,'encoder':enc,'params':params},np.asarray(m.predict(ate),float)


def optimize_weights(y,P):
    y=np.asarray(y,float);P=np.asarray(P,float);n=P.shape[1]
    equal=np.ones(n)/n
    # R2 maximization == squared-error minimization for a fixed response vector.
    def objective(w): return float(np.mean((y-P@w)**2))
    res=minimize(objective,equal,method='SLSQP',bounds=[(0.0,1.0)]*n,
                 constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1.0}],
                 options={'maxiter':300,'ftol':1e-9})
    w=res.x if res.success and np.all(np.isfinite(res.x)) else equal
    w=np.clip(w,0,1);return w/w.sum()


def select_configs_on_holdout(X,y,phase,seed):
    ti,vi=split_tuning_holdout(y,seed)
    Xtr,Xva=X.iloc[ti].reset_index(drop=True),X.iloc[vi].reset_index(drop=True)
    ytr,yva=y.iloc[ti].reset_index(drop=True),y.iloc[vi].reset_index(drop=True)
    selected={}; rows=[]
    for name in MODEL_NAMES:
        cands=[]
        for ci,cfg in enumerate(CONFIGS[name],1):
            log(f'  {phase}: candidate {name} {ci}/{len(CONFIGS[name])}')
            pred,it=fit_eval(name,cfg,Xtr,ytr,Xva,yva)
            row={'phase':phase,'model':name,'config_id':ci,'tuning_holdout_r2':get_r2(yva,pred),
                 'tuning_holdout_rmse':get_rmse(yva,pred),'tuning_holdout_mae':float(mean_absolute_error(yva,pred)),
                 'best_iteration':it,'parameters':json.dumps(cfg,sort_keys=True)}
            rows.append(row);cands.append((row,cfg))
            log(f'    {name} c{ci}: holdout R2={row["tuning_holdout_r2"]:.5f}, RMSE={row["tuning_holdout_rmse"]:.3f}, iter={it}')
        best=max(cands,key=lambda x:(x[0]['tuning_holdout_r2'],-x[0]['tuning_holdout_rmse']))
        selected[name]={'summary':best[0],'cfg':best[1]}
        log(f'  {phase}: selected {name} config {best[0]["config_id"]}')
    return selected,rows


def selected_oof_predictions(X,y,selected,phase,seed):
    folds=make_folds(X,y,INNER_SPLITS,seed)
    P=np.full((len(y),3),np.nan); outrows=[]; median_iters={}
    for j,name in enumerate(MODEL_NAMES):
        cfg=selected[name]['cfg'];its=[]
        for fi,(ti,vi) in enumerate(folds,1):
            pred,it=fit_eval(name,cfg,X.iloc[ti].reset_index(drop=True),y.iloc[ti].reset_index(drop=True),
                             X.iloc[vi].reset_index(drop=True),y.iloc[vi].reset_index(drop=True))
            P[vi,j]=pred;its.append(it)
            outrows.append({'phase':phase,'model':name,'fold':fi,'r2':get_r2(y.iloc[vi],pred),
                            'rmse':get_rmse(y.iloc[vi],pred),'mae':float(mean_absolute_error(y.iloc[vi],pred)),
                            'best_iteration':it,'config_id':selected[name]['summary']['config_id']})
            log(f'    {phase}: selected {name} inner OOF fold {fi}/{len(folds)} R2={outrows[-1]["r2"]:.5f}, it={it}')
        median_iters[name]=max(50,int(np.median(its)))
        selected[name]['median_best_iteration']=median_iters[name]
    if np.isnan(P).any(): raise RuntimeError(f'NaN OOF predictions in {phase}')
    return P,median_iters,outrows


def main():
    train0=pd.read_csv(TRAIN_PATH);test0=pd.read_csv(TEST_PATH);sample=pd.read_csv(SAMPLE_PATH)
    raw_n=len(train0)
    removed_nonpositive=int((train0.price<=0).sum());removed_high=int((train0.price>5000).sum())
    cleaned=train0.loc[(train0.price>0)&(train0.price<=5000)].copy().reset_index(drop=True)
    y=cleaned.price.astype(float).reset_index(drop=True)
    X=build_features(cleaned).reset_index(drop=True);Xtest=build_features(test0).reset_index(drop=True)
    for c in CAT_COLS:
        X[c]=X[c].astype(str).fillna('__MISSING__');Xtest[c]=Xtest[c].astype(str).fillna('__MISSING__')
    log(f'Data train={raw_n}, cleaned={len(cleaned)}, test={len(test0)}; removed price<=0={removed_nonpositive}, price>5000={removed_high}')
    log(f'Assumed metric R2; nested design outer={OUTER_SPLITS}, inner={INNER_SPLITS}, CatBoost candidates={len(CAT_CONFIGS)}')

    # The outer held-out rows are not used for tuning, base-model OOF weights, or iteration choice.
    outer_folds=make_folds(X,y,OUTER_SPLITS,SEED)
    nested_base=np.full((len(y),3),np.nan);nested_equal=np.full(len(y),np.nan);nested_optimized=np.full(len(y),np.nan)
    outer_rows=[];weight_rows=[];candidate_rows=[];inner_oof_rows=[]
    for oi,(oti,ovi) in enumerate(outer_folds,1):
        phase=f'outer{oi}'
        Xtr=X.iloc[oti].reset_index(drop=True);ytr=y.iloc[oti].reset_index(drop=True)
        Xva=X.iloc[ovi].reset_index(drop=True);yva=y.iloc[ovi].reset_index(drop=True)
        log(f'\n========== OUTER FOLD {oi}/{OUTER_SPLITS}: train={len(oti)}, validation={len(ovi)} ==========')
        selected,crows=select_configs_on_holdout(Xtr,ytr,phase,SEED+oi*101)
        candidate_rows.extend(crows)
        Pin,median_iters,orows=selected_oof_predictions(Xtr,ytr,selected,phase,SEED+oi*1000)
        inner_oof_rows.extend(orows)
        weights=optimize_weights(ytr,Pin)
        log(f'{phase}: optimized inner weights ' + ', '.join(f'{m}={w:.4f}' for m,w in zip(MODEL_NAMES,weights)))
        Pout=[]
        for j,name in enumerate(MODEL_NAMES):
            log(f'{phase}: outer refit {name} with {median_iters[name]} iterations')
            _,pred=fit_final(name,selected[name]['cfg'],median_iters[name],Xtr,ytr,Xva)
            Pout.append(pred)
        Pout=np.column_stack(Pout)
        nested_base[ovi]=Pout
        nested_equal[ovi]=Pout.mean(axis=1)
        nested_optimized[ovi]=Pout@weights
        row={'outer_fold':oi,'n_validation':len(ovi),'selected_lgb_config':selected['LightGBM']['summary']['config_id'],
             'selected_cat_config':selected['CatBoost']['summary']['config_id'],'selected_xgb_config':selected['XGBoost']['summary']['config_id'],
             'weight_lightgbm':weights[0],'weight_catboost':weights[1],'weight_xgboost':weights[2],
             'equal_blend_r2':get_r2(yva,nested_equal[ovi]),'equal_blend_rmse':get_rmse(yva,nested_equal[ovi]),
             'optimized_blend_r2':get_r2(yva,nested_optimized[ovi]),'optimized_blend_rmse':get_rmse(yva,nested_optimized[ovi])}
        for j,name in enumerate(MODEL_NAMES):
            row[f'{name}_r2']=get_r2(yva,Pout[:,j]);row[f'{name}_rmse']=get_rmse(yva,Pout[:,j])
        outer_rows.append(row)
        weight_rows.append({'outer_fold':oi,**{f'weight_{name.lower()}':weights[j] for j,name in enumerate(MODEL_NAMES)}})
        log(f'{phase}: held-out R2={row["optimized_blend_r2"]:.5f}, RMSE={row["optimized_blend_rmse"]:.3f}')

    metrics=[]
    for j,name in enumerate(MODEL_NAMES):
        metrics.append({'model':name,'nested_oof_r2':get_r2(y,nested_base[:,j]),'nested_oof_rmse':get_rmse(y,nested_base[:,j]),
                        'nested_oof_mae':float(mean_absolute_error(y,nested_base[:,j]))})
    metrics.append({'model':'Equal-weight blend','nested_oof_r2':get_r2(y,nested_equal),'nested_oof_rmse':get_rmse(y,nested_equal),
                    'nested_oof_mae':float(mean_absolute_error(y,nested_equal))})
    metrics.append({'model':'Nested optimized blend','nested_oof_r2':get_r2(y,nested_optimized),'nested_oof_rmse':get_rmse(y,nested_optimized),
                    'nested_oof_mae':float(mean_absolute_error(y,nested_optimized))})
    metrics_df=pd.DataFrame(metrics);outer_df=pd.DataFrame(outer_rows)
    metrics_df.to_csv(OUT/'flight_price_nested_cv_metrics.csv',index=False)
    outer_df.to_csv(OUT/'flight_price_nested_cv_outer_folds.csv',index=False)
    pd.DataFrame(weight_rows).to_csv(OUT/'flight_price_nested_cv_weights_by_outer_fold.csv',index=False)
    pd.DataFrame(candidate_rows).to_csv(OUT/'flight_price_nested_cv_candidate_results.csv',index=False)
    pd.DataFrame(inner_oof_rows).to_csv(OUT/'flight_price_nested_cv_inner_fold_details.csv',index=False)
    pd.DataFrame({'id':cleaned.id,'actual_price':y,'nested_oof_lightgbm':nested_base[:,0],
                  'nested_oof_catboost':nested_base[:,1],'nested_oof_xgboost':nested_base[:,2],
                  'nested_oof_equal_blend':nested_equal,'nested_oof_optimized_blend':nested_optimized}).to_csv(
                    OUT/'flight_price_nested_cv_oof_predictions.csv',index=False)

    # For the final Kaggle predictor: retune using all available train rows, then refit.
    log('\n========== FINAL FULL-DATA TUNING FOR SUBMISSION ==========')
    final_selected,final_crows=select_configs_on_holdout(X,y,'final_full_data',SEED+999)
    final_candidates_oof,final_iters,final_orows=selected_oof_predictions(X,y,final_selected,'final_full_data',SEED+1999)
    final_weights=optimize_weights(y,final_candidates_oof)
    pd.DataFrame(final_crows).to_csv(OUT/'flight_price_nested_final_candidate_results.csv',index=False)
    pd.DataFrame(final_orows).to_csv(OUT/'flight_price_nested_final_inner_oof_details.csv',index=False)
    final_models={};test_mtx=[]
    for name in MODEL_NAMES:
        log(f'Final full-data refit {name}, config={final_selected[name]["summary"]["config_id"]}, iter={final_iters[name]}')
        bundle,pred=fit_final(name,final_selected[name]['cfg'],final_iters[name],X,y,Xtest)
        final_models[name]={**bundle,'config_id':final_selected[name]['summary']['config_id'],
                            'median_best_iteration':final_iters[name], 'feature_config':final_selected[name]['cfg']}
        test_mtx.append(pred)
    Ptest=np.column_stack(test_mtx);test_pred=np.maximum(Ptest@final_weights,0)
    pred_ids=pd.DataFrame({'id':test0.id.to_numpy(),'price':test_pred})
    submission=sample[['id']].merge(pred_ids,on='id',how='left',validate='one_to_one')
    if submission.price.isna().any(): raise RuntimeError('Missing predictions after joining to sample submission ids.')
    subpath=OUT/'submission_nested_lightgbm_catboost_xgboost.csv'
    submission[['id','price']].to_csv(subpath,index=False)
    pd.DataFrame({'model':MODEL_NAMES,'final_weight':final_weights,
                  'selected_config_id':[final_selected[m]['summary']['config_id'] for m in MODEL_NAMES],
                  'median_best_iteration':[final_iters[m] for m in MODEL_NAMES],
                  'parameters':[json.dumps(final_selected[m]['cfg'],sort_keys=True) for m in MODEL_NAMES]}).to_csv(
                    OUT/'flight_price_nested_final_blend.csv',index=False)
    pd.DataFrame({'id':test0.id.to_numpy(),'lightgbm_prediction':Ptest[:,0],
                  'catboost_prediction':Ptest[:,1],'xgboost_prediction':Ptest[:,2],
                  'blended_prediction':test_pred}).to_csv(OUT/'flight_price_nested_test_predictions.csv',index=False)
    joblib.dump({'models':final_models,'blend_weights':final_weights.tolist(),'categorical_features':CAT_COLS,
                 'numerical_features':NUM_COLS,'cleaning_rule':'train price > 0 and <= 5000',
                 'assumed_metric':'R2','final_configs':{m:final_selected[m]['cfg'] for m in MODEL_NAMES}},
                OUT/'flight_price_nested_ensemble_models.joblib')

    report=['FLIGHT PRICE ENSEMBLE — EXTENDED CATBOOST TUNING & NESTED VALIDATION','='*78,
            f'Original train={raw_n}; cleaned train={len(cleaned)}; test={len(test0)}; removed price<=0={removed_nonpositive}; removed price>5000={removed_high}.',
            'The >5000 threshold is retained from the previous experiment; it is a heuristic, not proof that all such values are invalid.',
            'Features: route, unordered city pair, route×stops, route×class, airline×route, route×departure, airline×class, log duration/days_left and stop interactions.',
            'Excluded id and flight from the feature matrix; no target-derived features or target encoding were used.',
            'Metric assumption: R², because the Kaggle evaluation specification was not supplied.',
            f'Nested CV: outer {OUTER_SPLITS}-fold binned StratifiedKFold; inner {INNER_SPLITS}-fold binned StratifiedKFold for OOF weights.',
            f'Within each outer training fold, candidate configs were compared on a separate 20% binned-stratified tuning holdout. CatBoost searched {len(CAT_CONFIGS)} configs across depth, learning rate, L2 regularization and random strength; LightGBM/XGBoost searched {len(LGB_CONFIGS)}/{len(XGB_CONFIGS)} configs.',
            'Blend weights were optimized only on inner OOF predictions and then applied to the outer held-out fold, which was not used for model selection or weight selection.',
            '', 'NESTED OUTER-OOF METRICS (primary estimate of generalization):',metrics_df.to_string(index=False),
            '', 'OUTER FOLD DETAILS:',outer_df.to_string(index=False), '', 'FINAL FULL-TRAIN CONFIGS AND BLEND WEIGHTS:']
    for name in MODEL_NAMES:
        s=final_selected[name]['summary']
        report.append(f'{name}: candidate={s["config_id"]}; tuning-holdout R²={s["tuning_holdout_r2"]:.6f}; iterations={final_iters[name]}; params={json.dumps(final_selected[name]["cfg"],sort_keys=True)}')
    report.append('Final weights (optimized on full-training OOF predictions for submission only): '+', '.join(f'{m}={w:.6f}' for m,w in zip(MODEL_NAMES,final_weights)))
    report.append(f'Submission={subpath.name}; rows={len(submission)}; missing={int(submission.price.isna().sum())}; min={submission.price.min():.3f}; median={submission.price.median():.3f}; max={submission.price.max():.3f}.')
    report.append('The nested R² is calculated on the cleaned target subset. Compare it with the reported public score only provisionally because the Kaggle metric and the handling of extreme rows are unconfirmed.')
    (OUT/'flight_price_nested_validation_report.txt').write_text('\n'.join(report)+'\n')
    log('\n========== NESTED CV SUMMARY ==========');log(metrics_df.to_string(index=False))
    log('Final weights: '+', '.join(f'{m}={w:.4f}' for m,w in zip(MODEL_NAMES,final_weights)))
    log(f'Submission written: {subpath}');log('DONE')

if __name__=='__main__': main()
