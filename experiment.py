"""King County: group-disjoint nested CV, encoded-column selection, split conformal."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hw2_matplotlib')
import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.stats.diagnostic import het_breuschpagan

FEATURES = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']
NUMERIC = [x for x in FEATURES if x != 'zipcode']
DISCRETE = {'bedrooms','floors','waterfront','view','condition','grade','yr_built','yr_renovated'}
METHODS = ['M0','M1','M2','M3','M4','M5']
COUNTS = [5,10,20,40,60]
ALPHAS = np.logspace(0,5,11).tolist()
SEED = 42
SOURCE = 'https://www.kaggle.com/datasets/harlfoxem/housesalesprediction'


def json_write(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple,np.ndarray)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, Path): return str(x.resolve())
        return x
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False), encoding='utf-8')


def validate_data(df, strict=True):
    expected = {'id','date','price',*FEATURES}
    if set(df.columns) != expected: raise ValueError('Unexpected dataset columns')
    if strict and df.shape != (21613,21): raise ValueError(f'Unexpected shape: {df.shape}')
    if df.id.isna().any(): raise ValueError('Missing property identifier')
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in ['price',*FEATURES]):
        raise ValueError('Non-numeric target/predictor')
    if not np.isfinite(df.price).all() or (df.price <= 0).any(): raise ValueError('Invalid target')
    if np.isinf(df[FEATURES].to_numpy()).any(): raise ValueError('Infinite predictor')
    pd.to_datetime(df.date, format='%Y%m%dT%H%M%S', errors='raise')
    assert len(FEATURES)==18 and not {'price','id','date'} & set(FEATURES)


def partitions(df):
    rest,test = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(df,groups=df.id))
    a,b = next(GroupShuffleSplit(n_splits=1,test_size=.15/.8,random_state=SEED).split(df.iloc[rest],groups=df.id.iloc[rest]))
    parts = {'Development':rest[a], 'Calibration':rest[b], 'Test':test}
    assert len(np.unique(np.concatenate(list(parts.values())))) == len(df)
    for k,v in parts.items():
        for k2,v2 in parts.items():
            if k!=k2: assert not set(df.id.iloc[v]) & set(df.id.iloc[v2])
    return parts


def folds(X, groups, n=3):
    result = list(GroupKFold(n_splits=n).split(X,groups=groups))
    for a,b in result:
        assert not set(np.asarray(groups)[a]) & set(np.asarray(groups)[b])
    return result


def preprocessing():
    # All encoded columns, including ZIP dummies, are centered and scaled to unit SD.
    return Pipeline([('encode',ColumnTransformer([
        ('num',SimpleImputer(strategy='median',keep_empty_features=True),NUMERIC),
        ('zip',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                        ('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))]),['zipcode'])
    ],verbose_feature_names_out=True)),('scale',StandardScaler())])


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method='M0', value=None): self.method,self.value=method,value
    def fit(self,X,y):
        p=X.shape[1]
        self.scores_=np.zeros(p); self.ranking_=np.ones(p,dtype=int)
        self.support_=np.ones(p,dtype=bool)
        if self.method in ('M1','M2'):
            if self.value>p: raise ValueError('Feature count exceeds encoded width')
            if self.method=='M1':
                xc=X-X.mean(0); yc=y-np.mean(y)
                denom=np.sqrt((xc*xc).sum(0)*(yc*yc).sum())
                self.scores_=np.divide(xc.T@yc,denom,out=np.zeros(p),where=denom>0)
            else:
                mask=np.array([c in DISCRETE for c in NUMERIC]+[True]*(p-len(NUMERIC)))
                self.scores_=mutual_info_regression(X,y,discrete_features=mask,random_state=SEED)
            order=np.argsort(-np.abs(self.scores_),kind='stable')
            self.support_[:]=False; self.support_[order[:int(self.value)]]=True
            self.ranking_[order]=np.arange(1,p+1)
        elif self.method in ('M3','M4'):
            if self.value>p: raise ValueError('Feature count exceeds encoded width')
            r=RFE(LinearRegression(),n_features_to_select=int(self.value),step=5).fit(X,y)
            self.support_,self.ranking_=r.support_,r.ranking_
        elif self.method=='M5':
            # A convergence warning invalidates a configuration rather than being hidden.
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                lasso=Lasso(alpha=self.value,max_iter=1000000,tol=1e-5,precompute=True,random_state=SEED).fit(X,y)
            self.scores_=lasso.coef_; self.support_=np.abs(lasso.coef_)>1e-8
            if not self.support_.any(): raise ValueError('LASSO selected zero columns')
        return self
    def transform(self,X): return X[:,self.support_]


def fit_model(X,y,method,value):
    steps=[('preprocess',preprocessing())]
    if method!='M0': steps.append(('select',ColumnSelector(method,value)))
    steps.append(('regressor',LinearRegression()))
    return Pipeline(steps).fit(X,y)


def tune(X,y,groups,method,context,logs,audit):
    if method=='M0': return fit_model(X,y,method,None),None
    splits=folds(X,groups)
    # Fit preprocessing separately inside EACH inner training fold; cache only within
    # that fold. No target/validation information enters fitting or recursive paths.
    cached=[]
    for j,(a,b) in enumerate(splits):
        pre=preprocessing(); A=pre.fit_transform(X.iloc[a]); B=pre.transform(X.iloc[b])
        cached.append((A,B,np.asarray(y)[a],np.asarray(y)[b]))
        audit.append({'context':context,'method':method,'inner_fold':j+1,'train_rows':len(a),'validation_rows':len(b),
                      'train_groups':len(np.unique(np.asarray(groups)[a])),'validation_groups':len(np.unique(np.asarray(groups)[b])),
                      'group_overlap':0,'train_row_indices':X.index[a].tolist(),'validation_row_indices':X.index[b].tolist()})
    min_p=min(z[0].shape[1] for z in cached)
    if method=='M5': candidates=ALPHAS
    elif method=='M4':
        # RFECV-equivalent recursive elimination with fold-local preprocessing.
        # Standard RFECV after a fitted preprocessor leaks inner validation statistics.
        # Align fold paths by their common encoded counts when ZIP categories differ.
        candidates=sorted(set([1]+list(range(min_p,0,-5))))
    else: candidates=[k for k in COUNTS if k<=min_p]
    scores={v:[] for v in candidates}
    for j,(A,B,ya,yb) in enumerate(cached):
        for value in candidates:
            row={'context':context,'method':method,'inner_fold':j+1,'value':value}
            try:
                selector=ColumnSelector(method,value).fit(A,ya)
                pred=LinearRegression().fit(selector.transform(A),ya).predict(selector.transform(B))
                score=float(np.sqrt(mean_squared_error(yb,pred)))
                scores[value].append(score); row.update(rmse=score,status='ok')
            except (ValueError,ConvergenceWarning) as e:
                scores[value].append(float('inf')); row.update(rmse=None,status='invalid',reason=str(e))
            logs.append(row)
    valid=[v for v in candidates if np.isfinite(scores[v]).all()]
    if not valid: raise RuntimeError(f'{context} {method}: no valid candidates')
    best=min(valid,key=lambda v:(np.mean(scores[v]),v))
    return fit_model(X,y,method,best),best


def metadata(model):
    names=model['preprocess'].get_feature_names_out()
    support=model['select'].support_ if 'select' in model.named_steps else np.ones(len(names),dtype=bool)
    return names,support


def design(model,X):
    A=model['preprocess'].transform(X)
    return model['select'].transform(A) if 'select' in model.named_steps else A


def effective_rank(A): return int(np.linalg.matrix_rank(np.column_stack([np.ones(len(A)),A]))-1)


def metrics(y,pred,rank):
    mse=mean_squared_error(y,pred); r2=r2_score(y,pred); n=len(y)
    return dict(mae=mean_absolute_error(y,pred),mse=mse,rmse=np.sqrt(mse),r2=r2,
                adjusted_r2=1-(1-r2)*(n-1)/(n-rank-1) if n>rank+1 else np.nan,effective_rank=rank)


def choose(summary):
    best=summary.rmse_mean.idxmin(); threshold=summary.loc[best,'rmse_mean']+summary.loc[best,'rmse_std']/np.sqrt(5)
    eligible=summary.loc[summary.rmse_mean<=threshold].sort_values(['selected_count_mean','rmse_mean','method'])
    return {'best_rmse_method':summary.loc[best,'method'],'threshold':threshold,'eligible_methods':eligible.method.tolist(),
            'selected_method':eligible.method.iloc[0], 'policy':'one SE; fewest mean selected encoded columns, then RMSE, then M0..M5',
            'selected_mean_count':eligible.selected_count_mean.iloc[0], 'frozen_before_test':True}


def conformal(scores,alpha=.05):
    k=int(np.ceil((len(scores)+1)*(1-alpha)))
    return (float(np.partition(scores,k-1)[k-1]) if k<=len(scores) else float('inf')),k


class Outputs:
    def __init__(self,path):
        self.path=Path(path)
        for d in ['tables','figures','models']: (self.path/d).mkdir(parents=True,exist_ok=True)
    def table(self,name,data):
        df=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
        df.to_csv(self.path/'tables'/f'{name}.csv',index=False); return df
    def fig(self,name):
        plt.tight_layout(); plt.savefig(self.path/'figures'/f'{name}.png',dpi=220,bbox_inches='tight'); plt.close()


def eda(df,dev,out):
    dates=pd.to_datetime(df.date,format='%Y%m%dT%H%M%S')
    identity=np.isclose(df.sqft_living,df.sqft_above+df.sqft_basement,rtol=0,atol=0)
    info={'rows':len(df),'columns':len(df.columns),'original_predictors':18,'unique_properties':df.id.nunique(),
          'repeated_id_rows':df.id.duplicated().sum(),'properties_with_repeated_ids':(df.id.value_counts()>1).sum(),
          'duplicate_rows':df.duplicated().sum(),'missing_values':df.isna().sum().sum(),
          'area_identity_rows':identity.sum(),'area_identity_discrepancies':(~identity).sum(),
          'date_min':str(dates.min()),'date_max':str(dates.max()),'development_target_skew':dev.price.skew()}
    out.table('data_quality_summary',[{'measure':k,'value':v} for k,v in info.items()])
    out.table('column_quality',[{'column':c,'dtype':str(df[c].dtype),'role':'categorical' if c=='zipcode' else 'numeric' if c in NUMERIC else 'metadata/target',
                                'missing_count':df[c].isna().sum(),'missing_pct':df[c].isna().mean()*100} for c in df])
    out.table('development_descriptive_statistics',dev[FEATURES+['price']].describe().T.rename_axis('feature').reset_index())
    out.table('development_price_correlations',dev[NUMERIC+['price']].corr()['price'].rename_axis('feature').reset_index())
    corr=dev[NUMERIC].corr(); out.table('development_feature_correlations',corr.rename_axis('feature').reset_index())
    out.table('high_correlation_pairs',[{'feature_a':a,'feature_b':b,'correlation':corr.loc[a,b]} for i,a in enumerate(NUMERIC) for b in NUMERIC[i+1:] if abs(corr.loc[a,b])>.8])
    out.table('unusual_values',[{'check':k,'count':v} for k,v in {
        'bedrooms_zero':(df.bedrooms==0).sum(),'bedrooms_over_10':(df.bedrooms>10).sum(),
        'bathrooms_zero':(df.bathrooms==0).sum(),'living_area_nonpositive':(df.sqft_living<=0).sum(),
        'renovation_zero_meaningful':(df.yr_renovated==0).sum(),'basement_zero_meaningful':(df.sqft_basement==0).sum()}.items()])
    plt.figure(); plt.hist(dev.price,bins=70); plt.xlabel('Sale price (USD)'); plt.ylabel('Development transactions'); plt.title(f'Development price distribution; skew={dev.price.skew():.2f}'); out.fig('01_target_distribution')
    plt.figure(); plt.boxplot(dev.price); plt.ylabel('Sale price (USD)'); plt.title('Development prices: all observations retained'); out.fig('02_price_boxplot')
    plt.figure(figsize=(11,9)); plt.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm'); plt.colorbar(label='Pearson correlation'); plt.xticks(range(len(NUMERIC)),NUMERIC,rotation=90); plt.yticks(range(len(NUMERIC)),NUMERIC); out.fig('03_feature_correlation_heatmap')
    plt.figure(); plt.scatter(dev.sqft_living,dev.price,s=4,alpha=.25); plt.xlabel('Living area (sq ft)'); plt.ylabel('Price (USD)'); plt.title('Development observations'); out.fig('04_living_area_vs_price')
    dev[NUMERIC].hist(figsize=(15,12),bins=35); out.fig('16_feature_distributions')
    return info


def run(data_path, output_path):
    start=time.perf_counter(); out=Outputs(output_path)
    df=pd.read_csv(data_path); validate_data(df); parts=partitions(df)
    dev=df.iloc[parts['Development']]; X=dev[FEATURES]; y=dev.price.to_numpy(); groups=dev.id.to_numpy()
    config={'objective':'Predict historical residential sale prices and compare feature selection for multiple linear regression.',
            'research_questions':['RQ1: accuracy improvement?','RQ2: complexity reduction with retained performance?','RQ3: reasonable calibrated 95% prediction intervals?'],
            'dataset_source':SOURCE,'dataset_path':str(Path(data_path).resolve()),'dataset_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            'shape':df.shape,'features':FEATURES,'seed':SEED,'split':'GroupShuffleSplit 20% then 18.75% of remaining groups',
            'outer_cv':'GroupKFold(5), shared assignments','inner_cv':'GroupKFold(3), fold-local preprocessing and selection',
            'models':dict(zip(METHODS,['All','Pearson','Mutual Information','RFE','RFECV equivalent fold-local recursive elimination','LASSO selection + OLS'])),
            'count_grid':COUNTS,'lasso_alpha_grid':ALPHAS,'lasso_threshold':1e-8,'lasso_max_iter':1000000,'lasso_tol':1e-5,'lasso_precompute':True,
            'rfecv_step':5,'rfecv_min_features':1,'scoring':'mean fold RMSE in original USD',
            'discrete_MI_predictors':sorted(DISCRETE|{'zipcode dummies'}),'continuous_MI_predictors':[c for c in NUMERIC if c not in DISCRETE],
            'assumptions':'ZIP reference-category one-hot; other numeric/ordinal variables unchanged. Impute within folds; scale every encoded column. No row removal or engineered features.',
            'selection':'best mean outer RMSE + its sample SD / sqrt(5), then smallest mean column count, RMSE, fixed method order',
            'conformal_alpha':.05,'python':platform.python_version(),
            'versions':{k:v for k,v in [('numpy',np.__version__),('pandas',pd.__version__),('sklearn',sklearn.__version__),('scipy',scipy.__version__),('statsmodels',statsmodels.__version__),('matplotlib',matplotlib.__version__)]},
            'timestamp_utc':datetime.now(timezone.utc).isoformat()}
    json_write(out.path/'research_configuration.json',config)
    split=out.table('data_split_summary',[{'partition':k,'rows':len(v),'groups':df.id.iloc[v].nunique(),'proportion':len(v)/len(df),'overlap_with_other_partitions':0} for k,v in parts.items()])
    out.table('partition_assignments',[{'row_index':int(i),'property_id':df.id.iloc[i],'partition':k} for k,v in parts.items() for i in v])
    info=eda(df,dev,out)
    outer=folds(X,groups,5); rows=[]; feature_rows=[]; logs=[]; audit=[]; oof=[]
    out.table('outer_fold_assignments',[{'row_index':int(X.index[i]),'property_id':groups[i],'outer_validation_fold':f+1} for f,(_,b) in enumerate(outer) for i in b])
    def record_features(model,method,context):
        names,support=metadata(model)
        sel=model.named_steps.get('select')
        for j,name in enumerate(names):
            feature_rows.append({'method':method,'context':context,'encoded_feature':name,
                'original_feature':'zipcode' if name.startswith('zip__') else name.removeprefix('num__'),
                'selected':bool(support[j]),'score_or_lasso_coefficient':sel.scores_[j] if sel is not None else np.nan,
                'ranking':sel.ranking_[j] if sel is not None else 1})
        return int(support.sum())
    for f,(a,b) in enumerate(outer):
        for method in METHODS:
            t=time.perf_counter(); print(f'Outer {f+1}/5 {method}',flush=True)
            model,value=tune(X.iloc[a],y[a],groups[a],method,f'outer_{f+1}',logs,audit)
            pred=model.predict(X.iloc[b]); rank=effective_rank(design(model,X.iloc[a]))
            count=record_features(model,method,f'outer_{f+1}')
            rows.append({'method':method,'outer_fold':f+1,'train_rows':len(a),'validation_rows':len(b),
                         'train_groups':len(np.unique(groups[a])),'validation_groups':len(np.unique(groups[b])),
                         'group_overlap':0,'selected_count':count,'parameter':value,'seconds':time.perf_counter()-t,**metrics(y[b],pred,rank)})
            oof.extend({'method':method,'outer_fold':f+1,'row_index':int(X.index[i]),'actual':float(y[i]),'prediction':float(p)} for i,p in zip(b,pred))
            out.table('cv_fold_metrics',rows); out.table('inner_cv_results',logs); out.table('selected_features',feature_rows)
            json_write(out.path/'fold_audit.json',audit)
    cv=pd.DataFrame(rows)
    summary=cv.groupby('method').agg({k:['mean','std'] for k in ['mae','mse','rmse','r2','adjusted_r2','selected_count','seconds']})
    summary.columns=['_'.join(c) for c in summary.columns]; summary=summary.reset_index()
    out.table('cv_summary',summary); out.table('out_of_fold_predictions',oof)
    pooled=[]
    for method,block in pd.DataFrame(oof).groupby('method'):
        pooled.append({'method':method,'pooled_oof_rmse':np.sqrt(mean_squared_error(block.actual,block.prediction))})
    out.table('pooled_oof_metrics',pooled)
    selection=choose(summary); json_write(out.path/'model_selection.json',selection)
    print('Frozen selection:',selection,flush=True)
    models={}; parameters=[]; train_metrics=[]
    for method in METHODS:
        t=time.perf_counter(); print('Development refit',method,flush=True)
        model,value=tune(X,y,groups,method,'development',logs,audit); models[method]=model
        count=record_features(model,method,'development'); rank=effective_rank(design(model,X))
        parameters.append({'method':method,'value':value,'selected_count':count,'encoded_count':len(metadata(model)[0]),'seconds':time.perf_counter()-t})
        train_metrics.append({'method':method,**metrics(y,model.predict(X),rank)})
        joblib.dump(model,out.path/'models'/f'{method}.joblib')
    out.table('model_hyperparameters',parameters); out.table('development_model_metrics',train_metrics)
    out.table('selected_features',feature_rows); out.table('inner_cv_results',logs); json_write(out.path/'fold_audit.json',audit)
    winner=selection['selected_method']; final=models[winner]; joblib.dump(final,out.path/'models'/'final_model.joblib')
    cal=df.iloc[parts['Calibration']]; cp=final.predict(cal[FEATURES]); scores=np.abs(cal.price.to_numpy()-cp); q,k=conformal(scores)
    cuts=np.unique(np.quantile(cp,[.2,.4,.6,.8])); edges=np.r_[-np.inf,cuts,np.inf]
    calibration={'sample_count':len(cal),'order_k':k,'q_hat':q,'alpha':.05,'predicted_price_cutpoints':cuts,'actual_price_bins':len(edges)-1}
    json_write(out.path/'calibration.json',calibration)
    out.table('calibration_scores',pd.DataFrame({'row_index':cal.index,'property_id':cal.id,'actual':cal.price,'prediction':cp,'absolute_residual':scores}))
    # Test targets first used for model evaluation only AFTER winner and calibration freeze.
    test=df.iloc[parts['Test']]; Xt=test[FEATURES]; yt=test.price.to_numpy(); test_rows=[]; pred_rows=[]
    for method,model in models.items():
        pred=model.predict(Xt); rank=effective_rank(design(model,X))
        test_rows.append({'method':method,'selected_count':int(metadata(model)[1].sum()),**metrics(yt,pred,rank)})
        pred_rows.extend({'method':method,'row_index':int(i),'actual':float(a),'prediction':float(p)} for i,a,p in zip(test.index,yt,pred))
    tests=out.table('test_model_metrics',test_rows); out.table('test_predictions',pred_rows)
    pred=final.predict(Xt); low=pred-q; high=pred+q; covered=(yt>=low)&(yt<=high)
    pi=pd.DataFrame({'row_index':test.index,'property_id':test.id.to_numpy(),'actual':yt,'prediction':pred,'lower':low,'upper':high,'covered':covered,
                     'predicted_price_group':pd.cut(pred,edges,include_lowest=True)})
    out.table('prediction_intervals_test',pi)
    im={'nominal_coverage':.95,'empirical_coverage':covered.mean(),'average_width':np.mean(high-low),'q_hat':q,'calibration_n':len(cal),'order_k':k,
        'test_n':len(test),'covered':covered.sum(),'uncovered':(~covered).sum(),'negative_lower_bounds':(low<0).sum()}
    out.table('interval_metrics',[im])
    by=pi.groupby('predicted_price_group',observed=False).agg(observations=('covered','size'),covered=('covered','sum'),coverage=('covered','mean')).reset_index()
    out.table('interval_coverage_by_price',by)
    diag,coef=diagnostics(final,X,y,out)
    make_figures(summary,tests,pd.DataFrame(feature_rows),pd.DataFrame(logs),parameters,pi,by,out)
    loaded=joblib.load(out.path/'models'/'final_model.joblib'); np.testing.assert_allclose(loaded.predict(Xt),pred,rtol=0,atol=1e-8)
    assert np.all(low<=high) and int(covered.sum())==im['covered'] and len(pi)==len(test)
    assert len(cv)==30 and all(cv.groupby('method').size()==5)
    for _,s in summary.iterrows():
        block=cv[cv.method==s.method]; np.testing.assert_allclose([s.rmse_mean,s.rmse_std],[block.rmse.mean(),block.rmse.std(ddof=1)])
    required_tables=['data_quality_summary','data_split_summary','cv_fold_metrics','cv_summary','selected_features','model_hyperparameters','test_model_metrics',
                     'model_coefficients','statistical_diagnostics','prediction_intervals_test','interval_metrics','interval_coverage_by_price']
    assert all((out.path/'tables'/f'{n}.csv').is_file() for n in required_tables)
    assert len(list((out.path/'figures').glob('*.png')))>=15
    result={'status':'completed','dataset':info,'configuration':config,'selection':selection,'calibration':calibration,'interval_metrics':im,
            'cv_summary':summary.to_dict('records'),'test_metrics':test_rows,'diagnostics':diag,'runtime_seconds':time.perf_counter()-start,
            'consistency_checks':'passed: group isolation, all rows, 30 outer scores, CV aggregates, interval arithmetic, saved-model prediction equality, required outputs',
            'limitations':['Historical generalization only','One-SE rule is a heuristic, not proof of equivalence',
                          'Transaction residual exchangeability is not established by group splitting; no conditional coverage guarantee',
                          'Rank-deficient baseline retained unchanged','Numeric and ordinal predictors modeled linearly',
                          'RFECV equivalent implementation uses fold-local preprocessing and step-5 RFE candidate paths',
                          'Student confirmed ID 7115056123; script and archive filenames now consistently use this ID, superseding the original prompt typo']}
    json_write(out.path/'experiment_summary.json',result)
    write_handoff(out,result,split,summary,tests,pd.DataFrame(parameters),pd.DataFrame(feature_rows),by,coef)
    print(f'Completed in {result["runtime_seconds"]:.1f}s; winner={winner}',flush=True)
    return result


def diagnostics(model,X,y,out):
    A=design(model,X); pred=model.predict(X); residual=y-pred
    names,support=metadata(model); names=names[support]
    rank=effective_rank(A); Z=np.column_stack([np.ones(len(A)),A])
    # BP auxiliary regression uses an orthonormal basis of the full column space.
    # This retains the intercept span and avoids invalid rank-deficient degrees of freedom.
    U,s,_=np.linalg.svd(Z,full_matrices=False); r=np.linalg.matrix_rank(Z)
    aux=LinearRegression().fit(U[:,:r],residual**2)
    lm=len(y)*r2_score(residual**2,aux.predict(U[:,:r])); bp_p=stats.chi2.sf(lm,rank)
    diag={'partition':'Development training residuals, descriptive post-selection diagnostics','design_columns_with_intercept':Z.shape[1],
          'design_rank_with_intercept':r,'effective_predictor_rank':rank,'condition_number':np.linalg.cond(Z),
          'residual_mean':residual.mean(),'residual_sd':residual.std(ddof=1),'residual_skew':stats.skew(residual),
          'residual_excess_kurtosis':stats.kurtosis(residual),'bp_lm':lm,'bp_df':rank,'bp_pvalue':bp_p,
          'bp_pvalue_note':'Numerical underflow: stored 0.0 means below floating-point resolution, not an exactly zero probability' if bp_p==0 else 'Representable floating-point survival probability',
          'max_absolute_residual':np.max(np.abs(residual)),
          'bp_method':'Koenker/Breusch-Pagan n*R2 on squared residual auxiliary regression over full design column space; df=rank-1'}
    out.table('statistical_diagnostics',[{'measure':k,'value':v} for k,v in diag.items()])
    vif=[]
    for j,name in enumerate(names):
        target=A[:,j]; B=np.delete(A,j,axis=1)
        if np.var(target)==0: v=np.nan
        else:
            rr=r2_score(target,LinearRegression().fit(B,target).predict(B)) if B.shape[1] else 0
            v=np.inf if 1-rr<=1e-12 else 1/(1-rr)
        vif.append({'encoded_feature':name,'vif':v,'status':'infinite' if np.isinf(v) else 'undefined' if np.isnan(v) else 'finite'})
    out.table('vif',vif)
    coef=out.table('model_coefficients',[{'encoded_feature':n,'coefficient_usd_per_scaled_unit':c} for n,c in zip(names,model['regressor'].coef_)]+[{'encoded_feature':'intercept','coefficient_usd_per_scaled_unit':model['regressor'].intercept_}])
    out.table('development_residuals',pd.DataFrame({'row_index':X.index,'actual':y,'fitted':pred,'residual':residual}))
    ix=np.argsort(-np.abs(residual))[:25]
    out.table('extreme_errors_development',pd.DataFrame({'row_index':X.index[ix],'actual':y[ix],'prediction':pred[ix],'residual':residual[ix]}))
    plt.figure(); plt.scatter(pred,residual,s=4,alpha=.2); plt.axhline(0,color='red'); plt.xlabel('Development fitted price (USD)'); plt.ylabel('Residual (USD)'); plt.title('Training residuals; post-selection descriptive diagnostic'); out.fig('12_residuals_vs_fitted')
    plt.figure(); stats.probplot(residual,plot=plt); plt.ylabel('Development residual (USD)'); plt.title('Normal Q-Q plot: Development residuals'); out.fig('13_residual_qq_plot')
    plt.figure(); plt.hist(residual,bins=70); plt.xlabel('Development residual (USD)'); plt.ylabel('Count'); out.fig('17_residual_histogram')
    return diag,coef


def make_figures(summary,tests,features,logs,parameters,pi,by,out):
    final=features[features.context=='development']
    pivot=final[final.method!='M0'].pivot(index='encoded_feature',columns='method',values='selected').astype(int)
    fig,axes=plt.subplots(1,2,figsize=(14,12))
    for ax,indices in zip(axes,np.array_split(np.arange(len(pivot)),2)):
        block=pivot.iloc[indices]; ax.imshow(block,aspect='auto',vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(5),block.columns); ax.set_yticks(range(len(block)),block.index,fontsize=8)
        ax.set_title('Selected encoded columns: blue=yes, white=no')
    fig.suptitle('Final Development selections; each ZIP dummy is a separate encoded column')
    out.fig('05_feature_selection_comparison')
    for key,number in [('rmse','06'),('r2','07')]:
        plt.figure(); plt.bar(summary.method,summary[key+'_mean'],yerr=summary[key+'_std'],capsize=5); plt.ylabel('RMSE (USD)' if key=='rmse' else 'R²'); plt.title('Outer group CV: mean ± sample SD, 5 folds'); out.fig(number+'_cv_'+key+'_comparison')
    plt.figure(); plt.bar([r['method'] for r in parameters],[r['selected_count'] for r in parameters]); plt.ylabel('Selected encoded columns'); plt.title('Final Development fits'); out.fig('08_feature_count_comparison')
    curve=logs[(logs.method=='M4')&(logs.context=='development')].groupby('value').rmse.agg(['mean','std']).reset_index()
    out.table('rfecv_curve',curve)
    plt.figure(); plt.errorbar(curve.value,curve['mean'],yerr=curve['std'],marker='o',capsize=3); plt.xlabel('Retained encoded columns'); plt.ylabel('Inner-CV RMSE (USD)'); plt.title('RFECV: final Development tuning, 3 group folds'); out.fig('09_rfecv_performance')
    lasso=final[(final.method=='M5')&final.selected].sort_values('score_or_lasso_coefficient')
    alpha=next(r['value'] for r in parameters if r['method']=='M5')
    fig,axes=plt.subplots(1,2,figsize=(15,12))
    for ax,indices in zip(axes,np.array_split(np.arange(len(lasso)),2)):
        block=lasso.iloc[indices]; ax.barh(block.encoded_feature,block.score_or_lasso_coefficient)
        ax.tick_params(axis='y',labelsize=8); ax.set_xlabel('USD per standardized encoded unit'); ax.axvline(0,color='grey',linewidth=.5)
    fig.suptitle(f'LASSO selector coefficients; alpha={alpha:g}; final predictions use OLS')
    out.fig('10_lasso_coefficients')
    plt.figure(); plt.scatter(pi.actual,pi.prediction,s=4,alpha=.3); lo=min(pi.actual.min(),pi.prediction.min()); hi=max(pi.actual.max(),pi.prediction.max()); plt.plot([lo,hi],[lo,hi],color='red'); plt.xlabel('Actual Test price (USD)'); plt.ylabel('Predicted price (USD)'); out.fig('11_actual_vs_predicted')
    subset=pi.sample(n=min(100,len(pi)),random_state=SEED).sort_values('prediction').reset_index(drop=True)
    out.table('prediction_interval_plot_data',subset)
    plt.figure(figsize=(12,6)); xx=np.arange(len(subset)); plt.fill_between(xx,subset.lower,subset.upper,alpha=.25,label='95% split conformal interval'); plt.plot(xx,subset.prediction,label='Prediction'); plt.scatter(xx,subset.actual,s=15,label='Actual'); bad=~subset.covered; plt.scatter(xx[bad],subset.actual[bad],color='red',marker='x',s=40,label='Outside interval'); plt.ylabel('Price (USD)'); plt.xlabel('Fixed-seed Test subset, ordered by prediction'); plt.title(f'Visualization only: {len(subset)} Test observations; coverage uses full Test set'); plt.legend(); out.fig('14_prediction_interval_95')
    plt.figure(figsize=(10,5)); plt.bar(np.arange(len(by)),by.coverage); plt.xticks(np.arange(len(by)),[f'Bin {i+1}\nn={n}' for i,n in enumerate(by.observations)]); plt.axhline(.95,color='red',linestyle='--',label='Nominal 95%'); plt.ylim(0,1.05); plt.ylabel('Full-Test empirical coverage'); plt.xlabel('Predicted-price bins fixed using Calibration predictions'); plt.legend(); out.fig('15_interval_coverage_by_price')


def write_handoff(out,result,split,summary,tests,parameters,features,by,coef):
    joined=summary.merge(tests,on='method',suffixes=('_cv','_test')).merge(parameters[['method','value']],on='method')
    cv_view=summary[['method','selected_count_mean','rmse_mean','rmse_std','mae_mean','mae_std','r2_mean','r2_std']]
    test_view=tests.merge(parameters[['method','value','seconds']],on='method')
    tradeoffs=summary[['method','rmse_mean','selected_count_mean']].copy()
    baseline=tradeoffs[tradeoffs.method=='M0'].iloc[0]
    tradeoffs['cv_rmse_difference_vs_baseline']=tradeoffs.rmse_mean-baseline.rmse_mean
    tradeoffs['mean_column_reduction_percent']=100*(1-tradeoffs.selected_count_mean/baseline.selected_count_mean)
    final=features[(features.context=='development')&features.selected]
    winner=result['selection']['selected_method']; d=result['diagnostics']; im=result['interval_metrics']
    text=f'''# HW2 Stage 4 handoff

## A. Execution Status
Completed all six actual King County experiments at {result['configuration']['timestamp_utc']}.
Python {platform.python_version()}; runtime {result['runtime_seconds']:.2f} seconds.
Dataset: {result['configuration']['dataset_path']}; dimensions and schema verified, official Kaggle download; SHA256 recorded.
Automated test results: see `engineering_test_results.txt` and `validation_results.txt`.
Numerical solver adjustments, preliminary-run exclusion and retained third-party warnings: see `engineering_notes.md`.
Consistency checks: {result['consistency_checks']}.

## B. Dataset Verification
{json.dumps(result['dataset'],default=lambda x:x.item() if isinstance(x,np.generic) else str(x),indent=2)}

{split.to_markdown(index=False)}

All partition group intersections and all CV train/validation group intersections are empty.
No observations removed. Integrity checks cover all rows; target-dependent EDA uses Development only.

## C. Preprocessing
18 original predictors. ZIP Code is categorical, reference-category one-hot with unknown categories ignored.
Median numeric imputation and most-frequent ZIP imputation are learned per training fold.
Every encoded column, including ZIP dummies, is standardized to mean zero and unit SD (constants remain zero).
Other numeric and ordinal predictors retain their numeric representation; meaningful zero values retained.
Encoded widths and final selected counts: `model_hyperparameters.csv`.
Complete encoded-to-original mapping and selections: `selected_features.csv`.
Feature selection operates on individual encoded columns, including individual ZIP dummies.
MI treats bedrooms, floors, waterfront, view, condition, grade and year fields as discrete; bathrooms and area/location measures as continuous.

## D. Six-Model Comparison
Sample SD uses ddof=1. CV RMSE is mean of five fold RMSEs, distinct from `pooled_oof_metrics.csv`.
Adjusted R² uses the training design's effective predictor rank; validation/Test adjusted R² is descriptive.

Outer CV comparison:

{cv_view.to_markdown(index=False)}

Final Development fits and descriptive Test comparison (seconds are Development refit/tuning times):

{test_view.to_markdown(index=False)}

Full precision, MSE CV aggregates and per-fold timings remain in the CSV tables.

## E. Feature Selection Findings
Development-CV accuracy/complexity differences relative to M0 (negative RMSE difference is better):

{tradeoffs.to_markdown(index=False)}

'''
    for method in METHODS:
        f=final[final.method==method]
        text+=f"\n- {method}: {len(f)} encoded columns; {f.original_feature.nunique()} original groups: "+', '.join(f.encoded_feature)+'.\n'
    freq=final[final.method!='M0'].groupby('encoded_feature').method.nunique().sort_values(ascending=False)
    text+='\nSelection frequency across five methods (descriptive association, no causal interpretation):\n'+freq.rename('method_count').reset_index().to_markdown(index=False)+'\n'
    text+=f'''
## F. Selected Final Model
{json.dumps(result['selection'],indent=2)}

Selection was frozen before Test metrics. The one-standard-error rule is a practical heuristic, not proof of equivalence.
Primary model: {winner}. Full standardized coefficients and intercept are in `model_coefficients.csv`.
Largest absolute coefficients (including any intercept):
{coef.iloc[np.argsort(-np.abs(coef.coefficient_usd_per_scaled_unit.to_numpy()))[:12]].to_markdown(index=False)}

## G. Statistical Diagnostics
Development training residuals are descriptive post-selection diagnostics.
Effective predictor rank: {d['effective_predictor_rank']}; augmented rank {d['design_rank_with_intercept']} of {d['design_columns_with_intercept']}.
Residual skew: {d['residual_skew']:.6g}; excess kurtosis: {d['residual_excess_kurtosis']:.6g}.
BP/Koenker LM: {d['bp_lm']:.6g}, df={d['bp_df']}, p={d['bp_pvalue']:.6g}.
P-value representation: {d.get('bp_pvalue_note','See statistical_diagnostics.csv')}.
Squared residuals are regressed on an orthonormal basis of the complete design column space, retaining intercept span.
Residual normality/homoscedasticity are not assumed verified. Small large-sample p-values are not proof that prediction is useless.
Inspect residual-vs-fitted and Q-Q figures for curvature/tails; VIF and extreme-error tables for collinearity and errors.
Baseline retains exact dependencies. Infinite VIFs are explicitly marked; no artificial finite replacements.

## H. 95% Prediction Interval
Calibration n={im['calibration_n']}; k=ceil((n+1)*0.95)={im['order_k']}; q_hat={im['q_hat']:.10g} USD.
Nominal coverage=0.95; full Test n={im['test_n']}; coverage={im['empirical_coverage']:.10g}; average width={im['average_width']:.10g} USD.
Covered={im['covered']}; uncovered={im['uncovered']}; negative lower bounds={im['negative_lower_bounds']} (retained).
Predicted-price bin boundaries were frozen using Calibration predictions. Duplicate cutpoints are collapsed.

{by.to_markdown(index=False)}

Intervals are marginal: conditional coverage is not guaranteed. Repeated transactions within a property can violate transaction-level exchangeability even with group-disjoint partitions.

## I. Artifacts
Primary Stage 4 files: `{(out.path/'handoff.md').resolve()}` and `{(out.path/'experiment_summary.json').resolve()}`.
CSV tables, figures, source and saved models:
'''
    for p in sorted(out.path.rglob('*')):
        if p.is_file(): text+=f'\n- `{p.resolve()}`'
    for p in sorted(out.path.parent.glob('*.py')): text+=f'\n- `{p.resolve()}`'
    text+='\n\n## J. Limitations and Unresolved Issues\n'+'\n'.join('- '+s for s in result['limitations'])
    text+='\n- Test comparisons are descriptive and did not change the winner. No formal academic PDF, NotebookLM summary or conversation transcript was generated.\n'
    text+='''
## K. Recommended Review Questions
- Are CV gains relative to the baseline meaningful compared with fold variability?
- Is the encoded-column reduction worth the accuracy tradeoff?
- Do residual plots and VIF results suggest limitations worth emphasizing?
- How does coverage change across the fixed predicted-price bins?
- Does within-property dependence limit the conformal interpretation?
- Which results require additional methodological authorization before rerunning?
'''
    (out.path/'handoff.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'data/kc_house_data.csv')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs_verified')
    args=parser.parse_args()
    try:
        with threadpool_limits(limits=1): run(args.data,args.output)
    except Exception as e:
        args.output.mkdir(parents=True,exist_ok=True)
        json_write(args.output/'experiment_summary.json',{'status':'failed','reason':repr(e),'timestamp':datetime.now(timezone.utc).isoformat()})
        raise
