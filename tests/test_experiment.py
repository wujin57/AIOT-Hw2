import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from threadpoolctl import threadpool_limits
import experiment as e

@pytest.fixture
def fixture():
    rng=np.random.RandomState(17); n=240
    df=pd.DataFrame({c:rng.normal(20,3,n) for c in e.FEATURES})
    for c in e.DISCRETE: df[c]=rng.randint(1,5,n)
    df['zipcode']=rng.choice([98001,98002,98003],n)
    df['price']=300000+5000*df.sqft_living+1000*df.grade+rng.normal(0,500,n)
    df['id']=np.repeat(np.arange(n//2),2); df['date']='20140502T000000'
    return df

def test_schema(fixture):
    e.validate_data(fixture,strict=False)
    with pytest.raises(ValueError): e.validate_data(fixture)
    with pytest.raises(ValueError): e.validate_data(fixture.drop(columns='grade'),False)
    fixture.loc[0,'price']=-1
    with pytest.raises(ValueError): e.validate_data(fixture,False)
    assert len(e.FEATURES)==18 and not {'price','id','date'} & set(e.FEATURES)

def test_partition_and_folds(fixture):
    p=e.partitions(fixture); q=e.partitions(fixture)
    assert set(np.concatenate(list(p.values())))==set(range(len(fixture)))
    for k in p: np.testing.assert_array_equal(p[k],q[k])
    dev=fixture.iloc[p['Development']]
    for a,b in e.folds(dev,dev.id,5):
        assert not set(dev.id.iloc[a]) & set(dev.id.iloc[b])
        for ia,ib in e.folds(dev.iloc[a],dev.id.iloc[a],3):
            assert not set(dev.id.iloc[a].iloc[ia]) & set(dev.id.iloc[a].iloc[ib])

def test_training_only_preprocessing(fixture):
    X=fixture[e.FEATURES]; pre=e.preprocessing().fit(X.iloc[:120])
    original=pre.transform(X.iloc[:120]).copy()
    X=X.copy(); X.iloc[120:,X.columns.get_loc('sqft_living')]=1e10
    X.iloc[120:,X.columns.get_loc('zipcode')]=99999
    with pytest.warns(UserWarning): pre.transform(X.iloc[120:])
    np.testing.assert_array_equal(original,pre.transform(X.iloc[:120]))
    assert not any('99999' in name for name in pre.get_feature_names_out())
    assert pre['encode'].named_transformers_['num'].statistics_[2]==fixture.sqft_living.iloc[:120].median()

@pytest.mark.parametrize('method',e.METHODS)
def test_six_methods_and_fold_audit(fixture,method):
    X=fixture[e.FEATURES]; y=fixture.price.to_numpy(); logs=[]; audit=[]
    with threadpool_limits(limits=1): model,value=e.tune(X,y,fixture.id.to_numpy(),method,'unit_test',logs,audit)
    names,support=e.metadata(model)
    assert len(names)==len(support) and support.any()
    assert isinstance(model['regressor'],LinearRegression)
    assert e.design(model,X).shape[1]==support.sum()
    assert np.isfinite(model.predict(X)).all()
    if method=='M0': assert support.all() and 'select' not in model.named_steps
    else:
        assert len(audit)==3
        for row in audit:
            a=row['train_row_indices']; b=row['validation_row_indices']
            assert not set(fixture.id.loc[a]) & set(fixture.id.loc[b])
    if method=='M5':
        np.testing.assert_array_equal(support,np.abs(model['select'].scores_)>1e-8)
        pred=LinearRegression().fit(e.design(model,X),y).predict(e.design(model,X))
        np.testing.assert_allclose(pred,model.predict(X))

def test_selector_train_only_and_constants():
    rng=np.random.RandomState(0); A=rng.normal(size=(100,5)); A[:,0]=1; y=A[:,3]*5
    s=e.ColumnSelector('M1',1).fit(A,y)
    assert s.scores_[0]==0 and np.flatnonzero(s.support_).tolist()==[3]
    before=s.support_.copy(); s.transform(A*100)
    np.testing.assert_array_equal(before,s.support_)
    with pytest.raises(ValueError): e.ColumnSelector('M5',1e9).fit(A,y)
    with pytest.raises(ValueError): e.ColumnSelector('M3',6).fit(A,y)

def test_metrics_rank_and_conformal():
    A=np.array([[1,2],[2,4],[3,6],[4,8]])
    assert e.effective_rank(A)==1
    m=e.metrics([1,2,3,4],[1,2,4,5],1)
    assert m['rmse']**2==pytest.approx(m['mse'])
    assert m['adjusted_r2']==pytest.approx(1-(1-m['r2'])*3/2)
    assert np.isnan(e.metrics([1,2],[1,2],2)['adjusted_r2'])
    q,k=e.conformal(np.arange(1,20)); assert k==19 and q==19
    q,k=e.conformal(np.arange(1,11)); assert k==11 and np.isinf(q)

def test_selection_is_one_se_and_complexity():
    df=pd.DataFrame({'method':['M0','M1','M2'],'rmse_mean':[100.,101.,120.], 'rmse_std':[5.,1.,2.], 'selected_count_mean':[80.,10.,5.]})
    choice=e.choose(df); assert choice['selected_method']=='M1'
    assert choice['threshold']==pytest.approx(100+5/np.sqrt(5))

def test_rfecv_equivalence_fixed_design():
    # On an already fixed design, explicit candidate evaluation must equal sklearn RFECV.
    rng=np.random.RandomState(42); A=rng.normal(size=(90,11)); y=A[:,0]*4+A[:,1]+rng.normal(size=90)
    groups=np.repeat(np.arange(45),2); splits=e.folds(A,groups)
    scores={}
    with threadpool_limits(limits=1):
        for k in [1,6,11]:
            ss=[]
            for a,b in splits:
                s=e.ColumnSelector('M4',k).fit(A[a],y[a]); p=LinearRegression().fit(s.transform(A[a]),y[a]).predict(s.transform(A[b]))
                ss.append(np.sqrt(np.mean((y[b]-p)**2)))
            scores[k]=np.mean(ss)
        standard=RFECV(LinearRegression(),step=5,min_features_to_select=1,cv=splits,scoring='neg_root_mean_squared_error').fit(A,y)
    assert min(scores,key=lambda k:(scores[k],k))==standard.n_features_
    np.testing.assert_allclose([scores[k] for k in sorted(scores)],-standard.cv_results_['mean_test_score'])
