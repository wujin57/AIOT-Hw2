"""Independent checks of saved real-experiment artifacts (does not retune models)."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from experiment import FEATURES, metrics, conformal

def validate(root):
    t=root/'tables'; read=lambda name:pd.read_csv(t/(name+'.csv'))
    summary=json.loads((root/'experiment_summary.json').read_text()); assert summary['status']=='completed'
    partitions=read('partition_assignments'); assert len(partitions)==21613 and partitions.row_index.nunique()==21613
    assert partitions.groupby('property_id').partition.nunique().max()==1
    development=set(partitions.loc[partitions.partition=='Development','row_index'])
    mapping=partitions.set_index('row_index').property_id
    for fold in json.loads((root/'fold_audit.json').read_text()):
        a=fold['train_row_indices']; b=fold['validation_row_indices']
        assert set(a)|set(b)<=development
        assert not set(mapping.loc[a]) & set(mapping.loc[b])
    outer=read('outer_fold_assignments'); assert set(outer.row_index)==development
    assert outer.groupby('property_id').outer_validation_fold.nunique().max()==1
    cv=read('cv_fold_metrics'); stored=read('cv_summary'); assert len(cv)==30
    oof=read('out_of_fold_predictions')
    expected_outer=outer.set_index('row_index').outer_validation_fold.sort_index()
    for method,block in oof.groupby('method'):
        assert block.row_index.nunique()==len(development) and len(block)==len(development)
        observed=block.set_index('row_index').outer_fold.sort_index()
        np.testing.assert_array_equal(expected_outer.index,observed.index)
        np.testing.assert_array_equal(expected_outer,observed)
    for method,block in cv.groupby('method'):
        assert set(block.outer_fold)=={1,2,3,4,5}
        row=stored.set_index('method').loc[method]
        for key in ['mae','mse','rmse','r2']:
            np.testing.assert_allclose([row[key+'_mean'],row[key+'_std']],[block[key].mean(),block[key].std(ddof=1)])
        np.testing.assert_allclose(block.rmse**2,block.mse)
    fs=read('selected_features'); test_pred=read('test_predictions'); tm=read('test_model_metrics').set_index('method')
    for method in tm.index:
        m=joblib.load(root/'models'/f'{method}.joblib'); assert isinstance(m['regressor'],LinearRegression)
        sub=test_pred[test_pred.method==method]; values=metrics(sub.actual,sub.prediction,int(tm.loc[method,'effective_rank']))
        for key in ['mae','mse','rmse','r2','adjusted_r2']: np.testing.assert_allclose(values[key],tm.loc[method,key])
        selected=fs[(fs.method==method)&(fs.context=='development')]
        assert selected.selected.sum()==tm.loc[method,'selected_count']
    pi=read('prediction_intervals_test'); cal=read('calibration_scores'); im=read('interval_metrics').iloc[0]
    assert set(pi.row_index)==set(partitions.loc[partitions.partition=='Test','row_index'])
    assert set(cal.row_index)==set(partitions.loc[partitions.partition=='Calibration','row_index'])
    q,k=conformal(cal.absolute_residual.to_numpy()); np.testing.assert_allclose(q,im.q_hat); assert k==im.order_k
    np.testing.assert_allclose(pi.lower,pi.prediction-q); np.testing.assert_allclose(pi.upper,pi.prediction+q)
    flags=(pi.actual>=pi.lower)&(pi.actual<=pi.upper); np.testing.assert_array_equal(flags,pi.covered)
    np.testing.assert_allclose(flags.mean(),im.empirical_coverage); np.testing.assert_allclose((pi.upper-pi.lower).mean(),im.average_width)
    assert flags.sum()==im.covered and len(pi)==im.test_n
    by=read('interval_coverage_by_price'); assert by.observations.sum()==len(pi) and by.covered.sum()==flags.sum()
    cfg=json.loads((root/'research_configuration.json').read_text()); data=pd.read_csv(cfg['dataset_path'])
    winner=joblib.load(root/'models'/'final_model.joblib')
    np.testing.assert_allclose(winner.predict(data.loc[pi.row_index,FEATURES]),pi.prediction)
    np.testing.assert_allclose(winner.predict(data.loc[cal.row_index,FEATURES]),cal.prediction)
    np.testing.assert_allclose(data.loc[pi.row_index,'price'],pi.actual)
    np.testing.assert_allclose(data.loc[cal.row_index,'price'],cal.actual)
    cutpoints=np.unique(np.quantile(cal.prediction,[.2,.4,.6,.8]))
    frozen=json.loads((root/'calibration.json').read_text())
    np.testing.assert_allclose(cutpoints,frozen['predicted_price_cutpoints'])
    frozen_bins=pd.cut(pi.prediction,np.r_[-np.inf,cutpoints,np.inf],include_lowest=True)
    recomputed=pd.DataFrame({'bin':frozen_bins,'covered':flags}).groupby('bin',observed=False).covered.agg(['size','sum','mean'])
    np.testing.assert_array_equal(recomputed['size'],by.observations)
    np.testing.assert_array_equal(recomputed['sum'],by.covered)
    np.testing.assert_allclose(recomputed['mean'],by.coverage,equal_nan=True)
    for method in tm.index:
        model=joblib.load(root/'models'/f'{method}.joblib')
        sub=test_pred[test_pred.method==method]
        np.testing.assert_allclose(model.predict(data.loc[sub.row_index,FEATURES]),sub.prediction)
    assert len(list((root/'figures').glob('*.png')))==17
    print('PASS: 21,613 rows; group isolation; inner/outer audit; 30 fold scores; CV mean/SD; six OLS models; feature counts; Test metrics; conformal rank and coverage; saved-model predictions; 17 figures.')
    print('Winner:',summary['selection']['selected_method'])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs_verified'); args=p.parse_args(); validate(args.output)
