"""Export frozen experiment artifacts for a static dashboard; never fit models."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import FEATURES, NUMERIC, json_write
OUT=ROOT/'outputs_verified'
DEST=ROOT/'dashboard'

def export():
    for d in ['data','figures','downloads']: (DEST/d).mkdir(parents=True,exist_ok=True)
    cfg=json.loads((OUT/'research_configuration.json').read_text())
    assert hashlib.sha256((ROOT/'data/kc_house_data.csv').read_bytes()).hexdigest()==cfg['dataset_sha256']
    df=pd.read_csv(ROOT/'data/kc_house_data.csv')
    read=lambda n:pd.read_csv(OUT/'tables'/f'{n}.csv')
    cv=read('cv_summary').set_index('method'); test=read('test_model_metrics').set_index('method')
    hp=read('model_hyperparameters').set_index('method'); selection=read('selected_features'); fold=read('cv_fold_metrics')
    pred=read('test_predictions').pivot(index='row_index',columns='method',values='prediction')
    pi=read('prediction_intervals_test').set_index('row_index')
    plot=read('prediction_interval_plot_data')
    models={}
    for method in cv.index:
        m=joblib.load(OUT/'models'/f'{method}.joblib'); pre=m['preprocess']; enc=pre['encode']; scale=pre['scale']
        ohe=enc.named_transformers_['zip']['ohe']; cats=ohe.categories_[0].astype(float).tolist()
        support=m['select'].support_ if 'select' in m.named_steps else np.ones(len(scale.mean_),dtype=bool)
        coefficients=np.zeros(len(support)); coefficients[support]=m['regressor'].coef_
        names=pre.get_feature_names_out().tolist()
        fs=selection[(selection.method==method)&(selection.context=='development')]
        models[method]={'cv':cv.loc[method].to_dict(),'test':test.loc[method].to_dict(),'parameters':hp.loc[method].to_dict(),
            'folds':fold[fold.method==method].to_dict('records'),'names':names,'selected':support.tolist(),
            'groups':fs.loc[fs.selected,'original_feature'].unique().tolist(),
            'predictor':{'numeric':NUMERIC,'medians':enc.named_transformers_['num'].statistics_.tolist(),
                'zip_categories':cats,'zip_fill':float(enc.named_transformers_['zip']['impute'].statistics_[0]),
                'zip_drop':int(ohe.drop_idx_[0]),'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),
                'coefficients':coefficients.tolist(),'intercept':float(m['regressor'].intercept_)}}
    samples=[]
    for i in plot.row_index:
        samples.append({'row_index':int(i),'property_id':int(df.loc[i,'id']),'actual':float(df.loc[i,'price']),
            'features':df.loc[i,FEATURES].to_dict(),'predictions':pred.loc[i].to_dict(),
            'lower':float(pi.loc[i,'lower']),'upper':float(pi.loc[i,'upper']),'covered':bool(pi.loc[i,'covered'])})
    result={'models':models,'samples':samples,'features':FEATURES,'splits':read('data_split_summary').to_dict('records'),
        'selection':json.loads((OUT/'model_selection.json').read_text()),'interval':read('interval_metrics').iloc[0].to_dict(),
        'coverage':read('interval_coverage_by_price').to_dict('records'),'rfecv':read('rfecv_curve').to_dict('records'),
        'dataset_sha256':cfg['dataset_sha256'],'execution_timestamp':cfg['timestamp_utc'],
        'feature_ranges':{c:[float(df[c].min()),float(df[c].max())] for c in FEATURES},
        'diagnostics':json.loads((OUT/'experiment_summary.json').read_text())['diagnostics']}
    # Copy published assets byte-for-byte. CSV download sources remain authoritative.
    for p in (OUT/'figures').glob('*.png'): shutil.copy2(p,DEST/'figures'/p.name)
    for n in ['cv_summary','cv_fold_metrics','test_model_metrics','selected_features','prediction_intervals_test','interval_metrics','interval_coverage_by_price','model_coefficients']:
        shutil.copy2(OUT/'tables'/f'{n}.csv',DEST/'downloads'/f'{n}.csv')
    for n in ['handoff.md','experiment_summary.json']: shutil.copy2(OUT/n,DEST/'downloads'/n)
    json_write(DEST/'data/results.json',result)
    print('Exported six frozen models and 100 fixed-seed Test examples; no model fitting.')

if __name__=='__main__': export()
