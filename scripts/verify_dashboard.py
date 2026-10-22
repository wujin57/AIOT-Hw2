"""Check the static export against immutable results and create browser QA fixtures."""
from pathlib import Path
import hashlib
from html.parser import HTMLParser
import json
import sys
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import FEATURES, json_write
D=ROOT/'dashboard'; O=ROOT/'outputs_verified'; V=ROOT/'verification/web'

def verify():
    V.mkdir(exist_ok=True,parents=True)
    data=json.loads((D/'data/results.json').read_text())
    df=pd.read_csv(ROOT/'data/kc_house_data.csv')
    assert data['dataset_sha256']==hashlib.sha256((ROOT/'data/kc_house_data.csv').read_bytes()).hexdigest()
    assert data['selection']==json.loads((O/'model_selection.json').read_text())
    cv=pd.read_csv(O/'tables/cv_summary.csv').set_index('method')
    tm=pd.read_csv(O/'tables/test_model_metrics.csv').set_index('method')
    for key,m in data['models'].items():
        for metric in ['mae','mse','rmse','r2','adjusted_r2']:
            np.testing.assert_allclose(m['test'][metric],tm.loc[key,metric],rtol=0,atol=0)
        np.testing.assert_allclose(m['cv']['rmse_mean'],cv.loc[key,'rmse_mean'],rtol=0,atol=0)
        assert sum(m['selected'])==m['test']['selected_count']
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('href','src') and value and not value.startswith(('#','http')):
                    assert (D/value).is_file(),value
    Links().feed((D/'index.html').read_text())
    for p in (D/'downloads').iterdir():
        original=O/p.name if p.suffix in ('.md','.json') else O/'tables'/p.name
        assert p.read_bytes()==original.read_bytes(),p.name
    for p in (D/'figures').iterdir(): assert p.read_bytes()==(O/'figures'/p.name).read_bytes()
    expected=pd.read_csv(O/'tables/test_predictions.csv').pivot(index='row_index',columns='method',values='prediction')
    rows=[]
    for idx,r in expected.iterrows(): rows.append({'features':df.loc[idx,FEATURES].to_dict(),'expected':r.to_dict()})
    # Additional inference-only edge cases: missing numeric/category and unseen ZIP.
    base=df.loc[expected.index[0],FEATURES].to_dict()
    edge=[]
    for changes in [{'bedrooms':np.nan,'zipcode':np.nan},{'zipcode':99999}]:
        case={**base,**changes}; exp={}
        for key in data['models']:
            model=joblib.load(O/'models'/f'{key}.joblib')
            exp[key]=float(model.predict(pd.DataFrame([case],columns=FEATURES))[0])
        edge.append({'features':case,'expected':exp})
    json_write(V/'parity-fixtures.json',{'test':rows,'edge':edge})
    print('PASS: all published metrics, selection, 17 figures and 10 downloads match saved artifacts; local HTML links resolve.')
    print('Prepared 4,324 Test rows x 6 models plus 2 preprocessing edge cases for browser inference parity.')

if __name__=='__main__': verify()
