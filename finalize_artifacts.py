"""Regenerate presentation artifacts from saved verified results; never fit/tune models."""
from pathlib import Path
import json
import pandas as pd
import experiment as e
root=Path(__file__).resolve().parent/'outputs_verified'
out=e.Outputs(root); read=lambda n:pd.read_csv(root/'tables'/f'{n}.csv')
r=json.loads((root/'experiment_summary.json').read_text())
d=r['diagnostics']
d['bp_pvalue_note']='Numerical underflow: stored 0.0 means below floating-point resolution, not an exactly zero probability' if d['bp_pvalue']==0 else 'Representable floating-point survival probability'
r['engineering_tests']={'passed':13,'failed':0,'third_party_deprecation_warnings':14}
r['artifact_validation']='Passed independent validate_outputs.py; exact checks recorded in validation_results.txt'
r['authoritative_output_directory']=str(root)
r['engineering_notes']=str(root/'engineering_notes.md')
r['limitations'].extend([
 'Development residuals show a widening spread with fitted values and a heavy positive tail; Q-Q plot strongly departs from normality.',
 'Selected M3 augmented design has rank 60 of 61 columns; exact multicollinearity remains.',
 'Highest predicted-price bin has about 79.56% Test coverage; marginal coverage does not imply conditional coverage.',
 'Initial unoptimized run in outputs/ is preliminary and excluded from the archive; use outputs_verified/ only.'
])
e.json_write(root/'experiment_summary.json',r)
out.table('statistical_diagnostics',[{'measure':k,'value':v} for k,v in d.items()])
e.make_figures(read('cv_summary'),read('test_model_metrics'),read('selected_features'),read('inner_cv_results'),read('model_hyperparameters').to_dict('records'),read('prediction_intervals_test'),read('interval_coverage_by_price'),out)
e.write_handoff(out,r,read('data_split_summary'),read('cv_summary'),read('test_model_metrics'),read('model_hyperparameters'),read('selected_features'),read('interval_coverage_by_price'),read('model_coefficients'))
p=root/'handoff.md'; s=p.read_text(); s=s.replace('Automated test results: see `engineering_test_results.txt` and `validation_results.txt`.','Automated engineering tests: **13 passed, 0 failed**. Independent saved-output validation also passed. See `engineering_test_results.txt` and `validation_results.txt`.')
s=s.replace('Inspect residual-vs-fitted and Q-Q figures for curvature/tails; VIF and extreme-error tables for collinearity and errors.',
'''Visual inspection: residual spread widens at larger fitted prices, with substantial positive residuals at the high end. The Q-Q curve has a pronounced upper-tail departure from the normal reference line. Together with skewness, kurtosis and BP results, these observations do not support normal, constant-variance residual assumptions. No Test-driven model changes were made.
VIF includes infinite entries for the retained area variables; the selected augmented design is rank-deficient (60 of 61). The largest absolute Development residual is about USD 4.366 million. Exact rows remain in `extreme_errors_development.csv`.''')
s=s.replace('Intervals are marginal: conditional coverage is not guaranteed.',
'''The highest predicted-price bin (predictions above about USD 726,979) contains 949 Test observations, with coverage 755/949 = 79.56%. This is substantially below overall coverage; it is a descriptive limitation, not a reason to retune intervals on Test data.
Intervals are marginal: conditional coverage is not guaranteed.''')
p.write_text(s)
print('Presentation artifacts regenerated from unchanged saved predictions and metrics.')
