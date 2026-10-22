'use strict';
const $ = id => document.getElementById(id);
const money = v => new Intl.NumberFormat('en-US', {style:'currency',currency:'USD',maximumFractionDigits:0}).format(v);
const num = (v,d=0) => Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const methods = {
 M0:{name:'全特徵基準',short:'Baseline',type:'全部編碼欄位',description:'保留全部 18 個原始特徵所產生的 86 個編碼欄位。M0 擁有最低平均外層 CV RMSE，是本次準確度基準。'},
 M1:{name:'Pearson',short:'Pearson',type:'線性相關篩選',description:'按訓練折內的絕對 Pearson 相關排序。內層群組 CV 從 5、10、20、40、60 欄中選擇，最終保留 60 欄。'},
 M2:{name:'Mutual Information',short:'MI',type:'相依性篩選',description:'用 Mutual Information 評估相依性，明確區分離散與連續欄位。搜尋上限為 60 個編碼欄位。'},
 M3:{name:'RFE',short:'RFE',type:'one-SE 選定方法',description:'以標準化後的 OLS 係數遞迴刪除欄位，step=5。最終保留 60 欄；依 one-SE 複雜度規則入選，五折誤差皆高於 M0。'},
 M4:{name:'RFECV',short:'RFECV',type:'遞迴刪除＋群組 CV',description:'每個內層訓練折獨立前處理，再以 step=5 的等價遞迴刪除 CV 選欄數。完整 Development 擬合保留 86 欄；外層平均為 78 欄。'},
 M5:{name:'LASSO',short:'LASSO',type:'L1 篩選後重新 OLS',description:'以 LASSO 選擇欄位，再重新擬合 LinearRegression。這裡的預測來自 OLS，不是 LASSO 自身的預測。'}
};
const labels={bedrooms:'臥室數',bathrooms:'衛浴數',sqft_living:'居住面積',sqft_lot:'土地面積',floors:'樓層數',waterfront:'水岸房屋',view:'景觀評級',condition:'屋況評級',grade:'建築等級',sqft_above:'地上面積',sqft_basement:'地下室面積',yr_built:'建造年份',yr_renovated:'翻修年份',zipcode:'ZIP Code',lat:'緯度',long:'經度',sqft_living15:'鄰近居住面積',sqft_lot15:'鄰近土地面積'};
let data, active='M3', sample=0, charts={};
const colors=['#b7b0d1','#b7b0d1','#b7b0d1','#6552d9','#b7b0d1','#b7b0d1'];
function chart(id,type,labels,datasets,extra={}) {
 if(charts[id]) charts[id].destroy();
 charts[id]=new Chart($(id), {type,data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,animation:false,
  interaction:{mode:'nearest',intersect:false},plugins:{legend:{display:datasets.length>1,position:'bottom',labels:{font:{size:10},boxWidth:12}},tooltip:{padding:10}},
  scales:{x:{grid:{display:false},ticks:{font:{size:10},color:'#8490a3'}},y:{grid:{color:'#edf0f6'},ticks:{font:{size:10},color:'#8490a3'}}}},...extra});
}
function render(){
 const m=data.models[active];
 document.querySelectorAll('[data-method]').forEach(b=>{b.classList.toggle('active',b.dataset.method===active);b.setAttribute('aria-pressed',String(b.dataset.method===active));});
 $('method-description').textContent=methods[active].description;
 $('selected-count').textContent=m.test.selected_count;
 $('original-count').textContent=m.groups.length;
 $('parameter').textContent=active==='M5'?`LASSO α = ${m.parameters.value}`:active==='M0'?'全部欄位，不篩選':`內層選定欄數：${m.parameters.value}`;
 $('active-label').textContent=`${active} · ${methods[active].short}`;
 $('metrics').innerHTML=[['Test RMSE',money(m.test.rmse),'均方根誤差 · 越低越好'],['Test MAE',money(m.test.mae),'平均絕對誤差'],['Test R²',num(m.test.r2,4),'解釋變異比例'],['CV RMSE',money(m.cv.rmse_mean),`五折 SD ${money(m.cv.rmse_std)}`],['編碼欄位',`${m.test.selected_count} / 86`,`${m.groups.length} 個原始特徵群組`]].map(([label,value,note])=>`<div class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
 $('comparison-body').innerHTML=Object.entries(data.models).map(([id,m])=>`<tr class="${id===active?'selected-row':''}"><td>${id} · ${methods[id].short}${id==='M3'?'<span class="badge purple">one-SE 選定</span>':id==='M0'?'<span class="badge green">最低 CV RMSE</span>':''}</td><td>${m.test.selected_count}</td><td>${num(m.cv.rmse_mean)} ± ${num(m.cv.rmse_std)}</td><td>${num(m.test.mae)}</td><td>${num(m.test.rmse)}</td><td>${num(m.test.r2,4)}</td></tr>`).join('');
 const palette=Object.keys(methods).map(id=>id===active?'#6552d9':'#c5bce7');
 chart('cv-chart','bar',Object.keys(methods),[{label:'CV RMSE (USD)',data:Object.values(data.models).map(m=>m.cv.rmse_mean),backgroundColor:palette,borderRadius:5}]);
 chart('fold-chart','line',['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'],[{label:'M0 基準',data:data.models.M0.folds.map(f=>f.rmse),borderColor:'#a9afbe',backgroundColor:'#a9afbe',pointRadius:4,borderWidth:2},...(active==='M0'?[]:[{label:`${active} ${methods[active].short}`,data:m.folds.map(f=>f.rmse),borderColor:'#6552d9',backgroundColor:'#6552d9',pointRadius:4,borderWidth:2}])]);
 $('features-count').textContent=`${m.test.selected_count} 個編碼欄位 / ${m.groups.length} 個原始群組`;
 $('feature-groups').innerHTML=data.features.map(f=>`<span class="chip ${m.groups.includes(f)?'selected':''}">${labels[f]}<small>${f}</small></span>`).join('');
 renderFeatureTable();
 const pts=data.samples.map(r=>({x:r.actual,y:r.predictions[active]}));
 const max=Math.max(...pts.flatMap(p=>[p.x,p.y]));
 chart('scatter-chart','scatter',[],[{label:'Test 抽樣',data:pts,backgroundColor:'#8571cf99',pointRadius:3}, {type:'line',label:'理想預測',data:[{x:0,y:0},{x:max,y:max}],borderColor:'#a5acb9',borderDash:[5,5],borderWidth:1,pointRadius:0}],{options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{position:'bottom',labels:{font:{size:10},boxWidth:12}}},scales:{x:{type:'linear',title:{display:true,text:'實際價格 (USD)',font:{size:10}},grid:{color:'#f0f1f6'},ticks:{callback:v=>`${(v/1e6).toFixed(1)}M`,font:{size:10}}},y:{title:{display:true,text:'預測價格 (USD)',font:{size:10}},grid:{color:'#f0f1f6'},ticks:{callback:v=>`${(v/1e6).toFixed(1)}M`,font:{size:10}}}}}});
 predictScenario();
}
function renderFeatureTable(){
 const m=data.models[active],q=$('feature-search').value.trim().toLowerCase();
 $('intercept').textContent=`截距：${money(m.predictor.intercept)}。此為標準化座標下的基準預測，獨立於下方特徵係數，不列入重要性排序。`;
 const rows=m.names.map((name,i)=>({name,i})).filter(r=>r.name.toLowerCase().includes(q));
 $('feature-body').innerHTML=rows.length?rows.map(({name,i})=>`<tr><td>${name}</td><td>${m.selected[i]?'保留':'未保留'}</td><td>${m.selected[i]?num(m.predictor.coefficients[i],2):'—'}</td></tr>`).join(''):'<tr><td colspan="3">沒有符合的欄位。</td></tr>';
}
function setupInputs(){
 const primary=['sqft_above','sqft_basement','sqft_living','zipcode','bedrooms','bathrooms'];
 function field(f){
  const unit=f.startsWith('sqft')?'sq ft':f==='yr_renovated'?'0 = 未翻修':'';
  const label=`${labels[f]} <small>${unit}</small>`;
  if(f==='zipcode') return `<label>${label}<select name="${f}">${data.models.M3.predictor.zip_categories.map(v=>`<option value="${v}">${v}</option>`).join('')}</select></label>`;
  if(['waterfront','view','condition','grade'].includes(f)) {
   const [min,max]=data.feature_ranges[f]; const opts=[];
   for(let i=min;i<=max;i++) opts.push(`<option value="${i}">${f==='waterfront'?(i?'是':'否'):i}</option>`);
   return `<label>${label}<select name="${f}">${opts.join('')}</select></label>`;
  }
  const [min,max]=data.feature_ranges[f];
  return `<label>${label}<input name="${f}" type="number" step="any" min="${min}" max="${max}" required ${f==='sqft_living'?'readonly aria-label="居住面積，自動加總"':''}></label>`;
 }
 $('primary-inputs').innerHTML=primary.map(field).join('');
 $('secondary-inputs').innerHTML=data.features.filter(f=>!primary.includes(f)).map(field).join('');
 $('sample-select').innerHTML=data.samples.slice(0,20).map((r,i)=>`<option value="${i}">範例 ${i+1} · ZIP ${r.features.zipcode} · ${num(r.features.sqft_living)} sq ft</option>`).join('');
 restoreSample();
}
function restoreSample(){
 const r=data.samples[sample];
 data.features.forEach(f=>{$('prediction-form').elements.namedItem(f).value=r.features[f];});
 predictScenario();
}
function predictScenario(){
 if(!data || !$('prediction-form').elements.namedItem('sqft_above')) return;
 const form=$('prediction-form'); const values={};
 data.features.forEach(f=>{const el=form.elements.namedItem(f); values[f]=el.value===''?null:Number(el.value);});
 const valid=form.checkValidity() && data.features.every(f=>values[f]!==null && values[f]>=data.feature_ranges[f][0] && values[f]<=data.feature_ranges[f][1]);
 if(!valid){$('prediction-value').textContent='請檢查輸入';$('scenario-interval').textContent='請填入有效的房屋條件。';$('sample-actual').textContent='';$('scenario-status').textContent='空白或超出資料範圍的輸入不會顯示預測。';return;}
 const predicted=HW2.predict(data.models[active],values);
 $('prediction-model').textContent=`${active} · ${methods[active].short}`;
 $('prediction-value').textContent=money(predicted);
 $('scenario-interval').innerHTML=active==='M3'?`M3 · 95% 歷史分布區間示範<strong>${money(predicted-data.interval.q_hat)} — ${money(predicted+data.interval.q_hat)}</strong>`:'此研究只為最終 M3 校準區間。<br>切換至 M3 可查看區間示範。';
 const same=data.features.every(f=>values[f]===data.samples[sample].features[f]);
 $('sample-actual').textContent=same?`此筆歷史實際成交價：${money(data.samples[sample].actual)}`:'已修改為自訂情境；沒有對應的已知實際成交價。';
 $('scenario-status').textContent=same?`原始資料 row index ${data.samples[sample].row_index} · 保存的 Test 範例`:'手動試算不會改變上方評估指標或已選定的模型。';
}
function init(){
 $('methods').innerHTML=Object.entries(methods).map(([id,m])=>`<button type="button" class="method-button" data-method="${id}" aria-pressed="false"><span class="method-id">${id}</span>${m.name}<small>${m.type}</small></button>`).join('');
 $('methods').addEventListener('click',e=>{const b=e.target.closest('[data-method]');if(b){active=b.dataset.method;render();}});
 $('feature-search').addEventListener('input',renderFeatureTable);
 $('sample-select').addEventListener('change',()=>{sample=Number($('sample-select').value);restoreSample();});
 $('reset-sample').addEventListener('click',restoreSample);
 $('prediction-form').addEventListener('submit',e=>e.preventDefault());
 $('prediction-form').addEventListener('input',e=>{
  if(['sqft_above','sqft_basement'].includes(e.target.name)){
   const f=$('prediction-form').elements;
   f.namedItem('sqft_living').value=Number(f.namedItem('sqft_above').value)+Number(f.namedItem('sqft_basement').value);
  }
  predictScenario();
 });
 $('threshold').textContent=money(data.selection.threshold);
 $('interval-metrics').innerHTML=[['實際 Test 涵蓋率',`${num(100*data.interval.empirical_coverage,2)}%`],['平均區間寬度',money(data.interval.average_width)],['Calibration 觀測數',num(data.interval.calibration_n)],['涵蓋 / 全部 Test',`${num(data.interval.covered)} / ${num(data.interval.test_n)}`]].map(([label,value])=>`<div><strong>${value}</strong><span>${label}</span></div>`).join('');
 $('split-details').innerHTML=data.splits.map(s=>`<div><strong>${s.partition}</strong><br>${num(s.rows)} 筆 · ${num(s.groups)} 個房屋</div>`).join('');
 $('loading').hidden=true; $('results').hidden=false;
 setupInputs();render();
 chart('coverage-chart','bar',data.coverage.map((g,i)=>`組 ${i+1} · n=${g.observations}`),[{label:'實際涵蓋率 (%)',data:data.coverage.map(g=>g.coverage*100),backgroundColor:data.coverage.map((g,i)=>i===4?'#d3a965':'#9d8bd4'),borderRadius:4},{type:'line',label:'名目 95%',data:data.coverage.map(()=>95),borderColor:'#c48177',borderDash:[5,4],pointRadius:0,borderWidth:2}],{options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{position:'bottom',labels:{font:{size:10},boxWidth:12}},tooltip:{callbacks:{afterLabel:ctx=>data.coverage[ctx.dataIndex].predicted_price_group+' USD'}}},scales:{x:{grid:{display:false},ticks:{font:{size:9}}},y:{min:0,max:100,ticks:{callback:v=>v+'%',font:{size:10}},grid:{color:'#edf0f6'}}}}});
}
fetch('data/results.json').then(r=>{if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json();}).then(d=>{data=d;init();}).catch(err=>{$('loading').hidden=true;$('error').hidden=false;$('error').textContent=`無法載入研究結果：${err.message}。請重新整理，或下載研究交接文件。`;});
