# Machine Learning HW2：King County 多元線性迴歸

## 線上互動展示

[開啟 King County Regression Explorer →](https://wujin57.github.io/AIOT-Hw2/)

![HW2 儀表板](docs/images/dashboard-desktop.png)

沿用 fifty startup 的紫色靜態儀表板風格：六模型切換、CV/Test 比較、編碼欄位搜尋、房價情境試算、M3 conformal 區間、研究圖表與結果下載。瀏覽器使用保存模型的補值、ZIP 編碼、標準化與 OLS 係數進行試算，不重新訓練；所有歷史評估結果與最終 M3 選擇保持凍結。

- 網頁原始碼：`dashboard/`，以 GitHub Pages 部署；Chart.js 隨專案提供，不依賴外部 CDN。
- 匯出展示資料：`.venv/bin/python scripts/export_dashboard.py`，只讀取既有模型與結果，不呼叫模型擬合。
- 核對匯出：`.venv/bin/python scripts/verify_dashboard.py`。
- 本機預覽：在儲存庫根目錄執行 `.venv/bin/python -m http.server 8765`，開啟 `http://localhost:8765/dashboard/`。
- 瀏覽器測試：匯出與核對後，開啟 `http://localhost:8765/verification/web/browser-qa.html`。全 Test 資料 × 六模型，以及缺值／未知 ZIP 測試，皆與 Python 保存預測核對。
- Web 驗證紀錄：`verification/web/README.md`。網站是本次另行授權的展示功能，不改變原始研究規格或數值產物。

依 `IMPLEMENTATION_PROMPT.md` 實作 CRISP-DM 研究流程。使用官方 Kaggle King County 資料，六種配置皆以 `LinearRegression` 作最終預測；不包含正式學術 PDF。

## ChatGPT Stage 4 審查入口

請優先閱讀 [交接文件](outputs_verified/handoff.md)、[實驗摘要 JSON](outputs_verified/experiment_summary.json)、[CV 比較](outputs_verified/tables/cv_summary.csv) 與 [Test 比較](outputs_verified/tables/test_model_metrics.csv)。

可以將儲存庫網址連同以下文字貼給 ChatGPT：

> 請閱讀 outputs_verified/handoff.md、experiment_summary.json 與相關 CSV，依 IMPLEMENTATION_PROMPT.md 進行 Stage 4 統計審查。檢查資料洩漏、群組交叉驗證、選模規則、RFECV 等價實作、有效秩與 conformal 區間；區分實測結果與統計推論，不要假設殘差假設或條件涵蓋率成立。正確學號為 7115056123。

交接文件中的本機絕對路徑是執行紀錄；在 GitHub 請使用上方相對連結。若 ChatGPT 無法直接讀取儲存庫，請上傳上述交接文件、摘要與 CSV。

## Stage 4：只驗證已發表結果

不需重新訓練。從任意工作目錄執行 `<repo>/validate_outputs.py`，預設會使用該程式所在儲存庫的 `data/kc_house_data.csv` 與 `outputs_verified/`。也可指定 `--data /path/to/kc_house_data.csv`；顯式指定的相對路徑以目前工作目錄為準。驗證器先核對 `research_configuration.json` 的 SHA256，再檢查保存的模型預測。原始 `dataset_path` 保留作為歷史執行來源紀錄，不再作為資料讀取的依賴。

```bash
.venv/bin/python validate_outputs.py
.venv/bin/python validate_outputs.py --data /path/to/kc_house_data.csv
.venv/bin/python -m pytest tests -q
```

搬移目錄、顯式資料路徑、錯誤 checksum 拒絕測試與本次實際驗證紀錄位於 `verification/stage4/`。下方主程式指令僅供完整重現；本次修正沒有重跑訓練或覆寫數值產物。

## 執行

建議 Python 3.9–3.12；本次實際環境與精確版本見 `requirements-lock.txt` 及輸出 JSON。

```bash
cd hw2
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python 7115056123_hw2.py --data data/kc_house_data.csv --output outputs_verified
.venv/bin/python validate_outputs.py --output outputs_verified
```

本機已建立 `.venv`，可直接執行上述測試與主程式。若要重建相同 Python 3.9 環境，可改安裝 `requirements-lock.txt`。所有路徑由命令列或專案位置決定，不依賴 notebook 狀態。主程式會覆寫指定輸出資料夾中的同名結果；若要保留舊結果，使用不同 `--output`。

目前 `outputs_verified/` 是正式結果；`outputs/` 為已停止的初次慢速執行，請勿混用。數值調整與警告詳見 `outputs_verified/engineering_notes.md`。

## 資料與方法

- 官方來源：https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
- CSV 由官方公開 API 取得；來源、SHA256、維度、欄位及版本記於 `outputs_verified/research_configuration.json`。
- 21,613 筆、21 欄、18 個原始候選特徵；`id` 僅分組，`date` 僅檢視，`price` 僅目標。
- 依房屋 ID 以固定 seed=42 切 Development/Calibration/Test，目標約 65%/15%/20%。群組大小造成比例微幅差異。
- Development 外層 GroupKFold(5)，內層 GroupKFold(3)。相同外層切分供全部六種方法使用。
- M0 全特徵、M1 Pearson、M2 mutual information、M3 RFE、M4 RFECV、M5 LASSO 篩選後重新 OLS 擬合。
- ZIP Code 用 drop-first one-hot；所有編碼欄位標準化。其他數值、序位欄位維持原始數值表示，不轉換目標、不刪除高價異常值。
- MI 將 ZIP dummy、bedrooms、floors、waterfront、view、condition、grade、yr_built、yr_renovated 視為離散；其他數值視為連續。這是預先指定的估計假設。
- M4 使用明確實作的等價遞迴刪除交叉驗證：每個內層訓練折獨立學習前處理，再以 step=5 的 RFE 路徑比較 RMSE，最低保留 1 欄。避免在 RFECV 前對完整外層訓練資料先擬合前處理，導致內層驗證洩漏。若 ZIP 類別數不同，採各折可用共同欄數；測試以固定設計矩陣核對 sklearn RFECV 分數及選擇結果。
- 內層快取只在同一訓練折重用已擬合的前處理結果，不跨折共用。最終模型仍是可序列化的 sklearn Pipeline。
- LASSO alpha=logspace(0,5,11)，選取 abs(coef)>1e-8；空選擇及未收斂候選記為無效，不自動加回特徵。
- 勝出方法只依外層 CV 的 one-standard-error 規則選擇；Test 不影響選擇。所有方法重擬合 Development 後才作 Test 比較。
- 95% split conformal 使用獨立 Calibration 絕對誤差的第 ceil((n+1)*0.95) 順序統計量。分箱界線來自 Calibration 預測。負下界保留。

## 輸出與檢查

優先交給 ChatGPT Stage 4：

1. `outputs_verified/handoff.md`：A–K 完整交接文件。
2. `outputs_verified/experiment_summary.json`：實際結果與執行狀態。

其他內容：

- `outputs_verified/tables/`：CV、Test、特徵、係數、VIF、資料切分、逐筆預測及區間。
- `outputs_verified/figures/`：15 張指定圖，加特徵分布和殘差直方圖。
- `outputs_verified/models/final_model.joblib`：最終模型；另存 M0–M5。
- `outputs_verified/model_selection.json`：Test 評估前凍結的選擇。
- `outputs_verified/calibration.json`：Test 評估前凍結的區間與分箱界線。
- `outputs_verified/fold_audit.json`：各內層訓練與驗證 row index，可連結 partition_assignments 核對群組邊界。
- `outputs_verified/engineering_test_results.txt`、`outputs_verified/validation_results.txt`：工程測試與實際結果一致性檢查。
- `outputs_verified/run.log`：實際執行記錄與警告。

載入模型時需讓 `experiment.py` 可匯入（在 hw2 資料夾執行即可）：

```python
import joblib, pandas as pd
from experiment import FEATURES
model = joblib.load('outputs_verified/models/final_model.joblib')
data = pd.read_csv('data/kc_house_data.csv')
prediction = model.predict(data[FEATURES].iloc[:5])
```

僅載入可信來源的 joblib 檔案。

## 解讀限制與繳交

- **預測準確度**：M0 的外層 CV 平均 RMSE 最低（165,873.34 USD）；M3 為 166,319.69 USD，且在全部五個外層折中 RMSE 都高於 M0。不能宣稱 RFE 提高了此次預測準確度。
- **模型複雜度**：M3 因預先指定的 one-standard-error 複雜度規則而被選中。86 → 60 欄的 30.23% 減少指的是編碼後模型欄位，並非原始 18 個特徵減少 30.23%。Pearson、MI、RFE 的搜尋網格皆僅為 `[5, 10, 20, 40, 60]`，沒有評估 60 欄以上的這三種篩選配置，不能據此推論全域最佳欄數。
- **係數可識別性**：M3 含截距的設計矩陣共有 61 欄、秩為 60。OLS 個別係數可能不是唯一解，不宜把係數大小當作唯一可解釋的特徵重要性或因果效果。截距是標準化特徵座標下的基準預測，須與特徵係數分開解釋。
- **區間涵蓋率**：整體 Test 涵蓋率約 94.36%，最高預測價格組僅約 79.56%。基本 split conformal 在適當 exchangeability 假設下提供邊際涵蓋性質，不保證每個房價群組的條件涵蓋率。依房屋分組切分不等於已證明交易層級 exchangeability。

調整後 R² 使用含截距設計矩陣秩減一，Test/validation 版本僅描述性使用。BP 採完整設計欄空間的正交基底處理 rank deficiency，使用有效秩作自由度。殘差診斷是選模後描述性結果，不證明假設成立。

本研究評估歷史同分布附近的房屋價格，不代表現今市場預測能力。群組分離不能保證交易層級 exchangeability；區間不保證各價格區間條件涵蓋率。one-SE 只是實用選模準則，不是統計等效性檢定。

依使用者確認的正確學號 `7115056123`，主程式為 `7115056123_hw2.py`，壓縮檔為 `7115056123_hw2.zip`。原始 prompt 的主程式學號是筆誤，已依使用者更正；`IMPLEMENTATION_PROMPT.md` 保留原始內容作為紀錄。壓縮檔排除 `.venv`、快取與壓縮檔本身，包含資料、程式、測試、環境版本及實際產物。
