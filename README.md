# 氷河湖水域抽出・マルチセンサー時系列監視

Sentinel-2 と Landsat Collection 2 Level-2 の画像から水域を抽出し、同じ湖へ永続IDを付けて時系列追跡し、面積急増の確認候補を出力する研究用プログラムです。

このコードが出力するアラートは GLOF の発生を断定するものではありません。画像、品質情報、前後観測、地形・現地情報を用いた確認が必要です。

## 主な処理

1. Sentinel-2 L2A の BOA 反射率から NDWI 水域を抽出
2. Landsat L2SP の地表面反射率から NDWI 水域を抽出
3. SCL または QA_PIXEL を品質情報として記録・利用
4. 両センサーの観測を共通カタログへ統合
5. コア面積、位置、重なり、距離を使って湖をグループ化
6. 結合・分裂を記録しながら永続湖IDを追跡
7. 条件を満たす部分欠損を前後観測から補完
8. 面積変化を計算し、要確認候補と確認用レポートを出力

## フォルダ構成

```text
glacial_lake_monitoring/
├─ sentinel_ndwi.py                 実行用: Sentinel-2水域抽出
├─ landsat_ndwi.py                  実行用: Landsat水域抽出
├─ multisensor_observations.py      実行用: 統合・永続ID追跡
├─ lake_area_alerts.py              実行用: 面積急増候補の判定
├─ glacial_lake/                    直接実行しない内部処理
│  ├─ multisensor_config.py         統合処理の入力・閾値・追跡設定
│  ├─ multisensor_catalog.py
│  ├─ multisensor_grouping.py       コア基準の湖候補グループ化
│  ├─ multisensor_shadow_validation.py
│  ├─ multisensor_tracking.py       永続湖IDの割当・過去方向遡及
│  ├─ multisensor_relationships.py  湖の結合・分裂関係の記録
│  ├─ multisensor_interpolation.py  部分欠損補完・補完後ID整合
│  ├─ multisensor_fragments.py      コア未満の小断片の面積追加
│  ├─ multisensor_quality.py        センサー別の観測品質判定
│  ├─ multisensor_missing.py        観測間の欠損レコード作成
│  ├─ multisensor_timeseries.py     採用時系列・変化指標の作成
│  ├─ multisensor_polygons.py       共通追跡ポリゴンの作成
│  ├─ multisensor_outputs.py        最終成果物の保存
│  └─ water_preprocessing_common.py 前処理の共通関数
├─ environment.yml
├─ data/
│  ├─ sentinel/   解凍済み Sentinel-2 SAFE 製品
│  ├─ landsat/    解凍済み Landsat L2SP 製品
│  ├─ dem/        標高マスクに使う GeoTIFF
│  └─ aoi/        analysis_area.geojson
├─ outputs/       実行時に作成される成果物
│  ├─ results/       普段確認する最終成果物
│  ├─ details/       詳細解析用CSV
│  └─ diagnostics/   ラスター、Shapefile、中間GeoPackage
├─ tests/         アラート判定等の単体テスト
└─ docs/          日本語の手順・引継ぎ資料
```

衛星画像、DEM、AOI、実行結果は `.gitignore` の対象です。GitHub へはコードと文書だけを共有し、入力データは利用者が別途用意してください。

## 環境構築

Anaconda または Miniconda をインストール後、このフォルダで次を実行します。

```powershell
conda env create -f environment.yml
conda activate glacial-lake-monitoring
```

Windows では GDAL、Rasterio、GeoPandas の依存関係をまとめて導入できるため、`conda-forge` の環境を推奨します。

## 入力データの準備

### Sentinel-2

`data/sentinel/` の直下へ、解凍済みの `*.SAFE` 製品フォルダを配置します。プログラムが製品内の B03、B08、SCL、メタデータを再帰的に探します。

### Landsat

`data/landsat/` の直下へ、解凍済みの Collection 2 Level-2 製品フォルダを配置します。各製品には少なくとも MTL、SR_B3、SR_B5 が必要です。QA_PIXEL と QA_RADSAT がある場合は品質判定に使用します。

### DEM と AOI

- DEM: `data/dem/` に対象範囲を覆う GeoTIFF を配置
- AOI: `data/aoi/analysis_area.geojson` を配置

AOI の CRS は任意ですが、面積・距離計算用の CRS は `multisensor_observations.py` の `MULTISENSOR_TRACKING_CRS` で指定します。対象地域に適したメートル単位の投影座標系を使用してください。

## 実行前に確認する設定

各スクリプト先頭の「設定」または「ユーザー設定」を確認します。

- `sentinel_ndwi.py`: NDWI 閾値、最低標高、SCL 条件、AOI
- `landsat_ndwi.py`: NDWI 閾値、最低標高、QA_PIXEL 条件、AOI
- `glacial_lake/multisensor_config.py`: AOI、追跡 CRS、コア・継続面積、距離、追跡・補完条件
- `multisensor_observations.py`: 統合処理の実行順と内部出力設定
- `glacial_lake/multisensor_grouping.py`: コア面積、継続面積、距離による湖候補のグループ化
- `glacial_lake/multisensor_shadow_validation.py`: SCL雲影・Landsat QA候補の前後観測による検証
- `glacial_lake/multisensor_tracking.py`: 形状の重なり、面積比、距離による永続湖IDの割当と過去候補への遡及
- `glacial_lake/multisensor_relationships.py`: 永続湖の結合・分裂イベントと関連IDの記録
- `glacial_lake/multisensor_interpolation.py`: 前後形状・関連湖ファミリーによる部分欠損補完と、補完後に接続したIDの整合
- `glacial_lake/multisensor_fragments.py`: コア未満の小断片を独立湖にせず、既存の永続湖面積へ追加
- `glacial_lake/multisensor_quality.py`: Sentinel-2・Landsatの有効画素、QA、面積比などによる観測品質判定
- `glacial_lake/multisensor_missing.py`: 観測間で一部または全部の永続湖が未検出になった日を欠損レコードとして明示
- `glacial_lake/multisensor_timeseries.py`: 採用観測を日付順に整理し、観測間隔・面積差・日変化率を計算
- `glacial_lake/multisensor_polygons.py`: センサー別ポリゴンをAOIで切り抜き、CRSと属性を共通追跡形式へ統一
- `glacial_lake/multisensor_outputs.py`: 出力フォルダと中間GeoPackageの初期化、最終CSV・GeoPackage・時系列グラフの作成
- `lake_area_alerts.py`: 比較間隔、センサー間補正、急増候補、確認条件

現在の基準値は Sentinel-2 NDWI 0.45、Landsat NDWI 0.26、最低標高 3000 m です。ほかの地域へ適用するときは、代表日を QGIS で確認して妥当性を再評価してください。

## 実行順

リポジトリのルートで次の順に実行します。
実行対象は次の4ファイルだけで、`glacial_lake/` 内のファイルは直接実行しません。

```powershell
python sentinel_ndwi.py
python landsat_ndwi.py
python multisensor_observations.py
python lake_area_alerts.py
```

`sentinel_ndwi.py` と `landsat_ndwi.py` は互いに独立して実行できます。`multisensor_observations.py` は両方の集計CSVを入力にします。`lake_area_alerts.py` は統合処理の `final_lake_observations.csv` を入力にします。

## 主な出力

出力は用途別に3つへ分けています。

### 通常確認用: `outputs/results/`

- `summaries/sentinel_area_summary.csv`: Sentinel-2前処理の結果一覧
- `summaries/landsat_area_summary.csv`: Landsat前処理の結果一覧
- `summaries/scene_catalog.csv`: 統合対象シーンの品質・採否一覧
- `geopackages/multisensor_final.gpkg`: QGISで確認する最終湖追跡結果
- `figures/`: 面積、湖数、変化率の最終グラフ
- `alerts/alert_review_report.csv`: 人が確認すべきアラート候補

### 詳細解析用: `outputs/details/`

- `multisensor/selected_timeseries.csv`: 採用された観測時系列
- `multisensor/final_lake_observations.csv`: 湖ID・日付ごとの詳細結果
- `multisensor/final_timeseries.csv`: 日付ごとの総面積・湖数・欠損情報
- `alerts/`: 全観測の変化指標、個別湖・関連湖グループの判定、センサー補正記録

### 診断用: `outputs/diagnostics/`

- `sentinel/rasters/`, `sentinel/shapefiles/`: Sentinel-2のシーン別中間成果物
- `landsat/rasters/`, `landsat/shapefiles/`: Landsatのシーン別中間成果物
- `multisensor/multisensor_intermediate.gpkg`: ID追跡、補完、結合・分裂を調査する中間レイヤー

## 結果の確認

1. `outputs/diagnostics/`の各日の水域マスクと元画像を QGIS で重ね、誤検出・見逃しを確認します。
2. `outputs/results/summaries/scene_catalog.csv` で `selected_quality`、有効画素率、QA/SCL の影響を確認します。
3. `outputs/results/geopackages/multisensor_final.gpkg`を永続湖IDで色分けし、時系列コントローラで ID の乗り移り、過剰分裂、誤結合がないか確認します。
4. `outputs/results/alerts/alert_review_report.csv`で候補理由、比較に使った観測、センサー、面積差、増加率、信頼度を確認します。
5. 候補日は前後の衛星画像を目視し、雲、影、雪氷、位置ずれ、解像度差の影響を除外します。

## テスト

入力画像を処理せず、アラート判定の主要ロジックを確認できます。

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## GitHub へ登録する場合

先に GitHub 上で空のリポジトリを作成し、このフォルダで次を実行します。`YOUR_REPOSITORY_URL` は作成したリポジトリの URL に置き換えます。

```powershell
git init
git add .
git commit -m "Initial release"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

公開前に、入力画像や個人情報を含むパスが `git status` に表示されていないことを確認してください。ライセンスはまだ同梱していないため、公開範囲に応じて MIT、BSD-3-Clause などを選択してください。

## 現段階の注意点

- 閾値は現在の研究地域と観測期間で調整した値です。
- 雲、影、積雪、浮氷、センサー解像度差を完全には除去できません。
- センサー間面積補正は十分な独立湖・近接観測がある場合のみ成立します。
- アラートはスクリーニング用であり、危険度や決壊時刻を直接予測するものではありません。
- 季節変動や平年値を使う複数年ベースラインは今後の拡張項目です。

詳しい操作と背景は `docs/氷河湖抽出_時系列追跡_引継ぎ手順書_現行版.docx` を参照してください。
