from __future__ import annotations

from pathlib import Path

# ============================================================
# 設定
# ============================================================

# 内部モジュールは glacial_lake/ 配下にありますが、入出力は
# 実行スクリプトと同じリポジトリ直下を基準にします。
SCRIPT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# ユーザー設定
# 通常変更するのは、ここから「内部出力設定」の直前までです。
# ============================================================

# ------------------------------------------------------------
# 1. 入力データ
# ------------------------------------------------------------

# Sentinel-2の二値化処理で作成したシーン別集計CSV
SENTINEL_SUMMARY_CSV = (
    SCRIPT_DIR
    / "outputs"
    / "results"
    / "summaries"
    / "sentinel_area_summary.csv"
)

# landsat_ndwi.pyで作成したシーン別集計CSV
LANDSAT_SUMMARY_CSV = (
    SCRIPT_DIR
    / "outputs"
    / "results"
    / "summaries"
    / "landsat_area_summary.csv"
)

# 湖を解析する範囲。Landsatの切り抜きにも使用する。
LANDSAT_AOI_PATH = (
    SCRIPT_DIR
    / "data"
    / "aoi"
    / "analysis_area.geojson"
)

# 出力属性と中間フォルダに使用するAOI名
MULTISENSOR_AOI_NAME = "analysis_area"

# ------------------------------------------------------------
# 2. 出力先と共通座標系
# ------------------------------------------------------------

# results・details・diagnosticsを格納する出力ルート。
MULTISENSOR_OUTPUT_FOLDER_NAME = (
    "outputs"
)

# 面積・距離計算に使用する投影座標系
MULTISENSOR_TRACKING_CRS = "EPSG:32643"


# ------------------------------------------------------------
# 3. 水域抽出と湖候補のグループ化
# ------------------------------------------------------------

# 「コア面積」以上なら、新しい永続湖IDを作成できる。
# 「継続面積」以上なら、新規IDは作れないが既存IDを継続できる。
# 「小ポリゴン距離」以内の小断片は、最寄りのコアへまとめる。
SENTINEL_CORE_AREA_M2 = 1_500.0
SENTINEL_CONTINUATION_AREA_M2 = 500.0
SENTINEL_SMALL_TO_CORE_DISTANCE_M = 40.0

LANDSAT_CORE_AREA_M2 = 3_600.0
LANDSAT_CONTINUATION_AREA_M2 = 900.0
LANDSAT_SMALL_TO_CORE_DISTANCE_M = 60.0

MULTISENSOR_GROUPING_SETTINGS = {
    "Sentinel-2": {
        "core_area_m2": SENTINEL_CORE_AREA_M2,
        "continuation_area_m2": SENTINEL_CONTINUATION_AREA_M2,
        "small_to_core_distance_m": SENTINEL_SMALL_TO_CORE_DISTANCE_M,
    },
    "Landsat": {
        "core_area_m2": LANDSAT_CORE_AREA_M2,
        "continuation_area_m2": LANDSAT_CONTINUATION_AREA_M2,
        "small_to_core_distance_m": LANDSAT_SMALL_TO_CORE_DISTANCE_M,
    },
}

# Landsatの水域候補のうち、QA_PIXELのCloud / Cirrus /
# Dilated Cloud / Cloud Shadowがこの割合以上を占める候補は、
# 新規IDを直ちに作らず、前後観測による確認対象にする。
LANDSAT_QA_REVIEW_MIN_RATIO = 0.50


# ------------------------------------------------------------
# 4. SCL雲影候補の時系列確認
# ------------------------------------------------------------

# 前後何日までを、雲影内水域候補の確認に使用するか。
MULTISENSOR_SHADOW_MAX_GAP_DAYS = 60

# True: 前後両方の観測で支持された候補だけ残す。
# False: 前後どちらか一方で支持されれば残す。
MULTISENSOR_SHADOW_REQUIRE_BOTH_SIDES = False

# 形状重なり・面積比・距離による支持判定条件
MULTISENSOR_SHADOW_MIN_OVERLAP_SMALLER_RATIO = 0.20
MULTISENSOR_SHADOW_MIN_IOU = 0.05
MULTISENSOR_SHADOW_MIN_AREA_RATIO = 0.20
MULTISENSOR_SHADOW_MATCH_DISTANCE_BY_SENSOR = {
    "Sentinel-2": 40.0,
    "Landsat": 60.0,
}

# QA_PIXELの影響が非常に大きいLandsat候補について、前後観測で
# 支持される範囲から大きく外れた部分だけを追跡形状から除く。
# 通常の面積変化を消さないよう、影響率と残存率の両方が厳しい
# 条件を満たした候補に限って適用する。
ENABLE_LANDSAT_TEMPORAL_GEOMETRY_CLIP = True
LANDSAT_TEMPORAL_GEOMETRY_CLIP_MIN_QA_RATIO = 0.80
LANDSAT_TEMPORAL_GEOMETRY_CLIP_BUFFER_M = 60.0
LANDSAT_TEMPORAL_GEOMETRY_CLIP_MAX_RETAINED_RATIO = 0.50
LANDSAT_TEMPORAL_GEOMETRY_CLIP_MIN_AREA_M2 = (
    LANDSAT_CONTINUATION_AREA_M2
)

# AOI内で観測可能とみなす最低有効画素率。
# これ未満のSentinel-2観測は、水域0ではなく欠損として扱う。
SENTINEL_MIN_AOI_VALID_PIXEL_RATIO = 0.50

# Landsat水域候補のうちQA確認対象画素がこの割合を超える場合、
# 観測をusableではなくpartialとして扱う。候補自体は時系列確認へ
# 渡し、QA_PIXELによる氷河・積雪域の過剰除外を避ける。
LANDSAT_USABLE_MAX_AOI_WATER_QA_REVIEW_PERCENT = 50.0


# ------------------------------------------------------------
# 5. 永続湖IDの時系列追跡
# ------------------------------------------------------------

# 過去の湖形状を追跡候補として保持する最大日数
MULTISENSOR_TRACKING_MAX_GAP_DAYS = 60

# 重なり、IoU、面積比、距離による同一湖判定条件
MULTISENSOR_TRACKING_MIN_OVERLAP_SMALLER_RATIO = 0.20
MULTISENSOR_TRACKING_MIN_IOU = 0.05
MULTISENSOR_TRACKING_MIN_AREA_RATIO = 0.20
MULTISENSOR_TRACKING_DISTANCE_BY_SENSOR = {
    "Sentinel-2": 40.0,
    "Landsat": 60.0,
}

# 形状が重ならず距離だけで既存IDを継続する場合に許容する、
# 重心移動距離の倍率。センサー別距離閾値にこの倍率を掛ける。
# 小断片だけが近い別の湖へIDが乗り移ることを防ぐ。
MULTISENSOR_TRACKING_DISTANCE_ONLY_MAX_CENTROID_MULTIPLIER = 1.50

# 過去湖の大部分を含む一方、面積が急激に増えて通常の面積比条件を
# 満たさない候補は、その場で新規IDにしない。次の観測でも同じ
# 拡大形状が確認された場合だけ、元の永続IDの継続として採用する。
MULTISENSOR_LARGE_CHANGE_MIN_REFERENCE_OVERLAP_RATIO = 0.80
MULTISENSOR_LARGE_CHANGE_MAX_AREA_RATIO = 0.20
MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MAX_GAP_DAYS = 30
MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MIN_OVERLAP_SMALLER_RATIO = 0.50
MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MIN_AREA_RATIO = 0.50

# 観測間隔が長い間にも湖が成長し続けた場合の確認条件。
# 通常条件を一律に緩和せず、後続観測でも拡大形状が維持された
# 場合だけ元の永続IDへ遡って採用する。
MULTISENSOR_LARGE_CHANGE_LONG_GAP_MIN_DAYS = 10
MULTISENSOR_LARGE_CHANGE_LONG_GAP_MIN_REFERENCE_OVERLAP_RATIO = 0.50
MULTISENSOR_LARGE_CHANGE_LONG_GAP_CONFIRM_MIN_PENDING_COVERAGE_RATIO = 0.80
MULTISENSOR_LARGE_CHANGE_LONG_GAP_CONFIRM_MIN_AREA_RATIO = 0.20

# 合流・分離関係として記録する最低交差面積
MULTISENSOR_RELATION_MIN_INTERSECTION_AREA_M2 = 100.0


# ------------------------------------------------------------
# 6. 前後観測による部分欠損補完
# ------------------------------------------------------------

# 補完の前後参照として使用できる最大日数
MULTISENSOR_FILL_MAX_REFERENCE_GAP_DAYS = 30

# 前後形状同士に要求する最低IoU
MULTISENSOR_FILL_SAME_SENSOR_MIN_REFERENCE_IOU = 0.50
MULTISENSOR_FILL_CROSS_SENSOR_MIN_REFERENCE_IOU = 0.35

# 対象日の形状が安定水域と重なる最低割合
# 現在形状と前後日に共通する安定水域のうち、面積が小さい方を
# 分母にした重なり率。現在形状が一時的に分断・拡大していても、
# 共通水域との位置的一致が十分なら補完候補として扱う。
MULTISENSOR_FILL_MIN_CURRENT_STABLE_OVERLAP_SMALLER_RATIO = 0.50

# これより小さい補完部分は追加しない。
MULTISENSOR_FILL_MIN_AREA_BY_SENSOR = {
    "Sentinel-2": 100.0,
    "Landsat": 900.0,
}

# 補完後面積のうち、補完部分が占めてよい最大割合
MULTISENSOR_FILL_MAX_RATIO = 0.50

# 関連湖ファミリーの補完面積をIDへ配分する格子サイズ
MULTISENSOR_RELATED_FAMILY_ALLOCATION_CELL_M = 10.0


# ------------------------------------------------------------
# 7. 欠損・変化率の記録
# ------------------------------------------------------------

# Sentinel-2の0 m2を欠損と判定するときに、
# 前後の正面積観測を探す最大日数
SENTINEL_INTERIOR_ZERO_MAX_NEIGHBOR_GAP_DAYS = 20.0

# これより長い観測間隔にはlong_gap=Trueを記録する。
MULTISENSOR_LONG_GAP_DAYS = 10.0


# ------------------------------------------------------------
# 8. Landsat観測の品質判定
# ------------------------------------------------------------

# MTLのシーン雲量はAOI外も含むため、現在は棄却条件には使わない。
# 値自体は監査用と品質スコア用にscene_catalog.csvへ残す。
LANDSAT_USE_SCENE_CLOUD_COVER_FOR_REJECTION = False
LANDSAT_USABLE_MAX_CLOUD_PERCENT = 5.0
LANDSAT_PARTIAL_MAX_CLOUD_PERCENT = 15.0

# 同日のSentinel-2面積に対するLandsat面積比
LANDSAT_USABLE_MIN_AREA_RATIO = 0.50
LANDSAT_USABLE_MAX_AREA_RATIO = 2.00
LANDSAT_REJECT_MIN_AREA_RATIO = 0.25
LANDSAT_REJECT_MAX_AREA_RATIO = 4.00

# Landsatで有効な面積観測とする最低水域画素数
LANDSAT_MIN_USABLE_PIXEL_COUNT = 4

# この画素数以下なら面積観測として棄却する。
LANDSAT_REJECT_PIXEL_COUNT = 1
