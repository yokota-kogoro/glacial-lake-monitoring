from __future__ import annotations

import re
import shutil
import subprocess
import warnings
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation, binary_propagation
from xml.etree import ElementTree as ET

from glacial_lake.water_preprocessing_common import (
    align_continuous_raster_to_reference as common_align_continuous_raster,
    align_dem_mosaic_to_reference as common_align_dem,
    calculate_ndwi as common_calculate_ndwi,
    create_aoi_mask as common_create_aoi_mask,
    polygonize_water_mask as common_polygonize_water_mask,
    remove_existing_shapefile as common_remove_existing_shapefile,
    save_categorical_raster as common_save_categorical_raster,
    save_float_raster as common_save_float_raster,
    save_mask_raster as common_save_mask_raster,
)


# ============================================================
# 設定
# ============================================================

# このPythonファイルが保存されているフォルダ
SCRIPT_DIR = Path(__file__).resolve().parent

# Sentinel-2の解凍済みSAFE製品を置くフォルダ
INPUT_ROOT = SCRIPT_DIR / "data" / "sentinel"

# DEMの入力フォルダ
DEM_ROOT = SCRIPT_DIR / "data" / "dem"

# シーン別ラスター・Shapefileは診断用として保存する。
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "diagnostics" / "sentinel"

# 日常確認と統合処理に使う集計CSVはresultsへ保存する。
SUMMARY_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "results" / "summaries"

# JP2をRasterioで直接開けない環境で使用するgdal_translate。
# Noneの場合はPATH、QGIS、OSGeo4Wの順に自動探索する。
GDAL_TRANSLATE_PATH: Path | None = None

# NDWIの閾値
DEFAULT_NDWI_THRESHOLD = 0.45

# 日付ごとに閾値を変えたい場合
THRESHOLDS_BY_DATE: dict[str, float] = {
    # "2025-05-17": 0.08,
    # "2025-05-29": 0.12,
}

# DEMの標高データの閾値
MIN_ELEVATION_M = 3000.0

# NDWI計算で使用するB03+B08反射率の最小値
MIN_REFLECTANCE_SUM = 0.02

# SCLが水域（クラス6）と判定した暗い画素について、BOA補正後の
# B08がわずかに負になったことだけが原因でNDWIを計算できない場合に救済する。
# 通常のNDWI判定とSCLの雲除外条件は変更しない。
ENABLE_SCL_DARK_WATER_RESCUE = True
SCL_WATER_CLASS = 6

# SCL水域を救済するときに必要な、0へ丸めた後のB03反射率の下限。
# 通常のNDWI分母下限とは分離し、非常に暗い湖面だけに使用する。
SCL_WATER_RESCUE_MIN_GREEN_REFLECTANCE = 0.001

# BOA補正後にB03とB08の両方が負となりNDWIを定義できない場合でも、
# SCLが水域（クラス6）で、同日のNDWI水域に近接する非常に暗い画素は
# 水域マスクへ直接追加する。NDWI GeoTIFFには人工的な値を代入しない。
ENABLE_SCL_CONNECTED_DARK_WATER_RESCUE = True
SCL_CONNECTED_DARK_WATER_RESCUE_MAX_DISTANCE_M = 40.0

# 支持されたSCL水域成分に直接接する非常に暗いクラス5画素は、
# 湖岸の誤分類とみなして水域マスクへ直接追加する。
SCL_CONNECTED_DARK_WATER_EDGE_CLASSES = {
    5,
}
SCL_CONNECTED_DARK_WATER_EDGE_MAX_DISTANCE_M = 10.0

# SCLクラス2（暗い地物・地形影）と11（雪氷）は無条件では水域化せず、
# 同じ日の有効な水域から指定距離以内にあるNDWI無効画素だけを救済する。
ENABLE_SCL_NEIGHBOR_WATER_RESCUE = True
SCL_NEIGHBOR_RESCUE_CLASSES = {
    2,
    11,
}
SCL_NEIGHBOR_RESCUE_MAX_DISTANCE_M = 40.0

SCL_HARD_EXCLUDED_CLASSES = {
    0,   # No Data
    1,   # Saturated or defective
    8,   # Medium probability cloud
    9,   # High probability cloud
    10,  # Thin cirrus
}

# 雲影クラス3を最終的に除外するか
EXCLUDE_CLOUD_SHADOW = False

SCL_REVIEW_CLASSES = {
    3,   # Cloud shadow
}

SCL_NODATA_VALUE = 255

# 解析範囲指定
# 画像全体：None
# 特定範囲：GeoJSONまたはShapefileで作成したAOIを指定する
# 
# 例：AOI_PATH = SCRIPT_DIR / "data" / "aoi" / "analysis_area.geojson"
AOI_PATH: Path | None = None

# 削除する最小面積の設定（平方メートル）
MIN_POLYGON_AREA_M2 = 0.0

# 検出された水域内で残すポリゴンの設定
# True：最も大きなポリゴンを残す
KEEP_LARGEST_ONLY = False

# ポリゴン化するときの画素の接続方法
# 8：斜めに接する画素も同領域
# 4：上下左右に接する画素だけを同領域
PIXEL_CONNECTIVITY = 8

# DEMで使用するGeoTIFF拡張子
RASTER_EXTENSIONS = {".tif", ".tiff"}

# Sentinel-2製品内から直接読み込める拡張子
# 解凍済みSAFE製品では各バンドがJP2で保存されている。
SENTINEL_RASTER_EXTENSIONS = RASTER_EXTENSIONS | {".jp2"}


# ============================================================
# 設定値の確認
# ============================================================

def validate_settings() -> None:
    if not -1.0 <= DEFAULT_NDWI_THRESHOLD <= 1.0:
        raise ValueError(
            "DEFAULT_NDWI_THRESHOLDは-1.0から1.0の範囲で指定してください。"
        )
    
    for scene_date, threshold in THRESHOLDS_BY_DATE.items():
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                f"{scene_date}の閾値が不適切です: {threshold}"
            )

    if MIN_ELEVATION_M < -500.0:
        raise ValueError(
            "MIN_ELEVATION_Mの値を確認してください。"
        )  
    
    if MIN_POLYGON_AREA_M2 < 0:
        raise ValueError(
            "MIN_POLYGON_AREA_M2は0以上にしてください。"
        )
    
    if PIXEL_CONNECTIVITY not in {4, 8}:
        raise ValueError(
            "PIXEL_CONNECTIVITYは4または8にしてください。"
        )

    valid_scl_classes = set(range(12))

    invalid_scl_classes = (
        set(SCL_HARD_EXCLUDED_CLASSES)
        - valid_scl_classes
    )

    if invalid_scl_classes:
        raise ValueError(
            f"存在しないSCLクラスが設定されています: {sorted(invalid_scl_classes)}"
        )

    if not 0 <= SCL_NODATA_VALUE <= 255:
        raise ValueError(
            "SCL_NODATA_VALUEは0～255で指定してください。"
        )

    if SCL_WATER_CLASS not in valid_scl_classes:
        raise ValueError(
            "SCL_WATER_CLASSは0～11で指定してください。"
        )

    if SCL_WATER_RESCUE_MIN_GREEN_REFLECTANCE <= 0.0:
        raise ValueError(
            "SCL_WATER_RESCUE_MIN_GREEN_REFLECTANCEは0より大きくしてください。"
        )

    if SCL_CONNECTED_DARK_WATER_RESCUE_MAX_DISTANCE_M < 0.0:
        raise ValueError(
            "SCL_CONNECTED_DARK_WATER_RESCUE_MAX_DISTANCE_Mは0以上にしてください。"
        )

    if SCL_CONNECTED_DARK_WATER_EDGE_MAX_DISTANCE_M < 0.0:
        raise ValueError(
            "SCL_CONNECTED_DARK_WATER_EDGE_MAX_DISTANCE_Mは0以上にしてください。"
        )

    invalid_edge_classes = (
        set(SCL_CONNECTED_DARK_WATER_EDGE_CLASSES)
        - set(range(12))
    )
    if invalid_edge_classes:
        raise ValueError(
            "SCL_CONNECTED_DARK_WATER_EDGE_CLASSESに存在しないクラスがあります: "
            f"{sorted(invalid_edge_classes)}"
        )

    invalid_neighbor_classes = (
        set(SCL_NEIGHBOR_RESCUE_CLASSES)
        - valid_scl_classes
    )
    if invalid_neighbor_classes:
        raise ValueError(
            "SCL_NEIGHBOR_RESCUE_CLASSESに存在しないクラスがあります: "
            f"{sorted(invalid_neighbor_classes)}"
        )

    if SCL_WATER_CLASS in SCL_NEIGHBOR_RESCUE_CLASSES:
        raise ValueError(
            "SCL_WATER_CLASSを近傍救済クラスへ重複指定しないでください。"
        )

    if SCL_NEIGHBOR_RESCUE_MAX_DISTANCE_M < 0.0:
        raise ValueError(
            "SCL_NEIGHBOR_RESCUE_MAX_DISTANCE_Mは0以上にしてください。"
        )


# ============================================================
# SAFE製品内JP2の読み込み準備
# ============================================================

def path_for_windows_api(path: Path) -> str:
    """Windowsの260文字制限を避けるための絶対パス文字列を返す。"""

    absolute_path = str(path.absolute())

    if os.name != "nt" or absolute_path.startswith("\\\\?\\"):
        return absolute_path

    if absolute_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute_path[2:]

    return "\\\\?\\" + absolute_path

def find_gdal_translate() -> Path:
    """JP2対応のgdal_translate実行ファイルを取得する。"""

    candidates: list[Path] = []

    if GDAL_TRANSLATE_PATH is not None:
        candidates.append(Path(GDAL_TRANSLATE_PATH))

    path_command = shutil.which("gdal_translate")

    if path_command:
        candidates.append(Path(path_command))

    program_files = Path(r"C:\Program Files")

    if program_files.is_dir():
        qgis_candidates = sorted(
            program_files.glob("QGIS*/bin/gdal_translate.exe"),
            reverse=True,
        )
        candidates.extend(qgis_candidates)

    candidates.append(
        Path(r"C:\OSGeo4W\bin\gdal_translate.exe")
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "JP2をGeoTIFFへ自動変換するgdal_translateが見つかりません。"
        "QGISまたはOSGeo4Wのインストール先を確認し、"
        "GDAL_TRANSLATE_PATHへ指定してください。"
    )


def convert_jp2_to_cached_geotiff(
    jp2_path: Path,
    cache_dir: Path,
) -> Path:
    """入力JP2を変更せず、解析用GeoTIFFを出力側へ作成する。"""

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tif_path = cache_dir / f"{jp2_path.stem}.tif"
    temporary_path = cache_dir / f"{jp2_path.stem}.__converting__.tif"
    staged_jp2_path = cache_dir / f"{jp2_path.stem}.__source__.jp2"

    if tif_path.is_file():
        cache_is_current = (
            tif_path.stat().st_mtime
            >= Path(path_for_windows_api(jp2_path)).stat().st_mtime
        )

        if cache_is_current:
            try:
                with rasterio.open(tif_path) as cached_src:
                    if (
                        cached_src.width > 0
                        and cached_src.height > 0
                        and cached_src.count > 0
                        and cached_src.crs is not None
                    ):
                        return tif_path
            except rasterio.errors.RasterioError:
                pass

        tif_path.unlink()

    if temporary_path.exists():
        temporary_path.unlink()

    if staged_jp2_path.exists():
        staged_jp2_path.unlink()

    gdal_translate_path = find_gdal_translate()

    try:
        shutil.copyfile(
            path_for_windows_api(jp2_path),
            staged_jp2_path,
        )
    except OSError as error:
        raise RuntimeError(
            "長いパスのJP2を変換用一時フォルダへコピーできませんでした: "
            f"input={jp2_path}, staging={staged_jp2_path}, "
            f"message={error}"
        ) from error

    command = [
        str(gdal_translate_path),
        "-of",
        "GTiff",
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=IF_SAFER",
        str(staged_jp2_path),
        str(temporary_path),
    ]

    creation_flags = (
        subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, "CREATE_NO_WINDOW")
        else 0
    )

    try:
        completed_process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )
    finally:
        if staged_jp2_path.exists():
            staged_jp2_path.unlink()

    if completed_process.returncode != 0:
        if temporary_path.exists():
            temporary_path.unlink()

        error_text = (
            completed_process.stderr.strip()
            or completed_process.stdout.strip()
            or "詳細メッセージなし"
        )

        raise RuntimeError(
            "JP2からGeoTIFFへの自動変換に失敗しました: "
            f"input={jp2_path}, gdal_translate={gdal_translate_path}, "
            f"message={error_text}"
        )

    try:
        with rasterio.open(temporary_path) as converted_src:
            if (
                converted_src.width <= 0
                or converted_src.height <= 0
                or converted_src.count <= 0
                or converted_src.crs is None
            ):
                raise RuntimeError(
                    f"変換後GeoTIFFの構造が不正です: {temporary_path}"
                )
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    temporary_path.replace(tif_path)

    print(f"JP2を解析用GeoTIFFへ変換: {jp2_path.name}")
    print(f"JP2変換キャッシュ: {tif_path}")

    return tif_path


def prepare_sentinel_raster(
    source_path: Path,
    cache_dir: Path,
) -> Path:
    """GeoTIFFはそのまま、JP2はキャッシュ変換して返す。"""

    if source_path.suffix.lower() == ".jp2":
        return convert_jp2_to_cached_geotiff(
            jp2_path=source_path,
            cache_dir=cache_dir,
        )

    return source_path
    

# ============================================================
# Sentinel-2のファイル探索
# ============================================================

def extract_sentinel_component_key(
    file_path: Path,
    component_name: str,
) -> str | None:
    """Sentinel-2標準ファイル名からバンド共通部分を取得する。"""

    match = re.fullmatch(
        rf"(?P<scene_key>.+)_{re.escape(component_name)}",
        file_path.stem,
        re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group("scene_key")


def index_sentinel_components(
    file_paths: list[Path],
    component_name: str,
    product_dir: Path,
) -> dict[str, Path]:
    """タイル・観測日時単位のキーで製品内ファイルを整理する。"""

    indexed_paths: dict[str, Path] = {}

    for file_path in file_paths:
        scene_key = extract_sentinel_component_key(
            file_path=file_path,
            component_name=component_name,
        )

        if scene_key is None:
            continue

        if scene_key in indexed_paths:
            raise ValueError(
                f"{component_name}候補が同じシーンキーで複数あります: "
                f"product={product_dir}, scene_key={scene_key}, "
                f"files={[str(indexed_paths[scene_key]), str(file_path)]}"
            )

        indexed_paths[scene_key] = file_path

    return indexed_paths

# input_data以下を再帰的に探索
def find_scenes(
    input_root: Path,
) -> list[dict[str, Path | str]]:
    
    if not input_root.exists():
        raise FileNotFoundError(
            f"入力フォルダが見つかりません: {input_root}"
        )
    
    scenes: list[dict[str, Path | str]] = []

    # --------------------------------------------------------
    # 1. 解凍済みSentinel-2 L2A製品を探索
    # --------------------------------------------------------

    product_metadata_paths = sorted(
        metadata_path
        for metadata_path in input_root.rglob("MTD_MSIL2A.xml")
        if (metadata_path.parent / "GRANULE").is_dir()
    )

    for metadata_path in product_metadata_paths:
        product_dir = metadata_path.parent

        product_rasters = sorted(
            file_path
            for file_path in product_dir.rglob("*")
            if (
                file_path.suffix.lower()
                in SENTINEL_RASTER_EXTENSIONS
            )
        )

        b03_index = index_sentinel_components(
            file_paths=product_rasters,
            component_name="B03_10m",
            product_dir=product_dir,
        )
        b08_index = index_sentinel_components(
            file_paths=product_rasters,
            component_name="B08_10m",
            product_dir=product_dir,
        )
        scl_index = index_sentinel_components(
            file_paths=product_rasters,
            component_name="SCL_20m",
            product_dir=product_dir,
        )

        if not b03_index:
            raise FileNotFoundError(
                f"製品内にB03 10mがありません: {product_dir}"
            )

        for scene_key, b03_path in sorted(b03_index.items()):
            b08_path = b08_index.get(scene_key)
            scl_path = scl_index.get(scene_key)

            missing_components: list[str] = []

            if b08_path is None:
                missing_components.append("B08_10m")

            if scl_path is None:
                missing_components.append("SCL_20m")

            if missing_components:
                raise FileNotFoundError(
                    "解凍済みSentinel-2製品の構成が不足しています: "
                    f"product={product_dir}, scene_key={scene_key}, "
                    f"missing={missing_components}"
                )

            scenes.append(
                {
                    "scene_dir": product_dir,
                    "scene_name": scene_key,
                    "b03": b03_path,
                    "b08": b08_path,
                    "scl": scl_path,
                    "metadata": metadata_path,
                    "input_layout": "unpacked_product",
                }
            )

    if not scenes:
        raise FileNotFoundError(
            f"{input_root}以下に解凍済みSentinel-2 L2A製品が見つかりません。"
            "MTD_MSIL2A.xmlとGRANULEフォルダを含む製品構成を確認してください。"
        )
    
    return scenes


def read_sentinel2_l2a_metadata(
    metadata_path: Path,
    expected_scene_timestamp: str,
) -> dict[str, str | float]:

    try:
        root = ET.parse(metadata_path).getroot()
    except ET.ParseError as error:
        raise ValueError(
            f"XMLを解析できません: {metadata_path}"
        ) from error
    except OSError as error:
        raise OSError(
            f"XMLを読み込めません: {metadata_path}"
        )from error

    def local_name(tag: str) -> str:
        return tag.rsplit("}", maxsplit=1)[-1]

    def get_required_text(element_name: str) -> str:
        for element in root.iter():
            if (
                local_name(element.tag) == element_name
                and element.text is not None
            ):
                value = element.text.strip()

                if value:
                    return value

        raise ValueError(
            f"{element_name}がXMLにありません: {metadata_path}"
        )

    product_uri = get_required_text("PRODUCT_URI")
    product_start_time = get_required_text("PRODUCT_START_TIME")
    processing_baseline = get_required_text("PROCESSING_BASELINE")

    try:
        quantification_value = float(
            get_required_text(
                "BOA_QUANTIFICATION_VALUE"
            )
        )
    except ValueError as error:
        raise ValueError(
            "BOA_QUANTIFICATION_VALUEを数値へ"
            f"変換できません: {metadata_path}"
        ) from error

    if quantification_value <= 0:
        raise ValueError(
            "BOA_QUANTIFICATION_VALUEは0より"
            f"大きい必要があります: {quantification_value}"
        )

    offsets_by_band_id: dict[str, float] = {}

    for element in root.iter():
        if local_name(element.tag) != "BOA_ADD_OFFSET":
            continue

        band_id = element.attrib.get("band_id")

        if band_id is None or element.text is None:
            continue

        try:
            offsets_by_band_id[band_id] = float(
                element.text.strip()
            )
        except ValueError as error:
            raise ValueError(
                f"BOA_ADD_OFFSETを数値へ変換できません: {metadata_path}"
            ) from error

    required_band_ids = {
        "B03": "2",
        "B08": "7",
    }

    missing_bands = [
        band_name
        for band_name, band_id
        in required_band_ids.items()
        if band_id not in offsets_by_band_id
    ]

    if missing_bands:
        raise ValueError(
            "必要なBOA_ADD_OFFSETがありません。"
            f"対象バンド: {missing_bands}"
            f"XML: {metadata_path}"
        )

    if expected_scene_timestamp not in product_uri:
        raise ValueError(
            "バンド画像とXMLの観測日時が一致しません。"
            f"画像: {expected_scene_timestamp}, "
            f"PRODUCT_URI: {product_uri}"
        )

    return{
        "product_uri": product_uri,
        "product_start_time": product_start_time,
        "processing_baseline": processing_baseline,
        "quantification_value": quantification_value,
        "b03_boa_add_offset": offsets_by_band_id["2"],
        "b08_boa_add_offset": offsets_by_band_id["7"]
    }


# ============================================================
# DEMファイルの探索
# ============================================================

def find_dem_files(
    dem_root: Path,
) -> list[Path]:
    
    if not dem_root.exists():
        raise FileNotFoundError(
            f"DEMフォルダが見つかりません: {dem_root}"
        )
    
    dem_files = sorted(
        file_path
        for file_path in dem_root.rglob("*")
        if(
            file_path.is_file()
            and file_path.suffix.lower()
            in RASTER_EXTENSIONS
        )
    )

    if not dem_files:
        raise FileNotFoundError(
            f"{dem_root}以下にDEMのGeoTIFFが見つかりません。"
        )
    
    return dem_files


# ============================================================
# 日付取得
# ============================================================

def extract_scene_date(
    scene_dir: Path,
    band_file: Path,
) -> str:

    # 標準SAFE製品名とバンド名の両方から観測日を確認する。
    date_sources = (
        scene_dir.name,
        band_file.name,
    )

    for source_text in date_sources:
        compact_date_match = re.search(
            r"(?<!\d)"
            r"(20\d{2})"
            r"(\d{2})"
            r"(\d{2})"
            r"(?!\d)",
            source_text,
        )

        if compact_date_match:
            year, month, day = compact_date_match.groups()

            return f"{year}-{month}-{day}"

        dashed_date_match = re.search(
            r"(?<!\d)"
            r"(20\d{2})-"
            r"(\d{2})-"
            r"(\d{2})"
            r"(?!\d)",
            source_text,
        )

        if dashed_date_match:
            year, month, day = dashed_date_match.groups()

            return f"{year}-{month}-{day}"
    
    # 日付を取得できなかった場合
    return scene_dir.name

def get_threshold(
    scene_date: str,
) -> float:
    
    return float(
        THRESHOLDS_BY_DATE.get(
            scene_date,
            DEFAULT_NDWI_THRESHOLD,
        )
    )


# ============================================================
# Sentinel-2バンドの読み込み
# ============================================================

def read_band_on_reference_grid(
    band_path: Path,
    reference_profile: dict,
    reference_transform,
    reference_crs,
) -> np.ndarray:
    return common_align_continuous_raster(
        raster_path=band_path,
        reference_profile=reference_profile,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
    )


def read_scl_on_referece_grid(
    scl_path: Path,
    reference_profile: dict,
    reference_transform,
    reference_crs,
) -> np.ndarray:

    output_shape = (
        reference_profile["height"],
        reference_profile["width"],
    )

    aligned_scl = np.full(
        output_shape,
        SCL_NODATA_VALUE,
        dtype=np.uint8,
    )

    with rasterio.open(scl_path) as src:
        if src.crs is None:
            raise ValueError(
                f"SCLにCRSがありません: {scl_path}"
            )

        source = (
            src.read(1, masked=True)
            .filled(SCL_NODATA_VALUE)
            .astype(np.uint8)
        )

        same_grid = (
            src.width == reference_profile["width"]
            and src.height == reference_profile["height"]
            and src.transform == reference_transform
            and src.crs == reference_crs
        )

        if same_grid:
            return source

        print(
            f"SCLをB03の10mグリッドへ変換します: {scl_path.name}"
        )

        reproject(
            source=source,
            destination=aligned_scl,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=SCL_NODATA_VALUE,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=SCL_NODATA_VALUE,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )

    return aligned_scl


# ============================================================
# DEMをB03のグリッドへ合わせる
# ============================================================

def read_dem_on_reference_grid(
    dem_files: list[Path],
    reference_profile: dict,
    reference_transform,
    reference_crs,
) -> np.ndarray:
    return common_align_dem(
        dem_files=dem_files,
        reference_profile=reference_profile,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
    )


# ============================================================
# AOI処理
# ============================================================

def create_aoi_mask(
    aoi_path: Path | None,
    raster_crs,
    raster_shape: tuple[int, int],
    raster_transform,
) -> np.ndarray:
    return common_create_aoi_mask(
        aoi_path=aoi_path,
        raster_crs=raster_crs,
        raster_shape=raster_shape,
        raster_transform=raster_transform,
    )


def convert_sentinel2_dn_to_boa_reflectance(
        dn: np.ndarray,
        boa_add_offset: float,
        quantification_value: float,
) -> np.ndarray:

    dn = np.asarray(
        dn,
        dtype=np.float32,
    )

    valid_mask = (
        np.isfinite(dn)
        & (dn != 0)
    )

    reflectance = np.full(
        dn.shape,
        np.nan,
        dtype=np.float32,
    )

    reflectance[valid_mask] = (
        dn[valid_mask] + boa_add_offset
    ) / quantification_value

    return reflectance


# ============================================================
# NDWI計算
# ============================================================

def calculate_ndwi(
    b03: np.ndarray,
    b08: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return common_calculate_ndwi(
        green_reflectance=b03,
        nir_reflectance=b08,
        min_reflectance_sum=MIN_REFLECTANCE_SUM,
    )


def rescue_scl_dark_water_ndwi(
    b03: np.ndarray,
    b08: np.ndarray,
    aligned_scl: np.ndarray,
    ndwi: np.ndarray,
    spectral_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SCL水域のうち、負のB08で無効になった暗い水面だけを救済する。"""

    rescued_ndwi = ndwi.copy()
    rescued_valid_mask = spectral_valid_mask.copy()
    rescue_mask = np.zeros(
        spectral_valid_mask.shape,
        dtype=bool,
    )

    if not ENABLE_SCL_DARK_WATER_RESCUE:
        return rescued_ndwi, rescued_valid_mask, rescue_mask

    finite_reflectance_mask = (
        np.isfinite(b03)
        & np.isfinite(b08)
    )
    # 負の反射率を救済計算のときだけ0へ丸める。
    # 通常のNDWIには補正前の配列をそのまま使用する。
    clipped_b03 = np.maximum(b03, 0.0)
    clipped_b08 = np.maximum(b08, 0.0)
    clipped_denominator = clipped_b03 + clipped_b08

    rescue_mask = (
        ~spectral_valid_mask
        & finite_reflectance_mask
        & (aligned_scl == SCL_WATER_CLASS)
        & (b03 > 0.0)
        & (b08 < 0.0)
        & (
            clipped_denominator
            > SCL_WATER_RESCUE_MIN_GREEN_REFLECTANCE
        )
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rescued_values = (
            clipped_b03[rescue_mask]
            - clipped_b08[rescue_mask]
        ) / clipped_denominator[rescue_mask]

    physical_range_mask = (
        np.isfinite(rescued_values)
        & (rescued_values >= -1.0)
        & (rescued_values <= 1.0)
    )

    if not np.all(physical_range_mask):
        rescue_indices = np.flatnonzero(rescue_mask)
        invalid_indices = rescue_indices[~physical_range_mask]
        rescue_mask.flat[invalid_indices] = False
        rescued_values = rescued_values[physical_range_mask]

    rescued_ndwi[rescue_mask] = rescued_values
    rescued_valid_mask[rescue_mask] = True

    return rescued_ndwi, rescued_valid_mask, rescue_mask


def rescue_neighbor_scl_dark_water_ndwi(
    b03: np.ndarray,
    b08: np.ndarray,
    aligned_scl: np.ndarray,
    ndwi: np.ndarray,
    spectral_valid_mask: np.ndarray,
    seed_water_mask: np.ndarray,
    pixel_width_m: float,
    pixel_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """既存水域に近接するSCLクラス2・11のNDWI無効画素を救済する。"""

    rescued_ndwi = ndwi.copy()
    rescued_valid_mask = spectral_valid_mask.copy()
    rescue_mask = np.zeros(
        spectral_valid_mask.shape,
        dtype=bool,
    )

    if not ENABLE_SCL_NEIGHBOR_WATER_RESCUE:
        return rescued_ndwi, rescued_valid_mask, rescue_mask

    if not np.any(seed_water_mask):
        return rescued_ndwi, rescued_valid_mask, rescue_mask

    pixel_width_m = abs(float(pixel_width_m))
    pixel_height_m = abs(float(pixel_height_m))
    if pixel_width_m <= 0.0 or pixel_height_m <= 0.0:
        raise ValueError(
            "近傍救済に使用するピクセルサイズは0より大きくしてください。"
        )

    column_radius = int(
        np.ceil(
            SCL_NEIGHBOR_RESCUE_MAX_DISTANCE_M
            / pixel_width_m
        )
    )
    row_radius = int(
        np.ceil(
            SCL_NEIGHBOR_RESCUE_MAX_DISTANCE_M
            / pixel_height_m
        )
    )
    row_offsets = np.arange(
        -row_radius,
        row_radius + 1,
        dtype=np.float32,
    )[:, np.newaxis]
    column_offsets = np.arange(
        -column_radius,
        column_radius + 1,
        dtype=np.float32,
    )[np.newaxis, :]
    neighborhood_structure = (
        (row_offsets * pixel_height_m) ** 2
        + (column_offsets * pixel_width_m) ** 2
        <= SCL_NEIGHBOR_RESCUE_MAX_DISTANCE_M ** 2
    )

    near_seed_water_mask = binary_dilation(
        seed_water_mask,
        structure=neighborhood_structure,
        iterations=1,
        border_value=0,
    )

    finite_reflectance_mask = (
        np.isfinite(b03)
        & np.isfinite(b08)
    )
    clipped_b03 = np.maximum(b03, 0.0)
    clipped_b08 = np.maximum(b08, 0.0)
    clipped_denominator = clipped_b03 + clipped_b08

    rescue_mask = (
        ~spectral_valid_mask
        & finite_reflectance_mask
        & np.isin(
            aligned_scl,
            list(SCL_NEIGHBOR_RESCUE_CLASSES),
        )
        & near_seed_water_mask
        & (b03 > 0.0)
        & (b08 < 0.0)
        & (
            clipped_denominator
            > SCL_WATER_RESCUE_MIN_GREEN_REFLECTANCE
        )
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rescued_values = (
            clipped_b03[rescue_mask]
            - clipped_b08[rescue_mask]
        ) / clipped_denominator[rescue_mask]

    physical_range_mask = (
        np.isfinite(rescued_values)
        & (rescued_values >= -1.0)
        & (rescued_values <= 1.0)
    )

    if not np.all(physical_range_mask):
        rescue_indices = np.flatnonzero(rescue_mask)
        invalid_indices = rescue_indices[
            ~physical_range_mask
        ]
        rescue_mask.flat[invalid_indices] = False
        rescued_values = rescued_values[
            physical_range_mask
        ]

    rescued_ndwi[rescue_mask] = rescued_values
    rescued_valid_mask[rescue_mask] = True

    return rescued_ndwi, rescued_valid_mask, rescue_mask


def identify_connected_scl_dark_water_rescue(
    b03: np.ndarray,
    b08: np.ndarray,
    aligned_scl: np.ndarray,
    spectral_valid_mask: np.ndarray,
    seed_water_mask: np.ndarray,
    pixel_width_m: float,
    pixel_height_m: float,
) -> np.ndarray:
    """NDWIを定義できないSCL水域を、既存水域への近接条件で救済する。

    BOA補正後にB03とB08が非常に暗くなり、通常のNDWI計算にも
    SCLクラス6向けNDWI救済にも入らない画素が対象となる。
    NDWI値は生成せず、最終水域マスクに追加する画素だけを返す。
    """

    rescue_mask = np.zeros(
        spectral_valid_mask.shape,
        dtype=bool,
    )

    if not ENABLE_SCL_CONNECTED_DARK_WATER_RESCUE:
        return rescue_mask

    if not np.any(seed_water_mask):
        return rescue_mask

    pixel_width_m = abs(float(pixel_width_m))
    pixel_height_m = abs(float(pixel_height_m))
    if pixel_width_m <= 0.0 or pixel_height_m <= 0.0:
        raise ValueError(
            "SCLクラス6救済に使用するピクセルサイズは0より大きくしてください。"
        )

    column_radius = int(
        np.ceil(
            SCL_CONNECTED_DARK_WATER_RESCUE_MAX_DISTANCE_M
            / pixel_width_m
        )
    )
    row_radius = int(
        np.ceil(
            SCL_CONNECTED_DARK_WATER_RESCUE_MAX_DISTANCE_M
            / pixel_height_m
        )
    )
    row_offsets = np.arange(
        -row_radius,
        row_radius + 1,
        dtype=np.float32,
    )[:, np.newaxis]
    column_offsets = np.arange(
        -column_radius,
        column_radius + 1,
        dtype=np.float32,
    )[np.newaxis, :]
    neighborhood_structure = (
        (row_offsets * pixel_height_m) ** 2
        + (column_offsets * pixel_width_m) ** 2
        <= SCL_CONNECTED_DARK_WATER_RESCUE_MAX_DISTANCE_M ** 2
    )

    near_seed_water_mask = binary_dilation(
        seed_water_mask,
        structure=neighborhood_structure,
        iterations=1,
        border_value=0,
    )

    finite_reflectance_mask = (
        np.isfinite(b03)
        & np.isfinite(b08)
    )

    dark_scl_water_candidate_mask = (
        ~spectral_valid_mask
        & finite_reflectance_mask
        & (aligned_scl == SCL_WATER_CLASS)
        & (b08 < 0.0)
        & (b03 <= SCL_WATER_RESCUE_MIN_GREEN_REFLECTANCE)
    )

    # 近接画素だけを一度救済して終わるのではなく、確実なNDWI水域の
    # 近くにあるSCL=6の連結成分全体を支持された水域成分とする。
    # これにより、大きなSCL=6領域の外縁だけが残る状態を防ぐ。
    scl_water_component_mask = (
        aligned_scl == SCL_WATER_CLASS
    )
    component_markers = (
        scl_water_component_mask
        & near_seed_water_mask
    )
    supported_scl_water_component_mask = binary_propagation(
        component_markers,
        structure=np.ones((3, 3), dtype=bool),
        mask=scl_water_component_mask,
        border_value=0,
    )

    rescue_mask = (
        dark_scl_water_candidate_mask
        & supported_scl_water_component_mask
    )

    # SCL=6の支持された成分に直接接するクラス5のうち、B03とB08が
    # ともに負となった非常に暗い画素だけを湖岸として限定的に救済する。
    edge_column_radius = int(
        np.ceil(
            SCL_CONNECTED_DARK_WATER_EDGE_MAX_DISTANCE_M
            / pixel_width_m
        )
    )
    edge_row_radius = int(
        np.ceil(
            SCL_CONNECTED_DARK_WATER_EDGE_MAX_DISTANCE_M
            / pixel_height_m
        )
    )
    edge_row_offsets = np.arange(
        -edge_row_radius,
        edge_row_radius + 1,
        dtype=np.float32,
    )[:, np.newaxis]
    edge_column_offsets = np.arange(
        -edge_column_radius,
        edge_column_radius + 1,
        dtype=np.float32,
    )[np.newaxis, :]
    edge_structure = (
        (edge_row_offsets * pixel_height_m) ** 2
        + (edge_column_offsets * pixel_width_m) ** 2
        <= SCL_CONNECTED_DARK_WATER_EDGE_MAX_DISTANCE_M ** 2
    )
    near_supported_scl_water_mask = binary_dilation(
        supported_scl_water_component_mask,
        structure=edge_structure,
        iterations=1,
        border_value=0,
    )
    edge_rescue_mask = (
        ~spectral_valid_mask
        & finite_reflectance_mask
        & np.isin(
            aligned_scl,
            list(SCL_CONNECTED_DARK_WATER_EDGE_CLASSES),
        )
        & near_supported_scl_water_mask
        & (b03 <= 0.0)
        & (b08 < 0.0)
    )

    rescue_mask |= edge_rescue_mask

    return rescue_mask


# ============================================================
# float型GeoTIFF出力
# ============================================================

def save_float_raster(
    output_path: Path,
    array: np.ndarray,
    reference_profile: dict,
    nodata_value: float = -9999.0,
) -> None:
    common_save_float_raster(
        output_path=output_path,
        array=array,
        reference_profile=reference_profile,
        nodata_value=nodata_value,
    )

def save_mask_raster(
    output_path: Path,
    target_mask: np.ndarray,
    valid_mask:np.ndarray,
    reference_profile: dict,
) -> None:
    common_save_mask_raster(
        output_path=output_path,
        target_mask=target_mask,
        valid_mask=valid_mask,
        reference_profile=reference_profile,
    )


def save_categorical_raster(
        output_path: Path,
        array: np.ndarray,
        reference_profile: dict,
        nodata_value: int = SCL_NODATA_VALUE,
) -> None:
    common_save_categorical_raster(
        output_path=output_path,
        array=array,
        reference_profile=reference_profile,
        nodata_value=nodata_value,
    )


# ============================================================
# 面積計算用座標系
# ============================================================

def save_scene_shapefile(
    gdf: gpd.GeoDataframe,
    vector_dir: Path,
    scene_name: str,
) -> Path | None:
    
    if gdf.empty:
        print(
            "水域ポリゴンがないため、Shapefileは出力しません。"
        )
        return None
    
    shapefile_gdf = gdf.rename(
        columns={
            "threshold": "ndwi_thr",
            "polygon_id": "poly_id",
        }
    )

    shapefile_path = (
        vector_dir
        / f"{scene_name}_lakes.shp"
    )

    common_remove_existing_shapefile(shapefile_path)

    shapefile_gdf.to_file(
        shapefile_path,
        driver="ESRI Shapefile",
        encoding="UTF-8",
        index=False,
    )

    print(f"Shapefile出力: {shapefile_path}")

    return shapefile_path


# ============================================================
# 1シーンの処理
# ============================================================

def process_scene(
        scene: dict[str, Path | str],
        raster_root: Path,
        vector_root: Path,
        dem_files: list[Path],
) -> dict:
    
    scene_dir = Path(scene["scene_dir"])
    source_b03_path = Path(scene["b03"])
    source_b08_path = Path(scene["b08"])
    source_scl_path = Path(scene["scl"])
    metadata_path = Path(scene["metadata"])

    b03_timestamp_match = re.search(
        r"(\d{8}T\d{6})",
        source_b03_path.name,
    )
    b08_timestamp_match = re.search(
        r"(\d{8}T\d{6})",
        source_b08_path.name,
    )
    scl_timestamp_match = re.search(
        r"(\d{8}T\d{6})",
        source_scl_path.name,
    )

    if b03_timestamp_match is None:
        raise ValueError(
            f"B03ファイル名から観測日時を取得できません: {source_b03_path}"
        )

    if b08_timestamp_match is None:
        raise ValueError(
            f"B08ファイル名から観測日時を取得できません: {source_b08_path}"
        )

    if scl_timestamp_match is None:
        raise ValueError(
            f"SCLファイル名から観測日時を取得できません: {source_scl_path}"
        )

    b03_timestamp = b03_timestamp_match.group(1)
    b08_timestamp = b08_timestamp_match.group(1)
    scl_timestamp = scl_timestamp_match.group(1)

    if b03_timestamp != b08_timestamp:
        raise ValueError(
            "B03とB08の観測日時が一致しません。"
            f"B03: {b03_timestamp}, B08: {b08_timestamp}"
        )

    if scl_timestamp != b03_timestamp:
        raise ValueError(
            "B03とSCLの観測日時が一致しません。"
            f"B03: {b03_timestamp}, SCL: {scl_timestamp}"
        )

    metadata = read_sentinel2_l2a_metadata(
        metadata_path=metadata_path,
        expected_scene_timestamp=b03_timestamp,
    )

    quantification_value = float(
        metadata["quantification_value"]
    )
    b03_boa_add_offset = float(
        metadata["b03_boa_add_offset"]
    )
    b08_boa_add_offset = float(
        metadata["b08_boa_add_offset"]
    )

    scene_name = str(
        scene.get(
            "scene_name",
            scene_dir.name,
        )
    )
    input_layout = str(
        scene.get(
            "input_layout",
            "unknown",
        )
    )

    input_cache_dir = (
        raster_root.parent
        / "input_cache"
        / scene_name
    )

    b03_path = prepare_sentinel_raster(
        source_path=source_b03_path,
        cache_dir=input_cache_dir,
    )
    b08_path = prepare_sentinel_raster(
        source_path=source_b08_path,
        cache_dir=input_cache_dir,
    )
    scl_path = prepare_sentinel_raster(
        source_path=source_scl_path,
        cache_dir=input_cache_dir,
    )

    raster_dir = raster_root / scene_name
    vector_dir = vector_root / scene_name

    raster_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene_date = extract_scene_date(
        scene_dir,
        source_b03_path,
    )

    threshold = get_threshold(scene_date)

    print()
    print("=" * 70)
    print(f"処理中: {scene_name}")
    print(f"日付: {scene_date}")
    print(f"入力形式: {input_layout}")
    print(f"入力製品/シーンフォルダ: {scene_dir}")
    print(f"B03入力: {source_b03_path.name}")
    print(f"B08入力: {source_b08_path.name}")
    print(f"NDWI閾値: {threshold}")
    print(f"メタデータ: {metadata_path.name}")
    print(f"Processing Baseline: {metadata['processing_baseline']}")
    print(f"BOA量子化係数: {quantification_value}")
    print(f"B03 BOAオフセット: {b03_boa_add_offset}")
    print(f"B08 BOAオフセット: {b08_boa_add_offset}")
    print(f"SCL入力: {source_scl_path.name}")
    print(f"SCL除外クラス: {sorted(SCL_HARD_EXCLUDED_CLASSES)}")

    print(f"最低標高: {MIN_ELEVATION_M:.0f}m")
    print(f"Raster出力先: {raster_dir}")
    print(f"Shapefile出力先: {vector_dir}")

    # B03読み込み
    
    with rasterio.open(b03_path) as b03_src:

        if b03_src.crs is None:
            raise ValueError(
                f"B03にCRSが設定されていません: {b03_path}"
            )
        
        reference_profile = b03_src.profile.copy()
        reference_transform = b03_src.transform
        reference_crs = b03_src.crs

        b03_dn = (
            b03_src.read(1, masked=True)
            .astype(np.float32)
            .filled(np.nan)
        )

    # B08読み込み

    b08_dn = read_band_on_reference_grid(
        band_path=b08_path,
        reference_profile=reference_profile,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
    )

    b03 = convert_sentinel2_dn_to_boa_reflectance(
        dn=b03_dn,
        boa_add_offset=b03_boa_add_offset,
        quantification_value=quantification_value,
    )

    b08 = convert_sentinel2_dn_to_boa_reflectance(
        dn=b08_dn,
        boa_add_offset=b08_boa_add_offset,
        quantification_value=quantification_value,
    )

    aligned_scl = read_scl_on_referece_grid(
        scl_path=scl_path,
        reference_profile=reference_profile,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
    )

    scl_valid_mask = (
        aligned_scl != SCL_NODATA_VALUE
    )

    scl_hard_exclusion_mask = (
        scl_valid_mask
        & np.isin(
            aligned_scl,
            list(SCL_HARD_EXCLUDED_CLASSES),
        )
    )

    scl_shadow_mask = (
        scl_valid_mask
        & np.isin(
            aligned_scl,
            list(SCL_REVIEW_CLASSES),
        )
    )

    if EXCLUDE_CLOUD_SHADOW:
        scl_exclusion_mask = (
            scl_hard_exclusion_mask
            | scl_shadow_mask
        )
    else:
        scl_exclusion_mask = (
            scl_hard_exclusion_mask.copy()
        )

    scl_allowed_mask = (
        scl_valid_mask
        & ~scl_exclusion_mask
    )

    # DEMをB03グリッドへ変換

    aligned_dem = read_dem_on_reference_grid(
        dem_files=dem_files,
        reference_profile=reference_profile,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
    )

    dem_valid_mask = np.isfinite(
        aligned_dem
    )

    high_elevation_mask = (
        dem_valid_mask & (aligned_dem >= MIN_ELEVATION_M)
    )

    print(f"{MIN_ELEVATION_M:.0f}m以上の画素数: {int(high_elevation_mask.sum())}")

    # NDWI計算

    ndwi, spectral_valid_mask = calculate_ndwi(
        b03,
        b08,
    )

    #AOIマスク

    aoi_mask = create_aoi_mask(
        aoi_path=AOI_PATH,
        raster_crs=reference_crs,
        raster_shape=ndwi.shape,
        raster_transform=reference_transform,
    )

    # SCLによる救済を行う前の純粋なNDWI判定をQC用に保持する。
    pre_scl_valid_mask = (
        spectral_valid_mask
        & aoi_mask
        & high_elevation_mask
    )

    water_mask_before_scl = (
        pre_scl_valid_mask
        & (ndwi >= threshold)
    )

    (
        ndwi,
        spectral_valid_mask,
        scl_dark_water_rescue_mask,
    ) = rescue_scl_dark_water_ndwi(
        b03=b03,
        b08=b08,
        aligned_scl=aligned_scl,
        ndwi=ndwi,
        spectral_valid_mask=spectral_valid_mask,
    )

    scl_dark_water_rescue_mask &= (
        aoi_mask
        & high_elevation_mask
        & scl_allowed_mask
    )

    # 通常のNDWI水域とSCLクラス6の救済水域を、クラス2・11の
    # 近傍救済に使用する種水域とする。
    seed_water_mask = (
        spectral_valid_mask
        & aoi_mask
        & high_elevation_mask
        & scl_allowed_mask
        & (ndwi >= threshold)
    )

    # B03とB08がともに非常に暗くNDWIを定義できないSCLクラス6を、
    # 同日の確実な水域から指定距離以内に限って水域マスク用に救済する。
    # NDWI配列へ人工的な値は設定しない。
    scl_connected_dark_water_rescue_mask = (
        identify_connected_scl_dark_water_rescue(
            b03=b03,
            b08=b08,
            aligned_scl=aligned_scl,
            spectral_valid_mask=spectral_valid_mask,
            seed_water_mask=seed_water_mask,
            pixel_width_m=reference_transform.a,
            pixel_height_m=reference_transform.e,
        )
    )
    scl_connected_dark_water_rescue_mask &= (
        aoi_mask
        & high_elevation_mask
        & scl_allowed_mask
    )

    neighbor_seed_water_mask = (
        seed_water_mask
        | scl_connected_dark_water_rescue_mask
    )

    (
        ndwi,
        spectral_valid_mask,
        scl_neighbor_water_rescue_mask,
    ) = rescue_neighbor_scl_dark_water_ndwi(
        b03=b03,
        b08=b08,
        aligned_scl=aligned_scl,
        ndwi=ndwi,
        spectral_valid_mask=spectral_valid_mask,
        seed_water_mask=neighbor_seed_water_mask,
        pixel_width_m=reference_transform.a,
        pixel_height_m=reference_transform.e,
    )

    scl_neighbor_water_rescue_mask &= (
        aoi_mask
        & high_elevation_mask
        & scl_allowed_mask
    )

    scl_water_class_rescued_pixel_count = int(
        scl_dark_water_rescue_mask.sum()
    )
    scl_neighbor_rescued_pixel_count = int(
        scl_neighbor_water_rescue_mask.sum()
    )
    scl_connected_dark_water_rescued_pixel_count = int(
        scl_connected_dark_water_rescue_mask.sum()
    )
    scl_dark_water_rescued_pixel_count = (
        scl_water_class_rescued_pixel_count
        + scl_neighbor_rescued_pixel_count
        + scl_connected_dark_water_rescued_pixel_count
    )
    print(
        "SCLクラス6として救済した暗い画素数: "
        f"{scl_water_class_rescued_pixel_count}"
    )
    print(
        "既存水域の近傍で救済したクラス2・11画素数: "
        f"{scl_neighbor_rescued_pixel_count}"
    )
    print(
        "既存水域の近傍で直接救済したSCLクラス6画素数: "
        f"{scl_connected_dark_water_rescued_pixel_count}"
    )
    print(
        "SCL条件による救済画素数の合計: "
        f"{scl_dark_water_rescued_pixel_count}"
    )

    base_valid_mask = (
        (
            spectral_valid_mask
            & aoi_mask
            & high_elevation_mask
        )
        | scl_connected_dark_water_rescue_mask
    )

    valid_mask = (
        base_valid_mask
        & scl_allowed_mask
    )

    water_removed_by_scl = (
        water_mask_before_scl
        & ~valid_mask
    )

    ndwi[~valid_mask] = np.nan

    original_water_mask = (
        (
            valid_mask
            & (ndwi >= threshold)
        )
        | scl_connected_dark_water_rescue_mask
    )

    non_shadow_water_mask = (
        original_water_mask
        & ~scl_shadow_mask
    )

    shadow_water_candidate_mask = (
        original_water_mask
        & scl_shadow_mask
    )

    original_water_pixel_count = int(
        original_water_mask.sum()
    )

    # ポリゴン化・面積計算

    water_gdf = common_polygonize_water_mask(
        water_mask=original_water_mask,
        transform=reference_transform,
        raster_crs=reference_crs,
        scene_name=scene_name,
        scene_date=scene_date,
        threshold=threshold,
        review_mask=scl_shadow_mask,
        min_polygon_area_m2=MIN_POLYGON_AREA_M2,
        keep_largest_only=KEEP_LARGEST_ONLY,
        connectivity=PIXEL_CONNECTIVITY,
    )

    if not water_gdf.empty:

        final_water_mask = rasterize(
            [
                (geometry, 1)
                for geometry in water_gdf.geometry
            ],
            out_shape=original_water_mask.shape,
            transform=reference_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        ).astype(bool)

    else:

        final_water_mask = np.zeros_like(
            original_water_mask,
            dtype=bool,
        )

    # 整列済みDEM出力

    aligned_dem_output_path = (
        raster_dir / f"{scene_name}_DEM_aligned.tif"
    )

    save_float_raster(
        output_path=aligned_dem_output_path,
        array=aligned_dem,
        reference_profile=reference_profile,
    )

    # 標高3000m以上マスク出力

    elevation_mask_output_path = (
        raster_dir
        / (f"{scene_name}_elevation_{int(MIN_ELEVATION_M)}m_mask.tif")
    )

    save_mask_raster(
        output_path=elevation_mask_output_path,
        target_mask=high_elevation_mask,
        valid_mask=dem_valid_mask,
        reference_profile=reference_profile,
    )

    aligned_scl_output_path = (
        raster_dir
        / f"{scene_name}_SCL_aligned_10m.tif"
    )

    save_categorical_raster(
        output_path=aligned_scl_output_path,
        array=aligned_scl,
        reference_profile=reference_profile,
    )

    scl_exclusion_output_path = (
        raster_dir
        / f"{scene_name}_SCL_exclusion_mask.tif"
    )

    save_mask_raster(
        output_path=scl_exclusion_output_path,
        target_mask=scl_exclusion_mask,
        valid_mask=scl_valid_mask,
        reference_profile=reference_profile,
    )

    before_scl_output_path = (
        raster_dir
        / f"{scene_name}_water_mask_before_SCL.tif"
    )

    save_mask_raster(
        output_path=before_scl_output_path,
        target_mask=water_mask_before_scl,
        valid_mask=pre_scl_valid_mask,
        reference_profile=reference_profile,
    )

    removed_by_scl_output_path = (
        raster_dir
        / f"{scene_name}_water_removed_by_SCL.tif"
    )

    save_mask_raster(
        output_path=removed_by_scl_output_path,
        target_mask=water_removed_by_scl,
        valid_mask=pre_scl_valid_mask,
        reference_profile=reference_profile,
    )

    shadow_water_output_path = (
        raster_dir
        / f"{scene_name}_shadow_water_candidate.tif"
    )

    save_mask_raster(
        output_path=shadow_water_output_path,
        target_mask=shadow_water_candidate_mask,
        valid_mask=base_valid_mask,
        reference_profile=reference_profile,
    )

    scl_shadow_output_path = (
        raster_dir
        / f"{scene_name}_SCL_shadow_mask.tif"
    )

    save_mask_raster(
        output_path=scl_shadow_output_path,
        target_mask=scl_shadow_mask,
        valid_mask=scl_valid_mask,
        reference_profile=reference_profile,
    )

    # NDWI GeoTIFF出力

    ndwi_output_path = (
        raster_dir
        / f"{scene_name}_NDWI.tif"
    )

    save_float_raster(
        output_path=ndwi_output_path,
        array=ndwi,
        reference_profile=reference_profile,
    )

    # 二値マスクGeoTIFF出力

    mask_output_path = (
        raster_dir
        / f"{scene_name}_water_mask.tif"
    )

    save_mask_raster(
        output_path=mask_output_path,
        target_mask=final_water_mask,
        valid_mask=valid_mask,
        reference_profile=reference_profile,
    )

    # Shapefile出力

    shapefile_path = save_scene_shapefile(
        gdf=water_gdf,
        vector_dir=vector_dir,
        scene_name=scene_name,
    )

    # 面積計算

    if not water_gdf.empty:

        total_area_m2 = float(
            water_gdf["area_m2"].sum()
        )

        largest_area_m2 = float(
            water_gdf["area_m2"].max()
        )

    else:

        total_area_m2 = 0.0
        largest_area_m2 = 0.0

    final_water_pixel_count = int(
        final_water_mask.sum()
    )

    summary = {
        "scene": scene_name,
        "date": scene_date,
        "input_layout": input_layout,
        "threshold": threshold,
        "min_elevation_m": MIN_ELEVATION_M,
        "dem_file_count": len(dem_files),
        "dem_valid_pixel_count": int(dem_valid_mask.sum()),
        "high_elevation_pixel_count": int(high_elevation_mask.sum()),
        "polygon_count": int(len(water_gdf)),
        "total_area_m2": total_area_m2,
        "total_area_km2": total_area_m2 / 1_000_000.0,
        "largest_area_m2": largest_area_m2,
        "largest_area_km2": largest_area_m2 / 1_000_000.0,
        "original_water_pixel_count": original_water_pixel_count,
        "final_water_pixel_count": final_water_pixel_count,
        "scl_dark_water_rescue_enabled": ENABLE_SCL_DARK_WATER_RESCUE,
        "scl_dark_water_rescued_pixel_count": (
            scl_dark_water_rescued_pixel_count
        ),
        "scl_water_rescue_min_green_reflectance": (
            SCL_WATER_RESCUE_MIN_GREEN_REFLECTANCE
        ),
        "scl_water_class_rescued_pixel_count": (
            scl_water_class_rescued_pixel_count
        ),
        "scl_connected_dark_water_rescue_enabled": (
            ENABLE_SCL_CONNECTED_DARK_WATER_RESCUE
        ),
        "scl_connected_dark_water_rescue_max_distance_m": (
            SCL_CONNECTED_DARK_WATER_RESCUE_MAX_DISTANCE_M
        ),
        "scl_connected_dark_water_rescued_pixel_count": (
            scl_connected_dark_water_rescued_pixel_count
        ),
        "scl_neighbor_water_rescue_enabled": (
            ENABLE_SCL_NEIGHBOR_WATER_RESCUE
        ),
        "scl_neighbor_rescue_classes": ",".join(
            str(value)
            for value in sorted(
                SCL_NEIGHBOR_RESCUE_CLASSES
            )
        ),
        "scl_neighbor_rescue_max_distance_m": (
            SCL_NEIGHBOR_RESCUE_MAX_DISTANCE_M
        ),
        "scl_neighbor_rescued_pixel_count": (
            scl_neighbor_rescued_pixel_count
        ),
        "b03_file": str(b03_path),
        "b08_file": str(b08_path),
        "scl_file": str(scl_path),
        "b03_source_file": str(source_b03_path),
        "b08_source_file": str(source_b08_path),
        "scl_source_file": str(source_scl_path),
        "aligned_dem_file": str(aligned_dem_output_path),
        "elevation_mask_file": str(elevation_mask_output_path),
        "ndwi_file": str(ndwi_output_path),
        "mask_file": str(mask_output_path),
        "shapefile": (
            str(shapefile_path)
            if shapefile_path is not None
            else ""
        ),
        "metadata_file": str(metadata_path),
        "product_uri": str(metadata["product_uri"]),
        "product_start_time": str(metadata["product_start_time"]),
        "processing_baseline": str(metadata["processing_baseline"]),
        "boa_quantification_value": quantification_value,
        "b03_boa_add_offset": b03_boa_add_offset,
        "b08_boa_add_offset": b08_boa_add_offset,
        "scl_file": str(scl_path),
        "scl_excluede_classes": ",".join(
            str(value)
            for value in sorted(
                SCL_HARD_EXCLUDED_CLASSES
            )
        ),
        "scl_valid_pixel_count": int(scl_valid_mask.sum()),
        "scl_excluded_pixel_count": int(scl_exclusion_mask.sum()),
        "water_pixel_count_before_scl": int(water_mask_before_scl.sum()),
        "water_pixel_count_removed_by_scl": int(water_removed_by_scl.sum()),
        "aligned_scl_file": str(aligned_scl_output_path),
        "scl_exclusion_mask_file": str(scl_exclusion_output_path),
        "water_mask_before_scl_file": str(before_scl_output_path),
        "water_removed_by_scl_file": str(removed_by_scl_output_path),
        "exclude_cloud_shadow": EXCLUDE_CLOUD_SHADOW,
        "shadow_water_pixel_count": int(shadow_water_candidate_mask.sum()),
        "shadow_water_area_m2": float(
            shadow_water_candidate_mask.sum()
            * abs(
                reference_transform.a
                * reference_transform.e
            )
        ),
        "shadow_water_candidate_file": str(shadow_water_output_path),
        "scl_shadow_mask_file": str(scl_shadow_output_path),
        "error": "",
    }

    print(f"検出ポリゴン数: {summary['polygon_count']}")

    print(
        f"水域面積: {summary['total_area_m2']:.2f} m2"
    )

    print(
        f"水域面積: {summary['total_area_km2']:.6f} km2"
    )

    return summary


# ============================================================
# メイン処理
# ============================================================

def main() -> None:

    validate_settings()

    # 出力フォルダ作成

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raster_root = OUTPUT_ROOT / "rasters"
    vecter_root = OUTPUT_ROOT / "shapefiles"

    raster_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    vecter_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # DEMファイル自動探索

    dem_files = find_dem_files(DEM_ROOT)

    # Sentinel-2シーン自動探索

    scenes = find_scenes(INPUT_ROOT)

    scenes = sorted(
        scenes,
        key=lambda scene: extract_scene_date(
            Path(scene["scene_dir"]),
            Path(scene["b03"]),
        ),
    )

    print("=" *  70)
    print("NDWIによる氷河湖面積抽出")
    print("=" * 70)
    print(f"入力フォルダ: {INPUT_ROOT}")
    print(f"DEMフォルダ: {DEM_ROOT}")
    print(f"検出したDEM数: {len(dem_files)}")
    print(f"最低標高: {MIN_ELEVATION_M:.0f}m")
    print(f"出力フォルダ: {OUTPUT_ROOT}")
    print(f"検出したシーン数: {len(scenes)}")

    for dem_file in dem_files:
        print(f"DEM: {dem_file.name}")

    summary_rows: list[dict] = []

    # 全シーンを処理

    for scene in scenes:

        try:

            summary = process_scene(
                scene=scene,
                raster_root=raster_root,
                vector_root=vecter_root,
                dem_files=dem_files
            )

            summary_rows.append(summary)

        except Exception as error:

            scene_dir = Path(scene["scene_dir"])
            b03_path = Path(scene["b03"])
            scene_name = str(
                scene.get(
                    "scene_name",
                    scene_dir.name,
                )
            )
            input_layout = str(
                scene.get(
                    "input_layout",
                    "unknown",
                )
            )

            error_messege = str(error)

            warnings.warn(
                f"{scene_name}の処理に失敗しました: {error_messege}"
            )

            summary_rows.append(
                {
                    "scene": scene_name,
                    "date": extract_scene_date(
                        scene_dir,
                        b03_path,
                    ),
                    "input_layout": input_layout,
                    "threshold": get_threshold(
                        extract_scene_date(
                            scene_dir,
                            b03_path,
                        )
                    ),
                    "min_elevation_m": MIN_ELEVATION_M,
                    "dem_file_count": len(dem_files),
                    "dem_valid_pixel_count": np.nan,
                    "high_elevation_pixel_count": np.nan,
                    "polygon_count": 0,
                    "total_area_m2": np.nan,
                    "total_area_km2": np.nan,
                    "largest_area_m2": np.nan,
                    "largest_area_km2": np.nan,
                    "original_water_pixel_count": np.nan,
                    "final_water_pixel_count": np.nan,
                    "b03_file": str(scene["b03"]),
                    "b08_file": str(scene["b08"]),
                    "aligned_dem_file": "",
                    "elevation_mask_file": "",
                    "ndwi_file": "",
                    "mask_file": "",
                    "shapefile": "",
                    "error": error_messege,
                }
            )

    # 面積集計CSV出力

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:

        summary_df = summary_df.sort_values(
            by=["date", "scene"],
        ).reset_index(drop=True)

    summary_csv_path = (
        SUMMARY_OUTPUT_DIR
        / "sentinel_area_summary.csv"
    )

    summary_df.to_csv(
        summary_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 完了表示

    successful_count = int(
        (summary_df["error"] == "").sum()
    )

    failed_count = int(
        (summary_df["error"] != "").sum()
    )

    print()
    print("=" * 70)
    print("処理が完了しました。")
    print(f"成功したシーン数: {successful_count}")
    print(f"失敗したシーン数: {failed_count}")
    print(f"面積集計CSV: {summary_csv_path}")
    print(f"Raster出力先: {raster_root}")
    print(f"shapefile出力先: {vecter_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
