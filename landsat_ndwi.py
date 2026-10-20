from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.warp import reproject

from glacial_lake.water_preprocessing_common import (
    align_continuous_raster_to_reference,
    align_dem_mosaic_to_reference,
    calculate_ndwi,
    create_aoi_mask,
    polygonize_water_mask,
    save_float_raster,
    save_mask_raster,
    save_water_polygons,
)


# ============================================================
# 設定
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_ROOT = SCRIPT_DIR / "data" / "landsat"
DEM_ROOT = SCRIPT_DIR / "data" / "dem"
# シーン別ラスター・Shapefileは診断用として保存する。
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "diagnostics" / "landsat"
# 日常確認と統合処理に使う集計CSVはresultsへ保存する。
SUMMARY_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "results" / "summaries"

DEFAULT_NDWI_THRESHOLD = 0.26
THRESHOLDS_BY_DATE: dict[str, float] = {}

MIN_ELEVATION_M = 3000.0
MIN_POLYGON_AREA_M2 = 900.0
PIXEL_CONNECTIVITY = 8
KEEP_LARGEST_ONLY = False

# Landsat Collection 2 QA_PIXELを水域判定に使用する。
USE_QA_PIXEL = True

# 氷河・積雪域ではCFMaskが湖面や明るい地表をCloud / Cirrus /
# Dilated Cloudへ誤分類する場合がある。この3区分は水域画素を
# 直ちにNoDataへせず、ポリゴン属性shd_*へ「時系列確認対象」
# として記録する。Trueへ変更した区分だけ即時除外される。
EXCLUDE_QA_DILATED_CLOUD = False
EXCLUDE_QA_CIRRUS = False
EXCLUDE_QA_CLOUD = False

# 雲影は最初は除外せず、水域候補として時系列検証へ渡す。
EXCLUDE_QA_CLOUD_SHADOW = False

# 氷河湖や浮氷を誤って除外しないよう、雪氷は除外しない。
EXCLUDE_QA_SNOW = False

# NDWIに使うBand 3・Band 5の放射飽和を除外する。
USE_QA_RADSAT = True

# 山体によって地表が観測できていない画素を除外する。
EXCLUDE_QA_TERRAIN_OCCLUSION = True

# QA_RADSATを再投影したとき、入力範囲外を示す値。
QA_RADSAT_OUTSIDE_VALUE = 65535

# Landsatで従来使用していた分母判定を維持する。
DENOMINATOR_EPSILON = 1.0e-6

# QA_PIXELが水域（bit 7）と判定した暗い画素について、地表面反射率
# 補正後のBand 5がわずかに負になったことだけが原因でNDWIを計算
# できない場合に救済する。FillやQAの即時除外画素は救済しない。
ENABLE_QA_WATER_DARK_PIXEL_RESCUE = True

# Noneならシーン全体。AOIを指定すると前処理段階で切り出す。
# 例：AOI_PATH = SCRIPT_DIR / "data" / "aoi" / "analysis_area.geojson"
AOI_PATH: Path | None = None

RASTER_EXTENSIONS = {".tif", ".tiff"}


def validate_settings() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Landsat入力フォルダが見つかりません: {INPUT_ROOT}"
        )
    if not DEM_ROOT.exists():
        raise FileNotFoundError(
            f"DEMフォルダが見つかりません: {DEM_ROOT}"
        )
    if not -1.0 <= DEFAULT_NDWI_THRESHOLD <= 1.0:
        raise ValueError(
            "DEFAULT_NDWI_THRESHOLDは-1.0～1.0にしてください。"
        )
    for scene_date, threshold in THRESHOLDS_BY_DATE.items():
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                f"{scene_date}の閾値が不適切です: {threshold}"
            )
    if MIN_POLYGON_AREA_M2 < 0.0:
        raise ValueError(
            "MIN_POLYGON_AREA_M2は0以上にしてください。"
        )
    if PIXEL_CONNECTIVITY not in {4, 8}:
        raise ValueError(
            "PIXEL_CONNECTIVITYは4または8にしてください。"
        )


def find_one_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"ファイルが見つかりません: folder={folder}, pattern={pattern}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            "同じ種類のファイルが複数見つかりました。"
            f" folder={folder}, pattern={pattern}, matches={matches}"
        )
    return matches[0]


def find_optional_file(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    if len(matches) > 1:
        raise RuntimeError(
            "同じ種類の任意ファイルが複数見つかりました。"
            f" folder={folder}, pattern={pattern}, matches={matches}"
        )
    return matches[0] if matches else None


def read_mtl_value(mtl_text: str, key: str) -> str:
    match = re.search(
        rf"^[ \t]*{re.escape(key)}[ \t]*=[ \t]*(.+?)[ \t]*$",
        mtl_text,
        flags=re.MULTILINE,
    )
    if match is None:
        return ""
    return match.group(1).strip().strip('"')


def read_mtl_number(mtl_text: str, key: str) -> float:
    value_text = read_mtl_value(mtl_text, key)
    if value_text == "":
        raise ValueError(
            f"MTL内に値が見つかりません: {key}"
        )
    try:
        return float(value_text)
    except ValueError as error:
        raise ValueError(
            f"MTLの数値を変換できません: {key}={value_text}"
        ) from error


def find_dem_files(dem_root: Path) -> list[Path]:
    dem_files = sorted(
        path
        for path in dem_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in RASTER_EXTENSIONS
    )
    if not dem_files:
        raise FileNotFoundError(
            f"DEMラスターが見つかりません: {dem_root}"
        )
    return dem_files


def find_scenes(input_root: Path) -> list[dict[str, Path | None]]:
    scenes: list[dict[str, Path | None]] = []
    for mtl_path in sorted(input_root.rglob("*_MTL.txt")):
        scene_dir = mtl_path.parent
        green_path = find_one_file(scene_dir, "*_SR_B3.TIF")
        nir_path = find_one_file(scene_dir, "*_SR_B5.TIF")
        qa_pixel_path = find_optional_file(scene_dir, "*_QA_PIXEL.TIF")
        qa_radsat_path = find_optional_file(
            scene_dir,
            "*_QA_RADSAT.TIF",
        )
        scenes.append(
            {
                "scene_dir": scene_dir,
                "green": green_path,
                "nir": nir_path,
                "metadata": mtl_path,
                "qa_pixel": qa_pixel_path,
                "qa_radsat": qa_radsat_path,
            }
        )
    if not scenes:
        raise FileNotFoundError(
            f"Landsatシーンが見つかりません: {input_root}"
        )
    return scenes


def get_scene_metadata(mtl_path: Path) -> dict[str, str | float]:
    mtl_text = mtl_path.read_text(
        encoding="utf-8",
        errors="strict",
    )
    product_id = read_mtl_value(
        mtl_text,
        "LANDSAT_PRODUCT_ID",
    )
    if product_id == "":
        product_id = mtl_path.name.removesuffix("_MTL.txt")

    date_acquired = read_mtl_value(
        mtl_text,
        "DATE_ACQUIRED",
    )
    if date_acquired == "":
        date_match = re.search(r"_(\d{8})_", product_id)
        if date_match is None:
            raise ValueError(
                f"観測日を取得できません: {mtl_path}"
            )
        date_text = date_match.group(1)
        date_acquired = (
            f"{date_text[0:4]}-{date_text[4:6]}-{date_text[6:8]}"
        )

    scene_center_time = read_mtl_value(
        mtl_text,
        "SCENE_CENTER_TIME",
    )
    observation_datetime = (
        f"{date_acquired}T{scene_center_time}"
        if scene_center_time != ""
        else date_acquired
    )
    cloud_cover_text = read_mtl_value(
        mtl_text,
        "CLOUD_COVER",
    )
    cloud_cover = (
        float(cloud_cover_text)
        if cloud_cover_text != ""
        else np.nan
    )

    return {
        "product_id": product_id,
        "date": date_acquired,
        "observation_datetime": observation_datetime,
        "platform": read_mtl_value(
            mtl_text,
            "SPACECRAFT_ID",
        ),
        "processing_level": read_mtl_value(
            mtl_text,
            "PROCESSING_LEVEL",
        ),
        "cloud_cover_scene_percent": cloud_cover,
        "green_multiplier": read_mtl_number(
            mtl_text,
            "REFLECTANCE_MULT_BAND_3",
        ),
        "green_additive_offset": read_mtl_number(
            mtl_text,
            "REFLECTANCE_ADD_BAND_3",
        ),
        "nir_multiplier": read_mtl_number(
            mtl_text,
            "REFLECTANCE_MULT_BAND_5",
        ),
        "nir_additive_offset": read_mtl_number(
            mtl_text,
            "REFLECTANCE_ADD_BAND_5",
        ),
    }


def get_threshold(scene_date: str) -> float:
    return float(
        THRESHOLDS_BY_DATE.get(
            scene_date,
            DEFAULT_NDWI_THRESHOLD,
        )
    )


def convert_landsat_dn_to_reflectance(
    dn: np.ndarray,
    multiplier: float,
    additive_offset: float,
) -> np.ndarray:
    dn = np.asarray(dn, dtype=np.float32)
    valid_mask = np.isfinite(dn) & (dn != 0)
    reflectance = np.full(dn.shape, np.nan, dtype=np.float32)
    reflectance[valid_mask] = (
        dn[valid_mask] * multiplier + additive_offset
    )
    return reflectance


def align_qa_bitmask_to_reference(
    qa_path: Path,
    reference_profile: dict,
    reference_transform,
    reference_crs,
    outside_value: int,
) -> np.ndarray:

    if not 0 <= outside_value <= 65535:
        raise ValueError(
            "outside_valueは0～65535にしてください。"
            f" value={outside_value}"
        )

    output_shape = (
        int(reference_profile["height"]),
        int(reference_profile["width"]),
    )

    destination = np.full(
        output_shape,
        outside_value,
        dtype=np.uint16,
    )

    with rasterio.open(qa_path) as qa_dataset:
        if qa_dataset.crs is None:
            raise ValueError(
                f"QAラスターにCRSが設定されていません: {qa_path}"
            )

        source = qa_dataset.read(
            1,
            masked=False,
        ).astype(np.uint16)

        same_grid = (
            qa_dataset.width == reference_profile["width"]
            and qa_dataset.height == reference_profile["height"]
            and qa_dataset.transform == reference_transform
            and qa_dataset.crs == reference_crs
        )

        if same_grid:
            return source

        reproject(
            source=source,
            destination=destination,
            src_transform=qa_dataset.transform,
            src_crs=qa_dataset.crs,
            src_nodata=None,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=outside_value,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )

    return destination


def decode_landsat_qa_pixel(
    qa_pixel: np.ndarray,
    exclude_dilated_cloud: bool,
    exclude_cirrus: bool,
    exclude_cloud: bool,
    exclude_cloud_shadow: bool,
    exclude_snow: bool,
) -> dict[str, np.ndarray]:

    qa_pixel = np.asarray(
        qa_pixel,
        dtype=np.uint16,
    )

    def bit_is_set(bit_index: int) -> np.ndarray:
        bit_value = np.uint16(1 << bit_index)
        return (qa_pixel & bit_value) != 0

    fill_mask = bit_is_set(0)
    dilated_cloud_mask = bit_is_set(1)
    cirrus_mask = bit_is_set(2)
    cloud_mask = bit_is_set(3)
    cloud_shadow_mask = bit_is_set(4)
    snow_mask = bit_is_set(5)
    water_mask = bit_is_set(7)

    atmospheric_cloud_mask = (
        dilated_cloud_mask
        | cirrus_mask
        | cloud_mask
    )

    # Fillは実データがないので常に除外する。一方、雲系フラグは
    # 氷河・積雪域での誤分類を考慮し、設定された区分だけ除外する。
    hard_exclusion_mask = fill_mask.copy()

    if exclude_dilated_cloud:
        hard_exclusion_mask |= dilated_cloud_mask

    if exclude_cirrus:
        hard_exclusion_mask |= cirrus_mask

    if exclude_cloud:
        hard_exclusion_mask |= cloud_mask

    if exclude_cloud_shadow:
        hard_exclusion_mask |= cloud_shadow_mask

    if exclude_snow:
        hard_exclusion_mask |= snow_mask

    usable_mask = ~hard_exclusion_mask

    # 即時除外しなかったCloud / Cirrus / Dilated Cloud / Cloud
    # Shadowは、NDWI水域候補に残した上で時系列確認へ渡す。
    quality_review_mask = (
        atmospheric_cloud_mask
        | cloud_shadow_mask
    ) & usable_mask

    return {
        "fill_mask": fill_mask,
        "dilated_cloud_mask": dilated_cloud_mask,
        "cirrus_mask": cirrus_mask,
        "cloud_mask": cloud_mask,
        "cloud_shadow_mask": cloud_shadow_mask,
        "snow_mask": snow_mask,
        "water_mask": water_mask,
        "atmospheric_cloud_mask": atmospheric_cloud_mask,
        "hard_exclusion_mask": hard_exclusion_mask,
        "qa_data_mask": ~fill_mask,
        "usable_mask": usable_mask,
        "quality_review_mask": quality_review_mask,
        # 既存の統合処理との互換性を保つ別名。
        "shadow_review_mask": quality_review_mask,
    }


def rescue_qa_water_dark_pixel_ndwi(
    green_reflectance: np.ndarray,
    nir_reflectance: np.ndarray,
    qa_water_mask: np.ndarray,
    qa_usable_mask: np.ndarray,
    ndwi: np.ndarray,
    spectral_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QA水域のうち、負のBand 5で無効になった暗い水面だけを救済する。"""

    rescued_ndwi = ndwi.copy()
    rescued_valid_mask = spectral_valid_mask.copy()
    rescue_mask = np.zeros(
        spectral_valid_mask.shape,
        dtype=bool,
    )

    if not ENABLE_QA_WATER_DARK_PIXEL_RESCUE:
        return rescued_ndwi, rescued_valid_mask, rescue_mask

    finite_reflectance_mask = (
        np.isfinite(green_reflectance)
        & np.isfinite(nir_reflectance)
    )
    original_denominator = (
        green_reflectance + nir_reflectance
    )

    # 負の反射率を救済計算のときだけ0へ丸める。
    # 通常のNDWI計算に使用した反射率配列は変更しない。
    clipped_green = np.maximum(
        green_reflectance,
        0.0,
    )
    clipped_nir = np.maximum(
        nir_reflectance,
        0.0,
    )
    clipped_denominator = clipped_green + clipped_nir

    rescue_mask = (
        ~spectral_valid_mask
        & finite_reflectance_mask
        & qa_water_mask
        & qa_usable_mask
        & (green_reflectance > 0.0)
        & (nir_reflectance < 0.0)
        & (original_denominator > 0.0)
        & (clipped_denominator > DENOMINATOR_EPSILON)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rescued_values = (
            clipped_green[rescue_mask]
            - clipped_nir[rescue_mask]
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


def decode_landsat_qa_radsat(
    qa_radsat: np.ndarray,
    exclude_terrain_occlusion: bool,
    outside_value: int,
) -> dict[str, np.ndarray]:

    qa_radsat = np.asarray(
        qa_radsat,
        dtype=np.uint16,
    )

    def bit_is_set(bit_index: int) -> np.ndarray:
        bit_value = np.uint16(1 << bit_index)
        return (qa_radsat & bit_value) != 0

    outside_mask = (
        qa_radsat == np.uint16(outside_value)
    )

    # Landsat 8・9 Collection 2 QA_RADSAT
    # bit 2 = Band 3 saturation
    # bit 4 = Band 5 saturation
    # bit 11 = Terrain occlusion
    band3_saturated_mask = bit_is_set(2)
    band5_saturated_mask = bit_is_set(4)
    terrain_occlusion_mask = bit_is_set(11)

    ndwi_band_saturated_mask = (
        band3_saturated_mask
        | band5_saturated_mask
    )
    hard_exclusion_mask = (
        outside_mask
        | ndwi_band_saturated_mask
    )

    if exclude_terrain_occlusion:
        hard_exclusion_mask |= terrain_occlusion_mask

    return {
        "outside_mask": outside_mask,
        "band3_saturated_mask": band3_saturated_mask,
        "band5_saturated_mask": band5_saturated_mask,
        "ndwi_band_saturated_mask": ndwi_band_saturated_mask,
        "terrain_occlusion_mask": terrain_occlusion_mask,
        "hard_exclusion_mask": hard_exclusion_mask,
        "qa_data_mask": ~outside_mask,
        "usable_mask": ~hard_exclusion_mask,
    }


def process_scene(
    scene: dict[str, Path | None],
    dem_files: list[Path],
    raster_root: Path,
    vector_root: Path,
) -> dict:
    scene_dir = Path(scene["scene_dir"])
    green_path = Path(scene["green"])
    nir_path = Path(scene["nir"])
    metadata_path = Path(scene["metadata"])
    qa_pixel_path_value = scene["qa_pixel"]
    qa_pixel_path = (
        Path(qa_pixel_path_value)
        if qa_pixel_path_value is not None
        else None
    )
    qa_radsat_path_value = scene["qa_radsat"]
    qa_radsat_path = (
        Path(qa_radsat_path_value)
        if qa_radsat_path_value is not None
        else None
    )

    if USE_QA_PIXEL and qa_pixel_path is None:
        raise FileNotFoundError(
            f"QA_PIXELが見つかりません: {scene_dir}"
        )
    if USE_QA_RADSAT and qa_radsat_path is None:
        raise FileNotFoundError(
            f"QA_RADSATが見つかりません: {scene_dir}"
        )

    metadata = get_scene_metadata(metadata_path)
    product_id = str(metadata["product_id"])
    scene_date = str(metadata["date"])
    threshold = get_threshold(scene_date)
    date_folder = scene_date.replace("-", "")
    raster_dir = raster_root / date_folder
    vector_dir = vector_root / date_folder
    raster_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print(f"処理中: {product_id}")
    print(f"日付: {scene_date}")
    print(f"B03: {green_path.name}")
    print(f"B05: {nir_path.name}")
    print(f"NDWI閾値: {threshold}")
    print(f"最低標高: {MIN_ELEVATION_M:.0f}m")
    print(f"最小ポリゴン面積: {MIN_POLYGON_AREA_M2:.0f}m2")
    print(f"QA_PIXEL使用: {USE_QA_PIXEL}")
    print(f"QA_RADSAT使用: {USE_QA_RADSAT}")

    with rasterio.open(green_path) as green_dataset:
        if green_dataset.crs is None:
            raise ValueError(
                f"B03にCRSが設定されていません: {green_path}"
            )
        reference_profile = green_dataset.profile.copy()
        reference_transform = green_dataset.transform
        reference_crs = green_dataset.crs
        green_dn = (
            green_dataset.read(1, masked=True)
            .astype(np.float32)
            .filled(np.nan)
        )

    nir_dn = align_continuous_raster_to_reference(
        raster_path=nir_path,
        reference_profile=reference_profile,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
    )
    green_reflectance = convert_landsat_dn_to_reflectance(
        green_dn,
        float(metadata["green_multiplier"]),
        float(metadata["green_additive_offset"]),
    )
    nir_reflectance = convert_landsat_dn_to_reflectance(
        nir_dn,
        float(metadata["nir_multiplier"]),
        float(metadata["nir_additive_offset"]),
    )
    ndwi, spectral_valid_mask = calculate_ndwi(
        green_reflectance=green_reflectance,
        nir_reflectance=nir_reflectance,
        min_reflectance_sum=DENOMINATOR_EPSILON,
        use_absolute_denominator=True,
    )

    empty_qa_mask = np.zeros(
        ndwi.shape,
        dtype=bool,
    )

    if USE_QA_PIXEL:
        aligned_qa_pixel = align_qa_bitmask_to_reference(
            qa_path=qa_pixel_path,
            reference_profile=reference_profile,
            reference_transform=reference_transform,
            reference_crs=reference_crs,
            outside_value=1,
        )

        qa_flags = decode_landsat_qa_pixel(
            qa_pixel=aligned_qa_pixel,
            exclude_dilated_cloud=(
                EXCLUDE_QA_DILATED_CLOUD
            ),
            exclude_cirrus=EXCLUDE_QA_CIRRUS,
            exclude_cloud=EXCLUDE_QA_CLOUD,
            exclude_cloud_shadow=EXCLUDE_QA_CLOUD_SHADOW,
            exclude_snow=EXCLUDE_QA_SNOW,
        )
    else:
        qa_flags = {
            "fill_mask": empty_qa_mask.copy(),
            "dilated_cloud_mask": empty_qa_mask.copy(),
            "cirrus_mask": empty_qa_mask.copy(),
            "cloud_mask": empty_qa_mask.copy(),
            "cloud_shadow_mask": empty_qa_mask.copy(),
            "snow_mask": empty_qa_mask.copy(),
            "water_mask": empty_qa_mask.copy(),
            "atmospheric_cloud_mask": empty_qa_mask.copy(),
            "hard_exclusion_mask": empty_qa_mask.copy(),
            "qa_data_mask": np.ones(
                ndwi.shape,
                dtype=bool,
            ),
            "usable_mask": np.ones(
                ndwi.shape,
                dtype=bool,
            ),
            "quality_review_mask": empty_qa_mask.copy(),
            "shadow_review_mask": empty_qa_mask.copy(),
        }

    if USE_QA_RADSAT:
        aligned_qa_radsat = align_qa_bitmask_to_reference(
            qa_path=qa_radsat_path,
            reference_profile=reference_profile,
            reference_transform=reference_transform,
            reference_crs=reference_crs,
            outside_value=QA_RADSAT_OUTSIDE_VALUE,
        )
        qa_radsat_flags = decode_landsat_qa_radsat(
            qa_radsat=aligned_qa_radsat,
            exclude_terrain_occlusion=(
                EXCLUDE_QA_TERRAIN_OCCLUSION
            ),
            outside_value=QA_RADSAT_OUTSIDE_VALUE,
        )
    else:
        qa_radsat_flags = {
            "outside_mask": empty_qa_mask.copy(),
            "band3_saturated_mask": empty_qa_mask.copy(),
            "band5_saturated_mask": empty_qa_mask.copy(),
            "ndwi_band_saturated_mask": empty_qa_mask.copy(),
            "terrain_occlusion_mask": empty_qa_mask.copy(),
            "hard_exclusion_mask": empty_qa_mask.copy(),
            "qa_data_mask": np.ones(
                ndwi.shape,
                dtype=bool,
            ),
            "usable_mask": np.ones(
                ndwi.shape,
                dtype=bool,
            ),
        }

    (
        ndwi,
        spectral_valid_mask,
        qa_water_dark_rescue_mask,
    ) = rescue_qa_water_dark_pixel_ndwi(
        green_reflectance=green_reflectance,
        nir_reflectance=nir_reflectance,
        qa_water_mask=qa_flags["water_mask"],
        qa_usable_mask=qa_flags["usable_mask"],
        ndwi=ndwi,
        spectral_valid_mask=spectral_valid_mask,
    )

    aligned_dem = align_dem_mosaic_to_reference(
        dem_files=dem_files,
        reference_profile=reference_profile,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
    )
    dem_valid_mask = np.isfinite(aligned_dem)
    high_elevation_mask = (
        dem_valid_mask
        & (aligned_dem >= MIN_ELEVATION_M)
    )
    aoi_mask = create_aoi_mask(
        aoi_path=AOI_PATH,
        raster_crs=reference_crs,
        raster_shape=ndwi.shape,
        raster_transform=reference_transform,
    )
    pre_elevation_valid_mask = (
        spectral_valid_mask
        & aoi_mask
        & qa_flags["usable_mask"]
        & qa_radsat_flags["usable_mask"]
    )

    water_mask_before_elevation = (
        pre_elevation_valid_mask
        & (ndwi >= threshold)
    )

    final_valid_mask = (
        pre_elevation_valid_mask
        & high_elevation_mask
    )

    final_water_before_area_filter = (
        final_valid_mask
        & (ndwi >= threshold)
    )

    qa_water_dark_rescue_mask &= (
        final_valid_mask
        & (ndwi >= threshold)
    )
    qa_water_dark_rescued_pixel_count = int(
        qa_water_dark_rescue_mask.sum()
    )
    print(
        "QA水域として救済した暗い画素数: "
        f"{qa_water_dark_rescued_pixel_count}"
    )

    # QA_PIXELで雲・巻雲・膨張雲・雲影とされたNDWI水域候補。
    # 即時除外せず、ポリゴン属性shd_pxなどへ記録し、
    # multisensor_observations.pyの時系列確認へ渡す。
    qa_water_review_mask = (
        final_water_before_area_filter
        & qa_flags["quality_review_mask"]
    )

    water_gdf = polygonize_water_mask(
        water_mask=final_water_before_area_filter,
        transform=reference_transform,
        raster_crs=reference_crs,
        scene_name=date_folder,
        scene_date=scene_date,
        threshold=threshold,
        review_mask=qa_water_review_mask,
        min_polygon_area_m2=MIN_POLYGON_AREA_M2,
        keep_largest_only=KEEP_LARGEST_ONLY,
        connectivity=PIXEL_CONNECTIVITY,
    )
    if water_gdf.empty:
        final_water_mask = np.zeros_like(
            final_water_before_area_filter,
            dtype=bool,
        )
    else:
        final_water_mask = rasterize(
            [(geometry, 1) for geometry in water_gdf.geometry],
            out_shape=final_water_before_area_filter.shape,
            transform=reference_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        ).astype(bool)

    ndwi_for_output = ndwi.copy()
    ndwi_for_output[~final_valid_mask] = np.nan
    aligned_dem_path = raster_dir / f"{product_id}_DEM_aligned.tif"
    elevation_mask_path = (
        raster_dir
        / f"{product_id}_elevation_{int(MIN_ELEVATION_M)}m_mask.tif"
    )
    ndwi_path = raster_dir / f"{product_id}_NDWI.tif"
    before_elevation_mask_path = (
        raster_dir
        / f"{product_id}_water_mask_before_elevation.tif"
    )
    water_mask_path = raster_dir / f"{product_id}_water_mask.tif"
    polygon_path = vector_dir / f"{product_id}_lakes.shp"
    qa_hard_exclusion_mask_path = (
        raster_dir / f"{product_id}_QA_hard_exclusion_mask.tif"
        if USE_QA_PIXEL
        else None
    )

    qa_cloud_shadow_mask_path = (
        raster_dir / f"{product_id}_QA_cloud_shadow_mask.tif"
        if USE_QA_PIXEL
        else None
    )
    qa_radsat_exclusion_mask_path = (
        raster_dir / f"{product_id}_QA_RADSAT_exclusion_mask.tif"
        if USE_QA_RADSAT
        else None
    )

    save_float_raster(
        aligned_dem_path,
        aligned_dem,
        reference_profile,
    )
    save_mask_raster(
        elevation_mask_path,
        high_elevation_mask,
        dem_valid_mask,
        reference_profile,
    )
    save_float_raster(
        ndwi_path,
        ndwi_for_output,
        reference_profile,
    )
    save_mask_raster(
        before_elevation_mask_path,
        water_mask_before_elevation,
        pre_elevation_valid_mask,
        reference_profile,
    )
    save_mask_raster(
        water_mask_path,
        final_water_mask,
        final_valid_mask,
        reference_profile,
    )
    if USE_QA_PIXEL:
        qa_output_valid_mask = (
            qa_flags["qa_data_mask"]
            & aoi_mask
        )

        save_mask_raster(
            qa_hard_exclusion_mask_path,
            qa_flags["hard_exclusion_mask"],
            qa_output_valid_mask,
            reference_profile,
        )

        save_mask_raster(
            qa_cloud_shadow_mask_path,
            qa_flags["cloud_shadow_mask"],
            qa_output_valid_mask,
            reference_profile,
        )
    if USE_QA_RADSAT:
        qa_radsat_output_valid_mask = (
            qa_radsat_flags["qa_data_mask"]
            & aoi_mask
        )
        save_mask_raster(
            qa_radsat_exclusion_mask_path,
            qa_radsat_flags["hard_exclusion_mask"],
            qa_radsat_output_valid_mask,
            reference_profile,
        )
    saved_polygon_path = save_water_polygons(
        water_gdf,
        polygon_path,
    )

    total_area_m2 = (
        float(water_gdf["area_m2"].sum())
        if not water_gdf.empty
        else 0.0
    )
    largest_area_m2 = (
        float(water_gdf["area_m2"].max())
        if not water_gdf.empty
        else 0.0
    )
    qa_evaluation_mask = (
        spectral_valid_mask
        & aoi_mask
        & high_elevation_mask
        & qa_flags["qa_data_mask"]
    )

    qa_evaluation_pixel_count = int(
        qa_evaluation_mask.sum()
    )

    qa_hard_excluded_pixel_count = int(
        (
            qa_flags["hard_exclusion_mask"]
            & qa_evaluation_mask
        ).sum()
    )

    qa_cloud_shadow_pixel_count = int(
        (
            qa_flags["cloud_shadow_mask"]
            & qa_evaluation_mask
        ).sum()
    )

    qa_snow_pixel_count = int(
        (
            qa_flags["snow_mask"]
            & qa_evaluation_mask
        ).sum()
    )

    qa_shadow_water_pixel_count = int(
        (
            final_water_mask
            & qa_flags["cloud_shadow_mask"]
        ).sum()
    )

    qa_review_water_pixel_count = int(
        (
            final_water_mask
            & qa_flags["quality_review_mask"]
        ).sum()
    )

    qa_hard_excluded_percent = (
        qa_hard_excluded_pixel_count
        / qa_evaluation_pixel_count
        * 100.0
        if qa_evaluation_pixel_count > 0
        else np.nan
    )
    qa_radsat_evaluation_mask = (
        spectral_valid_mask
        & aoi_mask
        & high_elevation_mask
        & qa_radsat_flags["qa_data_mask"]
    )
    qa_radsat_evaluation_pixel_count = int(
        qa_radsat_evaluation_mask.sum()
    )
    qa_band3_saturated_pixel_count = int(
        (
            qa_radsat_flags["band3_saturated_mask"]
            & qa_radsat_evaluation_mask
        ).sum()
    )
    qa_band5_saturated_pixel_count = int(
        (
            qa_radsat_flags["band5_saturated_mask"]
            & qa_radsat_evaluation_mask
        ).sum()
    )
    qa_terrain_occlusion_pixel_count = int(
        (
            qa_radsat_flags["terrain_occlusion_mask"]
            & qa_radsat_evaluation_mask
        ).sum()
    )
    qa_radsat_excluded_pixel_count = int(
        (
            qa_radsat_flags["hard_exclusion_mask"]
            & qa_radsat_evaluation_mask
        ).sum()
    )
    qa_radsat_excluded_percent = (
        qa_radsat_excluded_pixel_count
        / qa_radsat_evaluation_pixel_count
        * 100.0
        if qa_radsat_evaluation_pixel_count > 0
        else np.nan
    )
    summary = {
        "scene": date_folder,
        "date": scene_date,
        "observation_datetime": metadata["observation_datetime"],
        "sensor_family": "Landsat",
        "platform": metadata["platform"],
        "product_id": product_id,
        "processing_level": metadata["processing_level"],
        "native_pixel_size_m": abs(reference_transform.a),
        "threshold": threshold,
        "min_elevation_m": MIN_ELEVATION_M,
        "min_polygon_area_m2": MIN_POLYGON_AREA_M2,
        "qa_pixel_used": USE_QA_PIXEL,
        "qa_water_dark_rescue_enabled": (
            ENABLE_QA_WATER_DARK_PIXEL_RESCUE
        ),
        "qa_water_dark_rescued_pixel_count": (
            qa_water_dark_rescued_pixel_count
        ),
        "qa_dilated_cloud_excluded": (
            EXCLUDE_QA_DILATED_CLOUD
        ),
        "qa_cirrus_excluded": EXCLUDE_QA_CIRRUS,
        "qa_cloud_excluded": EXCLUDE_QA_CLOUD,
        "qa_cloud_shadow_excluded": EXCLUDE_QA_CLOUD_SHADOW,
        "qa_snow_excluded": EXCLUDE_QA_SNOW,
        "qa_evaluation_pixel_count": qa_evaluation_pixel_count,
        "qa_hard_excluded_pixel_count": (
            qa_hard_excluded_pixel_count
        ),
        "qa_hard_excluded_percent": (
            qa_hard_excluded_percent
        ),
        "qa_cloud_shadow_pixel_count": (
            qa_cloud_shadow_pixel_count
        ),
        "qa_snow_pixel_count": qa_snow_pixel_count,
        "qa_shadow_water_pixel_count": (
            qa_shadow_water_pixel_count
        ),
        "qa_review_water_pixel_count": (
            qa_review_water_pixel_count
        ),
        "qa_radsat_used": USE_QA_RADSAT,
        "qa_terrain_occlusion_excluded": (
            EXCLUDE_QA_TERRAIN_OCCLUSION
        ),
        "qa_radsat_evaluation_pixel_count": (
            qa_radsat_evaluation_pixel_count
        ),
        "qa_band3_saturated_pixel_count": (
            qa_band3_saturated_pixel_count
        ),
        "qa_band5_saturated_pixel_count": (
            qa_band5_saturated_pixel_count
        ),
        "qa_terrain_occlusion_pixel_count": (
            qa_terrain_occlusion_pixel_count
        ),
        "qa_radsat_excluded_pixel_count": (
            qa_radsat_excluded_pixel_count
        ),
        "qa_radsat_excluded_percent": (
            qa_radsat_excluded_percent
        ),
        "cloud_cover_scene_percent": metadata[
            "cloud_cover_scene_percent"
        ],
        "dem_file_count": len(dem_files),
        "dem_valid_pixel_count": int(dem_valid_mask.sum()),
        "high_elevation_pixel_count": int(high_elevation_mask.sum()),
        "polygon_count": int(len(water_gdf)),
        "total_area_m2": total_area_m2,
        "total_area_km2": total_area_m2 / 1_000_000.0,
        "largest_area_m2": largest_area_m2,
        "largest_area_km2": largest_area_m2 / 1_000_000.0,
        "original_water_pixel_count": int(
            water_mask_before_elevation.sum()
        ),
        "final_water_pixel_count": int(final_water_mask.sum()),
        "b03_file": str(green_path.resolve()),
        "b05_file": str(nir_path.resolve()),
        "metadata_file": str(metadata_path.resolve()),
        "quality_mask_file": (
            str(Path(qa_pixel_path).resolve())
            if qa_pixel_path is not None
            else ""
        ),
        "qa_hard_exclusion_mask_file": (
            str(qa_hard_exclusion_mask_path.resolve())
            if qa_hard_exclusion_mask_path is not None
            else ""
        ),
        "qa_cloud_shadow_mask_file": (
            str(qa_cloud_shadow_mask_path.resolve())
            if qa_cloud_shadow_mask_path is not None
            else ""
        ),
        "qa_radsat_file": (
            str(qa_radsat_path.resolve())
            if qa_radsat_path is not None
            else ""
        ),
        "qa_radsat_exclusion_mask_file": (
            str(qa_radsat_exclusion_mask_path.resolve())
            if qa_radsat_exclusion_mask_path is not None
            else ""
        ),
        "aligned_dem_file": str(aligned_dem_path.resolve()),
        "elevation_mask_file": str(elevation_mask_path.resolve()),
        "ndwi_file": str(ndwi_path.resolve()),
        "water_mask_before_elevation_file": str(
            before_elevation_mask_path.resolve()
        ),
        "mask_file": str(water_mask_path.resolve()),
        "shapefile": (
            str(saved_polygon_path.resolve())
            if saved_polygon_path is not None
            else ""
        ),
        "green_multiplier": metadata["green_multiplier"],
        "green_additive_offset": metadata["green_additive_offset"],
        "nir_multiplier": metadata["nir_multiplier"],
        "nir_additive_offset": metadata["nir_additive_offset"],
        "error": "",
    }

    print(f"検出ポリゴン数: {summary['polygon_count']}")
    print(f"水域面積: {total_area_m2:.2f} m2")
    print(f"水域面積: {total_area_m2 / 1_000_000.0:.6f} km2")
    print(f"NDWI出力: {ndwi_path}")
    print(f"水域マスク出力: {water_mask_path}")
    print(f"水域ポリゴン出力: {saved_polygon_path or 'なし'}")
    if USE_QA_PIXEL:
        print(
            "QAによる除外画素数: "
            f"{qa_hard_excluded_pixel_count}"
        )
        print(
            "QAによる除外率: "
            f"{qa_hard_excluded_percent:.2f}%"
        )
        print(
            "雲影内の最終水域画素数: "
            f"{qa_shadow_water_pixel_count}"
        )
        print(
            "QA時系列確認対象の最終水域画素数: "
            f"{qa_review_water_pixel_count}"
        )
        print(
            "QA除外マスク出力: "
            f"{qa_hard_exclusion_mask_path}"
        )
        print(
            "QA雲影マスク出力: "
            f"{qa_cloud_shadow_mask_path}"
        )
    if USE_QA_RADSAT:
        print(
            "Band 3飽和画素数: "
            f"{qa_band3_saturated_pixel_count}"
        )
        print(
            "Band 5飽和画素数: "
            f"{qa_band5_saturated_pixel_count}"
        )
        print(
            "地形遮蔽画素数: "
            f"{qa_terrain_occlusion_pixel_count}"
        )
        print(
            "QA_RADSAT除外画素数: "
            f"{qa_radsat_excluded_pixel_count}"
        )
        print(
            "QA_RADSAT除外率: "
            f"{qa_radsat_excluded_percent:.4f}%"
        )
        print(
            "QA_RADSAT除外マスク出力: "
            f"{qa_radsat_exclusion_mask_path}"
        )
    return summary


def main() -> None:
    validate_settings()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raster_root = OUTPUT_ROOT / "rasters"
    vector_root = OUTPUT_ROOT / "shapefiles"
    raster_root.mkdir(parents=True, exist_ok=True)
    vector_root.mkdir(parents=True, exist_ok=True)

    dem_files = find_dem_files(DEM_ROOT)
    scenes = find_scenes(INPUT_ROOT)
    scenes = sorted(
        scenes,
        key=lambda scene: str(
            get_scene_metadata(Path(scene["metadata"]))["date"]
        ),
    )

    print("=" * 70)
    print("Landsat NDWIによる氷河湖面積抽出")
    print("=" * 70)
    print(f"入力フォルダ: {INPUT_ROOT}")
    print(f"DEMフォルダ: {DEM_ROOT}")
    print(f"出力フォルダ: {OUTPUT_ROOT}")
    print(f"検出したシーン数: {len(scenes)}")

    summary_rows: list[dict] = []
    for scene in scenes:
        try:
            summary_rows.append(
                process_scene(
                    scene=scene,
                    dem_files=dem_files,
                    raster_root=raster_root,
                    vector_root=vector_root,
                )
            )
        except Exception as error:
            scene_dir = Path(scene["scene_dir"])
            metadata_path = Path(scene["metadata"])
            error_message = str(error)
            warnings.warn(
                f"{scene_dir.name}の処理に失敗しました: {error_message}"
            )
            try:
                metadata = get_scene_metadata(metadata_path)
            except Exception:
                metadata = {
                    "product_id": scene_dir.name,
                    "date": scene_dir.parent.name,
                    "observation_datetime": "",
                    "platform": "",
                    "processing_level": "",
                    "cloud_cover_scene_percent": np.nan,
                }
            summary_rows.append(
                {
                    "scene": str(metadata["date"]).replace("-", ""),
                    "date": metadata["date"],
                    "observation_datetime": metadata[
                        "observation_datetime"
                    ],
                    "sensor_family": "Landsat",
                    "platform": metadata["platform"],
                    "product_id": metadata["product_id"],
                    "processing_level": metadata["processing_level"],
                    "native_pixel_size_m": 30.0,
                    "threshold": get_threshold(str(metadata["date"])),
                    "cloud_cover_scene_percent": metadata[
                        "cloud_cover_scene_percent"
                    ],
                    "metadata_file": str(metadata_path.resolve()),
                    "ndwi_file": "",
                    "mask_file": "",
                    "shapefile": "",
                    "error": error_message,
                }
            )

    summary_dataframe = pd.DataFrame(summary_rows)
    if not summary_dataframe.empty:
        summary_dataframe = (
            summary_dataframe.sort_values(["date", "product_id"])
            .reset_index(drop=True)
        )
    summary_path = SUMMARY_OUTPUT_DIR / "landsat_area_summary.csv"
    summary_dataframe.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    successful_count = int(
        (summary_dataframe["error"].fillna("") == "").sum()
    )
    failed_count = len(summary_dataframe) - successful_count
    print()
    print("=" * 70)
    print("Landsat前処理が完了しました。")
    print(f"成功したシーン数: {successful_count}")
    print(f"失敗したシーン数: {failed_count}")
    print(f"面積集計CSV: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
