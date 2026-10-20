from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask as raster_mask


CATALOG_COLUMNS = [
    "observation_id",
    "scene",
    "date",
    "observation_datetime",
    "sensor_family",
    "platform",
    "product_id",
    "processing_level",
    "native_pixel_size_m",
    "ndwi_threshold",
    "ndwi_file",
    "water_mask_file",
    "elevation_mask_file",
    "polygon_file",
    "metadata_file",
    "quality_mask_file",
    "minimum_elevation_m",
    "cloud_cover_scene_percent",
    "water_pixel_count",
    "water_area_m2",
    "catalog_status",
    "catalog_error",
]


def value_to_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def path_value_to_text(value) -> str:
    path_text = value_to_text(value)
    if path_text == "":
        return ""
    return str(Path(path_text).resolve())


def validate_summary_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    sensor_label: str,
) -> None:
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{sensor_label}集計CSVに必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )


def build_catalog_row(
    row: pd.Series,
    sensor_family: str,
) -> dict:
    scene_name = value_to_text(row["scene"])
    scene_date = pd.to_datetime(
        row["date"],
        errors="raise",
    ).date().isoformat()
    product_id = value_to_text(row.get("product_id", ""))
    if product_id == "":
        product_id = f"{sensor_family.upper()}_{scene_name}"

    ndwi_file = path_value_to_text(row["ndwi_file"])
    water_mask_file = path_value_to_text(row["mask_file"])
    elevation_mask_file = path_value_to_text(
        row["elevation_mask_file"]
    )
    polygon_file = path_value_to_text(row["shapefile"])
    metadata_file = path_value_to_text(row["metadata_file"])
    quality_mask_file = path_value_to_text(
        row.get(
            "aligned_scl_file"
            if sensor_family == "Sentinel-2"
            else "quality_mask_file",
            "",
        )
    )

    source_error = value_to_text(row["error"])
    source_total_area_m2 = pd.to_numeric(
        row["total_area_m2"],
        errors="coerce",
    )
    missing_files: list[str] = []
    for file_label, file_text in [
        ("NDWI", ndwi_file),
        ("水域マスク", water_mask_file),
        ("メタデータ", metadata_file),
    ]:
        if file_text == "" or not Path(file_text).exists():
            missing_files.append(file_label)

    if (
        pd.notna(source_total_area_m2)
        and float(source_total_area_m2) > 0.0
        and (
            polygon_file == ""
            or not Path(polygon_file).exists()
        )
    ):
        missing_files.append("水域ポリゴン")

    catalog_error_parts: list[str] = []
    if source_error != "":
        catalog_error_parts.append(source_error)
    if missing_files:
        catalog_error_parts.append(
            "不足ファイル: " + ", ".join(missing_files)
        )
    catalog_error = "; ".join(catalog_error_parts)

    if sensor_family == "Sentinel-2":
        platform = (
            product_id.split("_")[0]
            if "_" in product_id
            else "Sentinel-2"
        )
        observation_datetime = value_to_text(
            row.get("product_start_time", "")
        )
        processing_level = "L2A"
        native_pixel_size_m = 10.0
        cloud_cover = np.nan
    else:
        platform = value_to_text(row.get("platform", ""))
        observation_datetime = value_to_text(
            row.get("observation_datetime", scene_date)
        )
        processing_level = value_to_text(
            row.get("processing_level", "")
        )
        native_pixel_size_m = pd.to_numeric(
            row.get("native_pixel_size_m", 30.0),
            errors="coerce",
        )
        cloud_cover = pd.to_numeric(
            row.get("cloud_cover_scene_percent", np.nan),
            errors="coerce",
        )

    observation_prefix = (
        "SENTINEL2"
        if sensor_family == "Sentinel-2"
        else "LANDSAT"
    )

    return {
        "observation_id": f"{observation_prefix}::{product_id}",
        "scene": scene_name,
        "date": scene_date,
        "observation_datetime": observation_datetime,
        "sensor_family": sensor_family,
        "platform": platform,
        "product_id": product_id,
        "processing_level": processing_level,
        "native_pixel_size_m": native_pixel_size_m,
        "ndwi_threshold": float(row["threshold"]),
        "ndwi_file": ndwi_file,
        "water_mask_file": water_mask_file,
        "elevation_mask_file": elevation_mask_file,
        "polygon_file": polygon_file,
        "metadata_file": metadata_file,
        "quality_mask_file": quality_mask_file,
        "minimum_elevation_m": pd.to_numeric(
            row["min_elevation_m"],
            errors="coerce",
        ),
        "cloud_cover_scene_percent": cloud_cover,
        "water_pixel_count": pd.to_numeric(
            row["final_water_pixel_count"],
            errors="coerce",
        ),
        "water_area_m2": source_total_area_m2,
        "catalog_status": "ok" if catalog_error == "" else "invalid",
        "catalog_error": catalog_error,
    }


def build_multisensor_scene_catalog(
    sentinel_summary_csv: Path,
    landsat_summary_csv: Path,
) -> pd.DataFrame:
    """両センサーの前処理CSVを共通シーンカタログへ統合する。"""

    for sensor_label, summary_path in [
        ("Sentinel-2", sentinel_summary_csv),
        ("Landsat", landsat_summary_csv),
    ]:
        if not summary_path.exists():
            raise FileNotFoundError(
                f"{sensor_label}集計CSVが見つかりません: {summary_path}"
            )

    sentinel_dataframe = pd.read_csv(sentinel_summary_csv)
    landsat_dataframe = pd.read_csv(landsat_summary_csv)
    common_required_columns = {
        "scene",
        "date",
        "threshold",
        "ndwi_file",
        "mask_file",
        "shapefile",
        "metadata_file",
        "elevation_mask_file",
        "min_elevation_m",
        "final_water_pixel_count",
        "total_area_m2",
        "error",
    }
    validate_summary_columns(
        sentinel_dataframe,
        common_required_columns | {
            "product_uri",
            "product_start_time",
        },
        "Sentinel-2",
    )
    validate_summary_columns(
        landsat_dataframe,
        common_required_columns | {
            "product_id",
            "platform",
            "processing_level",
            "observation_datetime",
            "native_pixel_size_m",
            "cloud_cover_scene_percent",
        },
        "Landsat",
    )

    sentinel_for_catalog = sentinel_dataframe.copy()
    sentinel_for_catalog["product_id"] = sentinel_for_catalog[
        "product_uri"
    ]
    catalog_rows = [
        build_catalog_row(row, "Sentinel-2")
        for _, row in sentinel_for_catalog.iterrows()
    ]
    catalog_rows.extend(
        build_catalog_row(row, "Landsat")
        for _, row in landsat_dataframe.iterrows()
    )

    catalog_dataframe = pd.DataFrame(
        catalog_rows,
        columns=CATALOG_COLUMNS,
    )
    catalog_dataframe["date"] = pd.to_datetime(
        catalog_dataframe["date"],
        errors="raise",
    ).dt.normalize()
    duplicated_mask = catalog_dataframe[
        "observation_id"
    ].duplicated(keep=False)
    if duplicated_mask.any():
        duplicated_ids = sorted(
            catalog_dataframe.loc[
                duplicated_mask,
                "observation_id",
            ].astype(str).unique()
        )
        raise ValueError(
            "observation_idが重複しています: "
            + ", ".join(duplicated_ids)
        )

    return (
        catalog_dataframe.sort_values(
            [
                "date",
                "sensor_family",
                "platform",
                "observation_id",
            ]
        )
        .reset_index(drop=True)
    )


def measure_catalog_aoi_water_from_masks(
    catalog_dataframe: pd.DataFrame,
    aoi_path: Path,
    aoi_name: str,
) -> pd.DataFrame:
    """両センサーの前処理済みマスクとポリゴンを同じAOIで検査する。"""

    required_columns = {
        "observation_id",
        "sensor_family",
        "water_mask_file",
        "elevation_mask_file",
        "polygon_file",
        "catalog_status",
    }
    missing_columns = required_columns - set(catalog_dataframe.columns)
    if missing_columns:
        raise ValueError(
            "AOI面積計算に必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )
    if not aoi_path.exists():
        raise FileNotFoundError(
            f"AOIファイルが見つかりません: {aoi_path}"
        )

    aoi_gdf = gpd.read_file(aoi_path)
    if aoi_gdf.empty or aoi_gdf.crs is None:
        raise ValueError(
            f"AOIが空、またはCRSがありません: {aoi_path}"
        )
    aoi_gdf = aoi_gdf[
        aoi_gdf.geometry.notna()
        & ~aoi_gdf.geometry.is_empty
    ].copy()
    if aoi_gdf.empty:
        raise ValueError(
            f"AOIに有効な形状がありません: {aoi_path}"
        )

    output_dataframe = catalog_dataframe.copy().reset_index(drop=True)
    additional_columns = {
        "aoi_name": "",
        "aoi_file": "",
        "minimum_elevation_m": np.nan,
        "polygon_scope": "",
        "aoi_total_pixel_count": np.nan,
        "aoi_elevation_eligible_pixel_count": np.nan,
        "aoi_valid_pixel_count": np.nan,
        "aoi_valid_pixel_ratio": np.nan,
        "aoi_valid_pixel_ratio_of_total": np.nan,
        "aoi_qa_review_pixel_count": np.nan,
        "aoi_qa_review_percent": np.nan,
        "aoi_water_qa_review_pixel_count": np.nan,
        "aoi_water_qa_review_percent": np.nan,
        "aoi_water_pixel_count": np.nan,
        "aoi_water_area_m2": np.nan,
        "retained_polygon_count": np.nan,
        "retained_polygon_area_m2": np.nan,
        "polygon_status": "not_processed",
        "polygon_error": "",
    }
    for column_name, default_value in additional_columns.items():
        if column_name not in output_dataframe.columns:
            output_dataframe[column_name] = default_value

    for row_index, row in output_dataframe.iterrows():
        sensor_family = value_to_text(row["sensor_family"])
        output_dataframe.at[row_index, "aoi_name"] = aoi_name
        output_dataframe.at[row_index, "aoi_file"] = str(
            aoi_path.resolve()
        )
        output_dataframe.at[row_index, "polygon_scope"] = (
            "aoi_from_preprocessed_mask"
        )

        try:
            if value_to_text(row["catalog_status"]) != "ok":
                raise ValueError(
                    "catalog_statusがokではありません: "
                    f"{row['catalog_status']}"
                )
            water_mask_path = Path(
                value_to_text(row["water_mask_file"])
            )
            if not water_mask_path.exists():
                raise FileNotFoundError(
                    f"{sensor_family}水域マスクが見つかりません: "
                    f"{water_mask_path}"
                )

            elevation_mask_path = Path(
                value_to_text(row["elevation_mask_file"])
            )
            if not elevation_mask_path.exists():
                raise FileNotFoundError(
                    f"{sensor_family}標高マスクが見つかりません: "
                    f"{elevation_mask_path}"
                )

            with rasterio.open(water_mask_path) as mask_dataset:
                if mask_dataset.crs is None:
                    raise ValueError("水域マスクにCRSがありません。")
                if not mask_dataset.crs.is_projected:
                    raise ValueError(
                        "水域マスクが投影座標系ではありません: "
                        f"{mask_dataset.crs}"
                    )
                mask_transform = mask_dataset.transform
                if (
                    mask_transform.a <= 0.0
                    or mask_transform.e >= 0.0
                    or mask_transform.b != 0.0
                    or mask_transform.d != 0.0
                ):
                    raise ValueError(
                        "回転した水域マスクには対応していません。"
                    )
                aoi_in_mask_crs = aoi_gdf.to_crs(mask_dataset.crs)
                masked_array, clipped_transform = raster_mask(
                    mask_dataset,
                    [
                        geometry.__geo_interface__
                        for geometry in aoi_in_mask_crs.geometry
                    ],
                    indexes=1,
                    crop=True,
                    all_touched=False,
                    filled=False,
                )
                mask_crs = mask_dataset.crs

            masked_array = np.ma.asarray(masked_array)
            if masked_array.ndim == 3:
                masked_array = masked_array[0]
            raster_valid_mask = ~np.ma.getmaskarray(masked_array)
            water_values = np.asarray(masked_array)
            inside_aoi_mask = geometry_mask(
                [
                    geometry.__geo_interface__
                    for geometry in aoi_in_mask_crs.geometry
                ],
                out_shape=water_values.shape,
                transform=clipped_transform,
                invert=True,
                all_touched=False,
            )
            valid_values = water_values[raster_valid_mask]
            unique_values = set(np.unique(valid_values).tolist())
            if not unique_values.issubset({0, 1}):
                raise ValueError(
                    "水域マスクに0・1以外の有効値があります: "
                    f"{sorted(unique_values)}"
                )

            final_water_mask = raster_valid_mask & (water_values == 1)
            total_pixel_count = int(inside_aoi_mask.sum())
            valid_pixel_count = int(raster_valid_mask.sum())
            water_pixel_count = int(final_water_mask.sum())

            # 水域マスクは、標高条件を満たさない画素を
            # NoDataとして保存する。そのため全AOI画素を分母に
            # すると、広域解析で低標高域を雲欠損と誤判定する。
            # 観測可能率は3,000 m以上などの解析対象画素を
            # 分母にして計算する。
            with rasterio.open(elevation_mask_path) as elevation_dataset:
                if elevation_dataset.crs is None:
                    raise ValueError(
                        "標高マスクにCRSがありません。"
                    )
                aoi_in_elevation_crs = aoi_gdf.to_crs(
                    elevation_dataset.crs
                )
                elevation_array, elevation_transform = raster_mask(
                    elevation_dataset,
                    [
                        geometry.__geo_interface__
                        for geometry in aoi_in_elevation_crs.geometry
                    ],
                    indexes=1,
                    crop=True,
                    all_touched=False,
                    filled=False,
                )

            elevation_array = np.ma.asarray(elevation_array)
            if elevation_array.ndim == 3:
                elevation_array = elevation_array[0]
            elevation_values = np.asarray(elevation_array)
            elevation_valid_mask = ~np.ma.getmaskarray(
                elevation_array
            )

            if (
                elevation_values.shape != water_values.shape
                or not elevation_transform.almost_equals(
                    clipped_transform
                )
            ):
                raise ValueError(
                    "標高マスクと水域マスクの"
                    "AOI切り出し格子が一致しません: "
                    f"{elevation_mask_path}"
                )

            elevation_eligible_mask = (
                elevation_valid_mask
                & inside_aoi_mask
                & (elevation_values == 1)
            )
            elevation_eligible_pixel_count = int(
                elevation_eligible_mask.sum()
            )
            valid_pixel_ratio = (
                valid_pixel_count / elevation_eligible_pixel_count
                if elevation_eligible_pixel_count > 0
                else np.nan
            )
            valid_pixel_ratio_of_total = (
                valid_pixel_count / total_pixel_count
                if total_pixel_count > 0
                else np.nan
            )
            pixel_area_m2 = abs(
                clipped_transform.a * clipped_transform.e
            )
            water_area_m2 = float(water_pixel_count * pixel_area_m2)
            output_dataframe.at[
                row_index,
                "aoi_total_pixel_count",
            ] = total_pixel_count
            output_dataframe.at[
                row_index,
                "aoi_elevation_eligible_pixel_count",
            ] = elevation_eligible_pixel_count
            output_dataframe.at[
                row_index,
                "aoi_valid_pixel_count",
            ] = valid_pixel_count
            output_dataframe.at[
                row_index,
                "aoi_valid_pixel_ratio",
            ] = valid_pixel_ratio
            output_dataframe.at[
                row_index,
                "aoi_valid_pixel_ratio_of_total",
            ] = valid_pixel_ratio_of_total
            output_dataframe.at[
                row_index,
                "aoi_water_pixel_count",
            ] = water_pixel_count
            output_dataframe.at[
                row_index,
                "aoi_water_area_m2",
            ] = water_area_m2

            # LandsatのQA_PIXELとSentinel-2のSCLクラス3は、
            # ここでは水域から直接削除せず、AOI全体および水域候補内の
            # 時系列確認が必要な割合としてカタログへ記録する。
            quality_mask_text = value_to_text(
                row.get("quality_mask_file", "")
            )
            if (
                sensor_family in {"Sentinel-2", "Landsat"}
                and quality_mask_text != ""
                and Path(quality_mask_text).exists()
            ):
                with rasterio.open(quality_mask_text) as quality_dataset:
                    if quality_dataset.crs is None:
                        raise ValueError(
                            "Landsat QA_PIXELにCRSがありません。"
                        )
                    aoi_in_quality_crs = aoi_gdf.to_crs(
                        quality_dataset.crs
                    )
                    quality_array, quality_transform = raster_mask(
                        quality_dataset,
                        [
                            geometry.__geo_interface__
                            for geometry in aoi_in_quality_crs.geometry
                        ],
                        indexes=1,
                        crop=True,
                        all_touched=False,
                        filled=False,
                    )

                quality_array = np.ma.asarray(quality_array)
                if quality_array.ndim == 3:
                    quality_array = quality_array[0]
                quality_valid_mask = ~np.ma.getmaskarray(
                    quality_array
                )
                quality_values = np.asarray(
                    quality_array,
                    dtype=np.uint16,
                )

                if (
                    quality_values.shape != water_values.shape
                    or not quality_transform.almost_equals(
                        clipped_transform
                    )
                ):
                    raise ValueError(
                        "品質マスクと水域マスクの"
                        "AOI切り出し格子が一致しません: "
                        f"{quality_mask_text}"
                    )

                if sensor_family == "Landsat":
                    qa_review_mask = (
                        ((quality_values & np.uint16(1 << 1)) != 0)
                        | ((quality_values & np.uint16(1 << 2)) != 0)
                        | ((quality_values & np.uint16(1 << 3)) != 0)
                        | ((quality_values & np.uint16(1 << 4)) != 0)
                    ) & quality_valid_mask

                else:
                    # Sentinel-2ではSCLクラス3を雲影確認対象とする。
                    qa_review_mask = (
                        (quality_values == np.uint16(3))
                        & quality_valid_mask
                    )

                qa_valid_pixel_count = int(
                    quality_valid_mask.sum()
                )
                qa_review_pixel_count = int(
                    qa_review_mask.sum()
                )
                qa_review_percent = (
                    qa_review_pixel_count
                    / qa_valid_pixel_count
                    * 100.0
                    if qa_valid_pixel_count > 0
                    else np.nan
                )
                output_dataframe.at[
                    row_index,
                    "aoi_qa_review_pixel_count",
                ] = qa_review_pixel_count
                output_dataframe.at[
                    row_index,
                    "aoi_qa_review_percent",
                ] = qa_review_percent

                water_qa_review_pixel_count = int(
                    (
                        qa_review_mask
                        & final_water_mask
                    ).sum()
                )
                water_qa_review_percent = (
                    water_qa_review_pixel_count
                    / water_pixel_count
                    * 100.0
                    if water_pixel_count > 0
                    else 0.0
                )
                output_dataframe.at[
                    row_index,
                    "aoi_water_qa_review_pixel_count",
                ] = water_qa_review_pixel_count
                output_dataframe.at[
                    row_index,
                    "aoi_water_qa_review_percent",
                ] = water_qa_review_percent

            polygon_file_text = value_to_text(row["polygon_file"])
            if polygon_file_text == "":
                if water_pixel_count > 0:
                    raise ValueError(
                        "AOI内に水域画素がありますが、"
                        "polygon_fileが空です。"
                    )
                output_dataframe.at[
                    row_index,
                    "retained_polygon_count",
                ] = 0
                output_dataframe.at[
                    row_index,
                    "retained_polygon_area_m2",
                ] = 0.0
                output_dataframe.at[
                    row_index,
                    "polygon_status",
                ] = "no_retained_water_polygon"
                continue

            polygon_path = Path(polygon_file_text)
            if not polygon_path.exists():
                raise FileNotFoundError(
                    f"水域ポリゴンが見つかりません: {polygon_path}"
                )
            source_gdf = gpd.read_file(polygon_path)
            if source_gdf.empty:
                retained_polygon_count = 0
                retained_polygon_area_m2 = 0.0
            else:
                if source_gdf.crs is None:
                    raise ValueError(
                        f"水域ポリゴンにCRSがありません: {polygon_path}"
                    )
                source_gdf = source_gdf[
                    source_gdf.geometry.notna()
                    & ~source_gdf.geometry.is_empty
                ].to_crs(mask_crs)
                dissolved_aoi = (
                    aoi_in_mask_crs[["geometry"]]
                    .dissolve()
                    .reset_index(drop=True)
                )
                clipped_gdf = gpd.overlay(
                    source_gdf,
                    dissolved_aoi,
                    how="intersection",
                    keep_geom_type=True,
                    make_valid=True,
                )
                clipped_gdf = (
                    clipped_gdf.explode(index_parts=False)
                    .reset_index(drop=True)
                )
                clipped_gdf = clipped_gdf[
                    clipped_gdf.geometry.notna()
                    & ~clipped_gdf.geometry.is_empty
                    & clipped_gdf.geom_type.isin(
                        ["Polygon", "MultiPolygon"]
                    )
                ].copy()
                retained_polygon_count = len(clipped_gdf)
                retained_polygon_area_m2 = float(
                    clipped_gdf.geometry.area.sum()
                )

            output_dataframe.at[
                row_index,
                "retained_polygon_count",
            ] = retained_polygon_count
            output_dataframe.at[
                row_index,
                "retained_polygon_area_m2",
            ] = retained_polygon_area_m2
            if water_pixel_count > 0 and retained_polygon_count == 0:
                raise ValueError(
                    "AOI内に水域画素がありますが、"
                    "AOI内水域ポリゴンがありません。"
                )
            output_dataframe.at[row_index, "polygon_status"] = (
                "ok"
                if water_pixel_count > 0
                else "no_retained_water_polygon"
            )
            output_dataframe.at[row_index, "polygon_error"] = ""

        except Exception as error:
            output_dataframe.at[row_index, "polygon_status"] = "failed"
            output_dataframe.at[row_index, "polygon_error"] = str(error)

    return output_dataframe
