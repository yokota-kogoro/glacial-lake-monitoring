from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.warp import reproject
from shapely.geometry import shape


STANDARD_POLYGON_COLUMNS = [
    "scene",
    "date",
    "threshold",
    "polygon_id",
    "area_m2",
    "area_km2",
    "water_px",
    "shd_px",
    "clear_px",
    "shd_ratio",
    "shd_only",
    "geometry",
]


def align_continuous_raster_to_reference(
    raster_path: Path,
    reference_profile: dict,
    reference_transform,
    reference_crs,
) -> np.ndarray:
    """連続値ラスターを参照グリッドへbilinearで整列する。"""

    output_shape = (
        reference_profile["height"],
        reference_profile["width"],
    )
    destination = np.full(
        output_shape,
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(raster_path) as source_dataset:
        if source_dataset.crs is None:
            raise ValueError(
                f"CRSが設定されていません: {raster_path}"
            )

        source = (
            source_dataset.read(1, masked=True)
            .astype(np.float32)
            .filled(np.nan)
        )
        same_grid = (
            source_dataset.width == reference_profile["width"]
            and source_dataset.height == reference_profile["height"]
            and source_dataset.transform == reference_transform
            and source_dataset.crs == reference_crs
        )
        if same_grid:
            return source

        reproject(
            source=source,
            destination=destination,
            src_transform=source_dataset.transform,
            src_crs=source_dataset.crs,
            src_nodata=np.nan,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )

    return destination


def align_categorical_raster_to_reference(
    raster_path: Path,
    reference_profile: dict,
    reference_transform,
    reference_crs,
    nodata_value: int = 255,
) -> np.ndarray:
    """カテゴリ値ラスターを参照グリッドへnearestで整列する。"""

    output_shape = (
        reference_profile["height"],
        reference_profile["width"],
    )
    destination = np.full(
        output_shape,
        nodata_value,
        dtype=np.uint8,
    )

    with rasterio.open(raster_path) as source_dataset:
        if source_dataset.crs is None:
            raise ValueError(
                f"カテゴリラスターにCRSがありません: {raster_path}"
            )

        source = (
            source_dataset.read(1, masked=True)
            .filled(nodata_value)
            .astype(np.uint8)
        )
        same_grid = (
            source_dataset.width == reference_profile["width"]
            and source_dataset.height == reference_profile["height"]
            and source_dataset.transform == reference_transform
            and source_dataset.crs == reference_crs
        )
        if same_grid:
            return source

        reproject(
            source=source,
            destination=destination,
            src_transform=source_dataset.transform,
            src_crs=source_dataset.crs,
            src_nodata=nodata_value,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=nodata_value,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )

    return destination


def align_dem_mosaic_to_reference(
    dem_files: list[Path],
    reference_profile: dict,
    reference_transform,
    reference_crs,
) -> np.ndarray:
    """複数DEMを参照グリッドへ整列し、先に有効な値を優先して結合する。"""

    output_shape = (
        reference_profile["height"],
        reference_profile["width"],
    )
    aligned_dem = np.full(
        output_shape,
        np.nan,
        dtype=np.float32,
    )

    for dem_path in dem_files:
        print(f"DEM確認: {dem_path.name}")
        with rasterio.open(dem_path) as dem_dataset:
            if dem_dataset.crs is None:
                warnings.warn(
                    f"CRSがないためDEMをスキップします: {dem_path}"
                )
                continue

            temporary_dem = np.full(
                output_shape,
                np.nan,
                dtype=np.float32,
            )
            reproject(
                source=rasterio.band(dem_dataset, 1),
                destination=temporary_dem,
                src_transform=dem_dataset.transform,
                src_crs=dem_dataset.crs,
                src_nodata=dem_dataset.nodata,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
                init_dest_nodata=True,
            )

        fill_mask = (
            np.isfinite(temporary_dem)
            & ~np.isfinite(aligned_dem)
        )
        aligned_dem[fill_mask] = temporary_dem[fill_mask]

    if not np.any(np.isfinite(aligned_dem)):
        raise ValueError(
            "対象画像の範囲に重なるDEMがありません。"
            "DEMと画像のCRSおよび対象範囲を確認してください。"
        )

    return aligned_dem


def create_aoi_mask(
    aoi_path: Path | None,
    raster_crs,
    raster_shape: tuple[int, int],
    raster_transform,
) -> np.ndarray:
    """AOIをラスター化する。AOIがNoneなら全画素を対象とする。"""

    if aoi_path is None:
        return np.ones(raster_shape, dtype=bool)
    if not aoi_path.exists():
        raise FileNotFoundError(
            f"AOIファイルが見つかりません: {aoi_path}"
        )

    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(
            f"AOIファイルに地物がありません: {aoi_path}"
        )
    if aoi.crs is None:
        raise ValueError(
            f"AOIファイルにCRSが設定されていません: {aoi_path}"
        )

    aoi = aoi.to_crs(raster_crs)
    geometries = [
        geometry
        for geometry in aoi.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if not geometries:
        raise ValueError(
            f"AOIファイルに有効な形状がありません: {aoi_path}"
        )

    return rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=raster_shape,
        transform=raster_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    ).astype(bool)


def calculate_ndwi(
    green_reflectance: np.ndarray,
    nir_reflectance: np.ndarray,
    min_reflectance_sum: float,
    use_absolute_denominator: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """反射率からNDWIと有効画素マスクを返す。"""

    denominator = green_reflectance + nir_reflectance
    if use_absolute_denominator:
        denominator_is_valid = (
            np.abs(denominator) > min_reflectance_sum
        )
    else:
        denominator_is_valid = (
            denominator > min_reflectance_sum
        )

    valid_mask = (
        np.isfinite(green_reflectance)
        & np.isfinite(nir_reflectance)
        & np.isfinite(denominator)
        & denominator_is_valid
    )
    ndwi = np.full(
        green_reflectance.shape,
        np.nan,
        dtype=np.float32,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi[valid_mask] = (
            green_reflectance[valid_mask]
            - nir_reflectance[valid_mask]
        ) / denominator[valid_mask]

    physical_range_mask = (
        np.isfinite(ndwi)
        & (ndwi >= -1.0)
        & (ndwi <= 1.0)
    )
    valid_mask &= physical_range_mask
    ndwi[~valid_mask] = np.nan
    return ndwi, valid_mask


def save_float_raster(
    output_path: Path,
    array: np.ndarray,
    reference_profile: dict,
    nodata_value: float = -9999.0,
) -> None:
    """float32配列を圧縮GeoTIFFとして保存する。"""

    array = np.asarray(array, dtype=np.float32)
    expected_shape = (
        reference_profile["height"],
        reference_profile["width"],
    )
    if array.shape != expected_shape:
        raise ValueError(
            "出力する配列と参照ラスターの形状が一致しません。"
            f" array={array.shape}, reference={expected_shape}"
        )

    output_array = np.where(
        np.isfinite(array),
        array,
        nodata_value,
    ).astype(np.float32)
    output_profile = reference_profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata_value,
        compress="deflate",
        predictor=3,
        BIGTIFF="IF_SAFER",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **output_profile) as output_dataset:
        output_dataset.write(output_array, 1)


def save_mask_raster(
    output_path: Path,
    target_mask: np.ndarray,
    valid_mask: np.ndarray,
    reference_profile: dict,
    nodata_value: int = 255,
) -> None:
    """boolマスクを0=非対象、1=対象、255=NoDataで保存する。"""

    target_mask = np.asarray(target_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if target_mask.shape != valid_mask.shape:
        raise ValueError(
            "target_maskとvalid_maskの形状が一致しません。"
            f" target_mask={target_mask.shape}, valid_mask={valid_mask.shape}"
        )

    expected_shape = (
        reference_profile["height"],
        reference_profile["width"],
    )
    if target_mask.shape != expected_shape:
        raise ValueError(
            "マスクと参照ラスターの形状が一致しません。"
            f" mask={target_mask.shape}, reference={expected_shape}"
        )

    output_array = np.full(
        target_mask.shape,
        nodata_value,
        dtype=np.uint8,
    )
    output_array[valid_mask] = 0
    output_array[target_mask & valid_mask] = 1
    output_profile = reference_profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=nodata_value,
        compress="deflate",
        predictor=2,
        BIGTIFF="IF_SAFER",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # QGISなどで表示中の既存ファイルはWindows上で上書きできない。
    # 内容と空間情報が完全一致する場合だけ、安全に既存ファイルを再利用する。
    if output_path.exists():
        try:
            with rasterio.open(output_path) as existing_dataset:
                same_grid = (
                    existing_dataset.width == reference_profile["width"]
                    and existing_dataset.height == reference_profile["height"]
                    and existing_dataset.transform == reference_profile["transform"]
                    and existing_dataset.crs == reference_profile["crs"]
                    and existing_dataset.nodata == nodata_value
                )
                if same_grid:
                    existing_array = existing_dataset.read(1)
                    if np.array_equal(existing_array, output_array):
                        return
        except (OSError, rasterio.errors.RasterioError):
            pass

    with rasterio.open(output_path, "w", **output_profile) as output_dataset:
        output_dataset.write(output_array, 1)


def save_categorical_raster(
    output_path: Path,
    array: np.ndarray,
    reference_profile: dict,
    nodata_value: int = 255,
) -> None:
    """uint8カテゴリ配列をGeoTIFFとして保存する。"""

    array = np.asarray(array, dtype=np.uint8)
    expected_shape = (
        reference_profile["height"],
        reference_profile["width"],
    )
    if array.shape != expected_shape:
        raise ValueError(
            "カテゴリラスターと参照ラスターの形状が一致しません。"
            f" array={array.shape}, reference={expected_shape}"
        )

    output_profile = reference_profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=nodata_value,
        compress="deflate",
        predictor=2,
        BIGTIFF="IF_SAFER",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **output_profile) as output_dataset:
        output_dataset.write(array, 1)


def choose_area_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """面積をm²で計算できる投影座標系を選ぶ。"""

    if gdf.crs is None:
        raise ValueError("ポリゴンにCRSが設定されていません。")
    source_crs = CRS.from_user_input(gdf.crs)
    if source_crs.is_projected:
        axis_units = {
            axis.unit_name.lower()
            for axis in source_crs.axis_info
            if axis.unit_name
        }
        if "metre" in axis_units or "meter" in axis_units:
            return source_crs

    estimated_utm_crs = gdf.estimate_utm_crs()
    if estimated_utm_crs is not None:
        return CRS.from_user_input(estimated_utm_crs)
    return CRS.from_epsg(6933)


def create_empty_polygon_gdf(raster_crs) -> gpd.GeoDataFrame:
    """共通列を持つ空の水域GeoDataFrameを返す。"""

    return gpd.GeoDataFrame(
        columns=STANDARD_POLYGON_COLUMNS,
        geometry="geometry",
        crs=raster_crs,
    )


def polygonize_water_mask(
    water_mask: np.ndarray,
    transform,
    raster_crs,
    scene_name: str,
    scene_date: str,
    threshold: float,
    review_mask: np.ndarray | None = None,
    min_polygon_area_m2: float = 0.0,
    keep_largest_only: bool = False,
    connectivity: int = 8,
) -> gpd.GeoDataFrame:
    """水域マスクをポリゴン化し、センサー共通属性を付与する。"""

    water_mask = np.asarray(water_mask, dtype=bool)
    if review_mask is None:
        review_mask = np.zeros_like(water_mask, dtype=bool)
    else:
        review_mask = np.asarray(review_mask, dtype=bool)

    if water_mask.shape != review_mask.shape:
        raise ValueError(
            "water_maskとreview_maskの配列サイズが一致しません。"
        )
    if connectivity not in {4, 8}:
        raise ValueError("connectivityは4または8にしてください。")
    if min_polygon_area_m2 < 0.0:
        raise ValueError("min_polygon_area_m2は0以上にしてください。")
    if not np.any(water_mask):
        return create_empty_polygon_gdf(raster_crs)

    records: list[dict] = []
    polygon_generator = shapes(
        water_mask.astype(np.uint8),
        mask=water_mask,
        transform=transform,
        connectivity=connectivity,
    )
    raster_height, raster_width = water_mask.shape
    inverse_transform = ~transform

    for geometry_mapping, value in polygon_generator:
        if int(value) != 1:
            continue
        geometry = shape(geometry_mapping)
        if geometry.is_empty:
            continue

        min_x, min_y, max_x, max_y = geometry.bounds
        min_col_float, max_row_float = inverse_transform * (min_x, min_y)
        max_col_float, min_row_float = inverse_transform * (max_x, max_y)
        row_start = max(0, int(np.floor(min_row_float)))
        row_stop = min(raster_height, int(np.ceil(max_row_float)))
        col_start = max(0, int(np.floor(min_col_float)))
        col_stop = min(raster_width, int(np.ceil(max_col_float)))
        if row_stop <= row_start or col_stop <= col_start:
            continue

        window_transform = transform * rasterio.Affine.translation(
            col_start,
            row_start,
        )
        polygon_pixel_mask = rasterize(
            [(geometry, 1)],
            out_shape=(row_stop - row_start, col_stop - col_start),
            transform=window_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        ).astype(bool)
        water_window = water_mask[
            row_start:row_stop,
            col_start:col_stop,
        ]
        review_window = review_mask[
            row_start:row_stop,
            col_start:col_stop,
        ]
        polygon_water_mask = polygon_pixel_mask & water_window
        water_pixel_count = int(polygon_water_mask.sum())
        review_pixel_count = int(
            (polygon_water_mask & review_window).sum()
        )
        clear_pixel_count = water_pixel_count - review_pixel_count
        review_ratio = (
            review_pixel_count / water_pixel_count
            if water_pixel_count > 0
            else 0.0
        )
        records.append(
            {
                "geometry": geometry,
                "water_px": water_pixel_count,
                "shd_px": review_pixel_count,
                "clear_px": clear_pixel_count,
                "shd_ratio": review_ratio,
                "shd_only": bool(
                    water_pixel_count > 0
                    and review_pixel_count == water_pixel_count
                ),
            }
        )

    if not records:
        return create_empty_polygon_gdf(raster_crs)

    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=raster_crs,
    )
    area_crs = choose_area_crs(gdf)
    area_values = gdf.to_crs(area_crs).geometry.area.to_numpy()
    gdf["scene"] = scene_name
    gdf["date"] = scene_date
    gdf["threshold"] = float(threshold)
    gdf["area_m2"] = area_values
    gdf["area_km2"] = area_values / 1_000_000.0

    if min_polygon_area_m2 > 0.0:
        gdf = gdf[
            gdf["area_m2"] >= min_polygon_area_m2
        ].copy()
    if keep_largest_only and not gdf.empty:
        gdf = gdf.loc[[gdf["area_m2"].idxmax()]].copy()
    if gdf.empty:
        return create_empty_polygon_gdf(raster_crs)

    gdf = (
        gdf.sort_values("area_m2", ascending=False)
        .reset_index(drop=True)
    )
    gdf["polygon_id"] = np.arange(1, len(gdf) + 1)
    return gdf[STANDARD_POLYGON_COLUMNS]


def remove_existing_shapefile(shapefile_path: Path) -> None:
    """同名Shapefileを構成する既存ファイルを削除する。"""

    related_extensions = {
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".qix",
        ".sbn",
        ".sbx",
    }
    for extension in related_extensions:
        related_file = shapefile_path.with_suffix(extension)
        if related_file.exists():
            related_file.unlink()


def save_water_polygons(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> Path | None:
    """共通属性を短いShapefile列名へ変換して保存する。"""

    if gdf.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove_existing_shapefile(output_path)
    output_gdf = gdf.rename(
        columns={
            "threshold": "ndwi_thr",
            "polygon_id": "poly_id",
        }
    )
    output_gdf.to_file(
        output_path,
        driver="ESRI Shapefile",
        encoding="UTF-8",
        index=False,
    )
    return output_path
