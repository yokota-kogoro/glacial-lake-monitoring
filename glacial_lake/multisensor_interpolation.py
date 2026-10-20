from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union


def apply_multisensor_bracketed_partial_geometry_fill(
    persistent_tracking_gdf: gpd.GeoDataFrame,
    output_gpkg: Path,
    output_layer: str,
    max_reference_gap_days: int,
    same_sensor_min_reference_iou: float,
    cross_sensor_min_reference_iou: float,
    min_current_stable_overlap_smaller_ratio: float,
    min_fill_area_by_sensor: dict,
    max_fill_ratio: float,
) -> gpd.GeoDataFrame:

    required_columns = {
        "local_candidate_id",
        "date",
        "sensor_family",
        "persistent_lake_id",
        "persistent_counted_in_lake_count",
        "tracking_status",
        "area_m2",
        "area_km2",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(persistent_tracking_gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            "前後観測による部分補完に"
            "必要な列がありません: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if persistent_tracking_gdf.empty:
        raise ValueError(
            "部分補完対象が空です。"
        )

    if persistent_tracking_gdf.crs is None:
        raise ValueError(
            "部分補完対象にCRSがありません。"
        )

    if not persistent_tracking_gdf.crs.is_projected:
        raise ValueError(
            "部分補完の面積計算には"
            "投影座標系が必要です。"
        )

    if max_reference_gap_days <= 0:
        raise ValueError(
            "max_reference_gap_daysは"
            "1以上にしてください。"
        )

    for parameter_name, parameter_value in {
        "same_sensor_min_reference_iou": (
            same_sensor_min_reference_iou
        ),
        "cross_sensor_min_reference_iou": (
            cross_sensor_min_reference_iou
        ),
        "min_current_stable_overlap_smaller_ratio": (
            min_current_stable_overlap_smaller_ratio
        ),
        "max_fill_ratio": max_fill_ratio,
    }.items():
        if not 0.0 <= parameter_value <= 1.0:
            raise ValueError(
                f"{parameter_name}は"
                "0以上1以下にしてください。"
            )

    for (
        sensor_family,
        minimum_fill_area_m2,
    ) in min_fill_area_by_sensor.items():
        if float(minimum_fill_area_m2) < 0.0:
            raise ValueError(
                "最低補完面積は0以上にしてください: "
                f"{sensor_family}"
            )

    def normalize_polygonal_geometry(
        geometry,
    ):
        if geometry is None:
            return None

        if geometry.is_empty:
            return None

        try:
            if not geometry.is_valid:
                geometry = geometry.buffer(0)

        except Exception:
            return None

        if (
            geometry is None
            or geometry.is_empty
        ):
            return None

        geometry_type = geometry.geom_type

        if geometry_type in {
            "Polygon",
            "MultiPolygon",
        }:
            return geometry

        polygon_parts = []

        if hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                normalized_part = (
                    normalize_polygonal_geometry(
                        part
                    )
                )

                if normalized_part is None:
                    continue

                if (
                    normalized_part.geom_type
                    == "Polygon"
                ):
                    polygon_parts.append(
                        normalized_part
                    )

                elif (
                    normalized_part.geom_type
                    == "MultiPolygon"
                ):
                    polygon_parts.extend(
                        list(
                            normalized_part.geoms
                        )
                    )

        if not polygon_parts:
            return None

        try:
            normalized_geometry = unary_union(
                polygon_parts
            )

        except Exception:
            return None

        if (
            normalized_geometry is None
            or normalized_geometry.is_empty
        ):
            return None

        if not normalized_geometry.is_valid:
            try:
                normalized_geometry = (
                    normalized_geometry.buffer(0)
                )

            except Exception:
                return None

        if (
            normalized_geometry is None
            or normalized_geometry.is_empty
            or not normalized_geometry.is_valid
        ):
            return None

        return normalized_geometry

    output_gdf = (
        persistent_tracking_gdf
        .copy()
        .reset_index(drop=True)
    )

    output_gdf["_parsed_date"] = (
        pd.to_datetime(
            output_gdf["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if output_gdf["_parsed_date"].isna().any():
        invalid_ids = (
            output_gdf.loc[
                output_gdf[
                    "_parsed_date"
                ].isna(),
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "補完処理で日付を解釈できない"
            "候補があります: "
            + ", ".join(invalid_ids)
        )

    output_gdf["persistent_lake_id"] = (
        output_gdf["persistent_lake_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    output_gdf[
        "persistent_counted_in_lake_count"
    ] = (
        output_gdf[
            "persistent_counted_in_lake_count"
        ]
        .fillna(False)
        .astype(bool)
    )

    area_values = pd.to_numeric(
        output_gdf["area_m2"],
        errors="coerce",
    )

    assigned_direct_mask = (
        output_gdf["persistent_lake_id"]
        .ne("")
        & output_gdf[
            "persistent_counted_in_lake_count"
        ]
        & output_gdf.geometry.notna()
        & ~output_gdf.geometry.is_empty
        & area_values.notna()
    )

    output_gdf["direct_area_m2"] = (
        area_values
    )

    output_gdf["direct_area_km2"] = (
        area_values
        / 1_000_000.0
    )

    output_gdf[
        "temporally_filled_area_m2"
    ] = 0.0

    output_gdf["corrected_area_m2"] = (
        area_values
    )

    output_gdf["corrected_area_km2"] = (
        area_values
        / 1_000_000.0
    )

    output_gdf["temporal_fill_ratio"] = 0.0

    output_gdf[
        "temporal_fill_applied"
    ] = False

    output_gdf[
        "temporal_fill_status"
    ] = "not_applicable_unassigned"

    output_gdf.loc[
        assigned_direct_mask,
        "temporal_fill_status",
    ] = "not_filled_no_bracketing_reference"

    output_gdf[
        "temporal_fill_previous_date"
    ] = ""

    output_gdf[
        "temporal_fill_next_date"
    ] = ""

    output_gdf[
        "temporal_fill_previous_sensor"
    ] = ""

    output_gdf[
        "temporal_fill_current_sensor"
    ] = ""

    output_gdf[
        "temporal_fill_next_sensor"
    ] = ""

    output_gdf[
        "temporal_fill_cross_sensor_window"
    ] = False

    output_gdf[
        "temporal_reference_iou"
    ] = np.nan

    output_gdf[
        "temporal_reference_iou_threshold"
    ] = np.nan

    output_gdf[
        "temporal_current_stable_overlap_ratio"
    ] = np.nan

    output_gdf[
        "temporal_stable_overlap_smaller_ratio"
    ] = np.nan

    output_gdf[
        "temporal_min_fill_area_m2_used"
    ] = np.nan

    # 補完前の直接観測を固定する。
    #
    # ここで固定しないと、補完した形状が
    # 別の日付の補完根拠として連鎖的に使われる。
    original_direct_rows = (
        output_gdf.loc[
            assigned_direct_mask
        ]
        .copy()
    )

    for persistent_lake_id, lake_rows in (
        original_direct_rows.groupby(
            "persistent_lake_id",
            sort=False,
        )
    ):
        lake_rows = (
            lake_rows
            .sort_values(
                [
                    "_parsed_date",
                    "local_candidate_id",
                ]
            )
        )

        duplicated_date_mask = (
            lake_rows["_parsed_date"]
            .duplicated(
                keep=False
            )
        )

        if duplicated_date_mask.any():
            duplicate_rows = (
                lake_rows.loc[
                    duplicated_date_mask,
                    [
                        "_parsed_date",
                        "local_candidate_id",
                    ],
                ]
            )

            raise ValueError(
                "同じ永続IDが同じ日付に"
                "複数存在するため補完できません: "
                f"{persistent_lake_id}\n"
                + duplicate_rows.to_string(
                    index=False
                )
            )

        if len(lake_rows) < 3:
            continue

        lake_indices = list(
            lake_rows.index
        )

        for position in range(
            1,
            len(lake_indices) - 1,
        ):
            previous_index = (
                lake_indices[position - 1]
            )

            current_index = (
                lake_indices[position]
            )

            next_index = (
                lake_indices[position + 1]
            )

            previous_row = (
                original_direct_rows.loc[
                    previous_index
                ]
            )

            current_row = (
                original_direct_rows.loc[
                    current_index
                ]
            )

            next_row = (
                original_direct_rows.loc[
                    next_index
                ]
            )

            previous_date = (
                previous_row["_parsed_date"]
            )

            current_date = (
                current_row["_parsed_date"]
            )

            next_date = (
                next_row["_parsed_date"]
            )

            days_from_previous = int(
                (
                    current_date
                    - previous_date
                ).days
            )

            days_to_next = int(
                (
                    next_date
                    - current_date
                ).days
            )

            previous_sensor = str(
                previous_row["sensor_family"]
            )

            current_sensor = str(
                current_row["sensor_family"]
            )

            next_sensor = str(
                next_row["sensor_family"]
            )

            output_gdf.at[
                current_index,
                "temporal_fill_previous_date",
            ] = previous_date.strftime(
                "%Y-%m-%d"
            )

            output_gdf.at[
                current_index,
                "temporal_fill_next_date",
            ] = next_date.strftime(
                "%Y-%m-%d"
            )

            output_gdf.at[
                current_index,
                "temporal_fill_previous_sensor",
            ] = previous_sensor

            output_gdf.at[
                current_index,
                "temporal_fill_current_sensor",
            ] = current_sensor

            output_gdf.at[
                current_index,
                "temporal_fill_next_sensor",
            ] = next_sensor

            sensor_window = {
                previous_sensor,
                current_sensor,
                next_sensor,
            }

            cross_sensor_window = bool(
                len(sensor_window) > 1
            )

            output_gdf.at[
                current_index,
                "temporal_fill_cross_sensor_window",
            ] = cross_sensor_window

            if cross_sensor_window:
                reference_iou_threshold = float(
                    cross_sensor_min_reference_iou
                )

            else:
                reference_iou_threshold = float(
                    same_sensor_min_reference_iou
                )

            output_gdf.at[
                current_index,
                "temporal_reference_iou_threshold",
            ] = reference_iou_threshold

            minimum_fill_area_m2 = (
                min_fill_area_by_sensor.get(
                    current_sensor
                )
            )

            if minimum_fill_area_m2 is None:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_missing_"
                    "sensor_setting"
                )

                continue

            minimum_fill_area_m2 = float(
                minimum_fill_area_m2
            )

            output_gdf.at[
                current_index,
                "temporal_min_fill_area_m2_used",
            ] = minimum_fill_area_m2

            if (
                days_from_previous
                > max_reference_gap_days
                or days_to_next
                > max_reference_gap_days
            ):
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "reference_gap_too_long"
                )

                continue

            previous_geometry = (
                normalize_polygonal_geometry(
                    previous_row.geometry
                )
            )

            current_geometry = (
                normalize_polygonal_geometry(
                    current_row.geometry
                )
            )

            next_geometry = (
                normalize_polygonal_geometry(
                    next_row.geometry
                )
            )

            if (
                previous_geometry is None
                or current_geometry is None
                or next_geometry is None
            ):
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "invalid_geometry"
                )

                continue

            try:
                reference_intersection = (
                    previous_geometry
                    .intersection(
                        next_geometry
                    )
                )

                reference_intersection_area_m2 = (
                    float(
                        reference_intersection.area
                    )
                )

                reference_union_area_m2 = float(
                    previous_geometry.area
                    + next_geometry.area
                    - reference_intersection_area_m2
                )

            except Exception:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "reference_intersection_error"
                )

                continue

            reference_iou = (
                reference_intersection_area_m2
                / reference_union_area_m2
                if reference_union_area_m2 > 0.0
                else 0.0
            )

            output_gdf.at[
                current_index,
                "temporal_reference_iou",
            ] = reference_iou

            if (
                reference_iou
                < reference_iou_threshold
            ):
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "reference_disagreement"
                )

                continue

            stable_geometry = (
                normalize_polygonal_geometry(
                    reference_intersection
                )
            )

            if stable_geometry is None:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "no_stable_water"
                )

                continue

            stable_area_m2 = float(
                stable_geometry.area
            )

            if stable_area_m2 <= 0.0:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "no_stable_water"
                )

                continue

            current_area_m2 = float(
                current_geometry.area
            )

            if current_area_m2 <= 0.0:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "invalid_current_area"
                )

                continue

            try:
                current_stable_intersection_area_m2 = (
                    float(
                        stable_geometry
                        .intersection(
                            current_geometry
                        )
                        .area
                    )
                )

            except Exception:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_current_"
                    "intersection_error"
                )

                continue

            current_stable_overlap_ratio = (
                current_stable_intersection_area_m2
                / current_area_m2
            )

            smaller_current_stable_area_m2 = min(
                current_area_m2,
                stable_area_m2,
            )

            stable_overlap_smaller_ratio = (
                current_stable_intersection_area_m2
                / smaller_current_stable_area_m2
                if smaller_current_stable_area_m2 > 0.0
                else 0.0
            )

            output_gdf.at[
                current_index,
                "temporal_current_stable_overlap_ratio",
            ] = current_stable_overlap_ratio

            output_gdf.at[
                current_index,
                "temporal_stable_overlap_smaller_ratio",
            ] = stable_overlap_smaller_ratio

            if (
                stable_overlap_smaller_ratio
                < min_current_stable_overlap_smaller_ratio
            ):
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_current_stable_"
                    "overlap_too_low"
                )

                continue

            try:
                fill_geometry = (
                    stable_geometry
                    .difference(
                        current_geometry
                    )
                )

            except Exception:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "difference_error"
                )

                continue

            fill_geometry = (
                normalize_polygonal_geometry(
                    fill_geometry
                )
            )

            if fill_geometry is None:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "no_missing_part"
                )

                continue

            fill_area_m2 = float(
                fill_geometry.area
            )

            if (
                fill_area_m2
                < minimum_fill_area_m2
            ):
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "below_minimum_area"
                )

                continue

            provisional_corrected_area_m2 = (
                current_area_m2
                + fill_area_m2
            )

            if (
                provisional_corrected_area_m2
                <= 0.0
            ):
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "invalid_corrected_area"
                )

                continue

            fill_ratio = (
                fill_area_m2
                / provisional_corrected_area_m2
            )

            if fill_ratio > max_fill_ratio:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_"
                    "fill_ratio_too_large"
                )

                continue

            try:
                corrected_geometry = (
                    current_geometry.union(
                        fill_geometry
                    )
                )

            except Exception:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_union_error"
                )

                continue

            corrected_geometry = (
                normalize_polygonal_geometry(
                    corrected_geometry
                )
            )

            if corrected_geometry is None:
                output_gdf.at[
                    current_index,
                    "temporal_fill_status",
                ] = (
                    "not_filled_invalid_"
                    "corrected_geometry"
                )

                continue

            corrected_area_m2 = float(
                corrected_geometry.area
            )

            actual_filled_area_m2 = max(
                0.0,
                corrected_area_m2
                - current_area_m2,
            )

            actual_fill_ratio = (
                actual_filled_area_m2
                / corrected_area_m2
                if corrected_area_m2 > 0.0
                else 0.0
            )

            output_gdf.at[
                current_index,
                "geometry",
            ] = corrected_geometry

            output_gdf.at[
                current_index,
                "temporally_filled_area_m2",
            ] = actual_filled_area_m2

            output_gdf.at[
                current_index,
                "corrected_area_m2",
            ] = corrected_area_m2

            output_gdf.at[
                current_index,
                "corrected_area_km2",
            ] = (
                corrected_area_m2
                / 1_000_000.0
            )

            output_gdf.at[
                current_index,
                "temporal_fill_ratio",
            ] = actual_fill_ratio

            output_gdf.at[
                current_index,
                "temporal_fill_applied",
            ] = True

            output_gdf.at[
                current_index,
                "temporal_fill_status",
            ] = "partially_filled"

            output_gdf.at[
                current_index,
                "area_m2",
            ] = corrected_area_m2

            output_gdf.at[
                current_index,
                "area_km2",
            ] = (
                corrected_area_m2
                / 1_000_000.0
            )

    output_gdf = (
        output_gdf
        .drop(
            columns=[
                "_parsed_date",
            ]
        )
        .sort_values(
            [
                "date",
                "persistent_lake_id",
                "local_candidate_id",
            ]
        )
        .reset_index(drop=True)
    )

    invalid_output_geometry_mask = (
        output_gdf.geometry.notna()
        & ~output_gdf.geometry.is_empty
        & ~output_gdf.geometry.is_valid
    )

    if invalid_output_geometry_mask.any():
        invalid_ids = (
            output_gdf.loc[
                invalid_output_geometry_mask,
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "部分補完後に無効ジオメトリが"
            "あります: "
            + ", ".join(invalid_ids)
        )

    filled_mask = (
        output_gdf[
            "temporal_fill_applied"
        ]
        .fillna(False)
        .astype(bool)
    )

    if filled_mask.any():
        filled_geometry_area_difference = (
            output_gdf.loc[
                filled_mask
            ]
            .geometry.area
            - output_gdf.loc[
                filled_mask,
                "area_m2",
            ]
        ).abs()

        if (
            filled_geometry_area_difference
            .gt(1.0e-6)
            .any()
        ):
            raise ValueError(
                "部分補完後のジオメトリ面積と"
                "area_m2が一致しません。"
            )

    output_gpkg.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        output_gdf.to_file(
            output_gpkg,
            layer=output_layer,
            driver="GPKG",
            index=False,
            mode="w",
        )

    except PermissionError as error:
        raise PermissionError(
            "部分補完GeoPackageを"
            "保存できません。"
            "QGISで開いている場合は"
            "閉じてください: "
            f"{output_gpkg}"
        ) from error

    return output_gdf


def apply_multisensor_related_family_partial_fill(
    tracked_lakes_gdf: gpd.GeoDataFrame,
    output_gpkg: Path,
    output_layer: str,
    max_reference_gap_days: int,
    same_sensor_min_reference_iou: float,
    cross_sensor_min_reference_iou: float,
    min_current_stable_overlap_smaller_ratio: float,
    min_fill_area_by_sensor: dict,
    max_fill_ratio: float,
    allocation_cell_size_m: float,
) -> gpd.GeoDataFrame:

    required_columns = {
        "local_candidate_id",
        "date",
        "sensor_family",
        "persistent_lake_id",
        "persistent_counted_in_lake_count",
        "tracking_event",
        "related_previous_lake_ids",
        "matched_previous_lake_id",
        "merged_into_id_before_restore",
        "area_m2",
        "area_km2",
        "temporally_filled_area_m2",
        "corrected_area_m2",
        "corrected_area_km2",
        "temporal_fill_ratio",
        "temporal_fill_applied",
        "temporal_fill_status",
        "temporal_fill_previous_date",
        "temporal_fill_next_date",
        "temporal_reference_iou",
        "temporal_current_stable_overlap_ratio",
        "temporal_stable_overlap_smaller_ratio",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(tracked_lakes_gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            "関連湖ファミリー補完に必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )

    if tracked_lakes_gdf.empty:
        raise ValueError(
            "関連湖ファミリー補完対象が空です。"
        )

    if tracked_lakes_gdf.crs is None:
        raise ValueError(
            "関連湖ファミリー補完対象にCRSがありません。"
        )

    if not tracked_lakes_gdf.crs.is_projected:
        raise ValueError(
            "関連湖ファミリー補完の面積計算には"
            "投影座標系が必要です。"
        )

    if max_reference_gap_days <= 0:
        raise ValueError(
            "max_reference_gap_daysは1以上にしてください。"
        )

    if allocation_cell_size_m <= 0.0:
        raise ValueError(
            "allocation_cell_size_mは0より大きくしてください。"
        )

    for parameter_name, parameter_value in {
        "same_sensor_min_reference_iou": (
            same_sensor_min_reference_iou
        ),
        "cross_sensor_min_reference_iou": (
            cross_sensor_min_reference_iou
        ),
        "min_current_stable_overlap_smaller_ratio": (
            min_current_stable_overlap_smaller_ratio
        ),
        "max_fill_ratio": max_fill_ratio,
    }.items():
        if not 0.0 <= float(parameter_value) <= 1.0:
            raise ValueError(
                f"{parameter_name}は0以上1以下にしてください。"
            )

    for sensor_family, minimum_area in (
        min_fill_area_by_sensor.items()
    ):
        if float(minimum_area) < 0.0:
            raise ValueError(
                "最低補完面積は0以上にしてください: "
                f"{sensor_family}"
            )

    def normalize_polygonal_geometry(geometry):
        if geometry is None or geometry.is_empty:
            return None

        try:
            if not geometry.is_valid:
                geometry = geometry.buffer(0)

        except Exception:
            return None

        if geometry is None or geometry.is_empty:
            return None

        if geometry.geom_type in {
            "Polygon",
            "MultiPolygon",
        }:
            return geometry

        polygon_parts = []

        if hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                normalized_part = (
                    normalize_polygonal_geometry(part)
                )

                if normalized_part is None:
                    continue

                if normalized_part.geom_type == "Polygon":
                    polygon_parts.append(normalized_part)

                elif normalized_part.geom_type == "MultiPolygon":
                    polygon_parts.extend(
                        list(normalized_part.geoms)
                    )

        if not polygon_parts:
            return None

        try:
            union_geometry = unary_union(polygon_parts)

        except Exception:
            return None

        if union_geometry.is_empty:
            return None

        return union_geometry

    def union_rows(rows: gpd.GeoDataFrame):
        polygon_geometries = []

        for geometry in rows.geometry:
            normalized_geometry = (
                normalize_polygonal_geometry(geometry)
            )

            if normalized_geometry is not None:
                polygon_geometries.append(
                    normalized_geometry
                )

        if not polygon_geometries:
            return None

        try:
            return normalize_polygonal_geometry(
                unary_union(polygon_geometries)
            )

        except Exception:
            return None

    def calculate_iou(first_geometry, second_geometry):
        try:
            intersection_area = float(
                first_geometry.intersection(
                    second_geometry
                ).area
            )
            union_area = float(
                first_geometry.union(
                    second_geometry
                ).area
            )

        except Exception:
            return 0.0

        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area

    def split_relation_ids(value) -> set[str]:
        if value is None or pd.isna(value):
            return set()

        return {
            relation_id.strip()
            for relation_id in str(value).split(";")
            if relation_id.strip()
        }

    def create_relation_components(
        window_rows: gpd.GeoDataFrame,
    ) -> list[set[str]]:
        adjacency: dict[str, set[str]] = {}

        for _, row in window_rows.iterrows():
            relation_ids = {
                str(row["persistent_lake_id"]).strip()
            }

            for relation_column in [
                "related_previous_lake_ids",
                "matched_previous_lake_id",
                "merged_into_id_before_restore",
            ]:
                relation_ids.update(
                    split_relation_ids(
                        row.get(relation_column, "")
                    )
                )

            relation_ids.discard("")

            for relation_id in relation_ids:
                adjacency.setdefault(
                    relation_id,
                    set(),
                )
                adjacency[relation_id].update(
                    relation_ids - {relation_id}
                )

        components = []
        visited_ids: set[str] = set()

        for start_id in sorted(adjacency):
            if start_id in visited_ids:
                continue

            component_ids: set[str] = set()
            pending_ids = [start_id]

            while pending_ids:
                current_id = pending_ids.pop()

                if current_id in visited_ids:
                    continue

                visited_ids.add(current_id)
                component_ids.add(current_id)
                pending_ids.extend(
                    sorted(
                        adjacency.get(current_id, set())
                        - visited_ids
                    )
                )

            components.append(component_ids)

        return components

    def allocate_fill_geometry(
        fill_geometry,
        current_rows: gpd.GeoDataFrame,
    ) -> dict[int, object]:
        lake_geometries = {}
        lake_areas = {}
        lake_ids = {}

        for row_index, row in current_rows.iterrows():
            lake_geometry = normalize_polygonal_geometry(
                row.geometry
            )

            if lake_geometry is None:
                continue

            lake_geometries[row_index] = lake_geometry
            lake_areas[row_index] = float(
                lake_geometry.area
            )
            lake_ids[row_index] = str(
                row["persistent_lake_id"]
            )

        if not lake_geometries:
            return {}

        allocated_parts = {
            row_index: []
            for row_index in lake_geometries
        }

        min_x, min_y, max_x, max_y = (
            fill_geometry.bounds
        )
        start_x = (
            np.floor(min_x / allocation_cell_size_m)
            * allocation_cell_size_m
        )
        start_y = (
            np.floor(min_y / allocation_cell_size_m)
            * allocation_cell_size_m
        )

        for x_coordinate in np.arange(
            start_x,
            max_x,
            allocation_cell_size_m,
        ):
            for y_coordinate in np.arange(
                start_y,
                max_y,
                allocation_cell_size_m,
            ):
                cell_geometry = box(
                    x_coordinate,
                    y_coordinate,
                    x_coordinate + allocation_cell_size_m,
                    y_coordinate + allocation_cell_size_m,
                )

                try:
                    cell_fill_part = (
                        cell_geometry.intersection(
                            fill_geometry
                        )
                    )

                except Exception:
                    continue

                cell_fill_part = normalize_polygonal_geometry(
                    cell_fill_part
                )

                if cell_fill_part is None:
                    continue

                representative_point = (
                    cell_fill_part.representative_point()
                )
                nearest_index = min(
                    lake_geometries,
                    key=lambda candidate_index: (
                        representative_point.distance(
                            lake_geometries[candidate_index]
                        ),
                        -lake_areas[candidate_index],
                        lake_ids[candidate_index],
                    ),
                )
                allocated_parts[nearest_index].append(
                    cell_fill_part
                )

        allocated_geometries = {}

        for row_index, geometry_parts in (
            allocated_parts.items()
        ):
            if not geometry_parts:
                continue

            allocated_geometry = (
                normalize_polygonal_geometry(
                    unary_union(geometry_parts)
                )
            )

            if allocated_geometry is not None:
                allocated_geometries[row_index] = (
                    allocated_geometry
                )

        return allocated_geometries

    output_gdf = (
        tracked_lakes_gdf.copy()
        .reset_index(drop=True)
    )
    output_gdf["_parsed_date"] = (
        pd.to_datetime(
            output_gdf["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if output_gdf["_parsed_date"].isna().any():
        raise ValueError(
            "関連湖ファミリー補完で"
            "日付を解釈できない行があります。"
        )

    output_gdf["persistent_lake_id"] = (
        output_gdf["persistent_lake_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    output_gdf["related_family_fill_applied"] = False
    output_gdf["related_family_fill_status"] = (
        "not_applicable_unassigned"
    )
    output_gdf["related_family_fill_primary_id"] = ""
    output_gdf["related_family_members"] = ""
    output_gdf["related_family_direct_area_m2"] = np.nan
    output_gdf["related_family_filled_area_m2"] = 0.0
    output_gdf["related_family_corrected_area_m2"] = np.nan

    assigned_mask = (
        output_gdf["persistent_lake_id"].ne("")
        & output_gdf[
            "persistent_counted_in_lake_count"
        ].fillna(False).astype(bool)
        & output_gdf.geometry.notna()
        & ~output_gdf.geometry.is_empty
    )
    output_gdf.loc[
        assigned_mask,
        "related_family_fill_status",
    ] = "not_filled_no_family_window"

    direct_rows = output_gdf[assigned_mask].copy()
    event_names = {
        "merge",
        "merge_after_gap",
        "split_primary",
        "split_new",
        "complex_merge_split",
    }
    observation_dates = sorted(
        direct_rows["_parsed_date"].unique()
    )

    for date_position in range(
        1,
        len(observation_dates) - 1,
    ):
        previous_date = pd.Timestamp(
            observation_dates[date_position - 1]
        )
        current_date = pd.Timestamp(
            observation_dates[date_position]
        )
        next_date = pd.Timestamp(
            observation_dates[date_position + 1]
        )
        previous_rows = direct_rows[
            direct_rows["_parsed_date"].eq(previous_date)
        ]
        current_rows = direct_rows[
            direct_rows["_parsed_date"].eq(current_date)
        ]
        next_rows = direct_rows[
            direct_rows["_parsed_date"].eq(next_date)
        ]
        window_rows = gpd.GeoDataFrame(
            pd.concat(
                [previous_rows, current_rows, next_rows]
            ),
            geometry="geometry",
            crs=output_gdf.crs,
        )

        for component_ids in create_relation_components(
            window_rows
        ):
            component_previous_rows = previous_rows[
                previous_rows["persistent_lake_id"].isin(
                    component_ids
                )
            ]
            component_current_rows = current_rows[
                current_rows["persistent_lake_id"].isin(
                    component_ids
                )
            ]
            component_next_rows = next_rows[
                next_rows["persistent_lake_id"].isin(
                    component_ids
                )
            ]

            if (
                component_previous_rows.empty
                or component_current_rows.empty
                or component_next_rows.empty
            ):
                continue

            component_window_rows = pd.concat(
                [
                    component_previous_rows,
                    component_current_rows,
                    component_next_rows,
                ]
            )
            component_events = set(
                component_window_rows[
                    "tracking_event"
                ].astype(str)
            )

            if not component_events.intersection(event_names):
                continue

            current_indices = list(
                component_current_rows.index
            )
            family_members_text = ";".join(
                sorted(component_ids)
            )
            output_gdf.loc[
                current_indices,
                "related_family_members",
            ] = family_members_text

            if output_gdf.loc[
                current_indices,
                "temporal_fill_applied",
            ].fillna(False).astype(bool).any():
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_already_temporally_filled"
                continue

            if (
                (current_date - previous_date).days
                > max_reference_gap_days
                or (next_date - current_date).days
                > max_reference_gap_days
            ):
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_reference_gap_too_long"
                continue

            previous_geometry = union_rows(
                component_previous_rows
            )
            current_geometry = union_rows(
                component_current_rows
            )
            next_geometry = union_rows(
                component_next_rows
            )

            if (
                previous_geometry is None
                or current_geometry is None
                or next_geometry is None
            ):
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_invalid_family_geometry"
                continue

            sensor_set = set(
                pd.concat(
                    [
                        component_previous_rows[
                            "sensor_family"
                        ],
                        component_current_rows[
                            "sensor_family"
                        ],
                        component_next_rows[
                            "sensor_family"
                        ],
                    ]
                ).astype(str)
            )
            cross_sensor_window = len(sensor_set) > 1
            reference_iou_threshold = (
                cross_sensor_min_reference_iou
                if cross_sensor_window
                else same_sensor_min_reference_iou
            )
            reference_iou = calculate_iou(
                previous_geometry,
                next_geometry,
            )

            if reference_iou < reference_iou_threshold:
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_reference_disagreement"
                continue

            try:
                stable_geometry = (
                    normalize_polygonal_geometry(
                        previous_geometry.intersection(
                            next_geometry
                        )
                    )
                )

            except Exception:
                stable_geometry = None

            if stable_geometry is None:
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_no_stable_water"
                continue

            try:
                stable_intersection_area = float(
                    stable_geometry.intersection(
                        current_geometry
                    ).area
                )

            except Exception:
                stable_intersection_area = 0.0

            current_overlap_ratio = (
                stable_intersection_area
                / float(current_geometry.area)
                if float(current_geometry.area) > 0.0
                else 0.0
            )

            smaller_current_stable_area_m2 = min(
                float(current_geometry.area),
                float(stable_geometry.area),
            )

            stable_overlap_smaller_ratio = (
                stable_intersection_area
                / smaller_current_stable_area_m2
                if smaller_current_stable_area_m2 > 0.0
                else 0.0
            )

            if (
                stable_overlap_smaller_ratio
                < min_current_stable_overlap_smaller_ratio
            ):
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = (
                    "not_filled_current_family_"
                    "stable_overlap_too_low"
                )
                continue

            try:
                fill_geometry = normalize_polygonal_geometry(
                    stable_geometry.difference(
                        current_geometry
                    )
                )

            except Exception:
                fill_geometry = None

            if fill_geometry is None:
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_no_missing_part"
                continue

            fill_area_m2 = float(fill_geometry.area)
            current_sensor_names = set(
                component_current_rows[
                    "sensor_family"
                ].astype(str)
            )
            minimum_fill_area_m2 = max(
                float(
                    min_fill_area_by_sensor.get(
                        sensor_name,
                        min_fill_area_by_sensor.get(
                            "Sentinel-2",
                            0.0,
                        ),
                    )
                )
                for sensor_name in current_sensor_names
            )

            if fill_area_m2 < minimum_fill_area_m2:
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_below_minimum_area"
                continue

            family_direct_area_m2 = float(
                current_geometry.area
            )
            family_corrected_area_m2 = (
                family_direct_area_m2
                + fill_area_m2
            )
            family_fill_ratio = (
                fill_area_m2
                / family_corrected_area_m2
            )

            if family_fill_ratio > max_fill_ratio:
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_fill_ratio_too_large"
                continue

            allocated_fill_geometries = (
                allocate_fill_geometry(
                    fill_geometry,
                    component_current_rows,
                )
            )

            if not allocated_fill_geometries:
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_allocation_failed"
                continue

            allocated_total_area_m2 = sum(
                float(geometry.area)
                for geometry in (
                    allocated_fill_geometries.values()
                )
            )
            allocation_tolerance_m2 = max(
                0.01,
                fill_area_m2 * 1.0e-6,
            )

            if abs(
                allocated_total_area_m2
                - fill_area_m2
            ) > allocation_tolerance_m2:
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_allocation_area_mismatch"
                continue

            summary_index = (
                component_current_rows["area_m2"]
                .astype(float)
                .idxmax()
            )
            summary_id = str(
                output_gdf.at[
                    summary_index,
                    "persistent_lake_id",
                ]
            )
            output_gdf.loc[
                current_indices,
                "related_family_fill_primary_id",
            ] = summary_id
            filled_indices = []

            for row_index, allocated_geometry in (
                allocated_fill_geometries.items()
            ):
                original_geometry = (
                    normalize_polygonal_geometry(
                        output_gdf.at[
                            row_index,
                            "geometry",
                        ]
                    )
                )

                if original_geometry is None:
                    continue

                corrected_geometry = (
                    normalize_polygonal_geometry(
                        original_geometry.union(
                            allocated_geometry
                        )
                    )
                )

                if corrected_geometry is None:
                    continue

                corrected_area_m2 = float(
                    corrected_geometry.area
                )
                allocated_area_m2 = float(
                    allocated_geometry.area
                )
                previous_fill_area_m2 = float(
                    pd.to_numeric(
                        pd.Series(
                            [
                                output_gdf.at[
                                    row_index,
                                    "temporally_filled_area_m2",
                                ]
                            ]
                        ),
                        errors="coerce",
                    ).fillna(0.0).iloc[0]
                )
                total_fill_area_m2 = (
                    previous_fill_area_m2
                    + allocated_area_m2
                )

                output_gdf.at[
                    row_index,
                    "geometry",
                ] = corrected_geometry
                output_gdf.at[
                    row_index,
                    "area_m2",
                ] = corrected_area_m2
                output_gdf.at[
                    row_index,
                    "area_km2",
                ] = corrected_area_m2 / 1_000_000.0
                output_gdf.at[
                    row_index,
                    "temporally_filled_area_m2",
                ] = total_fill_area_m2
                output_gdf.at[
                    row_index,
                    "corrected_area_m2",
                ] = corrected_area_m2
                output_gdf.at[
                    row_index,
                    "corrected_area_km2",
                ] = corrected_area_m2 / 1_000_000.0
                output_gdf.at[
                    row_index,
                    "temporal_fill_ratio",
                ] = total_fill_area_m2 / corrected_area_m2
                output_gdf.at[
                    row_index,
                    "temporal_fill_applied",
                ] = True
                output_gdf.at[
                    row_index,
                    "temporal_fill_status",
                ] = "partially_filled_related_family_distributed"
                output_gdf.at[
                    row_index,
                    "temporal_fill_previous_date",
                ] = previous_date.strftime("%Y-%m-%d")
                output_gdf.at[
                    row_index,
                    "temporal_fill_next_date",
                ] = next_date.strftime("%Y-%m-%d")
                output_gdf.at[
                    row_index,
                    "temporal_reference_iou",
                ] = reference_iou
                output_gdf.at[
                    row_index,
                    "temporal_current_stable_overlap_ratio",
                ] = current_overlap_ratio
                output_gdf.at[
                    row_index,
                    "temporal_stable_overlap_smaller_ratio",
                ] = stable_overlap_smaller_ratio
                output_gdf.at[
                    row_index,
                    "related_family_fill_applied",
                ] = True
                output_gdf.at[
                    row_index,
                    "related_family_fill_status",
                ] = "partially_filled_nearest_lake"
                filled_indices.append(row_index)

            if not filled_indices:
                output_gdf.loc[
                    current_indices,
                    "related_family_fill_status",
                ] = "not_filled_no_valid_allocation"
                continue

            output_gdf.at[
                summary_index,
                "related_family_direct_area_m2",
            ] = family_direct_area_m2
            output_gdf.at[
                summary_index,
                "related_family_filled_area_m2",
            ] = fill_area_m2
            output_gdf.at[
                summary_index,
                "related_family_corrected_area_m2",
            ] = family_corrected_area_m2

            unfilled_indices = [
                row_index
                for row_index in current_indices
                if row_index not in filled_indices
            ]

            if unfilled_indices:
                output_gdf.loc[
                    unfilled_indices,
                    "related_family_fill_status",
                ] = (
                    "included_in_filled_family_without_allocation"
                )

    output_gdf = output_gdf.drop(
        columns=["_parsed_date"]
    )
    output_gdf["date"] = (
        pd.to_datetime(
            output_gdf["date"],
            errors="raise",
        ).dt.strftime("%Y-%m-%d")
    )

    invalid_geometry_mask = (
        output_gdf.geometry.notna()
        & ~output_gdf.geometry.is_empty
        & ~output_gdf.geometry.is_valid
    )

    if invalid_geometry_mask.any():
        invalid_ids = output_gdf.loc[
            invalid_geometry_mask,
            "local_candidate_id",
        ].astype(str).tolist()
        raise ValueError(
            "関連湖ファミリー補完後に"
            "無効ジオメトリがあります: "
            + ", ".join(invalid_ids)
        )

    output_gpkg.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        output_gdf.to_file(
            output_gpkg,
            layer=output_layer,
            driver="GPKG",
            index=False,
            mode="w",
        )

    except PermissionError as error:
        raise PermissionError(
            "関連湖ファミリー補完GeoPackageを"
            "保存できません。QGISで開いている場合は"
            "閉じてください: "
            f"{output_gpkg}"
        ) from error

    return output_gdf


def reconcile_multisensor_connected_ids_after_fill(
    tracked_lakes_gdf: gpd.GeoDataFrame,
    output_gpkg: Path,
    output_layer: str,
) -> gpd.GeoDataFrame:

    required_columns = {
        "local_candidate_id",
        "date",
        "persistent_lake_id",
        "persistent_counted_in_lake_count",
        "area_m2",
        "area_km2",
        "direct_area_m2",
        "direct_area_km2",
        "temporally_filled_area_m2",
        "corrected_area_m2",
        "corrected_area_km2",
        "temporal_fill_applied",
        "temporal_fill_status",
        "related_family_fill_applied",
        "related_family_fill_status",
        "related_family_direct_area_m2",
        "related_family_filled_area_m2",
        "related_family_corrected_area_m2",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(tracked_lakes_gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            "補完後トポロジー整合に必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )

    if tracked_lakes_gdf.empty:
        raise ValueError(
            "補完後トポロジー整合対象が空です。"
        )

    if tracked_lakes_gdf.crs is None:
        raise ValueError(
            "補完後トポロジー整合対象にCRSがありません。"
        )

    if not tracked_lakes_gdf.crs.is_projected:
        raise ValueError(
            "補完後トポロジー整合には投影座標系が必要です。"
        )

    def normalize_polygonal_geometry(geometry):
        if geometry is None or geometry.is_empty:
            return None

        try:
            if not geometry.is_valid:
                geometry = geometry.buffer(0)

        except Exception:
            return None

        if geometry is None or geometry.is_empty:
            return None

        if geometry.geom_type in {
            "Polygon",
            "MultiPolygon",
        }:
            return geometry

        polygon_parts = []

        if hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                normalized_part = (
                    normalize_polygonal_geometry(part)
                )

                if normalized_part is None:
                    continue

                if normalized_part.geom_type == "Polygon":
                    polygon_parts.append(normalized_part)

                elif normalized_part.geom_type == "MultiPolygon":
                    polygon_parts.extend(
                        list(normalized_part.geoms)
                    )

        if not polygon_parts:
            return None

        try:
            return normalize_polygonal_geometry(
                unary_union(polygon_parts)
            )

        except Exception:
            return None

    output_gdf = (
        tracked_lakes_gdf.copy()
        .reset_index(drop=True)
    )
    output_gdf["_parsed_date"] = (
        pd.to_datetime(
            output_gdf["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if output_gdf["_parsed_date"].isna().any():
        raise ValueError(
            "補完後トポロジー整合で"
            "日付を解釈できない行があります。"
        )

    output_gdf["persistent_lake_id"] = (
        output_gdf["persistent_lake_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    # この関数は補完後と小断片追加後の2回呼び出す。
    # 2回目に1回目の監査情報を消さないよう、既存列は保持する。
    if "topology_reconciled" not in output_gdf.columns:
        output_gdf["topology_reconciled"] = False
    else:
        output_gdf["topology_reconciled"] = (
            output_gdf["topology_reconciled"]
            .fillna(False)
            .astype(bool)
        )

    for topology_column in {
        "topology_event",
        "topology_survivor_id",
        "topology_merged_member_ids",
    }:
        if topology_column not in output_gdf.columns:
            output_gdf[topology_column] = ""
        else:
            output_gdf[topology_column] = (
                output_gdf[topology_column]
                .fillna("")
                .astype(str)
            )

    valid_geometry_mask = (
        output_gdf.geometry.notna()
        & ~output_gdf.geometry.is_empty
    )

    # 接続判定には未割当候補も含める。補完前には離れていた
    # 継続候補が、補完後の永続湖へ接続する場合があるため。
    reference_rows = output_gdf[
        valid_geometry_mask
    ].copy()

    assigned_reference_rows = output_gdf[
        valid_geometry_mask
        & output_gdf["persistent_lake_id"].ne("")
        & output_gdf[
            "persistent_counted_in_lake_count"
        ].fillna(False).astype(bool)
    ].copy()

    observation_ids = (
        reference_rows["observation_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if observation_ids.eq("").any():
        raise ValueError(
            "補完後トポロジー整合で"
            "observation_idが空の行があります。"
        )

    reference_rows["_observation_id"] = observation_ids

    # 同日の異なるセンサー観測を混ぜないよう、日付ではなく
    # observation_id単位で接続関係を判定する。
    for observation_id in sorted(
        reference_rows["_observation_id"].unique()
    ):
        current_rows = reference_rows[
            reference_rows["_observation_id"].eq(
                observation_id
            )
        ]

        if len(current_rows) < 2:
            continue

        current_dates = current_rows[
            "_parsed_date"
        ].dropna().unique()

        if len(current_dates) != 1:
            raise ValueError(
                "同じobservation_idに複数の日付があります: "
                f"{observation_id}"
            )

        current_date = pd.Timestamp(current_dates[0])

        current_indices = list(current_rows.index)
        adjacency = {
            row_index: set()
            for row_index in current_indices
        }

        for position, first_index in enumerate(
            current_indices
        ):
            first_geometry = normalize_polygonal_geometry(
                current_rows.at[first_index, "geometry"]
            )

            if first_geometry is None:
                continue

            for second_index in current_indices[
                position + 1:
            ]:
                second_geometry = (
                    normalize_polygonal_geometry(
                        current_rows.at[
                            second_index,
                            "geometry",
                        ]
                    )
                )

                if second_geometry is None:
                    continue

                try:
                    connected = first_geometry.intersects(
                        second_geometry
                    )

                except Exception:
                    connected = False

                if connected:
                    adjacency[first_index].add(second_index)
                    adjacency[second_index].add(first_index)

        connected_components = []
        visited_indices: set[int] = set()

        for start_index in current_indices:
            if start_index in visited_indices:
                continue

            component_indices = []
            pending_indices = [start_index]

            while pending_indices:
                row_index = pending_indices.pop()

                if row_index in visited_indices:
                    continue

                visited_indices.add(row_index)
                component_indices.append(row_index)
                pending_indices.extend(
                    adjacency[row_index]
                    - visited_indices
                )

            if len(component_indices) >= 2:
                connected_components.append(
                    component_indices
                )

        for component_indices in connected_components:
            component_rows = output_gdf.loc[
                component_indices
            ]

            assigned_component_mask = (
                component_rows[
                    "persistent_lake_id"
                ].astype(str).str.strip().ne("")
                & component_rows[
                    "persistent_counted_in_lake_count"
                ].fillna(False).astype(bool)
            )

            assigned_component_rows = (
                component_rows.loc[
                    assigned_component_mask
                ]
            )

            # 永続IDが一つもない連結成分は、ここでは新規湖に
            # せず、従来どおり未割当候補として残す。
            if assigned_component_rows.empty:
                continue

            fill_applied = (
                component_rows[
                    "temporal_fill_applied"
                ].fillna(False).astype(bool).any()
                or component_rows[
                    "related_family_fill_applied"
                ].fillna(False).astype(bool).any()
            )

            component_ids = set(
                assigned_component_rows[
                    "persistent_lake_id"
                ].astype(str)
            )

            # 直接観測または小断片追加で接続したIDでは、履歴の
            # 最も古いIDを残す。新しく発行されたIDが既存湖を
            # 上書きすることを防ぐ。
            if not fill_applied and len(component_ids) >= 2:
                component_history = assigned_reference_rows[
                    assigned_reference_rows[
                        "persistent_lake_id"
                    ].isin(component_ids)
                ]
                first_dates_by_id = (
                    component_history.groupby(
                        "persistent_lake_id"
                    )["_parsed_date"].min()
                )
                oldest_date = first_dates_by_id.min()
                oldest_ids = set(
                    first_dates_by_id[
                        first_dates_by_id.eq(oldest_date)
                    ].index.astype(str)
                )
            else:
                oldest_ids = set()

            future_rows = assigned_reference_rows[
                assigned_reference_rows["_parsed_date"].gt(
                    pd.Timestamp(current_date)
                )
                & assigned_reference_rows[
                    "persistent_lake_id"
                ].isin(component_ids)
            ]
            survivor_id = ""

            if len(oldest_ids) == 1:
                survivor_id = next(iter(oldest_ids))

            elif fill_applied and not future_rows.empty:
                nearest_future_date = future_rows[
                    "_parsed_date"
                ].min()
                future_ids = future_rows.loc[
                    future_rows["_parsed_date"].eq(
                        nearest_future_date
                    ),
                    "persistent_lake_id",
                ].astype(str).unique()

                if len(future_ids) == 1:
                    survivor_id = str(future_ids[0])

            if survivor_id not in component_ids:
                survivor_id = ""

            if survivor_id:
                survivor_index = assigned_component_rows.index[
                    assigned_component_rows[
                        "persistent_lake_id"
                    ].astype(str).eq(survivor_id)
                ][0]

            else:
                direct_areas = pd.to_numeric(
                    assigned_component_rows["direct_area_m2"],
                    errors="coerce",
                ).fillna(0.0)
                maximum_direct_area = direct_areas.max()
                survivor_candidates = assigned_component_rows.loc[
                    direct_areas.eq(maximum_direct_area)
                ].copy()
                survivor_candidates["_id_sort"] = (
                    survivor_candidates[
                        "persistent_lake_id"
                    ].astype(str)
                )
                survivor_index = (
                    survivor_candidates.sort_values(
                        ["_id_sort", "local_candidate_id"]
                    ).index[0]
                )
                survivor_id = str(
                    output_gdf.at[
                        survivor_index,
                        "persistent_lake_id",
                    ]
                )

            component_geometries = []

            for component_index in component_indices:
                component_geometry = (
                    normalize_polygonal_geometry(
                        output_gdf.at[
                            component_index,
                            "geometry",
                        ]
                    )
                )

                if component_geometry is not None:
                    component_geometries.append(
                        component_geometry
                    )

            if not component_geometries:
                continue

            merged_geometry = normalize_polygonal_geometry(
                unary_union(component_geometries)
            )

            if merged_geometry is None:
                continue

            merged_area_m2 = float(merged_geometry.area)
            total_direct_area_m2 = float(
                pd.to_numeric(
                    component_rows["direct_area_m2"],
                    errors="coerce",
                ).fillna(0.0).sum()
            )
            total_filled_area_m2 = float(
                pd.to_numeric(
                    component_rows[
                        "temporally_filled_area_m2"
                    ],
                    errors="coerce",
                ).fillna(0.0).sum()
            )
            merged_member_ids = ";".join(
                sorted(component_ids)
            )

            unassigned_member_exists = bool(
                (~assigned_component_mask).any()
            )

            if unassigned_member_exists:
                topology_survivor_event = (
                    "connected_unassigned_candidate_merged"
                )
            elif fill_applied:
                topology_survivor_event = (
                    "connected_ids_merged_after_fill"
                )
            else:
                topology_survivor_event = (
                    "direct_connected_ids_merged"
                )

            if fill_applied:
                topology_fill_status = (
                    "topology_merged_after_temporal_fill"
                )
            else:
                topology_fill_status = (
                    "topology_merged_direct_connection"
                )

            for column_name, column_value in {
                "geometry": merged_geometry,
                "area_m2": merged_area_m2,
                "area_km2": (
                    merged_area_m2 / 1_000_000.0
                ),
                "direct_area_m2": (
                    total_direct_area_m2
                ),
                "direct_area_km2": (
                    total_direct_area_m2
                    / 1_000_000.0
                ),
                "temporally_filled_area_m2": (
                    total_filled_area_m2
                ),
                "corrected_area_m2": merged_area_m2,
                "corrected_area_km2": (
                    merged_area_m2 / 1_000_000.0
                ),
                "temporal_fill_ratio": (
                    total_filled_area_m2 / merged_area_m2
                    if merged_area_m2 > 0.0
                    else 0.0
                ),
                "temporal_fill_status": (
                    topology_fill_status
                ),
                "related_family_direct_area_m2": (
                    total_direct_area_m2
                ),
                "related_family_filled_area_m2": (
                    total_filled_area_m2
                ),
                "related_family_corrected_area_m2": (
                    merged_area_m2
                ),
                "topology_reconciled": True,
                "topology_event": topology_survivor_event,
                "topology_survivor_id": survivor_id,
                "topology_merged_member_ids": (
                    merged_member_ids
                ),
            }.items():
                output_gdf.at[
                    survivor_index,
                    column_name,
                ] = column_value

            secondary_indices = [
                component_index
                for component_index in component_indices
                if component_index != survivor_index
            ]

            for secondary_index in secondary_indices:
                output_gdf.at[secondary_index, "geometry"] = None
                output_gdf.at[secondary_index, "area_m2"] = np.nan
                output_gdf.at[secondary_index, "area_km2"] = np.nan
                output_gdf.at[
                    secondary_index,
                    "corrected_area_m2",
                ] = np.nan
                output_gdf.at[
                    secondary_index,
                    "corrected_area_km2",
                ] = np.nan
                output_gdf.at[
                    secondary_index,
                    "temporally_filled_area_m2",
                ] = 0.0
                output_gdf.at[
                    secondary_index,
                    "temporal_fill_applied",
                ] = False
                output_gdf.at[
                    secondary_index,
                    "temporal_fill_status",
                ] = "merged_into_connected_lake"
                output_gdf.at[
                    secondary_index,
                    "related_family_fill_applied",
                ] = False
                output_gdf.at[
                    secondary_index,
                    "related_family_fill_status",
                ] = "merged_into_connected_lake"
                output_gdf.at[
                    secondary_index,
                    "persistent_counted_in_lake_count",
                ] = False
                output_gdf.at[
                    secondary_index,
                    "topology_reconciled",
                ] = True
                output_gdf.at[
                    secondary_index,
                    "topology_event",
                ] = "merged_into_connected_lake"
                output_gdf.at[
                    secondary_index,
                    "topology_survivor_id",
                ] = survivor_id
                output_gdf.at[
                    secondary_index,
                    "topology_merged_member_ids",
                ] = merged_member_ids

    output_gdf = output_gdf.drop(
        columns=["_parsed_date"]
    )
    output_gdf["date"] = (
        pd.to_datetime(
            output_gdf["date"],
            errors="raise",
        ).dt.strftime("%Y-%m-%d")
    )

    output_gpkg.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        output_gdf.to_file(
            output_gpkg,
            layer=output_layer,
            driver="GPKG",
            index=False,
            mode="w",
        )

    except PermissionError as error:
        raise PermissionError(
            "補完後トポロジー整合GeoPackageを"
            "保存できません。QGISで開いている場合は"
            "閉じてください: "
            f"{output_gpkg}"
        ) from error

    return output_gdf
