from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union


def validate_shadow_candidates_temporally(
    grouped_candidate_gdf: gpd.GeoDataFrame,
    output_gpkg: Path,
    output_layer: str,
    max_gap_days: int,
    min_overlap_smaller_ratio: float,
    min_iou: float,
    min_area_ratio: float,
    require_both_sides: bool,
    match_distance_by_sensor: dict,
    enable_landsat_geometry_clip: bool,
    landsat_geometry_clip_min_qa_ratio: float,
    landsat_geometry_clip_buffer_m: float,
    landsat_geometry_clip_max_retained_ratio: float,
    landsat_geometry_clip_min_area_m2: float,
) -> gpd.GeoDataFrame:

    required_columns = {
        "local_candidate_id",
        "observation_id",
        "date",
        "sensor_family",
        "candidate_role",
        "can_create_new_id",
        "counted_in_lake_count",
        "requires_shadow_validation",
        "area_m2",
        "shadow_only",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(grouped_candidate_gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            "雲影時系列確認に必要な列がありません: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if grouped_candidate_gdf.empty:
        raise ValueError(
            "雲影確認対象のGeoDataFrameが空です。"
        )

    if grouped_candidate_gdf.crs is None:
        raise ValueError(
            "雲影確認対象にCRSが"
            "設定されていません。"
        )

    if not grouped_candidate_gdf.crs.is_projected:
        raise ValueError(
            "雲影候補の距離確認には"
            "投影座標系が必要です。"
        )

    if max_gap_days <= 0:
        raise ValueError(
            "max_gap_daysは"
            "0より大きくしてください。"
        )

    if not 0.0 <= min_overlap_smaller_ratio <= 1.0:
        raise ValueError(
            "min_overlap_smaller_ratioは"
            "0から1の範囲で指定してください。"
        )

    if not 0.0 <= min_iou <= 1.0:
        raise ValueError(
            "min_iouは"
            "0から1の範囲で指定してください。"
        )

    if not 0.0 <= min_area_ratio <= 1.0:
        raise ValueError(
            "min_area_ratioは"
            "0から1の範囲で指定してください。"
        )

    if not 0.0 <= landsat_geometry_clip_min_qa_ratio <= 1.0:
        raise ValueError(
            "landsat_geometry_clip_min_qa_ratioは"
            "0から1の範囲で指定してください。"
        )

    if landsat_geometry_clip_buffer_m < 0.0:
        raise ValueError(
            "landsat_geometry_clip_buffer_mは"
            "0以上にしてください。"
        )

    if not 0.0 <= landsat_geometry_clip_max_retained_ratio <= 1.0:
        raise ValueError(
            "landsat_geometry_clip_max_retained_ratioは"
            "0から1の範囲で指定してください。"
        )

    if landsat_geometry_clip_min_area_m2 < 0.0:
        raise ValueError(
            "landsat_geometry_clip_min_area_m2は"
            "0以上にしてください。"
        )

    def value_to_bool(
        value,
    ) -> bool:

        if isinstance(
            value,
            (
                bool,
                np.bool_,
            ),
        ):
            return bool(value)

        if pd.isna(value):
            return False

        return (
            str(value)
            .strip()
            .lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

    working_gdf = (
        grouped_candidate_gdf
        .copy()
        .reset_index(drop=True)
    )

    working_gdf[
        "_parsed_date"
    ] = pd.to_datetime(
        working_gdf["date"],
        errors="coerce",
    )

    invalid_date_mask = (
        working_gdf[
            "_parsed_date"
        ]
        .isna()
    )

    if invalid_date_mask.any():
        invalid_ids = (
            working_gdf.loc[
                invalid_date_mask,
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "日付を解釈できない候補があります: "
            + ", ".join(
                invalid_ids
            )
        )

    working_gdf[
        "area_m2"
    ] = pd.to_numeric(
        working_gdf[
            "area_m2"
        ],
        errors="coerce",
    )

    invalid_area_mask = (
        working_gdf[
            "area_m2"
        ]
        .isna()
        | working_gdf[
            "area_m2"
        ]
        .le(0.0)
    )

    if invalid_area_mask.any():
        invalid_ids = (
            working_gdf.loc[
                invalid_area_mask,
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "面積が不正な候補があります: "
            + ", ".join(
                invalid_ids
            )
        )

    invalid_geometry_mask = (
        working_gdf.geometry.isna()
        | working_gdf.geometry.is_empty
        | ~working_gdf.geometry.is_valid
    )

    if invalid_geometry_mask.any():
        invalid_ids = (
            working_gdf.loc[
                invalid_geometry_mask,
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "無効なジオメトリがあります: "
            + ", ".join(
                invalid_ids
            )
        )

    working_gdf[
        "requires_shadow_validation"
    ] = (
        working_gdf[
            "requires_shadow_validation"
        ]
        .apply(
            value_to_bool
        )
    )

    working_gdf[
        "shadow_only"
    ] = (
        working_gdf[
            "shadow_only"
        ]
        .apply(
            value_to_bool
        )
    )

    working_gdf[
        "shadow_temporal_supported"
    ] = False

    working_gdf[
        "shadow_support_side"
    ] = ""

    working_gdf[
        "shadow_filter_status"
    ] = "not_shadow_candidate"

    working_gdf[
        "shadow_before_reference_ids"
    ] = ""

    working_gdf[
        "shadow_after_reference_ids"
    ] = ""

    working_gdf[
        "shadow_best_overlap_smaller_ratio"
    ] = np.nan

    working_gdf[
        "shadow_best_iou"
    ] = np.nan

    working_gdf[
        "shadow_min_distance_m"
    ] = np.nan

    working_gdf[
        "shadow_best_area_ratio"
    ] = np.nan

    # 時系列支持範囲によるLandsat形状クリップの監査列。
    # 適用されなかった候補にも初期値を持たせ、最終GPKGで
    # 形状が変更されたかを確認できるようにする。
    working_gdf[
        "qa_temporal_clip_applied"
    ] = False

    working_gdf[
        "qa_temporal_clip_status"
    ] = "not_applicable"

    working_gdf[
        "qa_temporal_clip_original_area_m2"
    ] = working_gdf["area_m2"].astype(float)

    working_gdf[
        "qa_temporal_clip_removed_area_m2"
    ] = 0.0

    working_gdf[
        "qa_temporal_clip_retained_ratio"
    ] = 1.0

    working_gdf[
        "_qa_temporal_clip_keep"
    ] = True

    shadow_candidate_indices = (
        working_gdf.index[
            working_gdf[
                "requires_shadow_validation"
            ]
        ]
        .tolist()
    )

    clear_reference_indices = (
        working_gdf.index[
            ~working_gdf[
                "requires_shadow_validation"
            ]
            & ~working_gdf[
                "shadow_only"
            ]
        ]
        .tolist()
    )

    # 全域解析では雲影候補と参照候補が数千件になる。
    # 全組み合わせで距離・交差を計算せず、時系列照合距離の
    # 最大値内にある候補だけを空間インデックスで取得する。
    clear_reference_gdf = working_gdf.loc[
        clear_reference_indices
    ]
    clear_reference_spatial_index = (
        clear_reference_gdf.sindex
    )
    maximum_configured_match_distance_m = max(
        float(distance_m)
        for distance_m in match_distance_by_sensor.values()
    )

    for current_index in (
        shadow_candidate_indices
    ):
        current_row = (
            working_gdf.loc[
                current_index
            ]
        )

        current_date = (
            current_row[
                "_parsed_date"
            ]
        )

        current_geometry = (
            current_row.geometry
        )

        current_area_m2 = float(
            current_row[
                "area_m2"
            ]
        )

        current_sensor = str(
            current_row[
                "sensor_family"
            ]
        )

        if (
            current_sensor
            not in match_distance_by_sensor
        ):
            raise ValueError(
                "雲影照合距離が設定されていない"
                "センサーです: "
                f"{current_sensor}"
            )

        supported_before = False
        supported_after = False

        before_reference_ids: list[str] = []
        after_reference_ids: list[str] = []

        best_overlap_smaller_ratio = 0.0
        best_iou = 0.0
        minimum_distance_m = np.inf
        best_area_ratio = 0.0

        spatial_query_geometry = (
            current_geometry.buffer(
                max(
                    float(
                        match_distance_by_sensor[
                            current_sensor
                        ]
                    ),
                    maximum_configured_match_distance_m,
                )
            )
        )

        candidate_reference_positions = (
            clear_reference_spatial_index.query(
                spatial_query_geometry,
                predicate="intersects",
            )
        )

        for reference_position in (
            candidate_reference_positions
        ):
            reference_index = (
                clear_reference_gdf.index[
                    int(reference_position)
                ]
            )
            reference_row = (
                working_gdf.loc[
                    reference_index
                ]
            )

            reference_date = (
                reference_row[
                    "_parsed_date"
                ]
            )

            if reference_date == current_date:
                continue

            day_difference = abs(
                (
                    reference_date
                    - current_date
                ).days
            )

            if day_difference > max_gap_days:
                continue

            reference_sensor = str(
                reference_row[
                    "sensor_family"
                ]
            )

            if (
                reference_sensor
                not in match_distance_by_sensor
            ):
                raise ValueError(
                    "雲影照合距離が設定されていない"
                    "センサーです: "
                    f"{reference_sensor}"
                )

            reference_geometry = (
                reference_row.geometry
            )

            reference_area_m2 = float(
                reference_row[
                    "area_m2"
                ]
            )

            intersection_area_m2 = float(
                current_geometry.intersection(
                    reference_geometry
                ).area
            )

            smaller_area_m2 = min(
                current_area_m2,
                reference_area_m2,
            )

            larger_area_m2 = max(
                current_area_m2,
                reference_area_m2,
            )

            union_area_m2 = (
                current_area_m2
                + reference_area_m2
                - intersection_area_m2
            )

            if smaller_area_m2 > 0.0:
                overlap_smaller_ratio = (
                    intersection_area_m2
                    / smaller_area_m2
                )

            else:
                overlap_smaller_ratio = 0.0

            if union_area_m2 > 0.0:
                iou = (
                    intersection_area_m2
                    / union_area_m2
                )

            else:
                iou = 0.0

            if larger_area_m2 > 0.0:
                area_ratio = (
                    smaller_area_m2
                    / larger_area_m2
                )

            else:
                area_ratio = 0.0

            geometry_distance_m = float(
                current_geometry.distance(
                    reference_geometry
                )
            )

            maximum_distance_m = max(
                float(
                    match_distance_by_sensor[
                        current_sensor
                    ]
                ),
                float(
                    match_distance_by_sensor[
                        reference_sensor
                    ]
                ),
            )

            # 雲影候補が正常な小湖を内包しただけで支持済みになることを
            # 防ぐため、重なり条件にも面積比の下限を適用する。
            overlap_match = bool(
                (
                    overlap_smaller_ratio
                    >= min_overlap_smaller_ratio
                    or iou >= min_iou
                )
                and area_ratio >= min_area_ratio
            )

            distance_match = bool(
                geometry_distance_m
                <= maximum_distance_m
                and area_ratio
                >= min_area_ratio
            )

            temporal_match = bool(
                overlap_match
                or distance_match
            )

            if not temporal_match:
                continue

            reference_id = str(
                reference_row[
                    "local_candidate_id"
                ]
            )

            if reference_date < current_date:
                supported_before = True

                before_reference_ids.append(
                    reference_id
                )

            elif reference_date > current_date:
                supported_after = True

                after_reference_ids.append(
                    reference_id
                )

            best_overlap_smaller_ratio = max(
                best_overlap_smaller_ratio,
                overlap_smaller_ratio,
            )

            best_iou = max(
                best_iou,
                iou,
            )

            minimum_distance_m = min(
                minimum_distance_m,
                geometry_distance_m,
            )

            best_area_ratio = max(
                best_area_ratio,
                area_ratio,
            )

        if require_both_sides:
            temporally_supported = bool(
                supported_before
                and supported_after
            )

        else:
            temporally_supported = bool(
                supported_before
                or supported_after
            )

        support_sides: list[str] = []

        if supported_before:
            support_sides.append(
                "before"
            )

        if supported_after:
            support_sides.append(
                "after"
            )

        working_gdf.at[
            current_index,
            "shadow_temporal_supported",
        ] = temporally_supported

        working_gdf.at[
            current_index,
            "shadow_support_side",
        ] = ";".join(
            support_sides
        )

        working_gdf.at[
            current_index,
            "shadow_before_reference_ids",
        ] = ";".join(
            sorted(
                set(
                    before_reference_ids
                )
            )
        )

        working_gdf.at[
            current_index,
            "shadow_after_reference_ids",
        ] = ";".join(
            sorted(
                set(
                    after_reference_ids
                )
            )
        )

        if (
            before_reference_ids
            or after_reference_ids
        ):
            working_gdf.at[
                current_index,
                "shadow_best_overlap_smaller_ratio",
            ] = best_overlap_smaller_ratio

            working_gdf.at[
                current_index,
                "shadow_best_iou",
            ] = best_iou

            working_gdf.at[
                current_index,
                "shadow_min_distance_m",
            ] = minimum_distance_m

            working_gdf.at[
                current_index,
                "shadow_best_area_ratio",
            ] = best_area_ratio

        if temporally_supported:
            working_gdf.at[
                current_index,
                "shadow_filter_status",
            ] = "kept_temporal_match"

        else:
            working_gdf.at[
                current_index,
                "shadow_filter_status",
            ] = "removed_unconfirmed"

    def normalize_polygonal_geometry(geometry):
        """交差結果から有効なPolygon/MultiPolygonだけを返す。"""

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
                normalized_part = normalize_polygonal_geometry(
                    part
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

    clipped_candidate_count = 0
    removed_by_clip_count = 0
    removed_by_clip_area_m2 = 0.0

    if enable_landsat_geometry_clip:
        reference_geometry_by_id = {
            str(row["local_candidate_id"]): row.geometry
            for _, row in working_gdf.iterrows()
        }

        supported_landsat_indices = working_gdf.index[
            working_gdf["sensor_family"].astype(str).eq(
                "Landsat"
            )
            & working_gdf["requires_shadow_validation"]
            & working_gdf["shadow_temporal_supported"]
        ].tolist()

        for current_index in supported_landsat_indices:
            current_qa_ratio = pd.to_numeric(
                pd.Series(
                    [
                        working_gdf.at[
                            current_index,
                            "shadow_ratio",
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if (
                pd.isna(current_qa_ratio)
                or float(current_qa_ratio)
                < landsat_geometry_clip_min_qa_ratio
            ):
                working_gdf.at[
                    current_index,
                    "qa_temporal_clip_status",
                ] = "kept_below_qa_ratio_threshold"
                continue

            reference_ids = set()

            for reference_column in (
                "shadow_before_reference_ids",
                "shadow_after_reference_ids",
            ):
                reference_value = working_gdf.at[
                    current_index,
                    reference_column,
                ]

                if pd.isna(reference_value):
                    continue

                reference_ids.update(
                    reference_id
                    for reference_id in str(
                        reference_value
                    ).split(";")
                    if reference_id
                )

            reference_geometries = [
                reference_geometry_by_id[reference_id]
                for reference_id in reference_ids
                if reference_id in reference_geometry_by_id
            ]

            if not reference_geometries:
                working_gdf.at[
                    current_index,
                    "qa_temporal_clip_status",
                ] = "kept_no_reference_geometry"
                continue

            support_geometry = normalize_polygonal_geometry(
                unary_union(reference_geometries)
            )

            if support_geometry is None:
                working_gdf.at[
                    current_index,
                    "qa_temporal_clip_status",
                ] = "kept_invalid_reference_geometry"
                continue

            support_envelope = support_geometry.buffer(
                landsat_geometry_clip_buffer_m
            )
            current_geometry = working_gdf.at[
                current_index,
                "geometry",
            ]
            clipped_geometry = normalize_polygonal_geometry(
                current_geometry.intersection(
                    support_envelope
                )
            )
            original_area_m2 = float(
                working_gdf.at[
                    current_index,
                    "area_m2",
                ]
            )
            clipped_area_m2 = (
                0.0
                if clipped_geometry is None
                else float(clipped_geometry.area)
            )
            retained_ratio = (
                clipped_area_m2 / original_area_m2
                if original_area_m2 > 0.0
                else 0.0
            )

            working_gdf.at[
                current_index,
                "qa_temporal_clip_retained_ratio",
            ] = retained_ratio

            # 支持範囲外が過半を占める場合だけ形状を変更する。
            if (
                retained_ratio
                >= landsat_geometry_clip_max_retained_ratio
            ):
                working_gdf.at[
                    current_index,
                    "qa_temporal_clip_status",
                ] = "kept_within_supported_extent"
                continue

            if (
                clipped_geometry is None
                or clipped_area_m2
                < landsat_geometry_clip_min_area_m2
            ):
                working_gdf.at[
                    current_index,
                    "_qa_temporal_clip_keep",
                ] = False
                working_gdf.at[
                    current_index,
                    "qa_temporal_clip_applied",
                ] = True
                working_gdf.at[
                    current_index,
                    "qa_temporal_clip_status",
                ] = "removed_supported_part_below_minimum"
                working_gdf.at[
                    current_index,
                    "qa_temporal_clip_removed_area_m2",
                ] = original_area_m2
                removed_by_clip_count += 1
                removed_by_clip_area_m2 += original_area_m2
                continue

            working_gdf.at[
                current_index,
                "geometry",
            ] = clipped_geometry
            working_gdf.at[
                current_index,
                "area_m2",
            ] = clipped_area_m2

            if "area_km2" in working_gdf.columns:
                working_gdf.at[
                    current_index,
                    "area_km2",
                ] = clipped_area_m2 / 1_000_000.0

            working_gdf.at[
                current_index,
                "qa_temporal_clip_applied",
            ] = True
            working_gdf.at[
                current_index,
                "qa_temporal_clip_status",
            ] = "clipped_unsupported_expansion"
            working_gdf.at[
                current_index,
                "qa_temporal_clip_removed_area_m2",
            ] = max(
                original_area_m2 - clipped_area_m2,
                0.0,
            )
            clipped_candidate_count += 1
            removed_by_clip_area_m2 += max(
                original_area_m2 - clipped_area_m2,
                0.0,
            )

    retained_mask = (
        (
            ~working_gdf[
                "requires_shadow_validation"
            ]
            | working_gdf[
                "shadow_temporal_supported"
            ]
        )
        & working_gdf[
            "_qa_temporal_clip_keep"
        ]
    )

    retained_gdf = (
        working_gdf.loc[
            retained_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    retained_gdf = (
        retained_gdf
        .drop(
            columns=[
                "_parsed_date",
                "_qa_temporal_clip_keep",
            ]
        )
    )

    retained_gdf = (
        retained_gdf
        .sort_values(
            [
                "date",
                "local_candidate_id",
            ]
        )
        .reset_index(drop=True)
    )

    if retained_gdf.empty:
        raise ValueError(
            "雲影確認後に追跡へ渡せる"
            "候補がありません。"
        )

    if (
        retained_gdf[
            "local_candidate_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "雲影確認後の候補IDが"
            "重複しています。"
        )

    invalid_retained_geometry_mask = (
        retained_gdf.geometry.isna()
        | retained_gdf.geometry.is_empty
        | ~retained_gdf.geometry.is_valid
    )

    if invalid_retained_geometry_mask.any():
        raise ValueError(
            "雲影確認後の候補に"
            "無効ジオメトリがあります。"
        )

    print(
        "Landsat時系列支持範囲で形状を縮小した候補数: "
        f"{clipped_candidate_count}"
    )
    print(
        "Landsat時系列支持範囲が最低面積未満で除外した候補数: "
        f"{removed_by_clip_count}"
    )
    print(
        "Landsat時系列形状クリップによる除外面積: "
        f"{removed_by_clip_area_m2:.2f} m2"
    )

    output_gpkg.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        retained_gdf.to_file(
            output_gpkg,
            layer=output_layer,
            driver="GPKG",
            index=False,
            mode="w",
        )

    except PermissionError as error:
        raise PermissionError(
            "雲影確認後GeoPackageを"
            "保存できません。"
            "QGISで開いている場合は"
            "閉じてください: "
            f"{output_gpkg}"
        ) from error

    return retained_gdf

