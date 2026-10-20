from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union


def group_multisensor_polygons_by_core(
    tracking_input_gdf: gpd.GeoDataFrame,
    grouping_settings: dict,
    landsat_qa_review_min_ratio: float,
    output_gpkg: Path,
    output_layer: str,
) -> gpd.GeoDataFrame:

    required_columns = {
        "common_polygon_id",
        "observation_id",
        "date",
        "sensor_family",
        "platform",
        "scene",
        "product_id",
        "quality_status",
        "quality_score",
        "native_pixel_size_m",
        "ndwi_threshold",
        "common_area_m2",
        "source_pixel_count",
        "shadow_pixel_count",
        "clear_pixel_count",
        "shadow_ratio",
        "shadow_only",
        "source_polygon_file",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(tracking_input_gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            "グループ化に必要な列がありません: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if not 0.0 <= landsat_qa_review_min_ratio <= 1.0:
        raise ValueError(
            "landsat_qa_review_min_ratioは"
            "0から1の範囲で指定してください。"
        )

    if tracking_input_gdf.crs is None:
        raise ValueError(
            "追跡入力GeoDataFrameに"
            "CRSが設定されていません。"
        )

    if not tracking_input_gdf.crs.is_projected:
        raise ValueError(
            "距離と面積の計算には"
            "投影座標系が必要です。"
        )

    if tracking_input_gdf.empty:
        raise ValueError(
            "グループ化するポリゴンがありません。"
        )

    working_gdf = (
        tracking_input_gdf
        .copy()
        .reset_index(drop=True)
    )

    working_gdf[
        "common_area_m2"
    ] = pd.to_numeric(
        working_gdf[
            "common_area_m2"
        ],
        errors="coerce",
    )

    invalid_area_mask = (
        working_gdf[
            "common_area_m2"
        ]
        .isna()
        | working_gdf[
            "common_area_m2"
        ]
        .le(0.0)
    )

    if invalid_area_mask.any():
        invalid_ids = (
            working_gdf.loc[
                invalid_area_mask,
                "common_polygon_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "面積が不正なポリゴンがあります: "
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
                "common_polygon_id",
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

    def sum_numeric_or_nan(
        values: pd.Series,
    ) -> float:

        numeric_values = pd.to_numeric(
            values,
            errors="coerce",
        )

        if not numeric_values.notna().any():
            return np.nan

        return float(
            numeric_values.sum()
        )

    output_records: list[dict] = []

    grouped_observations = (
        working_gdf.groupby(
            [
                "observation_id",
                "date",
                "sensor_family",
            ],
            sort=True,
            dropna=False,
        )
    )

    for (
        observation_key,
        observation_gdf,
    ) in grouped_observations:

        (
            observation_id,
            observation_date,
            sensor_family,
        ) = observation_key

        observation_id = str(
            observation_id
        )

        observation_date = str(
            observation_date
        )

        sensor_family = str(
            sensor_family
        )

        if (
            sensor_family
            not in grouping_settings
        ):
            raise ValueError(
                "グループ化条件が設定されていない"
                "センサーです: "
                f"{sensor_family}"
            )

        sensor_settings = (
            grouping_settings[
                sensor_family
            ]
        )

        core_area_m2 = float(
            sensor_settings[
                "core_area_m2"
            ]
        )

        continuation_area_m2 = float(
            sensor_settings[
                "continuation_area_m2"
            ]
        )

        small_to_core_distance_m = float(
            sensor_settings[
                "small_to_core_distance_m"
            ]
        )

        if core_area_m2 <= 0.0:
            raise ValueError(
                "core_area_m2は"
                "0より大きくしてください。"
            )

        if continuation_area_m2 <= 0.0:
            raise ValueError(
                "continuation_area_m2は"
                "0より大きくしてください。"
            )

        if small_to_core_distance_m < 0.0:
            raise ValueError(
                "small_to_core_distance_mは"
                "0以上にしてください。"
            )

        observation_gdf = (
            observation_gdf
            .sort_values(
                "common_polygon_id"
            )
            .reset_index(drop=True)
        )

        core_indices = [
            row_index
            for row_index, row
            in observation_gdf.iterrows()
            if float(
                row[
                    "common_area_m2"
                ]
            )
            >= core_area_m2
        ]

        core_index_set = set(
            core_indices
        )

        small_indices = [
            row_index
            for row_index
            in observation_gdf.index
            if row_index
            not in core_index_set
        ]

        group_members: dict[
            int,
            list[int],
        ] = {}

        for (
            local_group_number,
            core_index,
        ) in enumerate(
            core_indices,
            start=1,
        ):
            group_members[
                local_group_number
            ] = [
                core_index,
            ]

        unassigned_small_indices: list[
            int
        ] = []

        for small_index in small_indices:
            small_geometry = (
                observation_gdf.at[
                    small_index,
                    "geometry",
                ]
            )

            nearest_group_number = None
            nearest_distance_m = np.inf

            for (
                local_group_number,
                core_index,
            ) in enumerate(
                core_indices,
                start=1,
            ):
                core_geometry = (
                    observation_gdf.at[
                        core_index,
                        "geometry",
                    ]
                )

                distance_m = float(
                    small_geometry.distance(
                        core_geometry
                    )
                )

                if (
                    distance_m
                    < nearest_distance_m
                ):
                    nearest_distance_m = (
                        distance_m
                    )

                    nearest_group_number = (
                        local_group_number
                    )

            if (
                nearest_group_number
                is not None
                and nearest_distance_m
                <= small_to_core_distance_m
            ):
                group_members[
                    nearest_group_number
                ].append(
                    small_index
                )

            else:
                unassigned_small_indices.append(
                    small_index
                )

        first_row = (
            observation_gdf.iloc[0]
        )

        local_candidate_number = 1

        for local_group_number in sorted(
            group_members
        ):
            member_indices = (
                group_members[
                    local_group_number
                ]
            )

            member_gdf = (
                observation_gdf.loc[
                    member_indices
                ]
                .copy()
            )

            group_geometry = unary_union(
                member_gdf.geometry.tolist()
            )

            if (
                group_geometry is None
                or group_geometry.is_empty
            ):
                continue

            if not group_geometry.is_valid:
                group_geometry = (
                    group_geometry.buffer(0)
                )

            if (
                group_geometry is None
                or group_geometry.is_empty
                or not group_geometry.is_valid
            ):
                raise ValueError(
                    "コアグループのジオメトリを"
                    "修復できません: "
                    f"{observation_id}"
                )

            group_area_m2 = float(
                group_geometry.area
            )

            source_polygon_ids = ";".join(
                member_gdf[
                    "common_polygon_id"
                ]
                .astype(str)
                .tolist()
            )

            group_source_pixel_count = (
                sum_numeric_or_nan(
                    member_gdf[
                        "source_pixel_count"
                    ]
                )
            )

            group_shadow_pixel_count = (
                sum_numeric_or_nan(
                    member_gdf[
                        "shadow_pixel_count"
                    ]
                )
            )

            group_clear_pixel_count = (
                sum_numeric_or_nan(
                    member_gdf[
                        "clear_pixel_count"
                    ]
                )
            )

            if (
                pd.notna(
                    group_source_pixel_count
                )
                and pd.notna(
                    group_shadow_pixel_count
                )
                and group_source_pixel_count
                > 0.0
            ):
                group_shadow_ratio = (
                    group_shadow_pixel_count
                    / group_source_pixel_count
                )

            else:
                group_shadow_ratio = np.nan

            group_shadow_only = bool(
                member_gdf[
                    "shadow_only"
                ]
                .fillna(False)
                .astype(bool)
                .all()
            )

            # Sentinel-2の雲影のみ候補に加え、LandsatではQA_PIXELの
            # 雲系フラグが一定割合以上を占める候補も時系列確認する。
            requires_shadow_validation = bool(
                (
                    sensor_family == "Sentinel-2"
                    and group_shadow_only
                )
                or (
                    sensor_family == "Landsat"
                    and pd.notna(group_shadow_ratio)
                    and group_shadow_ratio
                    >= landsat_qa_review_min_ratio
                )
            )

            can_create_new_id = bool(
                group_area_m2
                >= core_area_m2
                and not requires_shadow_validation
            )

            counted_in_lake_count = (
                can_create_new_id
            )

            local_candidate_id = (
                f"{observation_id}_"
                f"C{local_candidate_number:03d}"
            )

            output_records.append(
                {
                    "local_candidate_id": (
                        local_candidate_id
                    ),
                    "observation_id": (
                        observation_id
                    ),
                    "date": (
                        observation_date
                    ),
                    "sensor_family": (
                        sensor_family
                    ),
                    "platform": str(
                        first_row[
                            "platform"
                        ]
                    ),
                    "scene": str(
                        first_row[
                            "scene"
                        ]
                    ),
                    "product_id": str(
                        first_row[
                            "product_id"
                        ]
                    ),
                    "quality_status": str(
                        first_row[
                            "quality_status"
                        ]
                    ),
                    "quality_score": float(
                        first_row[
                            "quality_score"
                        ]
                    ),
                    "candidate_source": (
                        "core_group"
                    ),
                    "candidate_role": (
                        "new_or_continuation"
                        if can_create_new_id
                        else "continuation_only"
                    ),
                    "can_create_new_id": (
                        can_create_new_id
                    ),
                    "counted_in_lake_count": (
                        counted_in_lake_count
                    ),
                    "requires_shadow_validation": (
                        requires_shadow_validation
                    ),
                    "core_polygon_count": 1,
                    "assigned_small_polygon_count": (
                        len(member_indices) - 1
                    ),
                    "source_polygon_count": (
                        len(member_indices)
                    ),
                    "source_polygon_ids": (
                        source_polygon_ids
                    ),
                    "core_area_threshold_m2": (
                        core_area_m2
                    ),
                    "continuation_area_threshold_m2": (
                        continuation_area_m2
                    ),
                    "small_to_core_distance_m": (
                        small_to_core_distance_m
                    ),
                    "area_m2": (
                        group_area_m2
                    ),
                    "area_km2": (
                        group_area_m2
                        / 1_000_000.0
                    ),
                    "water_pixel_count": (
                        group_source_pixel_count
                    ),
                    "shadow_pixel_count": (
                        group_shadow_pixel_count
                    ),
                    "clear_pixel_count": (
                        group_clear_pixel_count
                    ),
                    "shadow_ratio": (
                        group_shadow_ratio
                    ),
                    "shadow_only": (
                        group_shadow_only
                    ),
                    "source_polygon_file": str(
                        first_row[
                            "source_polygon_file"
                        ]
                    ),
                    "geometry": (
                        group_geometry
                    ),
                }
            )

            local_candidate_number += 1

        for small_index in (
            unassigned_small_indices
        ):
            small_row = (
                observation_gdf.loc[
                    small_index
                ]
            )

            small_area_m2 = float(
                small_row[
                    "common_area_m2"
                ]
            )

            if (
                small_area_m2
                < continuation_area_m2
            ):
                continue

            small_geometry = (
                small_row.geometry
            )

            if (
                small_geometry is None
                or small_geometry.is_empty
            ):
                continue

            if not small_geometry.is_valid:
                small_geometry = (
                    small_geometry.buffer(0)
                )

            if (
                small_geometry is None
                or small_geometry.is_empty
                or not small_geometry.is_valid
            ):
                raise ValueError(
                    "継続候補のジオメトリを"
                    "修復できません: "
                    f"{observation_id}"
                )

            small_shadow_only = bool(
                small_row[
                    "shadow_only"
                ]
            )

            small_shadow_ratio = pd.to_numeric(
                small_row["shadow_ratio"],
                errors="coerce",
            )

            requires_shadow_validation = bool(
                (
                    sensor_family == "Sentinel-2"
                    and small_shadow_only
                )
                or (
                    sensor_family == "Landsat"
                    and pd.notna(small_shadow_ratio)
                    and float(small_shadow_ratio)
                    >= landsat_qa_review_min_ratio
                )
            )

            local_candidate_id = (
                f"{observation_id}_"
                f"C{local_candidate_number:03d}"
            )

            output_records.append(
                {
                    "local_candidate_id": (
                        local_candidate_id
                    ),
                    "observation_id": (
                        observation_id
                    ),
                    "date": (
                        observation_date
                    ),
                    "sensor_family": (
                        sensor_family
                    ),
                    "platform": str(
                        first_row[
                            "platform"
                        ]
                    ),
                    "scene": str(
                        first_row[
                            "scene"
                        ]
                    ),
                    "product_id": str(
                        first_row[
                            "product_id"
                        ]
                    ),
                    "quality_status": str(
                        first_row[
                            "quality_status"
                        ]
                    ),
                    "quality_score": float(
                        first_row[
                            "quality_score"
                        ]
                    ),
                    "candidate_source": (
                        "unassigned_small"
                    ),
                    "candidate_role": (
                        "continuation_only"
                    ),
                    "can_create_new_id": False,
                    "counted_in_lake_count": False,
                    "requires_shadow_validation": (
                        requires_shadow_validation
                    ),
                    "core_polygon_count": 0,
                    "assigned_small_polygon_count": 0,
                    "source_polygon_count": 1,
                    "source_polygon_ids": str(
                        small_row[
                            "common_polygon_id"
                        ]
                    ),
                    "core_area_threshold_m2": (
                        core_area_m2
                    ),
                    "continuation_area_threshold_m2": (
                        continuation_area_m2
                    ),
                    "small_to_core_distance_m": (
                        small_to_core_distance_m
                    ),
                    "area_m2": (
                        small_area_m2
                    ),
                    "area_km2": (
                        small_area_m2
                        / 1_000_000.0
                    ),
                    "water_pixel_count": (
                        small_row[
                            "source_pixel_count"
                        ]
                    ),
                    "shadow_pixel_count": (
                        small_row[
                            "shadow_pixel_count"
                        ]
                    ),
                    "clear_pixel_count": (
                        small_row[
                            "clear_pixel_count"
                        ]
                    ),
                    "shadow_ratio": (
                        small_row[
                            "shadow_ratio"
                        ]
                    ),
                    "shadow_only": (
                        small_shadow_only
                    ),
                    "source_polygon_file": str(
                        small_row[
                            "source_polygon_file"
                        ]
                    ),
                    "geometry": (
                        small_geometry
                    ),
                }
            )

            local_candidate_number += 1

    if not output_records:
        raise ValueError(
            "グループ化後に残る"
            "湖候補がありません。"
        )

    output_gdf = gpd.GeoDataFrame(
        output_records,
        geometry="geometry",
        crs=tracking_input_gdf.crs,
    )

    output_gdf = (
        output_gdf
        .sort_values(
            [
                "date",
                "local_candidate_id",
            ]
        )
        .reset_index(drop=True)
    )

    duplicate_id_mask = (
        output_gdf[
            "local_candidate_id"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate_id_mask.any():
        duplicate_ids = (
            output_gdf.loc[
                duplicate_id_mask,
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "local_candidate_idが重複しています: "
            + ", ".join(
                duplicate_ids
            )
        )

    invalid_output_geometry_mask = (
        output_gdf.geometry.isna()
        | output_gdf.geometry.is_empty
        | ~output_gdf.geometry.is_valid
    )

    if invalid_output_geometry_mask.any():
        raise ValueError(
            "グループ化後の出力に"
            "無効ジオメトリがあります。"
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
            "グループ化GeoPackageを"
            "保存できません。"
            "QGISで開いている場合は"
            "閉じてください: "
            f"{output_gpkg}"
        ) from error

    return output_gdf

