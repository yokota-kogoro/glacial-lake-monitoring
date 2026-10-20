from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def add_multisensor_tracking_relationships(
    persistent_tracking_gdf: gpd.GeoDataFrame,
    output_gpkg: Path,
    output_layer: str,
    max_gap_days: int,
    min_intersection_area_m2: float,
    min_overlap_smaller_ratio: float,
    min_iou: float,
) -> gpd.GeoDataFrame:

    required_columns = {
        "local_candidate_id",
        "observation_id",
        "date",
        "sensor_family",
        "persistent_lake_id",
        "persistent_counted_in_lake_count",
        "tracking_status",
        "area_m2",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(persistent_tracking_gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            "合流・分離関係の判定に必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )

    if persistent_tracking_gdf.empty:
        raise ValueError("関係判定対象が空です。")

    if persistent_tracking_gdf.crs is None:
        raise ValueError("関係判定対象にCRSがありません。")

    if not persistent_tracking_gdf.crs.is_projected:
        raise ValueError(
            "交差面積を使った関係判定には"
            "投影座標系が必要です。"
        )

    if max_gap_days <= 0:
        raise ValueError("max_gap_daysは1以上にしてください。")

    if min_intersection_area_m2 < 0.0:
        raise ValueError(
            "min_intersection_area_m2は0以上にしてください。"
        )

    if not 0.0 <= min_overlap_smaller_ratio <= 1.0:
        raise ValueError(
            "min_overlap_smaller_ratioは0以上1以下にしてください。"
        )

    if not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iouは0以上1以下にしてください。")

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
                output_gdf["_parsed_date"].isna(),
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "関係判定で日付を解釈できない候補があります: "
            + ", ".join(invalid_ids)
        )

    output_gdf["persistent_lake_id"] = (
        output_gdf["persistent_lake_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    output_gdf["persistent_counted_in_lake_count"] = (
        output_gdf["persistent_counted_in_lake_count"]
        .fillna(False)
        .astype(bool)
    )

    invalid_geometry_mask = (
        output_gdf.geometry.isna()
        | output_gdf.geometry.is_empty
        | ~output_gdf.geometry.is_valid
    )

    if invalid_geometry_mask.any():
        invalid_ids = (
            output_gdf.loc[
                invalid_geometry_mask,
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "関係判定対象に無効ジオメトリがあります: "
            + ", ".join(invalid_ids)
        )

    assigned_mask = (
        output_gdf["persistent_lake_id"].ne("")
        & output_gdf["persistent_counted_in_lake_count"]
    )

    assigned_gdf = output_gdf.loc[assigned_mask].copy()

    duplicate_id_date_mask = assigned_gdf.duplicated(
        subset=[
            "_parsed_date",
            "persistent_lake_id",
        ],
        keep=False,
    )

    if duplicate_id_date_mask.any():
        duplicate_rows = assigned_gdf.loc[
            duplicate_id_date_mask,
            [
                "_parsed_date",
                "persistent_lake_id",
                "local_candidate_id",
            ],
        ]
        raise ValueError(
            "同じ日付に同じ永続IDが複数存在します:\n"
            + duplicate_rows.to_string(index=False)
        )

    output_gdf["tracking_event"] = "no_persistent_id"
    output_gdf.loc[
        assigned_mask,
        "tracking_event",
    ] = "continuation"

    created_mask = (
        assigned_mask
        & output_gdf["tracking_status"]
        .astype(str)
        .eq("created_new_id")
    )
    output_gdf.loc[created_mask, "tracking_event"] = "new"

    output_gdf["related_previous_lake_ids"] = ""
    output_gdf["matched_previous_lake_id"] = ""
    output_gdf["merged_into_id_before_restore"] = ""
    output_gdf["restored_after_merge"] = False
    output_gdf["relation_previous_id_count"] = 0
    output_gdf["relation_previous_dates"] = ""
    output_gdf["relation_split_source_ids"] = ""
    output_gdf["relation_best_intersection_area_m2"] = np.nan
    output_gdf["relation_best_overlap_smaller_ratio"] = np.nan
    output_gdf["relation_best_iou"] = np.nan
    output_gdf["relation_best_previous_overlap_ratio"] = np.nan
    output_gdf["relation_best_current_overlap_ratio"] = np.nan

    observation_dates = sorted(
        assigned_gdf["_parsed_date"]
        .drop_duplicates()
        .tolist()
    )

    for current_date in observation_dates:
        current_indices = assigned_gdf.index[
            assigned_gdf["_parsed_date"] == current_date
        ].tolist()

        if not current_indices:
            continue

        previous_rows = assigned_gdf.loc[
            assigned_gdf["_parsed_date"] < current_date
        ].copy()

        previous_reference_rows = {}

        if not previous_rows.empty:
            previous_rows["_relation_gap_days"] = (
                current_date
                - previous_rows["_parsed_date"]
            ).dt.days
            previous_rows = previous_rows.loc[
                previous_rows["_relation_gap_days"]
                .between(1, max_gap_days)
            ]
            previous_rows = (
                previous_rows
                .sort_values(
                    [
                        "_parsed_date",
                        "local_candidate_id",
                    ]
                )
                .groupby(
                    "persistent_lake_id",
                    sort=False,
                )
                .tail(1)
            )

            for previous_index, previous_row in previous_rows.iterrows():
                previous_reference_rows[
                    str(previous_row["persistent_lake_id"])
                ] = {
                    "index": previous_index,
                    "row": previous_row,
                }

        relation_records_by_current = {
            current_index: []
            for current_index in current_indices
        }

        for current_index in current_indices:
            current_row = assigned_gdf.loc[current_index]
            current_geometry = current_row.geometry
            current_area_m2 = float(current_geometry.area)
            current_persistent_id = str(
                current_row["persistent_lake_id"]
            )

            for (
                previous_persistent_id,
                previous_reference,
            ) in previous_reference_rows.items():
                previous_row = previous_reference["row"]
                previous_geometry = previous_row.geometry
                previous_area_m2 = float(previous_geometry.area)

                if current_area_m2 <= 0.0 or previous_area_m2 <= 0.0:
                    continue

                try:
                    intersection_area_m2 = float(
                        current_geometry
                        .intersection(previous_geometry)
                        .area
                    )
                except Exception:
                    continue

                smaller_area_m2 = min(
                    current_area_m2,
                    previous_area_m2,
                )
                union_area_m2 = (
                    current_area_m2
                    + previous_area_m2
                    - intersection_area_m2
                )
                overlap_smaller_ratio = (
                    intersection_area_m2 / smaller_area_m2
                    if smaller_area_m2 > 0.0
                    else 0.0
                )
                iou = (
                    intersection_area_m2 / union_area_m2
                    if union_area_m2 > 0.0
                    else 0.0
                )
                previous_overlap_ratio = (
                    intersection_area_m2 / previous_area_m2
                )
                current_overlap_ratio = (
                    intersection_area_m2 / current_area_m2
                )

                spatial_relation = bool(
                    intersection_area_m2
                    >= min_intersection_area_m2
                    and (
                        overlap_smaller_ratio
                        >= min_overlap_smaller_ratio
                        or iou >= min_iou
                    )
                )

                identity_relation = bool(
                    current_persistent_id
                    == previous_persistent_id
                    and str(current_row["tracking_status"])
                    == "continued_existing_id"
                )

                if not (spatial_relation or identity_relation):
                    continue

                relation_records_by_current[current_index].append(
                    {
                        "previous_persistent_id": previous_persistent_id,
                        "previous_date": previous_row["_parsed_date"],
                        "intersection_area_m2": intersection_area_m2,
                        "overlap_smaller_ratio": overlap_smaller_ratio,
                        "iou": iou,
                        "previous_overlap_ratio": previous_overlap_ratio,
                        "current_overlap_ratio": current_overlap_ratio,
                        "spatial_relation": spatial_relation,
                        "identity_relation": identity_relation,
                    }
                )

        previous_to_current_indices = {}

        for current_index, relation_records in (
            relation_records_by_current.items()
        ):
            for relation_record in relation_records:
                previous_persistent_id = relation_record[
                    "previous_persistent_id"
                ]
                previous_to_current_indices.setdefault(
                    previous_persistent_id,
                    set(),
                ).add(current_index)

        for current_index in current_indices:
            current_persistent_id = str(
                assigned_gdf.at[
                    current_index,
                    "persistent_lake_id",
                ]
            )
            current_tracking_status = str(
                assigned_gdf.at[
                    current_index,
                    "tracking_status",
                ]
            )
            relation_records = sorted(
                relation_records_by_current[current_index],
                key=lambda relation: (
                    -int(relation["spatial_relation"]),
                    -relation["previous_overlap_ratio"],
                    -relation["intersection_area_m2"],
                    -relation["iou"],
                    -relation["overlap_smaller_ratio"],
                    relation["previous_persistent_id"],
                ),
            )

            related_previous_ids = sorted(
                {
                    relation["previous_persistent_id"]
                    for relation in relation_records
                }
            )
            related_previous_dates = sorted(
                {
                    relation["previous_persistent_id"]
                    + ":"
                    + relation["previous_date"].strftime("%Y-%m-%d")
                    for relation in relation_records
                }
            )

            output_gdf.at[
                current_index,
                "related_previous_lake_ids",
            ] = ";".join(related_previous_ids)
            output_gdf.at[
                current_index,
                "relation_previous_dates",
            ] = ";".join(related_previous_dates)
            output_gdf.at[
                current_index,
                "relation_previous_id_count",
            ] = len(related_previous_ids)

            if relation_records:
                same_id_relations = [
                    relation
                    for relation in relation_records
                    if relation["previous_persistent_id"]
                    == current_persistent_id
                ]
                matched_relation = (
                    same_id_relations[0]
                    if same_id_relations
                    else relation_records[0]
                )
                output_gdf.at[
                    current_index,
                    "matched_previous_lake_id",
                ] = matched_relation["previous_persistent_id"]
                output_gdf.at[
                    current_index,
                    "relation_best_intersection_area_m2",
                ] = matched_relation["intersection_area_m2"]
                output_gdf.at[
                    current_index,
                    "relation_best_overlap_smaller_ratio",
                ] = matched_relation["overlap_smaller_ratio"]
                output_gdf.at[
                    current_index,
                    "relation_best_iou",
                ] = matched_relation["iou"]
                output_gdf.at[
                    current_index,
                    "relation_best_previous_overlap_ratio",
                ] = matched_relation["previous_overlap_ratio"]
                output_gdf.at[
                    current_index,
                    "relation_best_current_overlap_ratio",
                ] = matched_relation["current_overlap_ratio"]

            split_source_ids = sorted(
                {
                    previous_persistent_id
                    for previous_persistent_id in related_previous_ids
                    if len(
                        previous_to_current_indices.get(
                            previous_persistent_id,
                            set(),
                        )
                    ) > 1
                }
            )
            output_gdf.at[
                current_index,
                "relation_split_source_ids",
            ] = ";".join(split_source_ids)

            is_merge = len(related_previous_ids) > 1
            is_split = bool(split_source_ids)

            if is_merge and is_split:
                tracking_event = "complex_merge_split"
            elif is_merge:
                tracking_event = "merge"
            elif is_split:
                tracking_event = (
                    "split_primary"
                    if current_persistent_id in split_source_ids
                    else "split_new"
                )
            elif current_tracking_status == "created_new_id":
                tracking_event = "new"
            else:
                tracking_event = "continuation"

            output_gdf.at[
                current_index,
                "tracking_event",
            ] = tracking_event

    output_gdf = (
        output_gdf
        .drop(columns=["_parsed_date"])
        .sort_values(
            [
                "date",
                "persistent_lake_id",
                "local_candidate_id",
            ]
        )
        .reset_index(drop=True)
    )

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

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
            "合流・分離関係GeoPackageを保存できません。"
            "QGISで開いている場合は閉じてください: "
            f"{output_gpkg}"
        ) from error

    return output_gdf

