from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union


def attach_subthreshold_fragments_to_persistent_lakes(
    tracking_input_gdf: gpd.GeoDataFrame,
    persistent_lake_gdf: gpd.GeoDataFrame,
    grouping_settings: dict,
    output_gpkg: Path,
    output_layer: str,
    audit_layer: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    required_tracking_columns = {
        "common_polygon_id",
        "observation_id",
        "date",
        "sensor_family",
        "common_area_m2",
        "shadow_only",
        "geometry",
    }

    missing_tracking_columns = (
        required_tracking_columns
        - set(tracking_input_gdf.columns)
    )

    if missing_tracking_columns:
        raise ValueError(
            "小断片抽出に必要な元ポリゴン列がありません: "
            + ", ".join(
                sorted(missing_tracking_columns)
            )
        )

    required_lake_columns = {
        "local_candidate_id",
        "observation_id",
        "date",
        "sensor_family",
        "source_polygon_ids",
        "persistent_lake_id",
        "persistent_counted_in_lake_count",
        "area_m2",
        "area_km2",
        "requires_shadow_validation",
        "shadow_temporal_supported",
        "geometry",
    }

    missing_lake_columns = (
        required_lake_columns
        - set(persistent_lake_gdf.columns)
    )

    if missing_lake_columns:
        raise ValueError(
            "小断片追加に必要な永続湖列がありません: "
            + ", ".join(
                sorted(missing_lake_columns)
            )
        )

    if tracking_input_gdf.empty:
        raise ValueError(
            "元追跡ポリゴンが空です。"
        )

    if persistent_lake_gdf.empty:
        raise ValueError(
            "永続湖ポリゴンが空です。"
        )

    if tracking_input_gdf.crs is None:
        raise ValueError(
            "元追跡ポリゴンにCRSがありません。"
        )

    if persistent_lake_gdf.crs is None:
        raise ValueError(
            "永続湖ポリゴンにCRSがありません。"
        )

    if not persistent_lake_gdf.crs.is_projected:
        raise ValueError(
            "小断片の距離判定には"
            "投影座標系が必要です。"
        )

    raw_gdf = (
        tracking_input_gdf
        .copy()
        .reset_index(drop=True)
    )

    output_gdf = (
        persistent_lake_gdf
        .copy()
        .reset_index(drop=True)
    )

    if raw_gdf.crs != output_gdf.crs:
        raw_gdf = raw_gdf.to_crs(
            output_gdf.crs
        )

    raw_gdf["common_polygon_id"] = (
        raw_gdf["common_polygon_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    raw_gdf["common_area_m2"] = (
        pd.to_numeric(
            raw_gdf["common_area_m2"],
            errors="coerce",
        )
    )

    raw_gdf["shadow_only"] = (
        raw_gdf["shadow_only"]
        .fillna(False)
        .astype(bool)
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

    output_gdf[
        "requires_shadow_validation"
    ] = (
        output_gdf[
            "requires_shadow_validation"
        ]
        .fillna(False)
        .astype(bool)
    )

    output_gdf[
        "shadow_temporal_supported"
    ] = (
        output_gdf[
            "shadow_temporal_supported"
        ]
        .fillna(False)
        .astype(bool)
    )

    invalid_raw_geometry_mask = (
        raw_gdf.geometry.isna()
        | raw_gdf.geometry.is_empty
        | ~raw_gdf.geometry.is_valid
    )

    if invalid_raw_geometry_mask.any():
        invalid_ids = (
            raw_gdf.loc[
                invalid_raw_geometry_mask,
                "common_polygon_id",
            ]
            .tolist()
        )

        raise ValueError(
            "小断片候補に無効ジオメトリがあります: "
            + ", ".join(invalid_ids)
        )

    invalid_lake_geometry_mask = (
        output_gdf.geometry.isna()
        | output_gdf.geometry.is_empty
        | ~output_gdf.geometry.is_valid
    )

    if invalid_lake_geometry_mask.any():
        invalid_ids = (
            output_gdf.loc[
                invalid_lake_geometry_mask,
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "永続湖に無効ジオメトリがあります: "
            + ", ".join(invalid_ids)
        )

    # グループ化済み候補に使われた元ポリゴンIDを集める。
    #
    # 未割当の湖候補も含めることで、すでに候補として
    # 出力されているポリゴンの二重加算を防ぐ。
    used_source_polygon_ids = set()

    for source_id_text in (
        output_gdf["source_polygon_ids"]
        .fillna("")
        .astype(str)
    ):
        for source_polygon_id in (
            source_id_text.split(";")
        ):
            source_polygon_id = (
                source_polygon_id.strip()
            )

            if source_polygon_id:
                used_source_polygon_ids.add(
                    source_polygon_id
                )

    # 前段の時系列補完で記録した直接観測面積を保持する。
    # area_m2はすでに補完後面積の場合があるため、ここで
    # direct_area_m2へ代入し直すと補完内訳が失われる。
    if "direct_area_m2" in output_gdf.columns:
        output_gdf["direct_area_m2"] = (
            pd.to_numeric(
                output_gdf["direct_area_m2"],
                errors="coerce",
            )
        )
    else:
        output_gdf["direct_area_m2"] = (
            pd.to_numeric(
                output_gdf["area_m2"],
                errors="coerce",
            )
        )

    output_gdf["direct_area_km2"] = (
        output_gdf["direct_area_m2"]
        / 1_000_000.0
    )

    output_gdf[
        "area_only_fragment_count"
    ] = 0

    output_gdf[
        "area_only_shadow_fragment_count"
    ] = 0

    output_gdf[
        "area_only_raw_fragment_area_m2"
    ] = 0.0

    output_gdf[
        "area_only_added_area_m2"
    ] = 0.0

    output_gdf[
        "area_only_source_polygon_ids"
    ] = ""

    output_gdf["area_only_status"] = (
        "no_fragment_attached"
    )

    fragment_assignments: dict[int, list[int]] = {}

    audit_records = []

    for raw_index, raw_row in raw_gdf.iterrows():
        common_polygon_id = str(
            raw_row["common_polygon_id"]
        ).strip()

        if not common_polygon_id:
            continue

        # すでにコアグループまたは継続候補として
        # 使用された元ポリゴンは処理しない。
        if (
            common_polygon_id
            in used_source_polygon_ids
        ):
            continue

        sensor_family = str(
            raw_row["sensor_family"]
        )

        sensor_settings = (
            grouping_settings.get(
                sensor_family
            )
        )

        if sensor_settings is None:
            raise ValueError(
                "小断片追加条件が設定されていない"
                "センサーがあります: "
                f"{sensor_family}"
            )

        continuation_area_m2 = float(
            sensor_settings[
                "continuation_area_m2"
            ]
        )

        maximum_distance_m = float(
            sensor_settings[
                "small_to_core_distance_m"
            ]
        )

        fragment_area_m2 = float(
            raw_row["common_area_m2"]
        )

        if not np.isfinite(fragment_area_m2):
            continue

        if fragment_area_m2 <= 0.0:
            continue

        # 今回の対象は継続面積閾値未満だけ。
        if (
            fragment_area_m2
            >= continuation_area_m2
        ):
            continue

        observation_id = str(
            raw_row["observation_id"]
        )

        fragment_shadow_only = bool(
            raw_row["shadow_only"]
        )

        assigned_lake_mask = (
            (
                output_gdf["observation_id"]
                .astype(str)
                == observation_id
            )
            & (
                output_gdf[
                    "persistent_lake_id"
                ]
                .ne("")
            )
            & (
                output_gdf[
                    "persistent_counted_in_lake_count"
                ]
            )
        )

        assigned_lake_indices = (
            output_gdf.index[
                assigned_lake_mask
            ]
            .tolist()
        )

        nearby_matches = []

        for lake_index in assigned_lake_indices:
            lake_geometry = (
                output_gdf.at[
                    lake_index,
                    "geometry",
                ]
            )

            distance_m = float(
                raw_row.geometry.distance(
                    lake_geometry
                )
            )

            if distance_m > maximum_distance_m:
                continue

            target_shadow_supported = bool(
                output_gdf.at[
                    lake_index,
                    "requires_shadow_validation",
                ]
                and output_gdf.at[
                    lake_index,
                    "shadow_temporal_supported",
                ]
            )

            # 雲影だけで構成された小断片は、
            # 時系列的に確認された雲影湖へだけ追加する。
            if (
                fragment_shadow_only
                and not target_shadow_supported
            ):
                continue

            nearby_matches.append(
                {
                    "lake_index": lake_index,
                    "distance_m": distance_m,
                    "persistent_lake_id": str(
                        output_gdf.at[
                            lake_index,
                            "persistent_lake_id",
                        ]
                    ),
                    "local_candidate_id": str(
                        output_gdf.at[
                            lake_index,
                            "local_candidate_id",
                        ]
                    ),
                }
            )

        audit_record = {
            "common_polygon_id": (
                common_polygon_id
            ),
            "observation_id": observation_id,
            "date": str(raw_row["date"]),
            "sensor_family": sensor_family,
            "fragment_area_m2": (
                fragment_area_m2
            ),
            "fragment_shadow_only": (
                fragment_shadow_only
            ),
            "continuation_threshold_m2": (
                continuation_area_m2
            ),
            "maximum_distance_m": (
                maximum_distance_m
            ),
            "assignment_status": "",
            "assigned_persistent_lake_id": "",
            "assigned_local_candidate_id": "",
            "assigned_distance_m": np.nan,
            "geometry": raw_row.geometry,
        }

        if not assigned_lake_indices:
            audit_record[
                "assignment_status"
            ] = "no_assigned_lake_on_date"

            audit_records.append(
                audit_record
            )

            continue

        if not nearby_matches:
            if fragment_shadow_only:
                audit_record[
                    "assignment_status"
                ] = (
                    "no_temporally_supported_"
                    "shadow_lake_within_distance"
                )

            else:
                audit_record[
                    "assignment_status"
                ] = (
                    "no_assigned_lake_"
                    "within_distance"
                )

            audit_records.append(
                audit_record
            )

            continue

        nearby_matches = sorted(
            nearby_matches,
            key=lambda match: (
                match["distance_m"],
                match["persistent_lake_id"],
                match["local_candidate_id"],
            ),
        )

        minimum_distance_m = (
            nearby_matches[0][
                "distance_m"
            ]
        )

        nearest_matches = [
            match
            for match in nearby_matches
            if abs(
                match["distance_m"]
                - minimum_distance_m
            )
            <= 1.0e-9
        ]

        nearest_lake_ids = {
            match["persistent_lake_id"]
            for match in nearest_matches
        }

        if len(nearest_lake_ids) != 1:
            audit_record[
                "assignment_status"
            ] = "ambiguous_equal_distance"

            audit_record[
                "assigned_distance_m"
            ] = minimum_distance_m

            audit_records.append(
                audit_record
            )

            continue

        selected_match = nearest_matches[0]
        selected_lake_index = selected_match[
            "lake_index"
        ]

        fragment_assignments.setdefault(
            selected_lake_index,
            [],
        ).append(
            raw_index
        )

        audit_record[
            "assignment_status"
        ] = "attached_to_persistent_lake"

        audit_record[
            "assigned_persistent_lake_id"
        ] = selected_match[
            "persistent_lake_id"
        ]

        audit_record[
            "assigned_local_candidate_id"
        ] = selected_match[
            "local_candidate_id"
        ]

        audit_record[
            "assigned_distance_m"
        ] = selected_match[
            "distance_m"
        ]

        audit_records.append(
            audit_record
        )

    # すべての割当先を決めてから結合する。
    #
    # 逐次的に湖を拡張しながら距離判定すると、
    # 断片が鎖状に40 mずつ連結される可能性があるため、
    # 距離判定時には元の湖形状だけを使用している。
    for lake_index, raw_indices in (
        fragment_assignments.items()
    ):
        direct_geometry = (
            output_gdf.at[
                lake_index,
                "geometry",
            ]
        )

        fragment_geometries = (
            raw_gdf.loc[
                raw_indices,
                "geometry",
            ]
            .tolist()
        )

        completed_geometry = unary_union(
            [
                direct_geometry,
                *fragment_geometries,
            ]
        )

        if (
            completed_geometry is None
            or completed_geometry.is_empty
        ):
            raise ValueError(
                "小断片結合後のジオメトリが"
                "空になりました: "
                f"{output_gdf.at[lake_index, 'local_candidate_id']}"
            )

        if not completed_geometry.is_valid:
            completed_geometry = (
                completed_geometry.buffer(0)
            )

        if (
            completed_geometry is None
            or completed_geometry.is_empty
            or not completed_geometry.is_valid
        ):
            raise ValueError(
                "小断片結合後のジオメトリを"
                "修復できません: "
                f"{output_gdf.at[lake_index, 'local_candidate_id']}"
            )

        # 小断片だけの追加量を求めるため、時系列補完を含む
        # 小断片追加直前のジオメトリ面積を基準にする。
        area_before_fragment_m2 = float(
            direct_geometry.area
        )

        completed_area_m2 = float(
            completed_geometry.area
        )

        added_area_m2 = max(
            0.0,
            completed_area_m2
            - area_before_fragment_m2,
        )

        fragment_rows = (
            raw_gdf.loc[
                raw_indices
            ]
        )

        fragment_ids = (
            fragment_rows[
                "common_polygon_id"
            ]
            .astype(str)
            .tolist()
        )

        raw_fragment_area_m2 = float(
            fragment_rows[
                "common_area_m2"
            ].sum()
        )

        shadow_fragment_count = int(
            fragment_rows[
                "shadow_only"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        output_gdf.at[
            lake_index,
            "geometry",
        ] = completed_geometry

        output_gdf.at[
            lake_index,
            "area_m2",
        ] = completed_area_m2

        output_gdf.at[
            lake_index,
            "area_km2",
        ] = (
            completed_area_m2
            / 1_000_000.0
        )

        output_gdf.at[
            lake_index,
            "area_only_fragment_count",
        ] = len(raw_indices)

        output_gdf.at[
            lake_index,
            "area_only_shadow_fragment_count",
        ] = shadow_fragment_count

        output_gdf.at[
            lake_index,
            "area_only_raw_fragment_area_m2",
        ] = raw_fragment_area_m2

        output_gdf.at[
            lake_index,
            "area_only_added_area_m2",
        ] = added_area_m2

        output_gdf.at[
            lake_index,
            "area_only_source_polygon_ids",
        ] = ";".join(fragment_ids)

        output_gdf.at[
            lake_index,
            "area_only_status",
        ] = "fragments_attached"

    invalid_output_geometry_mask = (
        output_gdf.geometry.isna()
        | output_gdf.geometry.is_empty
        | ~output_gdf.geometry.is_valid
    )

    if invalid_output_geometry_mask.any():
        raise ValueError(
            "小断片追加後の出力に"
            "無効ジオメトリがあります。"
        )

    geometry_area_difference = (
        output_gdf.geometry.area
        - pd.to_numeric(
            output_gdf["area_m2"],
            errors="coerce",
        )
    ).abs()

    if (
        geometry_area_difference
        .fillna(np.inf)
        .gt(1.0e-6)
        .any()
    ):
        raise ValueError(
            "小断片追加後のジオメトリ面積と"
            "area_m2が一致しません。"
        )

    output_gdf = (
        output_gdf
        .sort_values(
            [
                "date",
                "persistent_lake_id",
                "local_candidate_id",
            ]
        )
        .reset_index(drop=True)
    )

    audit_columns = [
        "common_polygon_id",
        "observation_id",
        "date",
        "sensor_family",
        "fragment_area_m2",
        "fragment_shadow_only",
        "continuation_threshold_m2",
        "maximum_distance_m",
        "assignment_status",
        "assigned_persistent_lake_id",
        "assigned_local_candidate_id",
        "assigned_distance_m",
        "geometry",
    ]

    fragment_audit_gdf = gpd.GeoDataFrame(
        audit_records,
        columns=audit_columns,
        geometry="geometry",
        crs=output_gdf.crs,
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

        if not fragment_audit_gdf.empty:
            fragment_audit_gdf.to_file(
                output_gpkg,
                layer=audit_layer,
                driver="GPKG",
                index=False,
                mode="a",
            )

    except PermissionError as error:
        raise PermissionError(
            "小断片追加後GeoPackageを"
            "保存できません。"
            "QGISで開いている場合は"
            "閉じてください: "
            f"{output_gpkg}"
        ) from error

    return (
        output_gdf,
        fragment_audit_gdf,
    )

