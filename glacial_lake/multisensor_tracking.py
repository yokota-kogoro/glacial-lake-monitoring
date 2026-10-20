from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def _get_tracking_anchor_geometry(geometry):
    """永続ID照合に使う最大の連結Polygon部分を返す。"""

    if geometry is None or geometry.is_empty:
        return geometry

    if geometry.geom_type == "Polygon":
        return geometry

    polygon_parts = []

    if geometry.geom_type == "MultiPolygon":
        polygon_parts.extend(
            part
            for part in geometry.geoms
            if not part.is_empty and part.area > 0.0
        )

    elif geometry.geom_type == "GeometryCollection":
        for part in geometry.geoms:
            if part.geom_type == "Polygon":
                if not part.is_empty and part.area > 0.0:
                    polygon_parts.append(part)
            elif part.geom_type == "MultiPolygon":
                polygon_parts.extend(
                    polygon
                    for polygon in part.geoms
                    if (
                        not polygon.is_empty
                        and polygon.area > 0.0
                    )
                )

    if not polygon_parts:
        return geometry

    return max(
        polygon_parts,
        key=lambda part: float(part.area),
    )


def _select_maximum_cardinality_tracking_matches(
    match_candidates: list[dict],
    candidate_index_key: str,
) -> dict:
    """候補と永続IDを、成立する対応数が最大になるよう一対一化する。"""

    if not match_candidates:
        return {}

    matches_by_candidate: dict[object, list[dict]] = {}
    first_rank_by_candidate: dict[object, int] = {}
    seen_pairs = set()

    # 呼出し側で品質順に並べられた順序を、各候補内でも維持する。
    for rank, match in enumerate(match_candidates):
        candidate_index = match[candidate_index_key]
        persistent_lake_id = match["persistent_lake_id"]
        pair_key = (
            candidate_index,
            persistent_lake_id,
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        matches_by_candidate.setdefault(
            candidate_index,
            [],
        ).append(match)
        first_rank_by_candidate.setdefault(
            candidate_index,
            rank,
        )

    # 対応できるIDが少ない候補から処理する。例えば再分離時に、
    # 「LAKE_002だけに一致する上側湖」を先に確保し、
    # 「LAKE_001とLAKE_002の両方に一致する下側湖」が
    # LAKE_002を先取りすることを防ぐ。
    candidate_order = sorted(
        matches_by_candidate,
        key=lambda candidate_index: (
            len(matches_by_candidate[candidate_index]),
            first_rank_by_candidate[candidate_index],
            str(candidate_index),
        ),
    )

    selected_match_by_candidate: dict[object, dict] = {}
    candidate_by_track_id: dict[str, object] = {}

    def try_assign_candidate(
        candidate_index,
        visited_candidate_indices: set,
        visited_track_ids: set,
    ) -> bool:
        if candidate_index in visited_candidate_indices:
            return False

        visited_candidate_indices.add(candidate_index)

        for match in matches_by_candidate[candidate_index]:
            persistent_lake_id = match[
                "persistent_lake_id"
            ]

            if persistent_lake_id in visited_track_ids:
                continue

            visited_track_ids.add(persistent_lake_id)
            occupied_candidate_index = (
                candidate_by_track_id.get(
                    persistent_lake_id
                )
            )

            if (
                occupied_candidate_index is None
                or try_assign_candidate(
                    occupied_candidate_index,
                    visited_candidate_indices,
                    visited_track_ids,
                )
            ):
                candidate_by_track_id[
                    persistent_lake_id
                ] = candidate_index
                selected_match_by_candidate[
                    candidate_index
                ] = match
                return True

        return False

    # 増加路を探索することで、単純な先着順ではなく、成立する
    # 一対一対応の件数が最大になる組合せを求める。
    for candidate_index in candidate_order:
        try_assign_candidate(
            candidate_index,
            visited_candidate_indices=set(),
            visited_track_ids=set(),
        )

    return selected_match_by_candidate


def assign_multisensor_persistent_ids(
    validated_candidate_gdf: gpd.GeoDataFrame,
    output_gpkg: Path,
    output_layer: str,
    max_gap_days: int,
    min_overlap_smaller_ratio: float,
    min_iou: float,
    min_area_ratio: float,
    match_distance_by_sensor: dict,
    distance_only_max_centroid_multiplier: float,
    large_change_min_reference_overlap_ratio: float = 0.80,
    large_change_max_area_ratio: float = 0.20,
    large_change_confirmation_max_gap_days: int = 30,
    large_change_confirmation_min_overlap_smaller_ratio: float = 0.50,
    large_change_confirmation_min_area_ratio: float = 0.50,
    large_change_long_gap_min_days: int = 10,
    large_change_long_gap_min_reference_overlap_ratio: float = 0.50,
    large_change_long_gap_confirm_min_pending_coverage_ratio: float = 0.80,
    large_change_long_gap_confirm_min_area_ratio: float = 0.20,
) -> gpd.GeoDataFrame:

    required_columns = {
        "local_candidate_id",
        "observation_id",
        "date",
        "sensor_family",
        "quality_status",
        "area_m2",
        "candidate_role",
        "can_create_new_id",
        "requires_shadow_validation",
        "shadow_only",
        "shadow_temporal_supported",
        "shadow_filter_status",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(validated_candidate_gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            "永続ID追跡に必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )

    if validated_candidate_gdf.empty:
        raise ValueError(
            "永続ID追跡対象が空です。"
        )

    if validated_candidate_gdf.crs is None:
        raise ValueError(
            "永続ID追跡対象にCRSがありません。"
        )

    if not validated_candidate_gdf.crs.is_projected:
        raise ValueError(
            "距離と面積を使った永続ID追跡には"
            "投影座標系が必要です。"
        )

    if max_gap_days <= 0:
        raise ValueError(
            "max_gap_daysは0より大きくしてください。"
        )

    if distance_only_max_centroid_multiplier <= 0.0:
        raise ValueError(
            "distance_only_max_centroid_multiplierは"
            "0より大きくしてください。"
        )

    if not 0.0 <= min_overlap_smaller_ratio <= 1.0:
        raise ValueError(
            "min_overlap_smaller_ratioは"
            "0以上1以下にしてください。"
        )

    if not 0.0 <= min_iou <= 1.0:
        raise ValueError(
            "min_iouは0以上1以下にしてください。"
        )

    if not 0.0 <= min_area_ratio <= 1.0:
        raise ValueError(
            "min_area_ratioは0以上1以下にしてください。"
        )

    ratio_settings = {
        "large_change_min_reference_overlap_ratio": (
            large_change_min_reference_overlap_ratio
        ),
        "large_change_max_area_ratio": (
            large_change_max_area_ratio
        ),
        "large_change_confirmation_min_overlap_smaller_ratio": (
            large_change_confirmation_min_overlap_smaller_ratio
        ),
        "large_change_confirmation_min_area_ratio": (
            large_change_confirmation_min_area_ratio
        ),
        "large_change_long_gap_min_reference_overlap_ratio": (
            large_change_long_gap_min_reference_overlap_ratio
        ),
        "large_change_long_gap_confirm_min_pending_coverage_ratio": (
            large_change_long_gap_confirm_min_pending_coverage_ratio
        ),
        "large_change_long_gap_confirm_min_area_ratio": (
            large_change_long_gap_confirm_min_area_ratio
        ),
    }

    for setting_name, setting_value in ratio_settings.items():
        if not 0.0 <= setting_value <= 1.0:
            raise ValueError(
                f"{setting_name}は0以上1以下にしてください。"
            )

    if large_change_confirmation_max_gap_days <= 0:
        raise ValueError(
            "large_change_confirmation_max_gap_daysは"
            "0より大きくしてください。"
        )

    if large_change_long_gap_min_days <= 0:
        raise ValueError(
            "large_change_long_gap_min_daysは"
            "0より大きくしてください。"
        )

    working_gdf = (
        validated_candidate_gdf
        .copy()
        .reset_index(drop=True)
    )

    if (
        working_gdf["local_candidate_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "local_candidate_idが重複しています。"
        )

    working_gdf["_parsed_date"] = (
        pd.to_datetime(
            working_gdf["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if working_gdf["_parsed_date"].isna().any():
        invalid_ids = (
            working_gdf.loc[
                working_gdf["_parsed_date"].isna(),
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "日付を解釈できない候補があります: "
            + ", ".join(invalid_ids)
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
            "追跡できない無効ジオメトリがあります: "
            + ", ".join(invalid_ids)
        )

    working_gdf["can_create_new_id"] = (
        working_gdf["can_create_new_id"]
        .fillna(False)
        .astype(bool)
    )

    working_gdf["quality_status"] = (
        working_gdf["quality_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # AOIを十分に観測できていないLandsatシーンは、既存湖の
    # 継続確認には利用するが、新しい永続IDの起点にはしない。
    # 部分観測の端や雲マスク境界を新規湖と誤認することを防ぐ。
    incomplete_landsat_mask = (
        working_gdf["sensor_family"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("landsat")
        & working_gdf["quality_status"].ne("usable")
    )
    working_gdf.loc[
        incomplete_landsat_mask,
        "can_create_new_id",
    ] = False

    working_gdf["shadow_only"] = (
        working_gdf["shadow_only"]
        .fillna(False)
        .astype(bool)
    )

    working_gdf["shadow_temporal_supported"] = (
        working_gdf["shadow_temporal_supported"]
        .fillna(False)
        .astype(bool)
    )

    working_gdf["shadow_filter_status"] = (
        working_gdf["shadow_filter_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    working_gdf["persistent_lake_id"] = ""
    working_gdf["tracking_status"] = ""
    working_gdf["tracking_reference_date"] = ""
    working_gdf["tracking_gap_days"] = np.nan
    working_gdf["tracking_overlap_smaller_ratio"] = np.nan
    working_gdf["tracking_iou"] = np.nan
    working_gdf["tracking_distance_m"] = np.nan
    working_gdf["tracking_centroid_distance_m"] = np.nan
    working_gdf["tracking_area_ratio"] = np.nan
    working_gdf["tracking_match_method"] = ""
    working_gdf["tracking_anchor_area_m2"] = np.nan
    working_gdf["persistent_counted_in_lake_count"] = False
    working_gdf["large_change_status"] = "not_applicable"
    working_gdf["large_change_reference_id"] = ""
    working_gdf["large_change_reference_date"] = ""
    working_gdf["large_change_reference_overlap_ratio"] = np.nan
    working_gdf["large_change_initial_area_ratio"] = np.nan
    working_gdf["large_change_confirmation_date"] = ""
    working_gdf["large_change_confirmation_gap_days"] = np.nan
    working_gdf["large_change_confirmation_overlap_ratio"] = np.nan
    working_gdf["large_change_confirmation_area_ratio"] = np.nan
    working_gdf["large_change_initial_rule"] = ""
    working_gdf["large_change_confirmation_rule"] = ""

    # IDごとの最後に確認された形状を保存する。
    #
    # 合流などでその日に割り当てられなかったIDも、
    # max_gap_days以内であれば削除しない。
    active_tracks: dict[str, dict] = {}

    # 面積が急増したため通常照合から外れた候補を一時保持する。
    # 次の観測で同様の形状が確認されるまで、ID・湖数・面積には
    # 採用しない。
    pending_large_changes: dict[str, dict] = {}

    next_id_number = 1

    observation_dates = sorted(
        working_gdf["_parsed_date"]
        .drop_duplicates()
        .tolist()
    )

    for current_date in observation_dates:
        expired_pending_ids = []

        for persistent_lake_id, pending in (
            pending_large_changes.items()
        ):
            pending_gap_days = int(
                (
                    current_date
                    - pending["candidate_date"]
                ).days
            )

            if (
                pending_gap_days
                > large_change_confirmation_max_gap_days
            ):
                pending_index = pending["candidate_index"]
                working_gdf.at[
                    pending_index,
                    "large_change_status",
                ] = "rejected_no_temporal_confirmation"
                expired_pending_ids.append(
                    persistent_lake_id
                )

        for persistent_lake_id in expired_pending_ids:
            pending_large_changes.pop(
                persistent_lake_id,
                None,
            )

        current_indices = (
            working_gdf.index[
                working_gdf["_parsed_date"]
                == current_date
            ]
            .tolist()
        )

        # 処理順によりIDが変わりにくいように、
        # コア候補を先、その後に継続専用候補を並べる。
        current_indices = sorted(
            current_indices,
            key=lambda index: (
                not bool(
                    working_gdf.at[
                        index,
                        "can_create_new_id",
                    ]
                ),
                str(
                    working_gdf.at[
                        index,
                        "local_candidate_id",
                    ]
                ),
            ),
        )

        eligible_track_ids = []

        for persistent_lake_id, track in (
            active_tracks.items()
        ):
            gap_days = int(
                (
                    current_date
                    - track["last_date"]
                ).days
            )

            if 0 < gap_days <= max_gap_days:
                eligible_track_ids.append(
                    persistent_lake_id
        )

        match_candidates = []
        large_change_relations_by_candidate = {
            current_index: []
            for current_index in current_indices
        }

        for current_index in current_indices:
            current_geometry = (
                working_gdf.at[
                    current_index,
                    "geometry",
                ]
            )

            # 面積集計には候補全体を使う一方、ID照合には最大の
            # 連結Polygon部分だけを使う。これにより、別の湖に
            # 近い小断片がMultiPolygonへ含まれていても、その
            # 小断片だけを根拠に旧IDが乗り移ることを防ぐ。
            current_anchor_geometry = (
                _get_tracking_anchor_geometry(
                    current_geometry
                )
            )
            current_anchor_area_m2 = float(
                current_anchor_geometry.area
            )

            working_gdf.at[
                current_index,
                "tracking_anchor_area_m2",
            ] = current_anchor_area_m2

            current_sensor = str(
                working_gdf.at[
                    current_index,
                    "sensor_family",
                ]
            )

            current_shadow_only = bool(
                working_gdf.at[
                    current_index,
                    "shadow_only",
                ]
            )

            current_shadow_temporally_supported = bool(
                working_gdf.at[
                    current_index,
                    "shadow_temporal_supported",
                ]
            )

            current_shadow_filter_status = str(
                working_gdf.at[
                    current_index,
                    "shadow_filter_status",
                ]
            )

            shadow_large_change_allowed = bool(
                not current_shadow_only
                or (
                    current_shadow_temporally_supported
                    and current_shadow_filter_status
                    == "kept_temporal_match"
                )
            )

            current_max_distance_m = float(
                match_distance_by_sensor.get(
                    current_sensor,
                    0.0,
                )
            )

            for persistent_lake_id in (
                eligible_track_ids
            ):
                track = active_tracks[
                    persistent_lake_id
                ]

                previous_geometry = (
                    track["geometry"]
                )

                previous_anchor_geometry = (
                    track.get(
                        "anchor_geometry",
                        _get_tracking_anchor_geometry(
                            previous_geometry
                        ),
                    )
                )

                previous_anchor_area_m2 = float(
                    previous_anchor_geometry.area
                )

                if (
                    current_anchor_area_m2 <= 0.0
                    or previous_anchor_area_m2 <= 0.0
                ):
                    continue

                previous_sensor = str(
                    track["sensor_family"]
                )

                previous_max_distance_m = float(
                    match_distance_by_sensor.get(
                        previous_sensor,
                        0.0,
                    )
                )

                # 異なるセンサー同士では、
                # 粗い方の位置誤差を考慮する。
                allowed_distance_m = max(
                    current_max_distance_m,
                    previous_max_distance_m,
                )

                intersection_area_m2 = float(
                    current_anchor_geometry
                    .intersection(
                        previous_anchor_geometry
                    )
                    .area
                )

                smaller_area_m2 = min(
                    current_anchor_area_m2,
                    previous_anchor_area_m2,
                )

                union_area_m2 = float(
                    current_anchor_geometry
                    .union(
                        previous_anchor_geometry
                    )
                    .area
                )

                overlap_smaller_ratio = (
                    intersection_area_m2
                    / smaller_area_m2
                    if smaller_area_m2 > 0.0
                    else 0.0
                )

                iou = (
                    intersection_area_m2
                    / union_area_m2
                    if union_area_m2 > 0.0
                    else 0.0
                )

                area_ratio = (
                    smaller_area_m2
                    / max(
                        current_anchor_area_m2,
                        previous_anchor_area_m2,
                    )
                )

                distance_m = float(
                    current_anchor_geometry.distance(
                        previous_anchor_geometry
                    )
                )

                centroid_distance_m = float(
                    current_anchor_geometry
                    .centroid
                    .distance(
                        previous_anchor_geometry.centroid
                    )
                )

                max_centroid_distance_m = float(
                    allowed_distance_m
                    * distance_only_max_centroid_multiplier
                )

                # 過去形状のうち、現在候補と重なる割合。
                # 通常照合の面積比を満たさない急拡大候補の検出にも
                # 使用するため、照合可否の判定前に計算する。
                previous_overlap_ratio = (
                    intersection_area_m2
                    / previous_anchor_area_m2
                    if previous_anchor_area_m2 > 0.0
                    else 0.0
                )

                reference_gap_days = int(
                    (
                        current_date
                        - track["last_date"]
                    ).days
                )

                standard_initial_rule = bool(
                    previous_overlap_ratio
                    >= large_change_min_reference_overlap_ratio
                )

                long_gap_initial_rule = bool(
                    reference_gap_days
                    >= large_change_long_gap_min_days
                    and previous_overlap_ratio
                    >= (
                        large_change_long_gap_min_reference_overlap_ratio
                    )
                )

                is_large_change_relation = bool(
                    shadow_large_change_allowed
                    and current_anchor_area_m2
                    > previous_anchor_area_m2
                    and intersection_area_m2 > 0.0
                    and (
                        standard_initial_rule
                        or long_gap_initial_rule
                    )
                    and area_ratio < min_area_ratio
                    and area_ratio
                    <= large_change_max_area_ratio
                )

                if is_large_change_relation:
                    large_change_relations_by_candidate[
                        current_index
                    ].append(
                        {
                            "persistent_lake_id": (
                                persistent_lake_id
                            ),
                            "reference_date": (
                                track["last_date"]
                            ),
                            "previous_overlap_ratio": (
                                previous_overlap_ratio
                            ),
                            "intersection_area_m2": (
                                intersection_area_m2
                            ),
                            "overlap_smaller_ratio": (
                                overlap_smaller_ratio
                            ),
                            "iou": iou,
                            "distance_m": distance_m,
                            "centroid_distance_m": (
                                centroid_distance_m
                            ),
                            "area_ratio": area_ratio,
                            "initial_rule": (
                                "long_gap_growth"
                                if (
                                    long_gap_initial_rule
                                    and not standard_initial_rule
                                )
                                else "standard"
                            ),
                        }
                    )

                if current_shadow_only:
                    # 雲影だけで構成された候補は、接触・近接だけでは
                    # 既存IDへ接続しない。形状・面積のすべてが一定以上
                    # 一致した場合だけ継続を許可する。
                    overlap_match = bool(
                        overlap_smaller_ratio
                        >= min_overlap_smaller_ratio
                        and iou >= min_iou
                        and area_ratio >= min_area_ratio
                    )
                    distance_match = False

                else:
                    overlap_match = bool(
                        (
                            overlap_smaller_ratio
                            >= min_overlap_smaller_ratio
                            or iou >= min_iou
                        )
                        and area_ratio >= min_area_ratio
                    )

                    distance_match = bool(
                        allowed_distance_m > 0.0
                        and distance_m
                        <= allowed_distance_m
                        and centroid_distance_m
                        <= max_centroid_distance_m
                        and area_ratio
                        >= min_area_ratio
                    )

                if not (
                    overlap_match
                    or distance_match
                ):
                    continue

                gap_days = int(
                    (
                        current_date
                        - track["last_date"]
                    ).days
                )

                # 重なり一致を距離一致より優先する。
                if overlap_match:
                    match_type_priority = 2
                else:
                    match_type_priority = 1

                match_candidates.append(
                    {
                        "current_index": (
                            current_index
                        ),
                        "persistent_lake_id": (
                            persistent_lake_id
                        ),
                        "match_type_priority": (
                            match_type_priority
                        ),
                        "reference_date": (
                            track["last_date"]
                        ),
                        "gap_days": gap_days,
                        "intersection_area_m2": (
                            intersection_area_m2
                        ),
                        "previous_overlap_ratio": (
                            previous_overlap_ratio
                        ),
                        "overlap_smaller_ratio": (
                            overlap_smaller_ratio
                        ),
                        "iou": iou,
                        "distance_m": distance_m,
                        "centroid_distance_m": (
                            centroid_distance_m
                        ),
                        "area_ratio": area_ratio,
                        "match_method": (
                            "overlap"
                            if overlap_match
                            else "distance"
                        ),
                    }
                )

        # 候補と永続IDの組合せを、以下の順番で優先する。
        #
        # 1. 重なり照合
        # 2. 観測間隔（直近の追跡形状を優先）
        # 3. 過去形状に対する交差率
        # 4. 交差面積
        # 5. IoU
        # 6. 面積比
        # 7. 小さい方を基準とした重なり率
        # 8. 距離
        #
        # 観測間隔を重なり率より後にすると、現在の大きな
        # ポリゴンが古い小ポリゴンを完全に包含した場合に、
        # 直近の同一湖IDではなく休眠中の古いIDを取得してしまう。
        # 直近日付を先にすることで、そのIDの入れ替わりを防ぐ。
        match_candidates = sorted(
            match_candidates,
            key=lambda match: (
                -match[
                    "match_type_priority"
                ],
                match["gap_days"],
                -match[
                    "previous_overlap_ratio"
                ],
                -match[
                    "intersection_area_m2"
                ],
                -match["iou"],
                -match["area_ratio"],
                -match[
                    "overlap_smaller_ratio"
                ],
                match["distance_m"],
                match["centroid_distance_m"],
                str(
                    working_gdf.at[
                        match["current_index"],
                        "local_candidate_id",
                    ]
                ),
                match["persistent_lake_id"],
            ),
        )

        # 同じ日付では候補とIDを一対一にし、成立する対応数を
        # 最大化する。これにより、合流後の再分離時に一方の
        # 候補が複数IDを先取りし、もう一方へ新規IDが作られる
        # ことを防ぐ。
        accepted_matches = (
            _select_maximum_cardinality_tracking_matches(
                match_candidates=match_candidates,
                candidate_index_key="current_index",
            )
        )

        used_track_ids = {
            match["persistent_lake_id"]
            for match in accepted_matches.values()
        }
        pending_confirmation_candidates = []

        for current_index in current_indices:
            if current_index in accepted_matches:
                continue

            current_geometry = working_gdf.at[
                current_index,
                "geometry",
            ]
            current_anchor_geometry = (
                _get_tracking_anchor_geometry(
                    current_geometry
                )
            )
            current_anchor_area_m2 = float(
                current_anchor_geometry.area
            )

            if current_anchor_area_m2 <= 0.0:
                continue

            for persistent_lake_id, pending in (
                pending_large_changes.items()
            ):
                if persistent_lake_id in used_track_ids:
                    continue

                confirmation_gap_days = int(
                    (
                        current_date
                        - pending["candidate_date"]
                    ).days
                )

                if not (
                    0 < confirmation_gap_days
                    <= large_change_confirmation_max_gap_days
                ):
                    continue

                pending_anchor_geometry = pending[
                    "anchor_geometry"
                ]
                pending_anchor_area_m2 = float(
                    pending_anchor_geometry.area
                )

                if pending_anchor_area_m2 <= 0.0:
                    continue

                intersection_area_m2 = float(
                    current_anchor_geometry
                    .intersection(
                        pending_anchor_geometry
                    )
                    .area
                )
                smaller_area_m2 = min(
                    current_anchor_area_m2,
                    pending_anchor_area_m2,
                )
                larger_area_m2 = max(
                    current_anchor_area_m2,
                    pending_anchor_area_m2,
                )
                union_area_m2 = float(
                    current_anchor_geometry
                    .union(
                        pending_anchor_geometry
                    )
                    .area
                )
                overlap_smaller_ratio = (
                    intersection_area_m2
                    / smaller_area_m2
                    if smaller_area_m2 > 0.0
                    else 0.0
                )
                iou = (
                    intersection_area_m2
                    / union_area_m2
                    if union_area_m2 > 0.0
                    else 0.0
                )
                area_ratio = (
                    smaller_area_m2
                    / larger_area_m2
                    if larger_area_m2 > 0.0
                    else 0.0
                )

                pending_coverage_ratio = (
                    intersection_area_m2
                    / pending_anchor_area_m2
                    if pending_anchor_area_m2 > 0.0
                    else 0.0
                )

                standard_confirmation = bool(
                    overlap_smaller_ratio
                    >= large_change_confirmation_min_overlap_smaller_ratio
                    and area_ratio
                    >= large_change_confirmation_min_area_ratio
                )

                # 長い観測間隔の間にも成長が続くと、後続形状との
                # 面積比が通常の0.50を下回ることがある。保留形状の
                # 大部分を包含し、さらに成長している場合だけ確認する。
                long_gap_growth_confirmation = bool(
                    confirmation_gap_days
                    >= large_change_long_gap_min_days
                    and current_anchor_area_m2
                    >= pending_anchor_area_m2
                    and pending_coverage_ratio
                    >= (
                        large_change_long_gap_confirm_min_pending_coverage_ratio
                    )
                    and area_ratio
                    >= large_change_long_gap_confirm_min_area_ratio
                )

                if not (
                    standard_confirmation
                    or long_gap_growth_confirmation
                ):
                    continue

                distance_m = float(
                    current_anchor_geometry.distance(
                        pending_anchor_geometry
                    )
                )
                centroid_distance_m = float(
                    current_anchor_geometry
                    .centroid
                    .distance(
                        pending_anchor_geometry.centroid
                    )
                )

                pending_confirmation_candidates.append(
                    {
                        "current_index": current_index,
                        "persistent_lake_id": (
                            persistent_lake_id
                        ),
                        "reference_date": (
                            pending["candidate_date"]
                        ),
                        "gap_days": (
                            confirmation_gap_days
                        ),
                        "intersection_area_m2": (
                            intersection_area_m2
                        ),
                        "previous_overlap_ratio": (
                            intersection_area_m2
                            / pending_anchor_area_m2
                        ),
                        "overlap_smaller_ratio": (
                            overlap_smaller_ratio
                        ),
                        "iou": iou,
                        "distance_m": distance_m,
                        "centroid_distance_m": (
                            centroid_distance_m
                        ),
                        "area_ratio": area_ratio,
                        "match_method": (
                            "confirmed_long_gap_continued_growth"
                            if (
                                long_gap_growth_confirmation
                                and not standard_confirmation
                            )
                            else "confirmed_large_change"
                        ),
                        "pending_index": (
                            pending["candidate_index"]
                        ),
                        "is_large_change_confirmation": True,
                        "confirmation_rule": (
                            "long_gap_continued_growth"
                            if (
                                long_gap_growth_confirmation
                                and not standard_confirmation
                            )
                            else "standard"
                        ),
                        "initial_rule": (
                            pending["initial_relation"].get(
                                "initial_rule",
                                "standard",
                            )
                        ),
                    }
                )

        pending_confirmation_candidates = sorted(
            pending_confirmation_candidates,
            key=lambda match: (
                match["gap_days"],
                -match["overlap_smaller_ratio"],
                -match["area_ratio"],
                -match["iou"],
                match["centroid_distance_m"],
                str(
                    working_gdf.at[
                        match["current_index"],
                        "local_candidate_id",
                    ]
                ),
                match["persistent_lake_id"],
            ),
        )

        confirmed_pending_matches = (
            _select_maximum_cardinality_tracking_matches(
                match_candidates=(
                    pending_confirmation_candidates
                ),
                candidate_index_key="current_index",
            )
        )

        for current_index, match in (
            confirmed_pending_matches.items()
        ):
            persistent_lake_id = match[
                "persistent_lake_id"
            ]
            pending = pending_large_changes[
                persistent_lake_id
            ]
            pending_index = pending["candidate_index"]
            initial_relation = pending[
                "initial_relation"
            ]

            # 確認待ちだった初回急拡大候補を、確認日の情報により
            # 元の永続IDへ遡って採用する。
            working_gdf.at[
                pending_index,
                "persistent_lake_id",
            ] = persistent_lake_id
            working_gdf.at[
                pending_index,
                "tracking_status",
            ] = "confirmed_large_change"
            working_gdf.at[
                pending_index,
                "tracking_reference_date",
            ] = initial_relation[
                "reference_date"
            ].strftime("%Y-%m-%d")
            working_gdf.at[
                pending_index,
                "tracking_gap_days",
            ] = int(
                (
                    pending["candidate_date"]
                    - initial_relation["reference_date"]
                ).days
            )
            working_gdf.at[
                pending_index,
                "tracking_overlap_smaller_ratio",
            ] = initial_relation[
                "overlap_smaller_ratio"
            ]
            working_gdf.at[
                pending_index,
                "tracking_iou",
            ] = initial_relation["iou"]
            working_gdf.at[
                pending_index,
                "tracking_distance_m",
            ] = initial_relation["distance_m"]
            working_gdf.at[
                pending_index,
                "tracking_centroid_distance_m",
            ] = initial_relation[
                "centroid_distance_m"
            ]
            working_gdf.at[
                pending_index,
                "tracking_area_ratio",
            ] = initial_relation["area_ratio"]
            working_gdf.at[
                pending_index,
                "tracking_match_method",
            ] = "large_change_temporally_confirmed"
            working_gdf.at[
                pending_index,
                "persistent_counted_in_lake_count",
            ] = True
            working_gdf.at[
                pending_index,
                "large_change_status",
            ] = "confirmed_from_next_observation"
            working_gdf.at[
                pending_index,
                "large_change_confirmation_date",
            ] = current_date.strftime("%Y-%m-%d")
            working_gdf.at[
                pending_index,
                "large_change_confirmation_gap_days",
            ] = match["gap_days"]
            working_gdf.at[
                pending_index,
                "large_change_confirmation_overlap_ratio",
            ] = match["overlap_smaller_ratio"]
            working_gdf.at[
                pending_index,
                "large_change_confirmation_area_ratio",
            ] = match["area_ratio"]
            working_gdf.at[
                pending_index,
                "large_change_initial_rule",
            ] = initial_relation.get(
                "initial_rule",
                "standard",
            )
            working_gdf.at[
                pending_index,
                "large_change_confirmation_rule",
            ] = match.get(
                "confirmation_rule",
                "standard",
            )

            accepted_matches[current_index] = match
            pending_large_changes.pop(
                persistent_lake_id,
                None,
            )

        # 確認待ちの後に元の大きさへ戻り、通常条件で旧IDが継続
        # できた場合、最初の急拡大は一時的誤検出として確定する。
        normally_matched_track_ids = {
            match["persistent_lake_id"]
            for match in accepted_matches.values()
            if not match.get(
                "is_large_change_confirmation",
                False,
            )
        }

        for persistent_lake_id in sorted(
            normally_matched_track_ids
        ):
            pending = pending_large_changes.pop(
                persistent_lake_id,
                None,
            )

            if pending is None:
                continue

            working_gdf.at[
                pending["candidate_index"],
                "large_change_status",
            ] = "rejected_next_observation_did_not_confirm"

        assigned_track_ids_this_date = {
            match["persistent_lake_id"]
            for match in accepted_matches.values()
        }

        for current_index in current_indices:
            current_geometry = (
                working_gdf.at[
                    current_index,
                    "geometry",
                ]
            )

            # この候補をactive_tracksへ保存するときも、照合時と
            # 同じ最大連結部分を基準形状として保持する。
            current_anchor_geometry = (
                _get_tracking_anchor_geometry(
                    current_geometry
                )
            )

            current_sensor = str(
                working_gdf.at[
                    current_index,
                    "sensor_family",
                ]
            )

            if current_index in accepted_matches:
                match = accepted_matches[
                    current_index
                ]

                persistent_lake_id = match[
                    "persistent_lake_id"
                ]

                working_gdf.at[
                    current_index,
                    "persistent_lake_id",
                ] = persistent_lake_id

                working_gdf.at[
                    current_index,
                    "tracking_status",
                ] = (
                    "continued_confirmed_large_change"
                    if match.get(
                        "is_large_change_confirmation",
                        False,
                    )
                    else "continued_existing_id"
                )

                working_gdf.at[
                    current_index,
                    "tracking_reference_date",
                ] = match[
                    "reference_date"
                ].strftime("%Y-%m-%d")

                working_gdf.at[
                    current_index,
                    "tracking_gap_days",
                ] = match["gap_days"]

                working_gdf.at[
                    current_index,
                    "tracking_overlap_smaller_ratio",
                ] = match[
                    "overlap_smaller_ratio"
                ]

                working_gdf.at[
                    current_index,
                    "tracking_iou",
                ] = match["iou"]

                working_gdf.at[
                    current_index,
                    "tracking_distance_m",
                ] = match["distance_m"]

                working_gdf.at[
                    current_index,
                    "tracking_centroid_distance_m",
                ] = match[
                    "centroid_distance_m"
                ]

                working_gdf.at[
                    current_index,
                    "tracking_area_ratio",
                ] = match["area_ratio"]

                working_gdf.at[
                    current_index,
                    "tracking_match_method",
                ] = match["match_method"]

                working_gdf.at[
                    current_index,
                    "persistent_counted_in_lake_count",
                ] = True

                if match.get(
                    "is_large_change_confirmation",
                    False,
                ):
                    working_gdf.at[
                        current_index,
                        "large_change_status",
                    ] = "confirmation_observation"
                    working_gdf.at[
                        current_index,
                        "large_change_reference_id",
                    ] = persistent_lake_id
                    working_gdf.at[
                        current_index,
                        "large_change_reference_date",
                    ] = match[
                        "reference_date"
                    ].strftime("%Y-%m-%d")
                    working_gdf.at[
                        current_index,
                        "large_change_confirmation_date",
                    ] = current_date.strftime("%Y-%m-%d")
                    working_gdf.at[
                        current_index,
                        "large_change_confirmation_gap_days",
                    ] = match["gap_days"]
                    working_gdf.at[
                        current_index,
                        "large_change_confirmation_overlap_ratio",
                    ] = match[
                        "overlap_smaller_ratio"
                    ]
                    working_gdf.at[
                        current_index,
                        "large_change_confirmation_area_ratio",
                    ] = match["area_ratio"]
                    working_gdf.at[
                        current_index,
                        "large_change_initial_rule",
                    ] = match.get(
                        "initial_rule",
                        "standard",
                    )
                    working_gdf.at[
                        current_index,
                        "large_change_confirmation_rule",
                    ] = match.get(
                        "confirmation_rule",
                        "standard",
                    )

                active_tracks[
                    persistent_lake_id
                ] = {
                    "geometry": current_geometry,
                    "anchor_geometry": (
                        current_anchor_geometry
                    ),
                    "last_date": current_date,
                    "sensor_family": current_sensor,
                    "local_candidate_id": str(
                        working_gdf.at[
                            current_index,
                            "local_candidate_id",
                        ]
                    ),
                }

                continue

            can_create_new_id = bool(
                working_gdf.at[
                    current_index,
                    "can_create_new_id",
                ]
            )

            large_change_relations = sorted(
                large_change_relations_by_candidate.get(
                    current_index,
                    [],
                ),
                key=lambda relation: (
                    -relation[
                        "previous_overlap_ratio"
                    ],
                    -relation[
                        "intersection_area_m2"
                    ],
                    -relation["area_ratio"],
                    relation["persistent_lake_id"],
                ),
            )

            # 継続専用候補も新規IDは作らないまま、既存IDに対する
            # 急増確認待ちへ進める。後続観測で確認されなければ
            # 従来どおり面積・湖数には採用しない。
            if large_change_relations:
                related_track_ids = sorted(
                    {
                        relation["persistent_lake_id"]
                        for relation in large_change_relations
                    }
                )
                best_relation = large_change_relations[0]

                working_gdf.at[
                    current_index,
                    "large_change_reference_id",
                ] = ";".join(related_track_ids)
                working_gdf.at[
                    current_index,
                    "large_change_reference_date",
                ] = best_relation[
                    "reference_date"
                ].strftime("%Y-%m-%d")
                working_gdf.at[
                    current_index,
                    "large_change_reference_overlap_ratio",
                ] = best_relation[
                    "previous_overlap_ratio"
                ]
                working_gdf.at[
                    current_index,
                    "large_change_initial_area_ratio",
                ] = best_relation["area_ratio"]
                working_gdf.at[
                    current_index,
                    "large_change_initial_rule",
                ] = best_relation.get(
                    "initial_rule",
                    "standard",
                )
                working_gdf.at[
                    current_index,
                    "persistent_counted_in_lake_count",
                ] = False

                if len(related_track_ids) > 1:
                    working_gdf.at[
                        current_index,
                        "tracking_status",
                    ] = "ambiguous_large_change"
                    working_gdf.at[
                        current_index,
                        "tracking_match_method",
                    ] = "large_change_multiple_prior_ids"
                    working_gdf.at[
                        current_index,
                        "large_change_status",
                    ] = "rejected_ambiguous_multiple_prior_ids"
                    continue

                related_track_id = related_track_ids[0]

                if (
                    related_track_id
                    in assigned_track_ids_this_date
                ):
                    working_gdf.at[
                        current_index,
                        "tracking_status",
                    ] = "blocked_large_change"
                    working_gdf.at[
                        current_index,
                        "tracking_match_method",
                    ] = "large_change_prior_id_already_used"
                    working_gdf.at[
                        current_index,
                        "large_change_status",
                    ] = "rejected_prior_id_already_used_same_date"
                    continue

                if related_track_id in pending_large_changes:
                    working_gdf.at[
                        current_index,
                        "tracking_status",
                    ] = "blocked_large_change"
                    working_gdf.at[
                        current_index,
                        "tracking_match_method",
                    ] = "large_change_conflicts_with_pending"
                    working_gdf.at[
                        current_index,
                        "large_change_status",
                    ] = "rejected_conflicting_pending_candidate"
                    continue

                working_gdf.at[
                    current_index,
                    "tracking_status",
                ] = "pending_large_change"
                working_gdf.at[
                    current_index,
                    "tracking_match_method",
                ] = "pending_large_change_confirmation"
                working_gdf.at[
                    current_index,
                    "large_change_status",
                ] = "pending_temporal_confirmation"

                pending_large_changes[related_track_id] = {
                    "candidate_index": current_index,
                    "candidate_date": current_date,
                    "geometry": current_geometry,
                    "anchor_geometry": (
                        current_anchor_geometry
                    ),
                    "sensor_family": current_sensor,
                    "initial_relation": best_relation,
                }
                continue

            if can_create_new_id:
                persistent_lake_id = (
                    f"LAKE_{next_id_number:03d}"
                )

                next_id_number += 1

                working_gdf.at[
                    current_index,
                    "persistent_lake_id",
                ] = persistent_lake_id

                working_gdf.at[
                    current_index,
                    "tracking_status",
                ] = "created_new_id"

                working_gdf.at[
                    current_index,
                    "tracking_match_method",
                ] = "new"

                working_gdf.at[
                    current_index,
                    "persistent_counted_in_lake_count",
                ] = True

                active_tracks[
                    persistent_lake_id
                ] = {
                    "geometry": current_geometry,
                    "anchor_geometry": (
                        current_anchor_geometry
                    ),
                    "last_date": current_date,
                    "sensor_family": current_sensor,
                    "local_candidate_id": str(
                        working_gdf.at[
                            current_index,
                            "local_candidate_id",
                        ]
                    ),
                }

            else:
                working_gdf.at[
                    current_index,
                    "tracking_status",
                ] = "unmatched_continuation"

                working_gdf.at[
                    current_index,
                    "tracking_match_method",
                ] = "unmatched"

                working_gdf.at[
                    current_index,
                    "persistent_counted_in_lake_count",
                ] = False

    for pending in pending_large_changes.values():
        pending_index = pending["candidate_index"]

        if (
            working_gdf.at[
                pending_index,
                "large_change_status",
            ]
            == "pending_temporal_confirmation"
        ):
            working_gdf.at[
                pending_index,
                "large_change_status",
            ] = "rejected_end_of_series_unconfirmed"

    output_gdf = (
        working_gdf
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

    assigned_gdf = (
        output_gdf.loc[
            output_gdf[
                "persistent_counted_in_lake_count"
            ]
            .astype(bool)
        ]
    )

    duplicate_id_date_mask = (
        assigned_gdf.duplicated(
            subset=[
                "date",
                "persistent_lake_id",
            ],
            keep=False,
        )
    )

    if duplicate_id_date_mask.any():
        duplicate_rows = (
            assigned_gdf.loc[
                duplicate_id_date_mask,
                [
                    "date",
                    "persistent_lake_id",
                    "local_candidate_id",
                ],
            ]
        )

        raise ValueError(
            "同じ日付に同じ永続IDが"
            "複数回割り当てられています:\n"
            + duplicate_rows.to_string(
                index=False
            )
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
            "永続ID追跡GeoPackageを"
            "保存できません。"
            "QGISで開いている場合は"
            "閉じてください: "
            f"{output_gpkg}"
        ) from error

    return output_gdf


def backfill_multisensor_persistent_ids(
    persistent_tracking_gdf: gpd.GeoDataFrame,
    output_gpkg: Path,
    output_layer: str,
    max_gap_days: int,
    min_overlap_smaller_ratio: float,
    min_iou: float,
    min_area_ratio: float,
    match_distance_by_sensor: dict,
    distance_only_max_centroid_multiplier: float,
) -> gpd.GeoDataFrame:

    required_columns = {
        "local_candidate_id",
        "date",
        "sensor_family",
        "area_m2",
        "can_create_new_id",
        "persistent_lake_id",
        "tracking_status",
        "persistent_counted_in_lake_count",
        "shadow_only",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(persistent_tracking_gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            "過去方向のID遡及に必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )

    if persistent_tracking_gdf.empty:
        raise ValueError(
            "過去方向のID遡及対象が空です。"
        )

    if persistent_tracking_gdf.crs is None:
        raise ValueError(
            "遡及対象にCRSが設定されていません。"
        )

    if not persistent_tracking_gdf.crs.is_projected:
        raise ValueError(
            "距離を使用するID遡及には"
            "投影座標系が必要です。"
        )

    if max_gap_days <= 0:
        raise ValueError(
            "max_gap_daysは0より大きくしてください。"
        )

    if distance_only_max_centroid_multiplier <= 0.0:
        raise ValueError(
            "distance_only_max_centroid_multiplierは"
            "0より大きくしてください。"
        )

    if not 0.0 <= min_overlap_smaller_ratio <= 1.0:
        raise ValueError(
            "min_overlap_smaller_ratioは"
            "0以上1以下にしてください。"
        )

    if not 0.0 <= min_iou <= 1.0:
        raise ValueError(
            "min_iouは0以上1以下にしてください。"
        )

    if not 0.0 <= min_area_ratio <= 1.0:
        raise ValueError(
            "min_area_ratioは0以上1以下にしてください。"
        )

    working_gdf = (
        persistent_tracking_gdf
        .copy()
        .reset_index(drop=True)
    )

    if (
        working_gdf["local_candidate_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "local_candidate_idが重複しています。"
        )

    working_gdf["_parsed_date"] = (
        pd.to_datetime(
            working_gdf["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if working_gdf["_parsed_date"].isna().any():
        invalid_ids = (
            working_gdf.loc[
                working_gdf["_parsed_date"].isna(),
                "local_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "日付を解釈できない候補があります: "
            + ", ".join(invalid_ids)
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
            "遡及できない無効ジオメトリがあります: "
            + ", ".join(invalid_ids)
        )

    working_gdf["persistent_lake_id"] = (
        working_gdf["persistent_lake_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    working_gdf["can_create_new_id"] = (
        working_gdf["can_create_new_id"]
        .fillna(False)
        .astype(bool)
    )

    working_gdf[
        "persistent_counted_in_lake_count"
    ] = (
        working_gdf[
            "persistent_counted_in_lake_count"
        ]
        .fillna(False)
        .astype(bool)
    )

    working_gdf["shadow_only"] = (
        working_gdf["shadow_only"]
        .fillna(False)
        .astype(bool)
    )

    assigned_mask = (
        working_gdf["persistent_lake_id"]
        .ne("")
    )

    working_gdf["retrospective_backfill"] = False

    working_gdf["retrospective_status"] = np.where(
        assigned_mask,
        "already_assigned",
        "no_future_match",
    )

    working_gdf["retrospective_anchor_date"] = ""
    working_gdf["retrospective_reference_date"] = ""
    working_gdf["retrospective_gap_days"] = np.nan
    working_gdf[
        "retrospective_intersection_area_m2"
    ] = np.nan
    working_gdf[
        "retrospective_reference_overlap_ratio"
    ] = np.nan
    working_gdf[
        "retrospective_overlap_smaller_ratio"
    ] = np.nan
    working_gdf["retrospective_iou"] = np.nan
    working_gdf["retrospective_distance_m"] = np.nan
    working_gdf[
        "retrospective_centroid_distance_m"
    ] = np.nan
    working_gdf["retrospective_area_ratio"] = np.nan
    working_gdf["retrospective_match_method"] = ""
    working_gdf["retrospective_anchor_area_m2"] = np.nan

    # 各IDについて、最も古い割当済み形状を
    # 過去方向追跡の開始地点にする。
    backward_tracks = {}

    persistent_lake_ids = sorted(
        working_gdf.loc[
            assigned_mask,
            "persistent_lake_id",
        ]
        .drop_duplicates()
        .tolist()
    )

    for persistent_lake_id in persistent_lake_ids:
        lake_rows = (
            working_gdf.loc[
                working_gdf["persistent_lake_id"]
                == persistent_lake_id
            ]
            .sort_values(
                [
                    "_parsed_date",
                    "local_candidate_id",
                ]
            )
        )

        if lake_rows.empty:
            continue

        anchor_row = lake_rows.iloc[0]
        anchor_date = anchor_row["_parsed_date"]

        anchor_geometry = _get_tracking_anchor_geometry(
            anchor_row.geometry
        )

        backward_tracks[persistent_lake_id] = {
            "geometry": anchor_row.geometry,
            "anchor_geometry": anchor_geometry,
            "reference_date": anchor_date,
            "anchor_date": anchor_date,
            "sensor_family": str(
                anchor_row["sensor_family"]
            ),
        }

    observation_dates = sorted(
        working_gdf["_parsed_date"]
        .drop_duplicates()
        .tolist(),
        reverse=True,
    )

    for current_date in observation_dates:
        candidate_indices = (
            working_gdf.index[
                (
                    working_gdf["_parsed_date"]
                    == current_date
                )
                & (
                    working_gdf["persistent_lake_id"]
                    == ""
                )
                & (
                    ~working_gdf["can_create_new_id"]
                )
            ]
            .tolist()
        )

        if not candidate_indices:
            continue

        eligible_track_ids = []

        for persistent_lake_id, track in (
            backward_tracks.items()
        ):
            gap_days = int(
                (
                    track["reference_date"]
                    - current_date
                ).days
            )

            if 0 < gap_days <= max_gap_days:
                eligible_track_ids.append(
                    persistent_lake_id
                )

        if not eligible_track_ids:
            continue

        match_candidates = []
        candidate_matching_ids = {
            candidate_index: set()
            for candidate_index in candidate_indices
        }

        for candidate_index in candidate_indices:
            candidate_geometry = (
                working_gdf.at[
                    candidate_index,
                    "geometry",
                ]
            )

            candidate_anchor_geometry = (
                _get_tracking_anchor_geometry(
                    candidate_geometry
                )
            )

            candidate_anchor_area_m2 = float(
                candidate_anchor_geometry.area
            )

            working_gdf.at[
                candidate_index,
                "retrospective_anchor_area_m2",
            ] = candidate_anchor_area_m2

            candidate_sensor = str(
                working_gdf.at[
                    candidate_index,
                    "sensor_family",
                ]
            )

            candidate_anchor_geometry = (
                _get_tracking_anchor_geometry(
                    candidate_geometry
                )
            )

            candidate_shadow_only = bool(
                working_gdf.at[
                    candidate_index,
                    "shadow_only",
                ]
            )

            candidate_max_distance_m = float(
                match_distance_by_sensor.get(
                    candidate_sensor,
                    0.0,
                )
            )

            for persistent_lake_id in (
                eligible_track_ids
            ):
                track = backward_tracks[
                    persistent_lake_id
                ]

                reference_geometry = (
                    track["geometry"]
                )

                reference_anchor_geometry = (
                    track.get(
                        "anchor_geometry",
                        _get_tracking_anchor_geometry(
                            reference_geometry
                        ),
                    )
                )

                reference_anchor_area_m2 = float(
                    reference_anchor_geometry.area
                )

                if (
                    candidate_anchor_area_m2 <= 0.0
                    or reference_anchor_area_m2 <= 0.0
                ):
                    continue

                reference_sensor = str(
                    track["sensor_family"]
                )

                reference_max_distance_m = float(
                    match_distance_by_sensor.get(
                        reference_sensor,
                        0.0,
                    )
                )

                # 異なるセンサー間では、
                # 粗い方の最大照合距離を使用する。
                allowed_distance_m = max(
                    candidate_max_distance_m,
                    reference_max_distance_m,
                )

                intersection_area_m2 = float(
                    candidate_anchor_geometry
                    .intersection(
                        reference_anchor_geometry
                    )
                    .area
                )

                smaller_area_m2 = min(
                    candidate_anchor_area_m2,
                    reference_anchor_area_m2,
                )

                larger_area_m2 = max(
                    candidate_anchor_area_m2,
                    reference_anchor_area_m2,
                )

                union_area_m2 = float(
                    candidate_anchor_geometry
                    .union(
                        reference_anchor_geometry
                    )
                    .area
                )

                overlap_smaller_ratio = (
                    intersection_area_m2
                    / smaller_area_m2
                    if smaller_area_m2 > 0.0
                    else 0.0
                )

                reference_overlap_ratio = (
                    intersection_area_m2
                    / reference_anchor_area_m2
                    if reference_anchor_area_m2 > 0.0
                    else 0.0
                )

                iou = (
                    intersection_area_m2
                    / union_area_m2
                    if union_area_m2 > 0.0
                    else 0.0
                )

                area_ratio = (
                    smaller_area_m2
                    / larger_area_m2
                    if larger_area_m2 > 0.0
                    else 0.0
                )

                distance_m = float(
                    candidate_anchor_geometry.distance(
                        reference_anchor_geometry
                    )
                )

                centroid_distance_m = float(
                    candidate_anchor_geometry
                    .centroid
                    .distance(
                        reference_anchor_geometry.centroid
                    )
                )

                max_centroid_distance_m = float(
                    allowed_distance_m
                    * distance_only_max_centroid_multiplier
                )

                if candidate_shadow_only:
                    # 雲影候補の遡及では、境界接触や近接だけによる
                    # 将来IDの付与を禁止する。
                    overlap_match = bool(
                        intersection_area_m2 > 0.0
                        and overlap_smaller_ratio
                        >= min_overlap_smaller_ratio
                        and iou >= min_iou
                        and area_ratio >= min_area_ratio
                    )
                    distance_match = False

                else:
                    overlap_match = bool(
                        (
                            overlap_smaller_ratio
                            >= min_overlap_smaller_ratio
                            or iou >= min_iou
                        )
                        and area_ratio >= min_area_ratio
                    )

                    distance_match = bool(
                        allowed_distance_m > 0.0
                        and distance_m
                        <= allowed_distance_m
                        and centroid_distance_m
                        <= max_centroid_distance_m
                        and area_ratio
                        >= min_area_ratio
                    )

                if not (
                    overlap_match
                    or distance_match
                ):
                    continue

                gap_days = int(
                    (
                        track["reference_date"]
                        - current_date
                    ).days
                )

                if overlap_match:
                    match_type_priority = 2
                else:
                    match_type_priority = 1

                match_record = {
                    "candidate_index": (
                        candidate_index
                    ),
                    "persistent_lake_id": (
                        persistent_lake_id
                    ),
                    "match_type_priority": (
                        match_type_priority
                    ),
                    "reference_date": (
                        track["reference_date"]
                    ),
                    "anchor_date": (
                        track["anchor_date"]
                    ),
                    "gap_days": gap_days,
                    "intersection_area_m2": (
                        intersection_area_m2
                    ),
                    "reference_overlap_ratio": (
                        reference_overlap_ratio
                    ),
                    "overlap_smaller_ratio": (
                        overlap_smaller_ratio
                    ),
                    "iou": iou,
                    "distance_m": distance_m,
                    "centroid_distance_m": (
                        centroid_distance_m
                    ),
                    "area_ratio": area_ratio,
                    "match_method": (
                        "overlap"
                        if overlap_match
                        else "distance"
                    ),
                }

                match_candidates.append(
                    match_record
                )

                candidate_matching_ids[
                    candidate_index
                ].add(
                    persistent_lake_id
                )

        # 同じ過去候補が複数の将来IDに一致する場合は、
        # 誤ったID統合を避けるため遡及しない。
        ambiguous_candidate_indices = {
            candidate_index
            for candidate_index, matching_ids
            in candidate_matching_ids.items()
            if len(matching_ids) > 1
        }

        for candidate_index in (
            ambiguous_candidate_indices
        ):
            working_gdf.at[
                candidate_index,
                "retrospective_status",
            ] = "ambiguous_multiple_future_ids"

        match_candidates = [
            match
            for match in match_candidates
            if match["candidate_index"]
            not in ambiguous_candidate_indices
        ]

        match_candidates = sorted(
            match_candidates,
            key=lambda match: (
                -match[
                    "match_type_priority"
                ],
                -match[
                    "reference_overlap_ratio"
                ],
                -match[
                    "intersection_area_m2"
                ],
                -match["iou"],
                -match["area_ratio"],
                -match[
                    "overlap_smaller_ratio"
                ],
                match["distance_m"],
                match["centroid_distance_m"],
                match["gap_days"],
                str(
                    working_gdf.at[
                        match["candidate_index"],
                        "local_candidate_id",
                    ]
                ),
                match["persistent_lake_id"],
            ),
        )

        # 過去方向の遡及でも、同じ日付の候補とIDを一対一にし、
        # 成立する対応数が最大になる組合せを使用する。
        accepted_matches = (
            _select_maximum_cardinality_tracking_matches(
                match_candidates=match_candidates,
                candidate_index_key="candidate_index",
            )
        )

        for candidate_index, match in (
            accepted_matches.items()
        ):
            persistent_lake_id = match[
                "persistent_lake_id"
            ]

            candidate_geometry = (
                working_gdf.at[
                    candidate_index,
                    "geometry",
                ]
            )

            candidate_sensor = str(
                working_gdf.at[
                    candidate_index,
                    "sensor_family",
                ]
            )

            working_gdf.at[
                candidate_index,
                "persistent_lake_id",
            ] = persistent_lake_id

            working_gdf.at[
                candidate_index,
                "tracking_status",
            ] = "retrospective_continuation"

            working_gdf.at[
                candidate_index,
                "tracking_reference_date",
            ] = match[
                "reference_date"
            ].strftime("%Y-%m-%d")

            working_gdf.at[
                candidate_index,
                "tracking_gap_days",
            ] = match["gap_days"]

            working_gdf.at[
                candidate_index,
                "tracking_overlap_smaller_ratio",
            ] = match[
                "overlap_smaller_ratio"
            ]

            working_gdf.at[
                candidate_index,
                "tracking_iou",
            ] = match["iou"]

            working_gdf.at[
                candidate_index,
                "tracking_distance_m",
            ] = match["distance_m"]

            working_gdf.at[
                candidate_index,
                "tracking_centroid_distance_m",
            ] = match[
                "centroid_distance_m"
            ]

            working_gdf.at[
                candidate_index,
                "tracking_area_ratio",
            ] = match["area_ratio"]

            working_gdf.at[
                candidate_index,
                "tracking_match_method",
            ] = (
                "retrospective_"
                + match["match_method"]
            )

            working_gdf.at[
                candidate_index,
                "persistent_counted_in_lake_count",
            ] = True

            working_gdf.at[
                candidate_index,
                "retrospective_backfill",
            ] = True

            working_gdf.at[
                candidate_index,
                "retrospective_status",
            ] = "backfilled_from_future_id"

            working_gdf.at[
                candidate_index,
                "retrospective_anchor_date",
            ] = match[
                "anchor_date"
            ].strftime("%Y-%m-%d")

            working_gdf.at[
                candidate_index,
                "retrospective_reference_date",
            ] = match[
                "reference_date"
            ].strftime("%Y-%m-%d")

            working_gdf.at[
                candidate_index,
                "retrospective_gap_days",
            ] = match["gap_days"]

            working_gdf.at[
                candidate_index,
                "retrospective_intersection_area_m2",
            ] = match[
                "intersection_area_m2"
            ]

            working_gdf.at[
                candidate_index,
                "retrospective_reference_overlap_ratio",
            ] = match[
                "reference_overlap_ratio"
            ]

            working_gdf.at[
                candidate_index,
                "retrospective_overlap_smaller_ratio",
            ] = match[
                "overlap_smaller_ratio"
            ]

            working_gdf.at[
                candidate_index,
                "retrospective_iou",
            ] = match["iou"]

            working_gdf.at[
                candidate_index,
                "retrospective_distance_m",
            ] = match["distance_m"]

            working_gdf.at[
                candidate_index,
                "retrospective_centroid_distance_m",
            ] = match[
                "centroid_distance_m"
            ]

            working_gdf.at[
                candidate_index,
                "retrospective_area_ratio",
            ] = match["area_ratio"]

            working_gdf.at[
                candidate_index,
                "retrospective_match_method",
            ] = match["match_method"]

            # 次の過去日付は、今回遡及した形状と比較する。
            backward_tracks[
                persistent_lake_id
            ] = {
                "geometry": candidate_geometry,
                "anchor_geometry": (
                    candidate_anchor_geometry
                ),
                "reference_date": current_date,
                "anchor_date": match[
                    "anchor_date"
                ],
                "sensor_family": candidate_sensor,
            }

    output_gdf = (
        working_gdf
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

    assigned_output_gdf = (
        output_gdf.loc[
            output_gdf["persistent_lake_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ]
    )

    duplicate_id_date_mask = (
        assigned_output_gdf.duplicated(
            subset=[
                "date",
                "persistent_lake_id",
            ],
            keep=False,
        )
    )

    if duplicate_id_date_mask.any():
        duplicate_rows = (
            assigned_output_gdf.loc[
                duplicate_id_date_mask,
                [
                    "date",
                    "persistent_lake_id",
                    "local_candidate_id",
                ],
            ]
        )

        raise ValueError(
            "遡及後に同じ日付へ同じIDが"
            "複数回割り当てられています:\n"
            + duplicate_rows.to_string(
                index=False
            )
        )

    invalid_output_geometry_mask = (
        output_gdf.geometry.isna()
        | output_gdf.geometry.is_empty
        | ~output_gdf.geometry.is_valid
    )

    if invalid_output_geometry_mask.any():
        raise ValueError(
            "遡及後の出力に"
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
            "ID遡及後GeoPackageを"
            "保存できません。"
            "QGISで開いている場合は"
            "閉じてください: "
            f"{output_gpkg}"
        ) from error

    return output_gdf
