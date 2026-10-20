from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 設定
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

ALERT_RESULTS_DIR = (
    SCRIPT_DIR / "outputs" / "results" / "alerts"
)
ALERT_DETAILS_DIR = (
    SCRIPT_DIR / "outputs" / "details" / "alerts"
)

INPUT_CSV = (
    SCRIPT_DIR
    / "outputs"
    / "details"
    / "multisensor"
    / "final_lake_observations.csv"
)

OUTPUT_CSV = (
    ALERT_DETAILS_DIR
    / "lake_change_metrics.csv"
)

# 前回の比較可能な観測までの最大日数（同一・異種センサー共通）。
MAX_COMPARISON_GAP_DAYS = 30
ENABLE_CROSS_SENSOR_COMPARISON = True
ENABLE_PARTIAL_FILL_COMPARISON = True
ENABLE_QA_SUPPORTED_DIRECT_COMPARISON = True
PARTIAL_FILL_MAX_RATIO = 0.20
PARTIAL_FILL_MIN_REFERENCE_IOU = 0.35
# センサー間補正後にも適用する、急増候補の最低面積差。
# 補正残差から求める誤差閾値の方が大きい場合は、そちらを優先する。
CROSS_SENSOR_MIN_INCREASE_M2 = {
    ("Sentinel-2", "Landsat"): 4500.0,
    ("Landsat", "Sentinel-2"): 4500.0,
}

# 補完を含まない同一湖の近接観測から、Landsat面積を
# Sentinel-2相当へ換算する係数を推定する。
SENSOR_CALIBRATION_REFERENCE = "Sentinel-2"
SENSOR_CALIBRATION_TARGET = "Landsat"
SENSOR_CALIBRATION_MAX_PAIR_GAP_DAYS = 7
SENSOR_CALIBRATION_MIN_PAIR_COUNT = 3
# 同じ湖の観測回数だけが多くても補正を成立させない。
SENSOR_CALIBRATION_MIN_INDEPENDENT_LAKE_COUNT = 3
SENSOR_CALIBRATION_MIN_RATIO = 0.25
SENSOR_CALIBRATION_MAX_RATIO = 4.0
# 湖ごとの代表比率を対数化し、MADに対して大きく外れる湖を除外する。
SENSOR_CALIBRATION_MAX_LAKE_RATIO_MODIFIED_Z = 3.5
SENSOR_CALIBRATION_ERROR_MULTIPLIER = 2.0

# この段階では補完を含まない観測だけを使う。
AREA_COLUMN = "direct_area_m2"

# 面積変化の要確認候補を抽出する試験値。危険度の基準ではない。
# 両条件を「以上」で満たす正の増加をWATCH_PENDINGとする。
WATCH_MIN_INCREASE_PERCENT = 20.0
WATCH_MIN_INCREASE_M2_BY_SENSOR = {
    "Sentinel-2": 500.0,
    "Landsat": 4500.0,
}
ALERT_OUTPUT_CSV = OUTPUT_CSV.with_name("lake_alert_events.csv")

# 急増検出日から確認を待つ日数。末尾に後続観測がなければ保留。
WATCH_CONFIRMATION_MAX_GAP_DAYS = 30

GROUP_OUTPUT_CSV = OUTPUT_CSV.with_name("lake_group_change_metrics.csv")
GROUP_ALERT_OUTPUT_CSV = OUTPUT_CSV.with_name("lake_group_alert_events.csv")
SENSOR_CALIBRATION_OUTPUT_CSV = OUTPUT_CSV.with_name(
    "sensor_area_calibration.csv"
)
SENSOR_CALIBRATION_PAIRS_OUTPUT_CSV = OUTPUT_CSV.with_name(
    "sensor_area_calibration_pairs.csv"
)
ALERT_REVIEW_REPORT_OUTPUT_CSV = (
    ALERT_RESULTS_DIR / "alert_review_report.csv"
)


def calculate_lake_group_metrics(
    observations,
    max_gap_days=30,
    area_column="direct_area_m2",
    allow_cross_sensor=False,
    allow_partial_fill=False,
    max_partial_fill_ratio=0.20,
    min_partial_reference_iou=0.35,
    allow_qa_supported_direct=False,
):
    """結合・分裂の関係を時系列順に取り込み、観測ごとの合計を比較。

    allow_cross_sensor=Trueなら過去と現在のセンサーが異なっても比較する。
    ただし、一つの合計に異なる日付・センサー・観測を混ぜない。

    グループは関係が発生した日から有効（過去全期間を一括統合しない）。
    初回結合時の参照だけは、過去の同一観測に揃った構成湖を合計する。
    補完、影、トポロジー修正などの品質条件は個別湖と同じまま維持する。
    """
    from shapely import wkt
    from shapely.errors import GEOSException
    from shapely.ops import unary_union

    required = {"related_previous_lake_ids", "relation_split_source_ids",
                "geometry_wkt"}
    if required - set(observations):
        raise ValueError("グループ比較に必要な列がありません: "
                         + ", ".join(sorted(required - set(observations))))
    table = calculate_lake_change_metrics(
        observations,
        max_gap_days,
        area_column,
        allow_cross_sensor=False,
        allow_partial_fill=allow_partial_fill,
        max_partial_fill_ratio=max_partial_fill_ratio,
        min_partial_reference_iou=min_partial_reference_iou,
        allow_qa_supported_direct=allow_qa_supported_direct,
    )
    events = {"merge", "complex_merge_split", "split_primary", "split_new"}
    groups = {}
    represented = {}
    active_ids = set()
    history = []
    output = []

    def ids(value):
        return set() if pd.isna(value) else {
            part.strip() for part in str(value).split(";") if part.strip()}

    def aggregate(rows, members):
        # CSVは同じ投影座標系の最終ポリゴンを保持している。
        # 直接面積とWKT面積が一致しない場合は、推測して足さない。
        reasons = set()
        coverage = set()
        geometries = []
        for row in rows:
            coverage.update(row["coverage"])
            reasons.update(row["reasons"])
            if row["geometry"] is not None:
                geometries.append(row["geometry"])
        if coverage != members:
            reasons.add("incomplete_group_members")
        if len({row["observation_id"] for row in rows}) != 1:
            reasons.add("mixed_observations")
        if len({row["sensor_family"] for row in rows}) != 1:
            reasons.add("mixed_sensors")
        if geometries:
            union_area = unary_union(geometries).area
            if sum(g.area for g in geometries) - union_area > 0.01:
                reasons.add("overlapping_member_geometries")
        estimated_area = sum(row["area"] for row in rows)
        direct_area = sum(row["direct_area"] for row in rows)
        observation_types = {row["observation_type"] for row in rows}
        is_partial = "partial_fill" in observation_types
        fill_ratio = (
            (estimated_area - direct_area) / estimated_area
            if estimated_area > 0 else np.nan
        )
        return (
            estimated_area, direct_area, is_partial, fill_ratio,
            observation_types, reasons,
        )

    for date, day in table.groupby("date", sort=True):
        previous_represented = {key: set(value) for key, value in represented.items()}
        removed_ids = set()
        event_ids = set()
        # 関係が初めて現れた時だけグループが拡張する。
        for _, row in day.iterrows():
            lake = row["persistent_lake_id"]
            if not lake:
                raise ValueError("グループ比較に空の永続湖IDがあります。")
            groups.setdefault(lake, {lake})
            if lake not in represented:
                active_ids.add(lake)
                represented[lake] = {lake}
            if row["tracking_event"] in events:
                relation = ids(row["related_previous_lake_ids"])
                relation |= ids(row["relation_split_source_ids"])
                relation.add(lake)
                removed_ids.update(relation)
                event_ids.add(lake)
                members = set().union(*(groups.get(x, {x}) for x in relation))
                for member in members:
                    groups[member] = members
        active_ids.difference_update(removed_ids)
        active_ids.update(event_ids)

        day_records = []
        for _, row in day.iterrows():
            lake = row["persistent_lake_id"]
            reasons = ids(row["metric_exclusion_reason"]) - {"identity_boundary"}
            coverage = set(represented.get(lake, {lake}))
            if row["tracking_event"] in events:
                relation = ids(row["related_previous_lake_ids"])
                relation |= ids(row["relation_split_source_ids"])
                if not relation:
                    reasons.add("missing_event_relations")
                coverage |= set().union(*(previous_represented.get(x, {x}) for x in relation))
                represented[lake] = coverage
            geometry = None
            try:
                geometry = wkt.loads(row["geometry_wkt"])
                if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {
                    "Polygon", "MultiPolygon"
                }:
                    raise ValueError("invalid polygon")
                if not np.isclose(geometry.area, row["metric_area_m2"],
                                  rtol=1e-6, atol=0.01):
                    reasons.add("direct_area_geometry_mismatch")
            except (ValueError, TypeError, GEOSException):
                geometry = None
                reasons.add("invalid_geometry")
            day_records.append(dict(
                lake=lake, coverage=coverage, reasons=reasons, geometry=geometry,
                area=row["metric_area_m2"],
                direct_area=row["metric_direct_area_m2"],
                observation_type=row["metric_observation_type"],
                date=date,
                observation_id=row["observation_id"], sensor_family=row["sensor_family"],
                event=row["tracking_event"],
            ))

        for membership in sorted({tuple(sorted(groups[r["lake"]])) for r in day_records}):
            members = set(membership)
            rows = [r for r in day_records if r["lake"] in members]
            if not (active_ids & members).issubset({r["lake"] for r in rows}):
                for r in rows:
                    r["reasons"].add("missing_active_group_member")
            (area, direct_area, is_partial, fill_ratio,
             observation_types, reasons) = aggregate(rows, members)
            if is_partial:
                group_observation_type = "partial_fill"
            elif "qa_supported_direct" in observation_types:
                group_observation_type = "qa_supported_direct"
            else:
                group_observation_type = "direct"
            record = dict(
                persistent_lake_id="GROUP:" + ";".join(membership),
                group_member_ids=";".join(membership),
                observed_lake_ids=";".join(sorted(r["lake"] for r in rows)),
                group_events=";".join(sorted({r["event"] for r in rows})),
                observation_id=rows[0]["observation_id"], date=date,
                sensor_family=rows[0]["sensor_family"],
                metric_area_source=("mixed_group_estimate" if is_partial else area_column),
                metric_area_m2=area, metric_direct_area_m2=direct_area,
                metric_estimated_area_m2=area,
                metric_observation_type=group_observation_type,
                metric_fill_ratio=fill_ratio,
                metric_partial_fill_eligible=(is_partial and not reasons),
                metric_qa_supported_direct_eligible=(
                    group_observation_type == "qa_supported_direct" and not reasons),
                metric_observation_eligible=not reasons,
                metric_exclusion_reason=";".join(sorted(reasons)),
                metric_comparison_valid=False,
                metric_comparison_status="excluded" if reasons else "baseline_no_same_sensor_reference",
                metric_reference_date=pd.NaT, metric_reference_observation_id="",
                metric_reference_sensor="", metric_sensor_pair="", metric_cross_sensor=False,
                metric_comparison_confidence="",
                metric_reference_area_m2=np.nan,
                metric_reference_direct_area_m2=np.nan,
                metric_interval_days=np.nan,
                metric_area_change_m2=np.nan, metric_area_change_percent=np.nan,
                metric_area_change_m2_per_day=np.nan,
                metric_conservative_growth_m2=np.nan,
                metric_conservative_growth_percent=np.nan,
            )
            if not reasons:
                for prior_date, prior_rows in reversed(history):
                    gap = (date - prior_date).days
                    if gap > max_gap_days:
                        break
                    candidates = [r for r in prior_rows if r["lake"] in members
                                  and (allow_cross_sensor or
                                       r["sensor_family"] == record["sensor_family"])]
                    (previous_area, previous_direct_area, previous_is_partial,
                     _, previous_observation_types,
                     previous_reasons) = aggregate(candidates, members)
                    if previous_reasons or not np.isfinite(previous_area) or previous_area <= 0:
                        continue
                    difference = area - previous_area
                    conservative_growth = direct_area - previous_area
                    combined_observation_types = (
                        observation_types | previous_observation_types
                    )
                    if "partial_fill" in combined_observation_types:
                        comparison_confidence = "conditional_partial_fill"
                    elif "qa_supported_direct" in combined_observation_types:
                        comparison_confidence = "conditional_qa_supported"
                    else:
                        comparison_confidence = "direct"
                    record.update(
                        metric_comparison_valid=True, metric_comparison_status="comparable",
                        metric_reference_date=prior_date,
                        metric_reference_observation_id=candidates[0]["observation_id"],
                        metric_reference_sensor=candidates[0]["sensor_family"],
                        metric_sensor_pair=candidates[0]["sensor_family"] + "->" + record["sensor_family"],
                        metric_cross_sensor=candidates[0]["sensor_family"] != record["sensor_family"],
                        metric_comparison_confidence=comparison_confidence,
                        metric_reference_area_m2=previous_area,
                        metric_reference_direct_area_m2=previous_direct_area,
                        metric_interval_days=gap,
                        metric_area_change_m2=difference,
                        metric_area_change_percent=difference / previous_area * 100,
                        metric_area_change_m2_per_day=difference / gap,
                        metric_conservative_growth_m2=conservative_growth,
                        metric_conservative_growth_percent=(
                            conservative_growth / previous_area * 100),
                    )
                    break
            output.append(record)
        history.append((date, day_records))
    return pd.DataFrame(output)


def calculate_lake_change_metrics(
    observations: pd.DataFrame,
    max_gap_days: int = 30,
    area_column: str = "direct_area_m2",
    allow_cross_sensor: bool = False,
    allow_partial_fill: bool = False,
    max_partial_fill_ratio: float = 0.20,
    min_partial_reference_iou: float = 0.35,
    allow_qa_supported_direct: bool = False,
) -> pd.DataFrame:

    if max_gap_days <= 0:
        raise ValueError("max_gap_daysは1以上にしてください。")
    if not 0 < max_partial_fill_ratio < 1:
        raise ValueError("部分補完率の上限は0より大きく1未満にしてください。")
    if not 0 <= min_partial_reference_iou <= 1:
        raise ValueError("部分補完のIoU閾値は0以上1以下にしてください。")

    required_columns ={
        "persistent_lake_id",
        "observation_id",
        "date",
        "sensor_family",
        "quality_status",
        "observation_status",
        "tracking_status",
        "tracking_event",
        "topology_reconciled",
        "temporal_fill_applied",
        "related_family_fill_applied",
        "requires_shadow_validation",
        "shadow_only",
        "large_change_status",
        area_column,
    }
    partial_columns = {
        "corrected_area_m2", "temporal_fill_ratio", "temporal_fill_status",
        "temporal_fill_previous_date", "temporal_fill_next_date",
        "temporal_reference_iou",
    }
    if allow_partial_fill:
        required_columns |= partial_columns
    qa_supported_columns = {
        "shadow_temporal_supported", "shadow_filter_status",
    }
    if allow_partial_fill or allow_qa_supported_direct:
        required_columns |= qa_supported_columns

    missing_columns = required_columns - set(observations.columns)

    if missing_columns:
        raise ValueError(
            "入力CSVに必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )

    if observations.empty:
        raise ValueError("入力CSVに観測行がありません。")

    table = observations.copy().reset_index(drop=True)

    text_columns = [
        "persistent_lake_id",
        "observation_id",
        "sensor_family",
        "quality_status",
        "observation_status",
        "tracking_status",
        "tracking_event",
        "large_change_status",
    ]
    if allow_partial_fill or allow_qa_supported_direct:
        text_columns += ["shadow_filter_status"]
    if allow_partial_fill:
        text_columns += ["temporal_fill_status"]

    for column in text_columns:
        table[column] = (
            table[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # CSVの文字列 "False" をbool型へ直接変換するとTrueになるため、
    # 認める表記を明示して変換する。
    boolean_columns = [
        "topology_reconciled",
        "temporal_fill_applied",
        "related_family_fill_applied",
        "requires_shadow_validation",
        "shadow_only",
    ]

    boolean_mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }

    for column in boolean_columns:
        values = (
            table[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        table[column] = (
            values.map(boolean_mapping).astype("boolean")
        )

    table["date"] = pd.to_datetime(
        table["date"],
        errors="coerce",
    ).dt.normalize()

    if table["date"].isna().any():
        raise ValueError("解釈できない観測日があります。")

    table["metric_area_m2"] = pd.to_numeric(
        table[area_column],
        errors="coerce",
    )
    table["metric_area_source"] = area_column
    table["metric_direct_area_m2"] = table["metric_area_m2"]
    table["metric_estimated_area_m2"] = table["metric_area_m2"]
    table["metric_observation_type"] = "direct"
    table["metric_fill_ratio"] = 0.0
    table["metric_partial_fill_eligible"] = False
    table["metric_qa_supported_direct_eligible"] = False
    if allow_partial_fill:
        for column in ["corrected_area_m2", "temporal_fill_ratio",
                       "temporal_reference_iou"]:
            table[column] = pd.to_numeric(table[column], errors="coerce")
        for column in ["temporal_fill_previous_date", "temporal_fill_next_date"]:
            table[column] = pd.to_datetime(table[column], errors="coerce").dt.normalize()
    if allow_partial_fill or allow_qa_supported_direct:
        supported_values = (table["shadow_temporal_supported"].astype("string")
                            .str.strip().str.lower())
        table["shadow_temporal_supported"] = supported_values.map(
            boolean_mapping).astype("boolean")

    duplicate_mask = (
        table["persistent_lake_id"].ne("")
        & table.duplicated(
            subset=["persistent_lake_id", "date"],
            keep=False,
        )
    )

    if duplicate_mask.any():
        raise ValueError(
            "同じ日付に同じ永続湖IDが複数あります。"
            "追跡結果を確認してください。"
        )

    table = table.sort_values(
        ["persistent_lake_id", "date", "observation_id"]
    ).reset_index(drop=True)

    table["metric_observation_eligible"] = False
    table["metric_exclusion_reason"] = ""
    table["metric_comparison_valid"] = False
    table["metric_comparison_status"] = "not_evaluated"
    table["metric_reference_observation_id"] = ""
    table["metric_reference_date"] = pd.NaT
    table["metric_reference_sensor"] = ""
    table["metric_sensor_pair"] = ""
    table["metric_cross_sensor"] = False
    table["metric_comparison_confidence"] = ""

    numeric_outputs = [
        "metric_reference_area_m2",
        "metric_interval_days",
        "metric_area_change_m2",
        "metric_area_change_percent",
        "metric_area_change_m2_per_day",
        "metric_reference_direct_area_m2",
        "metric_conservative_growth_m2",
        "metric_conservative_growth_percent",
    ]

    for column in numeric_outputs:
        table[column] = np.nan

    # 合流・分離の前後を、単一湖の面積変化として比較しない。
    boundary_events = {
        "merge",
        "complex_merge_split",
        "split_primary",
        "split_new",
    }

    accepted_tracking_statuses = {
        "created_new_id",
        "continued_existing_id",
        "retrospective_continuation",
        "confirmed_large_change",
        "continued_confirmed_large_change",
    }

    accepted_large_change_statuses = {
        "not_applicable",
        "confirmed_from_next_observation",
        "confirmation_observation",
    }

    # キー: (永続湖ID, センサー)
    # 値: 直近の比較可能な観測行のindex
    references = {}

    for index, row in table.iterrows():
        lake_id = row["persistent_lake_id"]
        sensor = row["sensor_family"]
        reasons = []
        is_partial = bool(
            allow_partial_fill
            and row["observation_status"] == "direct_with_partial_fill"
        )

        is_qa_supported_direct = bool(
            allow_qa_supported_direct
            and row["sensor_family"] == "Landsat"
            and row["observation_status"] == "direct"
            and row["quality_status"].lower() == "partial"
        )

        if not lake_id:
            reasons.append("missing_lake_id")

        if not sensor:
            reasons.append("missing_sensor")

        if is_partial:
            fill_ratio = row["temporal_fill_ratio"]
            reference_iou = row["temporal_reference_iou"]
            previous_fill_date = row["temporal_fill_previous_date"]
            next_fill_date = row["temporal_fill_next_date"]
            shadow_supported = row["shadow_temporal_supported"]
            shadow_filter_status = row["shadow_filter_status"]

            quality_supported = (
                row["quality_status"].lower() == "usable"
                or (
                    row["quality_status"].lower() == "partial"
                    and not pd.isna(shadow_supported)
                    and bool(shadow_supported)
                    and shadow_filter_status == "kept_temporal_match"
                )
            )

            if not quality_supported:
                reasons.append("partial_fill_quality_not_supported")

            if row["temporal_fill_status"] != "partially_filled":
                reasons.append("partial_fill_status_not_supported")

            if (
                pd.isna(row["temporal_fill_applied"])
                or not bool(row["temporal_fill_applied"])
            ):
                reasons.append("partial_fill_flag_not_set")

            if (
                not np.isfinite(fill_ratio)
                or not 0 < fill_ratio <= max_partial_fill_ratio
            ):
                reasons.append("partial_fill_ratio_out_of_range")

            if (
                not np.isfinite(reference_iou)
                or reference_iou < min_partial_reference_iou
            ):
                reasons.append("partial_fill_reference_iou_too_low")

            if (
                pd.isna(previous_fill_date)
                or pd.isna(next_fill_date)
                or not previous_fill_date < row["date"] < next_fill_date
            ):
                reasons.append(
                    "partial_fill_references_not_on_both_sides"
                )

            if (
                pd.isna(row["related_family_fill_applied"])
                or bool(row["related_family_fill_applied"])
            ):
                reasons.append("related_family_fill_not_supported")

            if (
                pd.isna(row["topology_reconciled"])
                or bool(row["topology_reconciled"])
            ):
                reasons.append("partial_fill_topology_not_stable")

            for column in [
                "requires_shadow_validation",
                "shadow_only",
            ]:
                value = row[column]

                if pd.isna(value):
                    reasons.append(f"unknown_{column}")

                elif bool(value) and not (
                    not pd.isna(shadow_supported)
                    and bool(shadow_supported)
                    and shadow_filter_status
                    == "kept_temporal_match"
                ):
                    reasons.append(f"unsupported_{column}")

            area = row["corrected_area_m2"]

            table.at[index, "metric_area_m2"] = area
            table.at[
                index,
                "metric_estimated_area_m2",
            ] = area
            table.at[
                index,
                "metric_observation_type",
            ] = "partial_fill"
            table.at[
                index,
                "metric_fill_ratio",
            ] = fill_ratio
            table.at[
                index,
                "metric_area_source",
            ] = "corrected_area_m2"

        elif is_qa_supported_direct:
            shadow_supported = row["shadow_temporal_supported"]
            shadow_filter_status = row["shadow_filter_status"]

            if (
                pd.isna(shadow_supported)
                or not bool(shadow_supported)
            ):
                reasons.append(
                    "qa_direct_not_temporally_supported"
                )

            if shadow_filter_status != "kept_temporal_match":
                reasons.append(
                    "qa_direct_shadow_filter_not_supported"
                )

            if (
                pd.isna(row["temporal_fill_applied"])
                or bool(row["temporal_fill_applied"])
            ):
                reasons.append(
                    "qa_direct_contains_temporal_fill"
                )

            if (
                pd.isna(row["related_family_fill_applied"])
                or bool(row["related_family_fill_applied"])
            ):
                reasons.append(
                    "qa_direct_contains_related_family_fill"
                )

            if (
                pd.isna(row["topology_reconciled"])
                or bool(row["topology_reconciled"])
            ):
                reasons.append(
                    "qa_direct_topology_not_stable"
                )

            area = row["metric_direct_area_m2"]

            table.at[
                index,
                "metric_area_m2",
            ] = area
            table.at[
                index,
                "metric_estimated_area_m2",
            ] = area
            table.at[
                index,
                "metric_observation_type",
            ] = "qa_supported_direct"
            table.at[
                index,
                "metric_fill_ratio",
            ] = 0.0
            table.at[
                index,
                "metric_area_source",
            ] = area_column

        else:
            if row["quality_status"].lower() != "usable":
                reasons.append("quality_not_usable")

            if row["observation_status"] != "direct":
                reasons.append("not_direct_observation")

            for column in boolean_columns:
                value = row[column]

                if pd.isna(value):
                    reasons.append(f"unknown_{column}")
                elif bool(value):
                    reasons.append(column)

            area = row["metric_area_m2"]

        if row["tracking_status"] not in accepted_tracking_statuses:
            reasons.append("tracking_status_requires_review")

        if (
            row["large_change_status"]
            not in accepted_large_change_statuses
        ):
            reasons.append("large_change_not_confirmed")

        if not np.isfinite(area) or area <= 0:
            reasons.append("invalid_or_zero_area")

        topology_value = row["topology_reconciled"]

        # 欠損行の空欄は、合流・分離が起きた根拠ではない。
        # 参照を保持し、次の有効観測との間隔を後段で確認する。
        is_missing_record = (
            row["observation_status"] == "missing_between_observations"
        )
        topology_unknown = pd.isna(topology_value)
        topology_changed = (
            not topology_unknown and bool(topology_value)
        )

        is_identity_boundary = (
            row["tracking_event"] in boundary_events
            or topology_changed
            # 実観測のトポロジー不明は、引き続き慎重に扱う。
            or (topology_unknown and not is_missing_record)
        )

        if is_identity_boundary:
            reasons.append("identity_boundary")

            # センサーをまたいだ古い参照も破棄する。
            keys_to_remove = [
                key for key in references
                if key[0] == lake_id
            ]

            for key in keys_to_remove:
                del references[key]

        if is_partial:
            # 部分補完固有の条件だけでなく、追跡・結合境界を含む
            # 最終的な比較可否と一致させる。
            table.at[index, "metric_partial_fill_eligible"] = not reasons

        if is_qa_supported_direct:
            table.at[index, "metric_qa_supported_direct_eligible"] = not reasons

        if reasons:
            table.at[index, "metric_exclusion_reason"] = (
                ";".join(dict.fromkeys(reasons))
            )
            table.at[index, "metric_comparison_status"] = (
                "excluded"
            )
            continue

        table.at[index, "metric_observation_eligible"] = True
        reference_key = (lake_id, "*" if allow_cross_sensor else sensor)
        reference_index = references.get(reference_key)

        if reference_index is None:
            # 部分補完観測は、その観測自体を条件付きで評価できるが、
            # 後続観測の面積変化を測る基準にはしない。補完値を基準に
            # すると、検出ばらつきの補正量が後続アラートへ伝播するため。
            if is_partial:
                table.at[index, "metric_comparison_status"] = (
                    "no_trusted_direct_reference"
                )
                continue

            table.at[index, "metric_comparison_status"] = (
                "baseline_no_same_sensor_reference"
            )
            references[reference_key] = index
            continue

        previous = table.loc[reference_index]
        previous_area = float(previous["metric_area_m2"])
        previous_direct_area = float(previous["metric_direct_area_m2"])
        gap_days = int((row["date"] - previous["date"]).days)

        table.at[index, "metric_reference_observation_id"] = (
            previous["observation_id"]
        )
        table.at[index, "metric_reference_date"] = previous["date"]
        table.at[index, "metric_reference_sensor"] = previous["sensor_family"]
        table.at[index, "metric_sensor_pair"] = previous["sensor_family"] + "->" + sensor
        table.at[index, "metric_cross_sensor"] = previous["sensor_family"] != sensor
        table.at[index, "metric_reference_area_m2"] = previous_area
        table.at[index, "metric_reference_direct_area_m2"] = previous_direct_area
        table.at[index, "metric_interval_days"] = gap_days

        if gap_days <= 0:
            raise ValueError(
                f"{lake_id}: 観測日の順序が不正です。"
            )

        if gap_days > max_gap_days:
            table.at[index, "metric_comparison_status"] = (
                "gap_too_long_new_baseline"
            )
            if not is_partial:
                references[reference_key] = index
            continue

        difference = float(area) - previous_area
        conservative_growth = float(row["metric_direct_area_m2"]) - previous_area

        table.at[index, "metric_area_change_m2"] = difference
        table.at[index, "metric_area_change_percent"] = (
            difference / previous_area * 100.0
        )
        table.at[index, "metric_area_change_m2_per_day"] = (
            difference / gap_days
        )
        table.at[index, "metric_conservative_growth_m2"] = conservative_growth
        table.at[index, "metric_conservative_growth_percent"] = (
            conservative_growth / previous_area * 100.0
        )
        current_observation_type = table.at[
            index,
            "metric_observation_type",
        ]
        previous_observation_type = previous[
            "metric_observation_type"
        ]

        comparison_types = {
            current_observation_type,
            previous_observation_type,
        }

        if "partial_fill" in comparison_types:
            comparison_confidence = (
                "conditional_partial_fill"
            )
        elif "qa_supported_direct" in comparison_types:
            comparison_confidence = (
                "conditional_qa_supported"
            )
        else:
            comparison_confidence = "direct"

        table.at[
            index,
            "metric_comparison_confidence",
        ] = comparison_confidence
        table.at[index, "metric_comparison_valid"] = True
        table.at[index, "metric_comparison_status"] = (
            "comparable_cross_sensor"
            if previous["sensor_family"] != sensor
            else "comparable_same_sensor"
        )

        # 補完観測は比較対象にだけ使用し、信頼できる直接観測の
        # 参照を保持する。次の直接観測は最後の直接観測と比較される。
        if not is_partial:
            references[reference_key] = index

    return table


def estimate_sensor_area_calibration(
    metrics: pd.DataFrame,
    reference_sensor: str = "Sentinel-2",
    target_sensor: str = "Landsat",
    max_pair_gap_days: int = 7,
    min_pair_count: int = 3,
    min_independent_lake_count: int = 3,
    min_ratio: float = 0.25,
    max_ratio: float = 4.0,
    max_lake_ratio_modified_z: float = 3.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """同一湖の近接観測からtarget_sensorの面積補正係数を推定する。

    補完を含まない比較可能な観測だけを使用し、各観測を一度だけ
    使用する一対一ペアを作る。まず湖ごとの比率中央値を求め、
    その中央値を湖間で集約するため、観測数が多い一湖に支配されない。
    """
    required = {
        "persistent_lake_id", "observation_id", "date", "sensor_family",
        "metric_area_m2", "metric_observation_type",
        "metric_observation_eligible",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(
            "センサー間補正に必要な列がありません: "
            + ", ".join(sorted(missing))
        )
    if not reference_sensor or not target_sensor:
        raise ValueError("補正元・補正先のセンサー名を指定してください。")
    if reference_sensor == target_sensor:
        raise ValueError("補正元と補正先には異なるセンサーを指定してください。")
    if max_pair_gap_days < 0:
        raise ValueError("補正ペアの最大日数差は0以上にしてください。")
    if min_pair_count < 2:
        raise ValueError("補正に必要な最小ペア数は2以上にしてください。")
    if min_independent_lake_count < 2:
        raise ValueError("補正に必要な独立湖数は2以上にしてください。")
    if not 0 < min_ratio < max_ratio:
        raise ValueError("補正係数の許容範囲が不正です。")
    if (
        not np.isfinite(max_lake_ratio_modified_z)
        or max_lake_ratio_modified_z <= 0
    ):
        raise ValueError("湖別比率の外れ値閾値は有限の正数にしてください。")

    table = metrics.copy().reset_index(drop=True)
    table["date"] = pd.to_datetime(table["date"], errors="coerce").dt.normalize()
    table["metric_area_m2"] = pd.to_numeric(
        table["metric_area_m2"], errors="coerce"
    )
    eligible = (
        table["metric_observation_eligible"]
        .astype("string").str.strip().str.lower()
        .isin(["true", "1", "1.0"])
    )
    # 部分補完はセンサー差と補完差を分離できないため使用しない。
    direct_type = table["metric_observation_type"].isin(
        {"direct", "qa_supported_direct"}
    )
    candidate_rows = table.loc[
        eligible
        & direct_type
        & table["sensor_family"].isin({reference_sensor, target_sensor})
        & table["persistent_lake_id"].fillna("").astype(str).ne("")
        & table["date"].notna()
        & table["metric_area_m2"].gt(0)
    ].copy()

    pair_records = []
    for lake_id, lake_rows in candidate_rows.groupby(
        "persistent_lake_id", sort=True
    ):
        reference_rows = lake_rows.loc[
            lake_rows["sensor_family"].eq(reference_sensor)
        ]
        target_rows = lake_rows.loc[
            lake_rows["sensor_family"].eq(target_sensor)
        ]
        candidates = []
        for reference_index, reference_row in reference_rows.iterrows():
            for target_index, target_row in target_rows.iterrows():
                gap_days = abs(
                    int((reference_row["date"] - target_row["date"]).days)
                )
                if gap_days <= max_pair_gap_days:
                    candidates.append((
                        gap_days,
                        max(reference_row["date"], target_row["date"]),
                        min(reference_row["date"], target_row["date"]),
                        reference_index,
                        target_index,
                    ))

        # 日数差が小さい組から採用し、同じ観測の重複利用を防ぐ。
        used_reference = set()
        used_target = set()
        for gap_days, _, _, reference_index, target_index in sorted(candidates):
            if (
                reference_index in used_reference
                or target_index in used_target
            ):
                continue
            used_reference.add(reference_index)
            used_target.add(target_index)
            reference_row = table.loc[reference_index]
            target_row = table.loc[target_index]
            ratio = (
                float(reference_row["metric_area_m2"])
                / float(target_row["metric_area_m2"])
            )
            accepted = bool(np.isfinite(ratio) and min_ratio <= ratio <= max_ratio)
            pair_records.append({
                "persistent_lake_id": lake_id,
                "reference_sensor": reference_sensor,
                "reference_observation_id": reference_row["observation_id"],
                "reference_date": reference_row["date"],
                "reference_area_m2": float(reference_row["metric_area_m2"]),
                "reference_observation_type": reference_row[
                    "metric_observation_type"
                ],
                "target_sensor": target_sensor,
                "target_observation_id": target_row["observation_id"],
                "target_date": target_row["date"],
                "target_area_m2": float(target_row["metric_area_m2"]),
                "target_observation_type": target_row[
                    "metric_observation_type"
                ],
                "pair_gap_days": gap_days,
                "area_ratio_reference_per_target": ratio,
                "pair_accepted": accepted,
                "pair_exclusion_reason": (
                    "" if accepted else "ratio_out_of_allowed_range"
                ),
            })

    pair_columns = [
        "persistent_lake_id", "reference_sensor",
        "reference_observation_id", "reference_date", "reference_area_m2",
        "reference_observation_type", "target_sensor",
        "target_observation_id", "target_date", "target_area_m2",
        "target_observation_type", "pair_gap_days",
        "area_ratio_reference_per_target", "pair_accepted",
        "pair_exclusion_reason", "lake_median_area_ratio",
        "lake_ratio_modified_z", "lake_ratio_outlier",
        "pair_used_for_calibration", "pair_calibration_exclusion_reason",
        "normalized_target_area_m2",
        "calibration_residual_m2", "absolute_calibration_residual_m2",
    ]
    pairs = pd.DataFrame(pair_records)
    if pairs.empty:
        pairs = pd.DataFrame(columns=pair_columns)
        accepted_pairs = pairs.copy()
        used_pairs = pairs.copy()
        lake_ratio_mad_log = np.nan
        factor = np.nan
    else:
        accepted_mask = pairs["pair_accepted"].astype(bool)
        accepted_pairs = pairs.loc[accepted_mask].copy()
        lake_ratios = (
            accepted_pairs.groupby("persistent_lake_id")[
                "area_ratio_reference_per_target"
            ].median()
        )
        pairs["lake_median_area_ratio"] = pairs["persistent_lake_id"].map(
            lake_ratios
        )
        if lake_ratios.empty:
            lake_modified_z = pd.Series(dtype=float)
            lake_outlier = pd.Series(dtype=bool)
            lake_ratio_mad_log = np.nan
        else:
            log_ratios = np.log(lake_ratios.astype(float))
            log_center = float(log_ratios.median())
            absolute_deviation = (log_ratios - log_center).abs()
            lake_ratio_mad_log = float(absolute_deviation.median())
            if lake_ratio_mad_log > 0:
                lake_modified_z = (
                    0.6745 * absolute_deviation / lake_ratio_mad_log
                )
                lake_outlier = lake_modified_z.gt(
                    max_lake_ratio_modified_z
                )
            else:
                # MAD=0は多数の湖で比率が一致している状態を含む。
                # 中央値自体が頑健なので、この場合は外れ値判定を行わない。
                lake_modified_z = pd.Series(0.0, index=lake_ratios.index)
                lake_outlier = pd.Series(False, index=lake_ratios.index)
        pairs["lake_ratio_modified_z"] = pairs["persistent_lake_id"].map(
            lake_modified_z
        )
        # 事前条件で除外されたペアはlake_outlierに存在しないため、
        # map後の欠損をFalseとして明示的にbool型へ統一する。
        pairs["lake_ratio_outlier"] = (
            pairs["persistent_lake_id"].map(lake_outlier).fillna(False)
            .astype(bool)
        )

        pair_accepted = (
            pairs["pair_accepted"].fillna(False).astype(bool)
        )
        pairs["pair_used_for_calibration"] = (
            pair_accepted & ~pairs["lake_ratio_outlier"]
        )

        # np.selectにはPandasのobject列ではなく、NumPyのbool配列を渡す。
        not_accepted_condition = (
            ~pair_accepted
        ).to_numpy(dtype=bool)
        lake_outlier_condition = pairs[
            "lake_ratio_outlier"
        ].to_numpy(dtype=bool)
        pair_exclusion_reason = (
            pairs["pair_exclusion_reason"]
            .fillna("")
            .astype(str)
            .to_numpy()
        )
        pairs["pair_calibration_exclusion_reason"] = np.select(
            [
                not_accepted_condition,
                lake_outlier_condition,
            ],
            [
                pair_exclusion_reason,
                "lake_ratio_outlier",
            ],
            default="",
        )
        used_pairs = pairs.loc[pairs["pair_used_for_calibration"]].copy()
        used_lake_ratios = (
            used_pairs.groupby("persistent_lake_id")[
                "area_ratio_reference_per_target"
            ].median()
        )
        # 各湖を一票として扱うため、観測数が多い湖も重みは増えない。
        factor = (
            float(used_lake_ratios.median())
            if not used_lake_ratios.empty else np.nan
        )

    pair_count = len(accepted_pairs)
    used_pair_count = len(used_pairs)
    accepted_lake_count = (
        int(accepted_pairs["persistent_lake_id"].nunique())
        if pair_count else 0
    )
    used_lake_count = (
        int(used_pairs["persistent_lake_id"].nunique())
        if used_pair_count else 0
    )
    calibration_applied = bool(
        used_pair_count >= min_pair_count
        and used_lake_count >= min_independent_lake_count
        and np.isfinite(factor)
        and min_ratio <= factor <= max_ratio
    )
    if pair_count == 0:
        status = "no_valid_pairs"
    elif pair_count < min_pair_count:
        status = "insufficient_pairs"
    elif used_pair_count < min_pair_count:
        status = "insufficient_pairs_after_lake_filter"
    elif used_lake_count < min_independent_lake_count:
        status = "insufficient_independent_lakes"
    elif calibration_applied:
        status = "calibrated"
    else:
        status = "invalid_factor"

    if not pairs.empty:
        pairs["normalized_target_area_m2"] = np.where(
            pairs["pair_accepted"] & np.isfinite(factor),
            pairs["target_area_m2"] * factor,
            np.nan,
        )
        pairs["calibration_residual_m2"] = np.where(
            pairs["pair_accepted"] & np.isfinite(factor),
            pairs["reference_area_m2"] - pairs["normalized_target_area_m2"],
            np.nan,
        )
        pairs["absolute_calibration_residual_m2"] = pairs[
            "calibration_residual_m2"
        ].abs()
        pairs = pairs.reindex(columns=pair_columns)

    residuals = (
        pairs.loc[
            pairs["pair_used_for_calibration"],
            "absolute_calibration_residual_m2",
        ]
        .dropna().to_numpy(dtype=float)
        if not pairs.empty else np.array([], dtype=float)
    )
    median_residual = float(np.median(residuals)) if residuals.size else np.nan
    p90_residual = float(np.percentile(residuals, 90)) if residuals.size else np.nan
    summary = pd.DataFrame([{
        "reference_sensor": reference_sensor,
        "target_sensor": target_sensor,
        "calibration_status": status,
        "calibration_applied": calibration_applied,
        "target_to_reference_area_factor": factor,
        "accepted_pair_count": pair_count,
        "used_pair_count": used_pair_count,
        "accepted_lake_count": accepted_lake_count,
        "used_lake_count": used_lake_count,
        "minimum_pair_count": min_pair_count,
        "minimum_independent_lake_count": min_independent_lake_count,
        "maximum_pair_gap_days": max_pair_gap_days,
        "minimum_allowed_ratio": min_ratio,
        "maximum_allowed_ratio": max_ratio,
        "maximum_lake_ratio_modified_z": max_lake_ratio_modified_z,
        "lake_ratio_mad_log": lake_ratio_mad_log,
        "outlier_lake_count": accepted_lake_count - used_lake_count,
        "median_absolute_residual_m2": median_residual,
        "p90_absolute_residual_m2": p90_residual,
    }])
    return summary, pairs


def apply_sensor_area_calibration(
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    error_multiplier: float = 2.0,
) -> pd.DataFrame:
    """補正が成立した場合だけ、面積を基準センサー相当へ換算する。"""
    if error_multiplier <= 0 or not np.isfinite(error_multiplier):
        raise ValueError("補正誤差の倍率は有限の正数にしてください。")
    if calibration is None or calibration.empty:
        raise ValueError("センサー間補正の集計行がありません。")

    result = metrics.copy()
    summary = calibration.iloc[0]
    reference_sensor = str(summary["reference_sensor"])
    target_sensor = str(summary["target_sensor"])
    status = str(summary["calibration_status"])
    factor = pd.to_numeric(
        summary["target_to_reference_area_factor"], errors="coerce"
    )
    applied_text = str(summary["calibration_applied"]).strip().lower()
    applied = (
        applied_text in {"true", "1", "1.0"}
        and np.isfinite(factor)
        and factor > 0
    )
    pair_count = int(
        summary.get("used_pair_count", summary["accepted_pair_count"])
    )
    p90_error = pd.to_numeric(
        summary["p90_absolute_residual_m2"], errors="coerce"
    )

    numeric_columns = [
        "metric_area_m2", "metric_direct_area_m2",
        "metric_estimated_area_m2", "metric_reference_area_m2",
        "metric_reference_direct_area_m2",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result["metric_raw_" + column.removeprefix("metric_")] = result[column]

    result["metric_area_scale"] = "native"
    result["metric_sensor_calibration_status"] = status
    result["metric_sensor_calibration_factor"] = factor
    result["metric_sensor_calibration_pair_count"] = pair_count
    result["metric_sensor_calibration_p90_error_m2"] = p90_error
    result["metric_current_scale_factor"] = 1.0
    result["metric_reference_scale_factor"] = 1.0
    result["metric_cross_sensor_calibrated"] = False
    result["metric_calibration_error_threshold_m2"] = np.nan

    if not applied:
        return result

    current_target = result["sensor_family"].eq(target_sensor)
    reference_target = result["metric_reference_sensor"].eq(target_sensor)
    known_current = result["sensor_family"].isin(
        {reference_sensor, target_sensor}
    )
    result.loc[current_target, "metric_current_scale_factor"] = factor
    result.loc[reference_target, "metric_reference_scale_factor"] = factor
    for column in [
        "metric_area_m2", "metric_direct_area_m2",
        "metric_estimated_area_m2",
    ]:
        result.loc[current_target, column] = (
            result.loc[current_target, column] * factor
        )
    for column in [
        "metric_reference_area_m2", "metric_reference_direct_area_m2",
    ]:
        result.loc[reference_target, column] = (
            result.loc[reference_target, column] * factor
        )
    result.loc[known_current, "metric_area_scale"] = (
        reference_sensor + "_equivalent"
    )

    valid = (
        result["metric_comparison_valid"].astype("string")
        .str.strip().str.lower().isin(["true", "1", "1.0"])
    )
    finite = (
        result["metric_area_m2"].notna()
        & result["metric_reference_area_m2"].notna()
        & result["metric_reference_area_m2"].gt(0)
        & result["metric_interval_days"].gt(0)
    )
    recalculate = valid & finite
    difference = (
        result.loc[recalculate, "metric_area_m2"]
        - result.loc[recalculate, "metric_reference_area_m2"]
    )
    conservative = (
        result.loc[recalculate, "metric_direct_area_m2"]
        - result.loc[recalculate, "metric_reference_area_m2"]
    )
    result.loc[recalculate, "metric_area_change_m2"] = difference
    result.loc[recalculate, "metric_area_change_percent"] = (
        difference
        / result.loc[recalculate, "metric_reference_area_m2"] * 100.0
    )
    result.loc[recalculate, "metric_area_change_m2_per_day"] = (
        difference / result.loc[recalculate, "metric_interval_days"]
    )
    result.loc[recalculate, "metric_conservative_growth_m2"] = conservative
    result.loc[recalculate, "metric_conservative_growth_percent"] = (
        conservative
        / result.loc[recalculate, "metric_reference_area_m2"] * 100.0
    )

    cross = (
        result["metric_cross_sensor"].astype("string")
        .str.strip().str.lower().isin(["true", "1", "1.0"])
    )
    supported_pair = (
        result["sensor_family"].isin({reference_sensor, target_sensor})
        & result["metric_reference_sensor"].isin(
            {reference_sensor, target_sensor}
        )
    )
    calibrated_cross = cross & supported_pair
    result.loc[calibrated_cross, "metric_cross_sensor_calibrated"] = True
    if np.isfinite(p90_error):
        result.loc[
            calibrated_cross,
            "metric_calibration_error_threshold_m2",
        ] = p90_error * error_multiplier
    return result


def classify_lake_area_watch(
    metrics: pd.DataFrame,
    min_increase_percent: float,
    min_increase_m2_by_sensor: dict[str, float],
    cross_sensor_thresholds=None,
) -> pd.DataFrame:
    """比較指標に観測ごとの試験的判定を追加する。後続確認は行わない。"""
    required = {
        "sensor_family", "metric_comparison_valid",
        "metric_comparison_status", "metric_exclusion_reason",
        "metric_area_change_m2", "metric_area_change_percent",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError("判定に必要な列がありません: " + ", ".join(sorted(missing)))
    if not np.isfinite(min_increase_percent) or min_increase_percent <= 0:
        raise ValueError("増加率の閾値は有限の正数にしてください。")
    for sensor, threshold in min_increase_m2_by_sensor.items():
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"{sensor}の増加面積閾値は有限の正数にしてください。")

    result = metrics.copy()
    # CSVから再読込した場合も、文字列FalseをTrueにしない。
    valid = (
        result["metric_comparison_valid"].astype("string")
        .str.strip().str.lower().isin(["true", "1", "1.0"])
    )
    change = pd.to_numeric(result["metric_area_change_m2"], errors="coerce")
    percent = pd.to_numeric(result["metric_area_change_percent"], errors="coerce")
    conservative_change = pd.to_numeric(
        result.get("metric_conservative_growth_m2", change), errors="coerce")
    conservative_percent = pd.to_numeric(
        result.get("metric_conservative_growth_percent", percent), errors="coerce")
    comparison_confidence = result.get(
        "metric_comparison_confidence",
        pd.Series("direct", index=result.index),
    )

    conditional = comparison_confidence.isin(
        {
            "conditional_partial_fill",
            "conditional_qa_supported",
        }
    )
    thresholds = (
        result["sensor_family"].fillna("").astype(str).str.strip()
        .map(min_increase_m2_by_sensor)
    )
    cross_sensor_thresholds = cross_sensor_thresholds or {}
    for pair, threshold in cross_sensor_thresholds.items():
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"{pair}の閾値は有限の正数にしてください。")
    cross = result.get("metric_cross_sensor", pd.Series(False, index=result.index))
    cross = cross.astype("string").str.lower().isin(["true", "1", "1.0"])
    calibrated = result.get(
        "metric_cross_sensor_calibrated",
        pd.Series(False, index=result.index),
    )
    calibrated = (
        calibrated.astype("string").str.lower().isin(["true", "1", "1.0"])
    )
    calibration_error_threshold = pd.to_numeric(
        result.get(
            "metric_calibration_error_threshold_m2",
            pd.Series(np.nan, index=result.index),
        ),
        errors="coerce",
    )
    for index in result.index[cross]:
        pair = (result.at[index, "metric_reference_sensor"], result.at[index, "sensor_family"])
        thresholds.at[index] = cross_sensor_thresholds.get(pair, np.nan)
        if calibrated.at[index] and np.isfinite(
            calibration_error_threshold.at[index]
        ):
            # 補正残差の2倍など、設定された誤差幅より小さい変化は
            # センサー差と区別できないため候補にしない。
            thresholds.at[index] = max(
                thresholds.at[index],
                calibration_error_threshold.at[index],
            )
    result["alert_cross_sensor_uncalibrated"] = cross & ~calibrated
    result["alert_sensor_calibration_applied"] = cross & calibrated
    result["alert_calibration_error_threshold_m2"] = (
        calibration_error_threshold
    )
    result["alert_rule_version"] = "area_watch_v3_sensor_calibrated_trial"
    result["alert_analysis_mode"] = "retrospective"
    result["alert_min_increase_m2"] = thresholds
    result["alert_min_increase_percent"] = min_increase_percent
    result["alert_evaluated"] = False
    result["alert_absolute_threshold_met"] = False
    result["alert_percent_threshold_met"] = False
    result["alert_level"] = "INVALID"
    result["alert_reason"] = ""
    result["alert_confidence"] = "NOT_EVALUATED"
    result["alert_conservative_absolute_threshold_met"] = False
    result["alert_conservative_percent_threshold_met"] = False

    for index in result.index:
        if not valid.at[index]:
            status = str(result.at[index, "metric_comparison_status"])
            if status in {
                "baseline_no_same_sensor_reference",
                "gap_too_long_new_baseline",
                "no_trusted_direct_reference",
            }:
                result.at[index, "alert_level"] = "NO_COMPARISON"
            reason = result.at[index, "metric_exclusion_reason"]
            detail = "" if pd.isna(reason) else str(reason).strip()
            result.at[index, "alert_reason"] = status + (
                ";" + detail if detail else ""
            )
            continue
        if pd.isna(thresholds.at[index]):
            result.at[index, "alert_reason"] = "sensor_threshold_not_configured"
            continue
        if not (np.isfinite(change.at[index]) and np.isfinite(percent.at[index])):
            result.at[index, "alert_reason"] = "invalid_change_metrics"
            continue
        if (change.at[index] > 0) != (percent.at[index] > 0):
            result.at[index, "alert_reason"] = "inconsistent_change_sign"
            continue

        absolute_met = change.at[index] >= thresholds.at[index]
        percent_met = percent.at[index] >= min_increase_percent
        result.at[index, "alert_evaluated"] = True
        confidence_value = comparison_confidence.at[index]
        if confidence_value == "conditional_partial_fill":
            alert_confidence = "CONDITIONAL_PARTIAL_FILL"
        elif confidence_value == "conditional_qa_supported":
            alert_confidence = "CONDITIONAL_QA_SUPPORTED"
        elif cross.at[index] and calibrated.at[index]:
            alert_confidence = "CROSS_SENSOR_CALIBRATED"
        else:
            alert_confidence = "DIRECT"
        result.at[index, "alert_confidence"] = alert_confidence
        result.at[index, "alert_absolute_threshold_met"] = absolute_met
        result.at[index, "alert_percent_threshold_met"] = percent_met
        conservative_absolute_met = (
            np.isfinite(conservative_change.at[index])
            and conservative_change.at[index] >= thresholds.at[index]
        )
        conservative_percent_met = (
            np.isfinite(conservative_percent.at[index])
            and conservative_percent.at[index] >= min_increase_percent
        )
        result.at[index, "alert_conservative_absolute_threshold_met"] = (
            conservative_absolute_met)
        result.at[index, "alert_conservative_percent_threshold_met"] = (
            conservative_percent_met)
        if absolute_met and percent_met:
            low_confidence = conditional.at[index] and not (
                conservative_absolute_met and conservative_percent_met)
            result.at[index, "alert_level"] = (
                "WATCH_PENDING_LOW_CONFIDENCE" if low_confidence
                else "WATCH_PENDING")
            if low_confidence:
                reason = "growth_depends_on_partial_fill_estimate"
            elif cross.at[index] and calibrated.at[index]:
                reason = "cross_sensor_growth_candidate_calibrated"
            elif cross.at[index]:
                reason = "cross_sensor_growth_candidate_uncalibrated"
            elif conditional.at[index]:
                reason = (
                    "qa_supported_growth_requires_clear_confirmation"
                    if confidence_value == "conditional_qa_supported"
                    else "growth_exceeds_threshold_without_filled_current_area"
                )
            else:
                reason = "both_growth_thresholds_met"
            result.at[index, "alert_reason"] = reason
        else:
            # NORMALは、この増加抽出条件に該当しないという意味。
            result.at[index, "alert_level"] = "NORMAL"
            if change.at[index] <= 0:
                reason = "no_positive_area_increase"
            else:
                reason = ";".join(
                    name for met, name in [
                        (absolute_met, "below_absolute_threshold"),
                        (percent_met, "below_percent_threshold"),
                    ] if not met
                )
            result.at[index, "alert_reason"] = reason
    return result


def resolve_lake_area_watches(
    alerts: pd.DataFrame,
    max_confirmation_gap_days: int = 30,
    confirmation_thresholds=None,
    calibration_error_multiplier: float = SENSOR_CALIBRATION_ERROR_MULTIPLIER,
) -> pd.DataFrame:
    """急増候補を後続の補完なし観測で事後確認する。

    共通面積スケールへ校正済みの候補は、候補が同一センサー比較から
    発生した場合も、いずれかの校正対象センサーで確認する。
    未校正の異種センサー候補は、元の基準センサーだけで確認する。

    初回のalert_levelは保持し、イベント発生行のwatch_statusへ
    確認・解除結果を記録する。確認日はwatch_resolution_dateに残す。
    """
    required = {
        "persistent_lake_id", "observation_id", "date", "sensor_family",
        "alert_level", "metric_observation_eligible", "metric_exclusion_reason",
        "metric_reference_area_m2", "metric_area_m2",
        "alert_min_increase_m2", "alert_min_increase_percent",
    }
    missing = required - set(alerts.columns)
    if missing:
        raise ValueError("後続確認に必要な列がありません: " + ", ".join(sorted(missing)))
    if max_confirmation_gap_days <= 0:
        raise ValueError("確認期限は1日以上にしてください。")
    if (
        not np.isfinite(calibration_error_multiplier)
        or calibration_error_multiplier <= 0
    ):
        raise ValueError("補正誤差の倍率は有限の正数にしてください。")
    result = alerts.copy().reset_index(drop=True)
    confirmation_thresholds = (WATCH_MIN_INCREASE_M2_BY_SENSOR
                               if confirmation_thresholds is None else confirmation_thresholds)
    for sensor, threshold in confirmation_thresholds.items():
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"{sensor}の確認閾値は有限の正数にしてください。")
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise ValueError("後続確認で解釈できない日付があります。")
    if result.duplicated(["persistent_lake_id", "date"]).any():
        raise ValueError("同一湖・同一日付の行が重複しています。")
    result = result.sort_values(
        ["persistent_lake_id", "date", "observation_id"]
    ).reset_index(drop=True)
    eligible = (
        result["metric_observation_eligible"].astype("string")
        .str.strip().str.lower().isin(["true", "1", "1.0"])
    )
    areas = pd.to_numeric(result["metric_area_m2"], errors="coerce")
    result["watch_rule_version"] = "calibrated_multisensor_confirmation_v4_trial"
    result["watch_required_sensor"] = ""
    result["watch_confirmation_min_increase_m2"] = np.nan
    result["watch_resolution_sensor"] = ""
    result["watch_status"] = "NOT_APPLICABLE"
    result["watch_reason"] = ""
    result["watch_event_id"] = ""
    result["watch_resolution_observation_id"] = ""
    result["watch_resolution_date"] = pd.NaT
    result["watch_deadline_date"] = pd.NaT
    result["watch_as_of_date"] = result["date"].max()
    result["watch_max_gap_days"] = max_confirmation_gap_days
    for column in [
        "watch_followup_area_m2", "watch_followup_gap_days",
        "watch_increase_from_baseline_m2", "watch_increase_from_baseline_percent",
    ]:
        result[column] = np.nan

    # 評価時点は今回入力した観測の末日。現在日時は使用しない。
    as_of_date = result["date"].max()
    watch_levels = {"WATCH_PENDING", "WATCH_PENDING_LOW_CONFIDENCE"}
    for index in result.index[result["alert_level"].isin(watch_levels)]:
        row = result.loc[index]
        baseline = pd.to_numeric(row["metric_reference_area_m2"], errors="coerce")
        absolute_min = pd.to_numeric(row["alert_min_increase_m2"], errors="coerce")
        percent_min = pd.to_numeric(row["alert_min_increase_percent"], errors="coerce")
        is_cross = str(row.get("metric_cross_sensor", False)).strip().lower() in {"true", "1", "1.0"}
        is_calibrated_cross = (
            is_cross
            and str(row.get("metric_cross_sensor_calibrated", False))
            .strip().lower() in {"true", "1", "1.0"}
        )
        candidate_calibration_ready = (
            str(row.get("metric_sensor_calibration_status", ""))
            .strip().lower() == "calibrated"
            and str(row.get("metric_area_scale", "native"))
            .strip().lower() != "native"
            and row["sensor_family"] in {"Sentinel-2", "Landsat"}
        )
        allow_calibrated_sensor_followup = bool(
            is_calibrated_cross or candidate_calibration_ready
        )
        required_sensor = (
            "any_calibrated_sensor"
            if allow_calibrated_sensor_followup
            else (
                row.get("metric_reference_sensor", "")
                if is_cross else row["sensor_family"]
            )
        )
        if is_cross and not is_calibrated_cross:
            absolute_min = confirmation_thresholds.get(required_sensor, np.nan)
        result.at[index, "watch_required_sensor"] = required_sensor
        result.at[index, "watch_confirmation_min_increase_m2"] = absolute_min
        result.at[index, "watch_event_id"] = (
            f"{row['persistent_lake_id']}::{row['observation_id']}"
        )
        if not all(np.isfinite(v) and v > 0 for v in [baseline, absolute_min, percent_min]):
            result.at[index, "watch_status"] = "INVALID"
            result.at[index, "watch_reason"] = "invalid_baseline_or_threshold"
            continue
        deadline = row["date"] + pd.Timedelta(days=max_confirmation_gap_days)
        result.at[index, "watch_deadline_date"] = deadline
        result.at[index, "watch_status"] = "WATCH_PENDING"
        if allow_calibrated_sensor_followup:
            result.at[index, "watch_reason"] = (
                "awaiting_eligible_calibrated_sensor_observation"
            )
        elif is_cross:
            result.at[index, "watch_reason"] = (
                "awaiting_original_baseline_sensor_uncalibrated_cross_sensor_candidate"
            )
        else:
            result.at[index, "watch_reason"] = (
                "awaiting_eligible_same_sensor_observation"
            )
        future = result.loc[
            result["persistent_lake_id"].eq(row["persistent_lake_id"])
            & result["date"].gt(row["date"])
            & result["date"].le(deadline)
        ]
        for future_index, followup in future.iterrows():
            raw_reason = followup["metric_exclusion_reason"]
            reasons = set(str(raw_reason).split(";")) if pd.notna(raw_reason) else set()
            followup_calibration_ready = (
                str(followup.get("metric_sensor_calibration_status", ""))
                .strip().lower() == "calibrated"
                and str(followup.get("metric_area_scale", "native")).strip()
                == str(row.get("metric_area_scale", "native")).strip()
                and followup["sensor_family"] in {"Sentinel-2", "Landsat"}
            )
            followup_type = str(
                followup.get("metric_observation_type", "direct")
            ).strip()
            # 別センサーで記録された合流・分離も、候補の比較を中断する。
            if "identity_boundary" in reasons:
                result.at[index, "watch_status"] = "INVALID"
                result.at[index, "watch_reason"] = "identity_changed_or_uncertain"
            elif (
                (
                    not allow_calibrated_sensor_followup
                    and followup["sensor_family"] != required_sensor
                )
                or (
                    allow_calibrated_sensor_followup
                    and not followup_calibration_ready
                )
                or not eligible.at[future_index]
                # 部分補完値は確認に使わない。時系列的に支持された
                # QA条件付き直接観測は、補完を含まないため利用できる。
                or followup_type not in {"direct", "qa_supported_direct"}
                or not np.isfinite(areas.at[future_index])
                or areas.at[future_index] <= 0
            ):
                continue
            else:
                effective_absolute_min = absolute_min
                if followup["sensor_family"] != row["sensor_family"]:
                    p90_error = pd.to_numeric(
                        followup.get(
                            "metric_sensor_calibration_p90_error_m2",
                            np.nan,
                        ),
                        errors="coerce",
                    )
                    if np.isfinite(p90_error):
                        # 異種センサーで確認するときは、初期候補の閾値より
                        # センサー間補正誤差の方が大きければ誤差側を優先する。
                        effective_absolute_min = max(
                            effective_absolute_min,
                            p90_error * calibration_error_multiplier,
                        )
                increase = areas.at[future_index] - baseline
                increase_percent = increase / baseline * 100.0
                # 発生直前の基準面積に対して増加が維持されたかを確認。
                # 後続日同士で増加が止まっても、増水を維持すれば確認済み。
                maintained = (
                    increase >= effective_absolute_min
                    and increase_percent >= percent_min
                )
                # 同一センサー比較から発生した候補を、別センサーの一観測
                # だけでは棄却しない。補正済み異種センサーは肯定的な確認に
                # 使用し、増加を確認できなければ次の有効観測を待つ。
                if (
                    not maintained
                    and not is_cross
                    and followup["sensor_family"] != row["sensor_family"]
                ):
                    continue
                result.at[
                    index, "watch_confirmation_min_increase_m2"
                ] = effective_absolute_min
                result.at[index, "watch_status"] = (
                    "REVIEW_CONFIRMED" if maintained else "DISMISSED"
                )
                result.at[index, "watch_reason"] = (
                    "growth_maintained_from_original_baseline" if maintained
                    else "growth_not_maintained_from_original_baseline"
                )
                result.at[index, "watch_followup_area_m2"] = areas.at[future_index]
                result.at[index, "watch_increase_from_baseline_m2"] = increase
                result.at[index, "watch_increase_from_baseline_percent"] = increase_percent
            result.at[index, "watch_resolution_date"] = followup["date"]
            result.at[index, "watch_resolution_sensor"] = followup["sensor_family"]
            result.at[index, "watch_resolution_observation_id"] = followup["observation_id"]
            result.at[index, "watch_followup_gap_days"] = (followup["date"] - row["date"]).days
            break
        if (
            result.at[index, "watch_status"] == "WATCH_PENDING"
            and as_of_date > deadline
        ):
            # 確認できないことは、面積が元に戻ったという証拠ではない。
            result.at[index, "watch_status"] = "EXPIRED_UNCONFIRMED"
            result.at[index, "watch_reason"] = "no_eligible_followup_within_deadline"
            result.at[index, "watch_resolution_date"] = deadline
    return result


def build_alert_outputs(observations, grouped=False, calibration=None):
    """主比較は直近センサー、same_sensor_*列は従来判定の補助記録。

    グループの基準は各観測内で全構成湖が揃うものだけを使う。
    個別湖の結合境界は従来通り無効で、グループ側で監視を継続する。
    """
    calculate = calculate_lake_group_metrics if grouped else calculate_lake_change_metrics
    same = calculate(
        observations, MAX_COMPARISON_GAP_DAYS, AREA_COLUMN,
        allow_partial_fill=ENABLE_PARTIAL_FILL_COMPARISON,
        max_partial_fill_ratio=PARTIAL_FILL_MAX_RATIO,
        min_partial_reference_iou=PARTIAL_FILL_MIN_REFERENCE_IOU,
        allow_qa_supported_direct=ENABLE_QA_SUPPORTED_DIRECT_COMPARISON,
    )
    if calibration is not None:
        same = apply_sensor_area_calibration(
            same,
            calibration,
            SENSOR_CALIBRATION_ERROR_MULTIPLIER,
        )
    same_alerts = classify_lake_area_watch(
        same, WATCH_MIN_INCREASE_PERCENT, WATCH_MIN_INCREASE_M2_BY_SENSOR)
    same_alerts = resolve_lake_area_watches(same_alerts, WATCH_CONFIRMATION_MAX_GAP_DAYS)
    primary = calculate(observations, MAX_COMPARISON_GAP_DAYS, AREA_COLUMN,
                        allow_cross_sensor=ENABLE_CROSS_SENSOR_COMPARISON,
                        allow_partial_fill=ENABLE_PARTIAL_FILL_COMPARISON,
                        max_partial_fill_ratio=PARTIAL_FILL_MAX_RATIO,
                        min_partial_reference_iou=PARTIAL_FILL_MIN_REFERENCE_IOU,
                        allow_qa_supported_direct=(
                            ENABLE_QA_SUPPORTED_DIRECT_COMPARISON))
    if calibration is not None:
        primary = apply_sensor_area_calibration(
            primary,
            calibration,
            SENSOR_CALIBRATION_ERROR_MULTIPLIER,
        )
    # 行順ではなく湖ID・日付・観測IDで対応させる。
    keys = ["persistent_lake_id", "date", "observation_id"]
    columns = [c for c in same_alerts if c.startswith(("metric_", "alert_", "watch_"))]
    supplement = same_alerts[keys + columns].rename(
        columns={c: "same_sensor_" + c for c in columns})
    primary = primary.merge(supplement, on=keys, how="left", validate="one_to_one")
    primary = primary.drop(columns=["geometry_wkt"], errors="ignore")
    alerts = classify_lake_area_watch(primary, WATCH_MIN_INCREASE_PERCENT,
                                    WATCH_MIN_INCREASE_M2_BY_SENSOR,
                                    CROSS_SENSOR_MIN_INCREASE_M2)
    alerts = resolve_lake_area_watches(alerts, WATCH_CONFIRMATION_MAX_GAP_DAYS)
    return primary, alerts


def build_alert_review_report(
    individual_alerts: pd.DataFrame,
    group_alerts: pd.DataFrame,
) -> pd.DataFrame:
    """個別湖と関連湖グループの急増候補を確認用の一表へまとめる。

    主比較が候補でない場合も、同一センサー比較が候補なら取りこぼさず
    掲載する。単独湖の関連グループが個別湖と重複する場合は個別湖を残す。
    manual_*列は確認者が記入する。
    """
    required = {
        "persistent_lake_id", "observation_id", "date", "sensor_family",
        "metric_reference_observation_id", "metric_reference_date",
        "metric_reference_sensor", "metric_area_m2",
        "metric_reference_area_m2", "metric_area_change_m2",
        "metric_area_change_percent", "metric_area_change_m2_per_day",
        "metric_observation_type", "metric_comparison_confidence",
        "alert_level", "alert_reason", "alert_confidence",
        "alert_min_increase_m2", "alert_min_increase_percent",
        "watch_event_id", "watch_status", "watch_reason",
        "watch_resolution_observation_id", "watch_resolution_date",
        "watch_resolution_sensor", "watch_followup_area_m2",
    }
    for name, table in (
        ("個別湖", individual_alerts),
        ("関連湖グループ", group_alerts),
    ):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(
                f"{name}の確認用レポートに必要な列がありません: "
                + ", ".join(sorted(missing))
            )

    report_columns = [
        "report_event_id", "review_scope", "review_scope_ja",
        "comparison_path", "comparison_path_ja",
        "persistent_lake_id", "group_member_ids", "observed_lake_ids",
        "alert_date", "alert_observation_id", "alert_sensor",
        "reference_date", "reference_observation_id", "reference_sensor",
        "raw_current_area_m2", "current_area_m2", "reference_area_m2",
        "area_scale", "area_change_m2", "area_change_percent",
        "area_change_m2_per_day", "minimum_increase_m2",
        "minimum_increase_percent", "absolute_threshold_met",
        "percent_threshold_met", "observation_type", "fill_ratio",
        "comparison_confidence", "cross_sensor",
        "sensor_calibration_applied", "sensor_calibration_status",
        "sensor_calibration_factor", "sensor_calibration_pair_count",
        "sensor_calibration_p90_error_m2",
        "calibration_error_threshold_m2", "same_sensor_comparison_valid",
        "same_sensor_reference_date", "same_sensor_area_change_m2",
        "same_sensor_area_change_percent", "alert_level", "alert_reason",
        "alert_reason_ja", "alert_confidence", "alert_confidence_ja",
        "candidate_condition_ja", "quality_note_ja", "watch_status",
        "watch_reason", "watch_reason_ja", "resolution_date",
        "confirmation_deadline_date", "followup_date",
        "followup_observation_id", "followup_sensor", "followup_area_m2",
        "increase_from_baseline_at_followup_m2",
        "increase_from_baseline_at_followup_percent", "review_priority",
        "recommended_review_action_ja", "screening_notice_ja",
        "manual_review_status", "manual_review_comment", "reviewer",
        "reviewed_at",
    ]
    watch_levels = {"WATCH_PENDING", "WATCH_PENDING_LOW_CONFIDENCE"}

    alert_reason_ja = {
        "growth_depends_on_partial_fill_estimate": (
            "補完後面積では急増条件を満たすが、直接観測面積だけでは確定できない"
        ),
        "cross_sensor_growth_candidate_calibrated": (
            "校正済みの異種センサー比較で面積急増条件を満たした"
        ),
        "cross_sensor_growth_candidate_uncalibrated": (
            "未校正の異種センサー比較で暫定的な面積急増条件を満たした"
        ),
        "qa_supported_growth_requires_clear_confirmation": (
            "時系列的に支持されたQA条件付き観測で急増条件を満たした"
        ),
        "growth_exceeds_threshold_without_filled_current_area": (
            "補完を除いた直接観測面積でも急増条件を満たした"
        ),
        "both_growth_thresholds_met": (
            "面積増加量と面積増加率の両方が閾値以上だった"
        ),
    }
    confidence_ja = {
        "DIRECT": "補完なしの直接観測",
        "CROSS_SENSOR_CALIBRATED": "センサー間補正済みの直接観測",
        "CONDITIONAL_PARTIAL_FILL": "部分補完を含む条件付き観測",
        "CONDITIONAL_QA_SUPPORTED": "QAを時系列確認した条件付き観測",
        "NOT_EVALUATED": "未評価",
    }
    watch_reason_ja = {
        "growth_maintained_from_original_baseline": (
            "後続観測でも急増前の基準面積からの増加が維持された"
        ),
        "growth_not_maintained_from_original_baseline": (
            "後続観測では急増前の基準面積からの増加が維持されなかった"
        ),
        "no_eligible_followup_within_deadline": (
            "確認期限内に条件を満たす後続観測がなかった"
        ),
        "awaiting_eligible_calibrated_sensor_observation": (
            "校正対象センサーの補完なし後続観測を待っている"
        ),
        "awaiting_original_baseline_sensor_uncalibrated_cross_sensor_candidate": (
            "未校正のため、元の基準センサーによる後続観測を待っている"
        ),
        "awaiting_eligible_same_sensor_observation": (
            "同じセンサーによる補完なし後続観測を待っている"
        ),
        "identity_changed_or_uncertain": (
            "結合・分裂などにより湖IDの対応関係が変化または不明確になった"
        ),
        "invalid_baseline_or_threshold": "基準面積または判定閾値が不正だった",
    }

    def series(table, column, default=np.nan):
        if column in table.columns:
            return table[column].reset_index(drop=True)
        return pd.Series(default, index=range(len(table)))

    def format_number(value, decimals=0):
        number = pd.to_numeric(value, errors="coerce")
        if not np.isfinite(number):
            return "不明"
        return f"{number:,.{decimals}f}"

    def is_true(value):
        return str(value).strip().lower() in {"true", "1", "1.0"}

    def make_scope(table, scope):
        primary_mask = table["alert_level"].isin(watch_levels)
        primary_selected = table.loc[primary_mask].copy()
        primary_selected["_comparison_path"] = "primary_multisensor"

        same_sensor_level = series(
            table, "same_sensor_alert_level", ""
        )
        same_sensor_mask = (
            same_sensor_level.isin(watch_levels) & ~primary_mask
        )
        same_sensor_selected = table.loc[same_sensor_mask].copy()
        if not same_sensor_selected.empty:
            # 同一センサー側で計算したmetric/alert/watch列を、確認用の
            # 主列へ移して一つの独立した候補として扱う。
            base_columns = [
                column for column in table.columns
                if column.startswith(("metric_", "alert_", "watch_"))
            ]
            for column in base_columns:
                source_column = "same_sensor_" + column
                if source_column in table.columns:
                    same_sensor_selected[column] = (
                        same_sensor_selected[source_column]
                    )
            same_sensor_selected["_comparison_path"] = (
                "same_sensor_fallback"
            )

        selected = pd.concat(
            [primary_selected, same_sensor_selected],
            ignore_index=True,
        )
        if selected.empty:
            return pd.DataFrame(columns=report_columns)
        selected = selected.reset_index(drop=True)

        report = pd.DataFrame(index=selected.index)
        report["review_scope"] = scope
        report["review_scope_ja"] = (
            "個別湖" if scope == "individual_lake" else "関連湖グループ"
        )
        report["comparison_path"] = series(
            selected, "_comparison_path", "primary_multisensor"
        )
        report["comparison_path_ja"] = report["comparison_path"].map({
            "primary_multisensor": "直近観測を使ったマルチセンサー比較",
            "same_sensor_fallback": "同一センサー比較による補足候補",
        }).fillna("比較経路を確認")
        report["persistent_lake_id"] = series(
            selected, "persistent_lake_id", ""
        )
        report["group_member_ids"] = series(selected, "group_member_ids", "")
        report["observed_lake_ids"] = series(
            selected, "observed_lake_ids", ""
        )
        report["alert_date"] = pd.to_datetime(
            series(selected, "date"), errors="coerce"
        ).dt.normalize()
        report["alert_observation_id"] = series(
            selected, "observation_id", ""
        )
        report["alert_sensor"] = series(selected, "sensor_family", "")
        report["reference_date"] = pd.to_datetime(
            series(selected, "metric_reference_date"), errors="coerce"
        ).dt.normalize()
        report["reference_observation_id"] = series(
            selected, "metric_reference_observation_id", ""
        )
        report["reference_sensor"] = series(
            selected, "metric_reference_sensor", ""
        )
        raw_area = pd.to_numeric(
            series(selected, "metric_raw_area_m2"), errors="coerce"
        )
        current_area = pd.to_numeric(
            series(selected, "metric_area_m2"), errors="coerce"
        )
        report["raw_current_area_m2"] = raw_area.where(
            raw_area.notna(), current_area
        )
        report["current_area_m2"] = current_area
        report["reference_area_m2"] = pd.to_numeric(
            series(selected, "metric_reference_area_m2"), errors="coerce"
        )
        report["area_scale"] = series(selected, "metric_area_scale", "native")
        report["area_change_m2"] = pd.to_numeric(
            series(selected, "metric_area_change_m2"), errors="coerce"
        )
        report["area_change_percent"] = pd.to_numeric(
            series(selected, "metric_area_change_percent"), errors="coerce"
        )
        report["area_change_m2_per_day"] = pd.to_numeric(
            series(selected, "metric_area_change_m2_per_day"), errors="coerce"
        )
        report["minimum_increase_m2"] = pd.to_numeric(
            series(selected, "alert_min_increase_m2"), errors="coerce"
        )
        report["minimum_increase_percent"] = pd.to_numeric(
            series(selected, "alert_min_increase_percent"), errors="coerce"
        )
        report["absolute_threshold_met"] = series(
            selected, "alert_absolute_threshold_met", False
        )
        report["percent_threshold_met"] = series(
            selected, "alert_percent_threshold_met", False
        )
        report["observation_type"] = series(
            selected, "metric_observation_type", ""
        )
        report["fill_ratio"] = pd.to_numeric(
            series(selected, "metric_fill_ratio"), errors="coerce"
        )
        report["comparison_confidence"] = series(
            selected, "metric_comparison_confidence", ""
        )
        report["cross_sensor"] = series(
            selected, "metric_cross_sensor", False
        )
        report["sensor_calibration_applied"] = series(
            selected, "alert_sensor_calibration_applied", False
        )
        report["sensor_calibration_status"] = series(
            selected, "metric_sensor_calibration_status", "not_available"
        )
        report["sensor_calibration_factor"] = pd.to_numeric(
            series(selected, "metric_sensor_calibration_factor"),
            errors="coerce",
        )
        report["sensor_calibration_pair_count"] = pd.to_numeric(
            series(selected, "metric_sensor_calibration_pair_count"),
            errors="coerce",
        )
        report["sensor_calibration_p90_error_m2"] = pd.to_numeric(
            series(selected, "metric_sensor_calibration_p90_error_m2"),
            errors="coerce",
        )
        report["calibration_error_threshold_m2"] = pd.to_numeric(
            series(selected, "alert_calibration_error_threshold_m2"),
            errors="coerce",
        )
        report["same_sensor_comparison_valid"] = series(
            selected, "same_sensor_metric_comparison_valid", False
        )
        report["same_sensor_reference_date"] = pd.to_datetime(
            series(selected, "same_sensor_metric_reference_date"),
            errors="coerce",
        ).dt.normalize()
        report["same_sensor_area_change_m2"] = pd.to_numeric(
            series(selected, "same_sensor_metric_area_change_m2"),
            errors="coerce",
        )
        report["same_sensor_area_change_percent"] = pd.to_numeric(
            series(selected, "same_sensor_metric_area_change_percent"),
            errors="coerce",
        )
        report["alert_level"] = series(selected, "alert_level", "")
        report["alert_reason"] = series(selected, "alert_reason", "")
        report["alert_reason_ja"] = report["alert_reason"].map(
            alert_reason_ja
        ).fillna("判定理由の詳細はalert_reasonを確認")
        report["alert_confidence"] = series(
            selected, "alert_confidence", ""
        )
        report["alert_confidence_ja"] = report["alert_confidence"].map(
            confidence_ja
        ).fillna("信頼度区分を確認")
        report["watch_status"] = series(selected, "watch_status", "")
        report["watch_reason"] = series(selected, "watch_reason", "")
        report["watch_reason_ja"] = report["watch_reason"].map(
            watch_reason_ja
        ).fillna("後続確認理由の詳細はwatch_reasonを確認")
        report["resolution_date"] = pd.to_datetime(
            series(selected, "watch_resolution_date"), errors="coerce"
        ).dt.normalize()
        report["confirmation_deadline_date"] = pd.to_datetime(
            series(selected, "watch_deadline_date"), errors="coerce"
        ).dt.normalize()
        report["followup_observation_id"] = series(
            selected, "watch_resolution_observation_id", ""
        )
        report["followup_date"] = report["resolution_date"].where(
            report["followup_observation_id"].fillna("").astype(str)
            .str.strip().ne("")
        )
        report["followup_sensor"] = series(
            selected, "watch_resolution_sensor", ""
        )
        report["followup_area_m2"] = pd.to_numeric(
            series(selected, "watch_followup_area_m2"), errors="coerce"
        )
        report["increase_from_baseline_at_followup_m2"] = pd.to_numeric(
            series(selected, "watch_increase_from_baseline_m2"),
            errors="coerce",
        )
        report["increase_from_baseline_at_followup_percent"] = pd.to_numeric(
            series(selected, "watch_increase_from_baseline_percent"),
            errors="coerce",
        )
        watch_event_id = series(selected, "watch_event_id", "").astype(str)
        fallback_event_id = (
            report["persistent_lake_id"].astype(str)
            + "::" + report["alert_observation_id"].astype(str)
        )
        watch_event_id = watch_event_id.where(
            watch_event_id.str.strip().ne(""), fallback_event_id
        )
        report["report_event_id"] = scope + "::" + watch_event_id

        report["candidate_condition_ja"] = report.apply(
            lambda row: (
                f"増加量 {format_number(row['area_change_m2'])} m² "
                f"（閾値 {format_number(row['minimum_increase_m2'])} m²）、"
                f"増加率 {format_number(row['area_change_percent'], 1)}% "
                f"（閾値 {format_number(row['minimum_increase_percent'], 1)}%）"
            ),
            axis=1,
        )

        def quality_note(row):
            notes = []
            if row["observation_type"] == "partial_fill":
                notes.append(
                    "部分補完を含む（補完率 "
                    + format_number(row["fill_ratio"] * 100, 1) + "%）"
                )
            elif row["observation_type"] == "qa_supported_direct":
                notes.append("QA条件付きだが時系列的に支持された直接面積")
            else:
                notes.append("補完なしの直接面積")
            if is_true(row["cross_sensor"]):
                if is_true(row["sensor_calibration_applied"]):
                    notes.append("異種センサー比較・補正適用済み")
                else:
                    notes.append("異種センサー比較・補正未適用")
            return "／".join(notes)

        report["quality_note_ja"] = report.apply(quality_note, axis=1)
        priority_map = {
            "REVIEW_CONFIRMED": "HIGH",
            "WATCH_PENDING": "MEDIUM",
            "EXPIRED_UNCONFIRMED": "MEDIUM",
            "DISMISSED": "LOW",
            "INVALID": "TECHNICAL_REVIEW",
        }
        report["review_priority"] = report["watch_status"].map(
            priority_map
        ).fillna("MEDIUM")

        def recommended_action(row):
            status = row["watch_status"]
            if status == "REVIEW_CONFIRMED":
                return (
                    "原画像で雲・影・雪氷・湖岸を確認し、真の増水なら"
                    "GLOF評価担当へ引き渡す"
                )
            if status == "DISMISSED":
                return "一時的変化または誤検出だったかを原画像で記録確認する"
            if status == "EXPIRED_UNCONFIRMED":
                return "有効な後続観測を追加し、確認判定を再実行する"
            if status == "INVALID":
                return "永続湖ID、結合・分裂関係、品質フラグを確認する"
            return "原画像を確認し、補完なしの後続観測を待つ"

        report["recommended_review_action_ja"] = report.apply(
            recommended_action, axis=1
        )
        report["screening_notice_ja"] = (
            "この結果はGLOF発生の確定ではなく、人による確認候補である"
        )
        for column in [
            "manual_review_status", "manual_review_comment",
            "reviewer", "reviewed_at",
        ]:
            report[column] = ""
        return report.reindex(columns=report_columns)

    report = pd.concat(
        [
            make_scope(individual_alerts, "individual_lake"),
            make_scope(group_alerts, "related_lake_group"),
        ],
        ignore_index=True,
    )
    if report.empty:
        return pd.DataFrame(columns=report_columns)

    # 一つの湖だけで構成される関連グループは、同じ観測・比較経路の
    # 個別湖候補が存在する場合に限り重複掲載しない。
    individual_keys = set(
        zip(
            report.loc[
                report["review_scope"].eq("individual_lake"),
                "persistent_lake_id",
            ].fillna("").astype(str),
            report.loc[
                report["review_scope"].eq("individual_lake"),
                "alert_observation_id",
            ].fillna("").astype(str),
            report.loc[
                report["review_scope"].eq("individual_lake"),
                "comparison_path",
            ].fillna("").astype(str),
        )
    )
    singleton_group = (
        report["review_scope"].eq("related_lake_group")
        & ~report["group_member_ids"].fillna("").astype(str).str.contains(
            ";", regex=False
        )
    )
    duplicate_singleton_group = pd.Series(False, index=report.index)
    for index in report.index[singleton_group]:
        logical_key = (
            str(report.at[index, "group_member_ids"]),
            str(report.at[index, "alert_observation_id"]),
            str(report.at[index, "comparison_path"]),
        )
        duplicate_singleton_group.at[index] = logical_key in individual_keys
    report = report.loc[~duplicate_singleton_group].copy()

    if report["report_event_id"].duplicated().any():
        raise ValueError("確認用レポートのイベントIDが重複しています。")
    priority_order = report["review_priority"].map({
        "HIGH": 0,
        "TECHNICAL_REVIEW": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }).fillna(4)
    return report.assign(_priority_order=priority_order).sort_values(
        ["_priority_order", "alert_date", "report_event_id"],
        ascending=[True, True, True],
    ).drop(columns="_priority_order").reset_index(drop=True)


def main() -> None:
    observations = pd.read_csv(
        INPUT_CSV,
        encoding="utf-8-sig",
    )

    calibration_source = calculate_lake_change_metrics(
        observations,
        MAX_COMPARISON_GAP_DAYS,
        AREA_COLUMN,
        allow_partial_fill=ENABLE_PARTIAL_FILL_COMPARISON,
        max_partial_fill_ratio=PARTIAL_FILL_MAX_RATIO,
        min_partial_reference_iou=PARTIAL_FILL_MIN_REFERENCE_IOU,
        allow_qa_supported_direct=ENABLE_QA_SUPPORTED_DIRECT_COMPARISON,
    )
    calibration, calibration_pairs = estimate_sensor_area_calibration(
        calibration_source,
        reference_sensor=SENSOR_CALIBRATION_REFERENCE,
        target_sensor=SENSOR_CALIBRATION_TARGET,
        max_pair_gap_days=SENSOR_CALIBRATION_MAX_PAIR_GAP_DAYS,
        min_pair_count=SENSOR_CALIBRATION_MIN_PAIR_COUNT,
        min_independent_lake_count=(
            SENSOR_CALIBRATION_MIN_INDEPENDENT_LAKE_COUNT
        ),
        min_ratio=SENSOR_CALIBRATION_MIN_RATIO,
        max_ratio=SENSOR_CALIBRATION_MAX_RATIO,
        max_lake_ratio_modified_z=(
            SENSOR_CALIBRATION_MAX_LAKE_RATIO_MODIFIED_Z
        ),
    )

    result, alerts = build_alert_outputs(
        observations,
        calibration=calibration,
    )
    group_metrics, group_alerts = build_alert_outputs(
        observations,
        grouped=True,
        calibration=calibration,
    )
    review_report = build_alert_review_report(alerts, group_alerts)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ALERT_REVIEW_REPORT_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration.to_csv(
        SENSOR_CALIBRATION_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    calibration_pairs.to_csv(
        SENSOR_CALIBRATION_PAIRS_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    calibration_row = calibration.iloc[0]
    print(
        "センサー間面積補正: "
        f"{calibration_row['calibration_status']}, "
        f"係数={calibration_row['target_to_reference_area_factor']:.6f}, "
        f"使用ペア数={int(calibration_row['used_pair_count'])}, "
        f"独立湖数={int(calibration_row['used_lake_count'])}"
    )
    print(f"センサー間補正CSV: {SENSOR_CALIBRATION_OUTPUT_CSV}")
    print(f"センサー間補正ペアCSV: {SENSOR_CALIBRATION_PAIRS_OUTPUT_CSV}")

    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print(f"入力観測数: {len(result)}")
    print("異種センサー比較数: " + str(int(
        (result["metric_comparison_valid"] & result["metric_cross_sensor"]).sum())))
    print("部分補完を含む条件付き比較数: " + str(int(
        (result["metric_comparison_valid"]
         & result["metric_comparison_confidence"].eq(
             "conditional_partial_fill")).sum())))
    print(
        "比較可能な観測数: "
        f"{int(result['metric_comparison_valid'].sum())}"
    )
    print(
        result["metric_comparison_status"]
        .value_counts()
        .to_string()
    )
    print(f"出力CSV: {OUTPUT_CSV}")
    alerts.to_csv(
        ALERT_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    print(alerts["alert_level"].value_counts().to_string())
    print(alerts["watch_status"].value_counts().to_string())
    print(f"要確認判定CSV: {ALERT_OUTPUT_CSV}")
    for table, path in ((group_metrics, GROUP_OUTPUT_CSV),
                        (group_alerts, GROUP_ALERT_OUTPUT_CSV)):
        table.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        print(f"関連湖グループ出力: {path}")
    print(group_metrics["metric_comparison_status"].value_counts().to_string())
    print("関連湖グループの異種センサー比較数: " + str(int(
        (group_metrics["metric_comparison_valid"] & group_metrics["metric_cross_sensor"]).sum())))
    print("関連湖グループの部分補完条件付き比較数: " + str(int(
        (group_metrics["metric_comparison_valid"]
         & group_metrics["metric_comparison_confidence"].eq(
             "conditional_partial_fill")).sum())))
    review_report.to_csv(
        ALERT_REVIEW_REPORT_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    print(f"確認用レポート候補数: {len(review_report)}")
    if not review_report.empty:
        print(review_report["watch_status"].value_counts().to_string())
    print(f"確認用レポートCSV: {ALERT_REVIEW_REPORT_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
