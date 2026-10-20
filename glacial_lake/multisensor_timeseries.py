from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def add_multisensor_change_metrics(
    timeseries_dataframe: pd.DataFrame,
    long_gap_days: float,
) -> pd.DataFrame:

    required_columns = {
        "date",
        "sensor_family",
        "timeseries_area_m2",
        "is_missing",
    }

    missing_columns = (
        required_columns
        - set(timeseries_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "変化指標の計算に必要な列がありません: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if long_gap_days <= 0.0:
        raise ValueError(
            "long_gap_daysは"
            "0より大きくしてください。"
        )

    output_dataframe = (
        timeseries_dataframe
        .copy()
        .reset_index(drop=True)
    )

    observation_dates = pd.to_datetime(
        output_dataframe["date"],
        errors="coerce",
    )

    invalid_date_mask = (
        observation_dates.isna()
    )

    if invalid_date_mask.any():
        invalid_rows = (
            output_dataframe.loc[
                invalid_date_mask,
                "date",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "変化指標を計算できない"
            "日付が含まれています: "
            + ", ".join(
                invalid_rows
            )
        )

    output_dataframe[
        "_metric_date"
    ] = observation_dates

    output_dataframe = (
        output_dataframe
        .sort_values(
            "_metric_date"
        )
        .reset_index(drop=True)
    )

    duplicated_date_mask = (
        output_dataframe[
            "_metric_date"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicated_date_mask.any():
        duplicated_dates = (
            output_dataframe.loc[
                duplicated_date_mask,
                "_metric_date",
            ]
            .dt.strftime(
                "%Y-%m-%d"
            )
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "変化指標の計算対象に"
            "重複日付があります: "
            + ", ".join(
                duplicated_dates
            )
        )

    timeseries_areas_m2 = pd.to_numeric(
        output_dataframe[
            "timeseries_area_m2"
        ],
        errors="coerce",
    )

    missing_values = (
        output_dataframe[
            "is_missing"
        ]
    )

    if pd.api.types.is_bool_dtype(
        missing_values
    ):
        is_missing = (
            missing_values
            .fillna(False)
            .astype(bool)
        )

    else:
        is_missing = (
            missing_values
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                {
                    "true",
                    "1",
                    "yes",
                }
            )
        )

    previous_dates = (
        output_dataframe[
            "_metric_date"
        ]
        .shift(1)
    )

    previous_sensor_family = (
        output_dataframe[
            "sensor_family"
        ]
        .astype(str)
        .shift(1)
    )

    previous_areas_m2 = (
        timeseries_areas_m2
        .shift(1)
    )

    previous_is_missing = (
        is_missing
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    observation_interval_days = (
        output_dataframe[
            "_metric_date"
        ]
        - previous_dates
    ).dt.total_seconds() / 86400.0

    invalid_interval_mask = (
        observation_interval_days
        .notna()
        & observation_interval_days
        .le(0.0)
    )

    if invalid_interval_mask.any():
        invalid_dates = (
            output_dataframe.loc[
                invalid_interval_mask,
                "_metric_date",
            ]
            .dt.strftime(
                "%Y-%m-%d"
            )
            .tolist()
        )

        raise ValueError(
            "0日以下の観測間隔が"
            "含まれています: "
            + ", ".join(
                invalid_dates
            )
        )

    current_sensor_family = (
        output_dataframe[
            "sensor_family"
        ]
        .astype(str)
    )

    sensor_changed = (
        previous_sensor_family
        .notna()
        & previous_sensor_family
        .ne(current_sensor_family)
    )

    sensor_transition = pd.Series(
        "",
        index=output_dataframe.index,
        dtype="object",
    )

    has_previous_sensor = (
        previous_sensor_family
        .notna()
    )

    sensor_transition.loc[
        has_previous_sensor
    ] = (
        previous_sensor_family.loc[
            has_previous_sensor
        ]
        + "->"
        + current_sensor_family.loc[
            has_previous_sensor
        ]
    )

    change_comparison_valid = (
        previous_dates.notna()
        & observation_interval_days.gt(0.0)
        & timeseries_areas_m2.notna()
        & previous_areas_m2.notna()
        & ~is_missing
        & ~previous_is_missing
    )

    area_change_m2 = (
        timeseries_areas_m2
        - previous_areas_m2
    ).where(
        change_comparison_valid,
        np.nan,
    )

    area_change_km2 = (
        area_change_m2
        / 1_000_000.0
    )

    area_change_m2_per_day = (
        area_change_m2
        / observation_interval_days
    ).where(
        change_comparison_valid,
        np.nan,
    )

    area_change_km2_per_day = (
        area_change_m2_per_day
        / 1_000_000.0
    )

    percent_comparison_valid = (
        change_comparison_valid
        & previous_areas_m2.gt(0.0)
    )

    area_change_percent = (
        (
            area_change_m2
            / previous_areas_m2
        )
        * 100.0
    ).where(
        percent_comparison_valid,
        np.nan,
    )

    change_comparison_reason = pd.Series(
        "valid_comparison",
        index=output_dataframe.index,
        dtype="object",
    )

    first_observation_mask = (
        previous_dates.isna()
    )

    change_comparison_reason.loc[
        first_observation_mask
    ] = "first_observation"

    current_missing_mask = (
        ~first_observation_mask
        & is_missing
    )

    change_comparison_reason.loc[
        current_missing_mask
    ] = "current_observation_missing"

    previous_missing_mask = (
        ~first_observation_mask
        & ~is_missing
        & previous_is_missing
    )

    change_comparison_reason.loc[
        previous_missing_mask
    ] = "previous_observation_missing"

    current_area_missing_mask = (
        ~first_observation_mask
        & ~is_missing
        & ~previous_is_missing
        & timeseries_areas_m2.isna()
    )

    change_comparison_reason.loc[
        current_area_missing_mask
    ] = "current_area_missing"

    previous_area_missing_mask = (
        ~first_observation_mask
        & ~is_missing
        & ~previous_is_missing
        & timeseries_areas_m2.notna()
        & previous_areas_m2.isna()
    )

    change_comparison_reason.loc[
        previous_area_missing_mask
    ] = "previous_area_missing"

    output_dataframe[
        "previous_date"
    ] = (
        previous_dates
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output_dataframe[
        "previous_sensor_family"
    ] = (
        previous_sensor_family
        .fillna("")
    )

    output_dataframe[
        "sensor_transition"
    ] = sensor_transition

    output_dataframe[
        "sensor_changed"
    ] = sensor_changed

    output_dataframe[
        "observation_interval_days"
    ] = observation_interval_days

    output_dataframe[
        "long_gap"
    ] = (
        observation_interval_days
        .gt(long_gap_days)
        .fillna(False)
    )

    output_dataframe[
        "previous_area_m2"
    ] = previous_areas_m2

    output_dataframe[
        "change_comparison_valid"
    ] = change_comparison_valid

    output_dataframe[
        "change_comparison_reason"
    ] = change_comparison_reason

    output_dataframe[
        "area_change_m2"
    ] = area_change_m2

    output_dataframe[
        "area_change_km2"
    ] = area_change_km2

    output_dataframe[
        "area_change_percent"
    ] = area_change_percent

    output_dataframe[
        "area_change_m2_per_day"
    ] = area_change_m2_per_day

    output_dataframe[
        "area_change_km2_per_day"
    ] = area_change_km2_per_day

    output_dataframe = (
        output_dataframe
        .drop(
            columns=[
                "_metric_date",
            ]
        )
    )

    return output_dataframe


def save_selected_multisensor_timeseries(
    catalog_dataframe: pd.DataFrame,
    output_csv: Path,
    long_gap_days: float,
) -> pd.DataFrame:

    required_columns = {
        "observation_id",
        "scene",
        "date",
        "observation_datetime",
        "sensor_family",
        "platform",
        "product_id",
        "native_pixel_size_m",
        "ndwi_threshold",
        "selected_for_timeseries",
        "selection_reason",
        "quality_status",
        "quality_score",
        "quality_reason",
        "aoi_water_area_m2",
        "timeseries_area_m2",
    }

    missing_columns = (
        required_columns
        - set(catalog_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "統合時系列CSVの作成に必要な列がありません: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    working_dataframe = (
        catalog_dataframe
        .copy()
        .reset_index(drop=True)
    )

    selected_values = (
        working_dataframe[
            "selected_for_timeseries"
        ]
    )

    if pd.api.types.is_bool_dtype(
        selected_values
    ):
        selected_mask = (
            selected_values
            .fillna(False)
        )

    else:
        selected_mask = (
            selected_values
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                {
                    "true",
                    "1",
                    "yes",
                }
            )
        )

    selected_dataframe = (
        working_dataframe.loc[
            selected_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    if selected_dataframe.empty:
        raise ValueError(
            "selected_for_timeseries=Trueの"
            "観測がありません。"
        )

    parsed_dates = pd.to_datetime(
        selected_dataframe["date"],
        errors="coerce",
    )

    invalid_date_mask = (
        parsed_dates.isna()
    )

    if invalid_date_mask.any():
        invalid_observation_ids = (
            selected_dataframe.loc[
                invalid_date_mask,
                "observation_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "採用観測の日付を解釈できません: "
            + ", ".join(
                invalid_observation_ids
            )
        )

    selected_dataframe[
        "_sort_date"
    ] = parsed_dates

    duplicated_date_mask = (
        selected_dataframe[
            "_sort_date"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicated_date_mask.any():
        duplicated_dates = (
            selected_dataframe.loc[
                duplicated_date_mask,
                "_sort_date",
            ]
            .dt.strftime(
                "%Y-%m-%d"
            )
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "同じ日付で複数の観測が"
            "時系列用に選択されています: "
            + ", ".join(
                duplicated_dates
            )
        )

    allowed_quality_statuses = {
        "usable",
        "partial",
        "missing",
    }

    unexpected_quality_mask = (
        ~selected_dataframe[
            "quality_status"
        ]
        .astype(str)
        .isin(
            allowed_quality_statuses
        )
    )

    if unexpected_quality_mask.any():
        unexpected_rows = (
            selected_dataframe.loc[
                unexpected_quality_mask,
                [
                    "date",
                    "sensor_family",
                    "quality_status",
                ],
            ]
            .astype(str)
            .agg(
                " / ".join,
                axis=1,
            )
            .tolist()
        )

        raise ValueError(
            "時系列用に選択できない"
            "品質状態が含まれています: "
            + ", ".join(
                unexpected_rows
            )
        )

    selected_dataframe[
        "detected_area_m2"
    ] = pd.to_numeric(
        selected_dataframe[
            "aoi_water_area_m2"
        ],
        errors="coerce",
    )

    selected_dataframe[
        "timeseries_area_m2"
    ] = pd.to_numeric(
        selected_dataframe[
            "timeseries_area_m2"
        ],
        errors="coerce",
    )

    missing_observation_mask = (
        selected_dataframe[
            "quality_status"
        ]
        .astype(str)
        .eq("missing")
    )

    non_missing_observation_mask = (
        ~missing_observation_mask
    )

    missing_area_in_valid_observation_mask = (
        non_missing_observation_mask
        & selected_dataframe[
            "timeseries_area_m2"
        ].isna()
    )

    if (
        missing_area_in_valid_observation_mask
        .any()
    ):
        invalid_rows = (
            selected_dataframe.loc[
                missing_area_in_valid_observation_mask,
                [
                    "date",
                    "sensor_family",
                    "observation_id",
                ],
            ]
            .astype(str)
            .agg(
                " / ".join,
                axis=1,
            )
            .tolist()
        )

        raise ValueError(
            "missing以外の採用観測で"
            "面積が空欄になっています: "
            + ", ".join(
                invalid_rows
            )
        )

    negative_area_mask = (
        selected_dataframe[
            "timeseries_area_m2"
        ].lt(0.0)
    )

    if negative_area_mask.any():
        invalid_rows = (
            selected_dataframe.loc[
                negative_area_mask,
                [
                    "date",
                    "sensor_family",
                    "timeseries_area_m2",
                ],
            ]
            .astype(str)
            .agg(
                " / ".join,
                axis=1,
            )
            .tolist()
        )

        raise ValueError(
            "負の水域面積が含まれています: "
            + ", ".join(
                invalid_rows
            )
        )

    selected_dataframe.loc[
        missing_observation_mask,
        "timeseries_area_m2",
    ] = np.nan

    selected_dataframe[
        "timeseries_area_km2"
    ] = (
        selected_dataframe[
            "timeseries_area_m2"
        ]
        / 1_000_000.0
    )

    selected_dataframe[
        "is_missing"
    ] = missing_observation_mask

    optional_column_defaults = {
        "polygon_file": "",
        "water_mask_file": "",
        "ndwi_file": "",
        "quality_mask_file": "",
        "cloud_cover_scene_percent": np.nan,
        "aoi_valid_pixel_ratio": np.nan,
        "aoi_water_qa_review_percent": np.nan,
        "same_day_area_ratio": np.nan,
        "reference_sentinel_area_m2": np.nan,
        "missing_evidence": "",
        "missing_previous_gap_days": np.nan,
        "missing_next_gap_days": np.nan,
    }

    for (
        column_name,
        default_value,
    ) in optional_column_defaults.items():
        if (
            column_name
            not in selected_dataframe.columns
        ):
            selected_dataframe[
                column_name
            ] = default_value

    selected_dataframe = (
        selected_dataframe
        .sort_values(
            "_sort_date"
        )
        .reset_index(drop=True)
    )

    selected_dataframe = (
        add_multisensor_change_metrics(
            timeseries_dataframe=(
                selected_dataframe
            ),
            long_gap_days=(
                long_gap_days
            ),
        )
    )

    selected_dataframe[
        "date"
    ] = (
        selected_dataframe[
            "_sort_date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output_columns = [
        "date",
        "observation_datetime",
        "sensor_family",
        "platform",
        "observation_id",
        "scene",
        "product_id",
        "native_pixel_size_m",
        "ndwi_threshold",
        "quality_status",
        "quality_score",
        "quality_reason",
        "selection_reason",
        "is_missing",
        "previous_date",
        "previous_sensor_family",
        "sensor_transition",
        "sensor_changed",
        "observation_interval_days",
        "long_gap",
        "previous_area_m2",
        "detected_area_m2",
        "timeseries_area_m2",
        "timeseries_area_km2",
        "change_comparison_valid",
        "change_comparison_reason",
        "area_change_m2",
        "area_change_km2",
        "area_change_percent",
        "area_change_m2_per_day",
        "area_change_km2_per_day",
        "cloud_cover_scene_percent",
        "aoi_valid_pixel_ratio",
        "aoi_water_qa_review_percent",
        "reference_sentinel_area_m2",
        "same_day_area_ratio",
        "missing_evidence",
        "missing_previous_gap_days",
        "missing_next_gap_days",
        "polygon_file",
        "water_mask_file",
        "ndwi_file",
        "quality_mask_file",
    ]

    output_dataframe = (
        selected_dataframe[
            output_columns
        ]
        .copy()
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        output_dataframe.to_csv(
            output_csv,
            index=False,
            encoding="utf-8-sig",
        )

    except PermissionError as error:
        raise PermissionError(
            "統合時系列CSVを保存できません。"
            "ExcelやQGISでCSVを開いている場合は"
            "閉じてから再実行してください: "
            f"{output_csv}"
        ) from error

    return output_dataframe

