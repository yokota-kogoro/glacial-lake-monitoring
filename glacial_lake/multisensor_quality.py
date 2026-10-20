from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def value_to_text(
    value,
) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return str(value).strip()


def path_value_to_text(
    value,
) -> str:

    path_text = value_to_text(value)

    if path_text == "":
        return ""

    return str(
        Path(path_text).resolve()
    )


def mark_interior_zero_observations_as_missing(
    quality_dataframe: pd.DataFrame,
    max_neighbor_gap_days: float,
) -> pd.DataFrame:

    required_columns = {
        "date",
        "sensor_family",
        "aoi_water_area_m2",
        "quality_status",
        "quality_score",
        "quality_reason",
    }

    missing_columns = (
        required_columns
        - set(quality_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "0 m²観測の欠損判定に必要な列がありません: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if max_neighbor_gap_days <= 0.0:
        raise ValueError(
            "max_neighbor_gap_daysは"
            "0より大きくしてください。"
        )

    output_dataframe = (
        quality_dataframe
        .copy()
    )

    if "missing_evidence" not in output_dataframe.columns:
        output_dataframe[
            "missing_evidence"
        ] = ""

    if (
        "missing_previous_gap_days"
        not in output_dataframe.columns
    ):
        output_dataframe[
            "missing_previous_gap_days"
        ] = np.nan

    if (
        "missing_next_gap_days"
        not in output_dataframe.columns
    ):
        output_dataframe[
            "missing_next_gap_days"
        ] = np.nan

    observation_dates = pd.to_datetime(
        output_dataframe["date"],
        errors="coerce",
    )

    water_areas_m2 = pd.to_numeric(
        output_dataframe["aoi_water_area_m2"],
        errors="coerce",
    )

    usable_mask = (
        output_dataframe["quality_status"]
        .astype(str)
        .eq("usable")
    )

    positive_area_mask = (
        usable_mask
        & observation_dates.notna()
        & water_areas_m2.notna()
        & water_areas_m2.gt(0.0)
    )

    positive_observation_dates = pd.DatetimeIndex(
        observation_dates.loc[
            positive_area_mask
        ].unique()
    ).sort_values()

    for row_index in output_dataframe.index:
        sensor_family = str(
            output_dataframe.at[
                row_index,
                "sensor_family",
            ]
        )

        quality_status = str(
            output_dataframe.at[
                row_index,
                "quality_status",
            ]
        )

        observation_date = (
            observation_dates.loc[
                row_index
            ]
        )

        water_area_m2 = (
            water_areas_m2.loc[
                row_index
            ]
        )

        if sensor_family != "Sentinel-2":
            continue

        if quality_status != "usable":
            continue

        if pd.isna(observation_date):
            continue

        if pd.isna(water_area_m2):
            continue

        if float(water_area_m2) > 0.0:
            continue

        same_day_positive_mask = (
            positive_area_mask
            & observation_dates.eq(
                observation_date
            )
        )

        same_day_positive_indices = (
            output_dataframe.index[
                same_day_positive_mask
            ]
        )

        if len(same_day_positive_indices) > 0:
            reference_index = (
                same_day_positive_indices[0]
            )

            reference_sensor = str(
                output_dataframe.at[
                    reference_index,
                    "sensor_family",
                ]
            )

            output_dataframe.at[
                row_index,
                "quality_status",
            ] = "missing"

            output_dataframe.at[
                row_index,
                "quality_score",
            ] = 0.0

            output_dataframe.at[
                row_index,
                "quality_reason",
            ] = (
                "sentinel_zero_with_same_day_"
                "usable_positive_observation"
            )

            output_dataframe.at[
                row_index,
                "missing_evidence",
            ] = (
                "same_day_positive:"
                f"{reference_sensor}"
            )

            continue

        previous_positive_dates = (
            positive_observation_dates[
                positive_observation_dates
                < observation_date
            ]
        )

        next_positive_dates = (
            positive_observation_dates[
                positive_observation_dates
                > observation_date
            ]
        )

        if len(previous_positive_dates) == 0:
            continue

        if len(next_positive_dates) == 0:
            continue

        previous_date = (
            previous_positive_dates[-1]
        )

        next_date = (
            next_positive_dates[0]
        )

        previous_gap_days = (
            observation_date
            - previous_date
        ).total_seconds() / 86400.0

        next_gap_days = (
            next_date
            - observation_date
        ).total_seconds() / 86400.0

        if (
            previous_gap_days
            > max_neighbor_gap_days
        ):
            continue

        if (
            next_gap_days
            > max_neighbor_gap_days
        ):
            continue

        output_dataframe.at[
            row_index,
            "quality_status",
        ] = "missing"

        output_dataframe.at[
            row_index,
            "quality_score",
        ] = 0.0

        output_dataframe.at[
            row_index,
            "quality_reason",
        ] = (
            "sentinel_interior_zero_between_"
            "positive_observations"
        )

        output_dataframe.at[
            row_index,
            "missing_evidence",
        ] = (
            f"bracketed:"
            f"{previous_date.strftime('%Y-%m-%d')}"
            f"/"
            f"{next_date.strftime('%Y-%m-%d')}"
        )

        output_dataframe.at[
            row_index,
            "missing_previous_gap_days",
        ] = previous_gap_days

        output_dataframe.at[
            row_index,
            "missing_next_gap_days",
        ] = next_gap_days

    return output_dataframe


def add_multisensor_quality_flags(
    catalog_dataframe: pd.DataFrame,
    sentinel_min_aoi_valid_pixel_ratio: float,
    landsat_use_scene_cloud_cover_for_rejection: bool,
    landsat_usable_max_cloud_percent: float,
    landsat_partial_max_cloud_percent: float,
    landsat_usable_min_area_ratio: float,
    landsat_usable_max_area_ratio: float,
    landsat_reject_min_area_ratio: float,
    landsat_reject_max_area_ratio: float,
    landsat_min_usable_pixel_count: int,
    landsat_reject_pixel_count: int,
    landsat_usable_max_aoi_water_qa_review_percent: float,
    select_partial_when_no_usable: bool,
    sentinel_interior_zero_max_neighbor_gap_days: float,
) -> pd.DataFrame:

    required_catalog_columns = {
        "observation_id",
        "date",
        "sensor_family",
        "platform",
        "catalog_status",
        "cloud_cover_scene_percent",
        "aoi_valid_pixel_ratio",
        "aoi_water_qa_review_percent",
        "aoi_water_pixel_count",
        "aoi_water_area_m2",
        "polygon_status",
    }

    missing_catalog_columns = (
        required_catalog_columns
        - set(catalog_dataframe.columns)
    )

    if missing_catalog_columns:
        raise ValueError(
            "品質判定に必要なカタログ列がありません: "
            + ", ".join(
                sorted(
                    missing_catalog_columns
                )
            )
        )

    if (
        landsat_usable_max_cloud_percent
        > landsat_partial_max_cloud_percent
    ):
        raise ValueError(
            "usableの雲量上限はpartialの"
            "雲量上限以下にしてください。"
        )

    if not 0.0 <= sentinel_min_aoi_valid_pixel_ratio <= 1.0:
        raise ValueError(
            "sentinel_min_aoi_valid_pixel_ratioは"
            "0以上1以下にしてください。"
        )

    if not (
        0.0
        <= landsat_usable_max_aoi_water_qa_review_percent
        <= 100.0
    ):
        raise ValueError(
            "landsat_usable_max_aoi_water_qa_review_percentは"
            "0以上100以下にしてください。"
        )

    if landsat_min_usable_pixel_count <= 0:
        raise ValueError(
            "landsat_min_usable_pixel_countは"
            "1以上にしてください。"
        )

    if landsat_reject_pixel_count < 0:
        raise ValueError(
            "landsat_reject_pixel_countは"
            "0以上にしてください。"
        )

    if (
        landsat_reject_min_area_ratio
        > landsat_usable_min_area_ratio
    ):
        raise ValueError(
            "棄却下限はusable下限以下に"
            "してください。"
        )

    if (
        landsat_usable_max_area_ratio
        > landsat_reject_max_area_ratio
    ):
        raise ValueError(
            "usable上限は棄却上限以下に"
            "してください。"
        )

    output_dataframe = (
        catalog_dataframe
        .copy()
        .reset_index(drop=True)
    )

    output_dataframe["date"] = (
        pd.to_datetime(
            output_dataframe["date"],
            errors="raise",
        )
        .dt.normalize()
    )

    sentinel_reference_by_date: dict[
        pd.Timestamp,
        dict,
    ] = {}

    sentinel_rows = output_dataframe[
        output_dataframe[
            "sensor_family"
        ]
        .astype(str)
        .eq("Sentinel-2")
    ].copy()

    sentinel_rows[
        "_aoi_area_m2"
    ] = pd.to_numeric(
        sentinel_rows[
            "aoi_water_area_m2"
        ],
        errors="coerce",
    )

    valid_sentinel_reference_mask = (
        sentinel_rows[
            "catalog_status"
        ]
        .astype(str)
        .eq("ok")
        & sentinel_rows[
            "polygon_status"
        ]
        .astype(str)
        .isin(
            {
                "ok",
                "no_retained_water_polygon",
            }
        )
        & sentinel_rows[
            "_aoi_area_m2"
        ]
        .notna()
    )

    valid_sentinel_rows = sentinel_rows[
        valid_sentinel_reference_mask
    ].copy()

    for (
        observation_date,
        same_day_sentinel_rows,
    ) in valid_sentinel_rows.groupby(
        "date",
        sort=True,
    ):
        same_day_area_values = pd.to_numeric(
            same_day_sentinel_rows[
                "_aoi_area_m2"
            ],
            errors="coerce",
        ).dropna()

        if same_day_area_values.empty:
            continue

        sentinel_reference_by_date[
            pd.Timestamp(
                observation_date
            ).normalize()
        ] = {
            "area_m2": float(
                same_day_area_values.max()
            ),
            "observation_count": int(
                len(
                    same_day_sentinel_rows
                )
            ),
        }

    quality_columns = {
        "reference_sentinel_area_m2": np.nan,
        "same_day_area_ratio": np.nan,
        "quality_status": "",
        "quality_score": np.nan,
        "quality_reason": "",
        "selected_for_timeseries": False,
        "selection_reason": "",
        "timeseries_area_m2": np.nan,
    }

    for column_name, default_value in (
        quality_columns.items()
    ):
        output_dataframe[
            column_name
        ] = default_value

    for row_index, row in (
        output_dataframe.iterrows()
    ):
        observation_date = pd.Timestamp(
            row["date"]
        ).normalize()

        sensor_family = value_to_text(
            row["sensor_family"]
        )

        catalog_status = value_to_text(
            row["catalog_status"]
        )

        if sensor_family == "Sentinel-2":
            sentinel_area_m2 = pd.to_numeric(
                row["aoi_water_area_m2"],
                errors="coerce",
            )

            sentinel_reference = (
                {
                    "area_m2": sentinel_area_m2,
                }
                if pd.notna(
                    sentinel_area_m2
                )
                else None
            )

        else:
            sentinel_reference = (
                sentinel_reference_by_date.get(
                    observation_date
                )
            )

            if sentinel_reference is not None:
                sentinel_area_m2 = (
                    sentinel_reference[
                        "area_m2"
                    ]
                )

            else:
                sentinel_area_m2 = np.nan

        if pd.notna(
            sentinel_area_m2
        ):
            output_dataframe.at[
                row_index,
                "reference_sentinel_area_m2",
            ] = float(
                sentinel_area_m2
            )

        # ----------------------------------------------------
        # Sentinel-2
        # ----------------------------------------------------

        if sensor_family == "Sentinel-2":
            if catalog_status != "ok":
                output_dataframe.at[
                    row_index,
                    "quality_status",
                ] = "rejected"

                output_dataframe.at[
                    row_index,
                    "quality_score",
                ] = 0.0

                output_dataframe.at[
                    row_index,
                    "quality_reason",
                ] = (
                    "sentinel_catalog_not_ok"
                )

                continue

            sentinel_valid_pixel_ratio = pd.to_numeric(
                row["aoi_valid_pixel_ratio"],
                errors="coerce",
            )

            if (
                pd.isna(sentinel_valid_pixel_ratio)
                or float(sentinel_valid_pixel_ratio)
                < sentinel_min_aoi_valid_pixel_ratio
            ):
                output_dataframe.at[
                    row_index,
                    "quality_status",
                ] = "missing"

                output_dataframe.at[
                    row_index,
                    "quality_score",
                ] = 0.0

                output_dataframe.at[
                    row_index,
                    "quality_reason",
                ] = (
                    "sentinel_aoi_valid_pixel_ratio_"
                    "below_minimum"
                )

                continue

            polygon_status = value_to_text(
                row["polygon_status"]
            )

            if polygon_status not in {
                "ok",
                "no_retained_water_polygon",
            }:
                output_dataframe.at[
                    row_index,
                    "quality_status",
                ] = "rejected"

                output_dataframe.at[
                    row_index,
                    "quality_score",
                ] = 0.0

                output_dataframe.at[
                    row_index,
                    "quality_reason",
                ] = (
                    "sentinel_aoi_processing_failed"
                )

                continue

            if pd.isna(
                sentinel_area_m2
            ):
                output_dataframe.at[
                    row_index,
                    "quality_status",
                ] = "rejected"

                output_dataframe.at[
                    row_index,
                    "quality_score",
                ] = 0.0

                output_dataframe.at[
                    row_index,
                    "quality_reason",
                ] = (
                    "sentinel_aoi_area_missing"
                )

                continue

            output_dataframe.at[
                row_index,
                "aoi_water_area_m2",
            ] = float(
                sentinel_area_m2
            )

            output_dataframe.at[
                row_index,
                "quality_status",
            ] = "usable"

            output_dataframe.at[
                row_index,
                "quality_score",
            ] = 100.0

            output_dataframe.at[
                row_index,
                "quality_reason",
            ] = (
                "sentinel_aoi_processing_ok"
            )

            continue

        # ----------------------------------------------------
        # Landsat
        # ----------------------------------------------------

        if sensor_family != "Landsat":
            output_dataframe.at[
                row_index,
                "quality_status",
            ] = "rejected"

            output_dataframe.at[
                row_index,
                "quality_score",
            ] = 0.0

            output_dataframe.at[
                row_index,
                "quality_reason",
            ] = "unsupported_sensor"

            continue

        polygon_status = value_to_text(
            row["polygon_status"]
        )

        if (
            catalog_status != "ok"
            or polygon_status != "ok"
        ):
            output_dataframe.at[
                row_index,
                "quality_status",
            ] = "rejected"

            output_dataframe.at[
                row_index,
                "quality_score",
            ] = 0.0

            output_dataframe.at[
                row_index,
                "quality_reason",
            ] = (
                "landsat_catalog_or_polygon_not_ok"
            )

            continue

        landsat_area_m2 = pd.to_numeric(
            row["aoi_water_area_m2"],
            errors="coerce",
        )

        landsat_pixel_count = pd.to_numeric(
            row["aoi_water_pixel_count"],
            errors="coerce",
        )

        cloud_cover_percent = pd.to_numeric(
            row[
                "cloud_cover_scene_percent"
            ],
            errors="coerce",
        )

        aoi_water_qa_review_percent = pd.to_numeric(
            row["aoi_water_qa_review_percent"],
            errors="coerce",
        )

        if (
            pd.isna(landsat_area_m2)
            or pd.isna(landsat_pixel_count)
        ):
            output_dataframe.at[
                row_index,
                "quality_status",
            ] = "rejected"

            output_dataframe.at[
                row_index,
                "quality_score",
            ] = 0.0

            output_dataframe.at[
                row_index,
                "quality_reason",
            ] = (
                "landsat_aoi_measurement_missing"
            )

            continue

        if (
            pd.notna(sentinel_area_m2)
            and sentinel_area_m2 > 0.0
        ):
            same_day_area_ratio = (
                float(landsat_area_m2)
                / float(sentinel_area_m2)
            )

            output_dataframe.at[
                row_index,
                "same_day_area_ratio",
            ] = same_day_area_ratio

        else:
            same_day_area_ratio = np.nan

        rejection_reasons: list[str] = []
        partial_reasons: list[str] = []

        if (
            landsat_pixel_count
            <= landsat_reject_pixel_count
        ):
            rejection_reasons.append(
                "too_few_landsat_water_pixels"
            )

        elif (
            landsat_pixel_count
            < landsat_min_usable_pixel_count
        ):
            partial_reasons.append(
                "small_landsat_water_pixel_count"
            )

        if (
            landsat_use_scene_cloud_cover_for_rejection
            and pd.notna(cloud_cover_percent)
        ):
            if (
                cloud_cover_percent
                > landsat_partial_max_cloud_percent
            ):
                rejection_reasons.append(
                    "landsat_cloud_cover_too_high"
                )

            elif (
                cloud_cover_percent
                > landsat_usable_max_cloud_percent
            ):
                partial_reasons.append(
                    "landsat_cloud_cover_moderate"
                )

        elif landsat_use_scene_cloud_cover_for_rejection:
            partial_reasons.append(
                "landsat_cloud_cover_missing"
            )

        if pd.notna(
            same_day_area_ratio
        ):
            if (
                same_day_area_ratio
                < landsat_reject_min_area_ratio
                or same_day_area_ratio
                > landsat_reject_max_area_ratio
            ):
                rejection_reasons.append(
                    "same_day_area_ratio_outside_reject_range"
                )

            elif (
                same_day_area_ratio
                < landsat_usable_min_area_ratio
                or same_day_area_ratio
                > landsat_usable_max_area_ratio
            ):
                partial_reasons.append(
                    "same_day_area_ratio_outside_usable_range"
                )

        if (
            pd.notna(aoi_water_qa_review_percent)
            and float(aoi_water_qa_review_percent)
            > landsat_usable_max_aoi_water_qa_review_percent
        ):
            partial_reasons.append(
                "landsat_aoi_water_qa_review_high"
            )

        quality_score = 90.0

        if pd.notna(
            cloud_cover_percent
        ):
            quality_score -= min(
                float(
                    cloud_cover_percent
                )
                * 2.0,
                30.0,
            )

        if (
            landsat_pixel_count
            < landsat_min_usable_pixel_count
        ):
            quality_score -= 15.0

        if pd.notna(aoi_water_qa_review_percent):
            quality_score -= min(
                float(aoi_water_qa_review_percent) * 0.30,
                30.0,
            )

        if (
            pd.notna(
                same_day_area_ratio
            )
            and (
                same_day_area_ratio
                < landsat_usable_min_area_ratio
                or same_day_area_ratio
                > landsat_usable_max_area_ratio
            )
        ):
            quality_score -= 20.0

        quality_score = max(
            0.0,
            min(
                100.0,
                quality_score,
            ),
        )

        if rejection_reasons:
            output_dataframe.at[
                row_index,
                "quality_status",
            ] = "rejected"

            output_dataframe.at[
                row_index,
                "quality_score",
            ] = 0.0

            output_dataframe.at[
                row_index,
                "quality_reason",
            ] = ";".join(
                rejection_reasons
            )

        elif partial_reasons:
            output_dataframe.at[
                row_index,
                "quality_status",
            ] = "partial"

            output_dataframe.at[
                row_index,
                "quality_score",
            ] = quality_score

            output_dataframe.at[
                row_index,
                "quality_reason",
            ] = ";".join(
                partial_reasons
            )

        else:
            output_dataframe.at[
                row_index,
                "quality_status",
            ] = "usable"

            output_dataframe.at[
                row_index,
                "quality_score",
            ] = quality_score

            output_dataframe.at[
                row_index,
                "quality_reason",
            ] = (
                "landsat_quality_conditions_met"
            )

    output_dataframe = (
        mark_interior_zero_observations_as_missing(
            quality_dataframe=(
                output_dataframe
            ),
            max_neighbor_gap_days=(
                sentinel_interior_zero_max_neighbor_gap_days
            ),
        )
    )

    # --------------------------------------------------------
    # 日付ごとに時系列へ採用する観測を1件選ぶ
    # --------------------------------------------------------

    sensor_priority = {
        "Sentinel-2": 0,
        "Landsat": 1,
    }

    grouped_indices = (
        output_dataframe.groupby(
            "date"
        ).groups
    )

    for (
        observation_date,
        date_indices,
    ) in grouped_indices.items():

        date_indices = list(
            date_indices
        )

        usable_indices = [
            row_index
            for row_index in date_indices
            if (
                output_dataframe.at[
                    row_index,
                    "quality_status",
                ]
                == "usable"
            )
        ]

        selected_index = None

        if usable_indices:
            selected_index = min(
                usable_indices,
                key=lambda row_index: (
                    sensor_priority.get(
                        output_dataframe.at[
                            row_index,
                            "sensor_family",
                        ],
                        99,
                    ),
                    -float(
                        output_dataframe.at[
                            row_index,
                            "quality_score",
                        ]
                    ),
                ),
            )

        elif select_partial_when_no_usable:
            partial_indices = [
                row_index
                for row_index in date_indices
                if (
                    output_dataframe.at[
                        row_index,
                        "quality_status",
                    ]
                    == "partial"
                )
            ]

            if partial_indices:
                selected_index = max(
                    partial_indices,
                    key=lambda row_index: float(
                        output_dataframe.at[
                            row_index,
                            "quality_score",
                        ]
                    ),
                )

        missing_indices = [
            row_index
            for row_index in date_indices
            if (
                output_dataframe.at[
                    row_index,
                    "quality_status",
                ]
                == "missing"
            )
        ]

        if (
            selected_index is None
            and missing_indices
        ):
            selected_index = min(
                missing_indices,
                key=lambda row_index: (
                    sensor_priority.get(
                        output_dataframe.at[
                            row_index,
                            "sensor_family",
                        ],
                        99,
                    )
                ),
            )

        for row_index in date_indices:
            quality_status = (
                output_dataframe.at[
                    row_index,
                    "quality_status",
                ]
            )

            if row_index == selected_index:
                output_dataframe.at[
                    row_index,
                    "selected_for_timeseries",
                ] = True

                if quality_status == "usable":
                    selection_reason = (
                        "preferred_usable_observation"
                    )

                    timeseries_area_m2 = pd.to_numeric(
                        output_dataframe.at[
                            row_index,
                            "aoi_water_area_m2",
                        ],
                        errors="coerce",
                    )

                elif quality_status == "partial":
                    selection_reason = (
                        "partial_selected_no_usable_observation"
                    )

                    timeseries_area_m2 = pd.to_numeric(
                        output_dataframe.at[
                            row_index,
                            "aoi_water_area_m2",
                        ],
                        errors="coerce",
                    )

                else:
                    selection_reason = (
                        "missing_observation_placeholder"
                    )

                    timeseries_area_m2 = np.nan

                output_dataframe.at[
                    row_index,
                    "selection_reason",
                ] = selection_reason

                output_dataframe.at[
                    row_index,
                    "timeseries_area_m2",
                ] = timeseries_area_m2

            elif quality_status == "usable":
                output_dataframe.at[
                    row_index,
                    "selection_reason",
                ] = (
                    "lower_priority_same_date"
                )

            elif quality_status == "partial":
                output_dataframe.at[
                    row_index,
                    "selection_reason",
                ] = (
                    "manual_review_required"
                )

            elif quality_status == "missing":
                output_dataframe.at[
                    row_index,
                    "selection_reason",
                ] = (
                    "missing_observation_not_selected"
                )

            else:
                output_dataframe.at[
                    row_index,
                    "selection_reason",
                ] = "quality_rejected"

    return output_dataframe

