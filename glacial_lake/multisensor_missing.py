from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def add_multisensor_bracketed_missing_records(
    tracked_lakes_gdf: gpd.GeoDataFrame,
    selected_timeseries_dataframe: pd.DataFrame,
    max_gap_days: int,
) -> gpd.GeoDataFrame:

    required_tracking_columns = {
        "local_candidate_id",
        "date",
        "persistent_lake_id",
        "area_m2",
        "area_km2",
        "geometry",
    }
    required_calendar_columns = {
        "observation_id",
        "date",
        "sensor_family",
        "platform",
    }
    missing_tracking_columns = (
        required_tracking_columns
        - set(tracked_lakes_gdf.columns)
    )
    missing_calendar_columns = (
        required_calendar_columns
        - set(selected_timeseries_dataframe.columns)
    )

    if missing_tracking_columns:
        raise ValueError(
            "欠損行作成に必要な追跡列がありません: "
            + ", ".join(sorted(missing_tracking_columns))
        )

    if missing_calendar_columns:
        raise ValueError(
            "欠損行作成に必要な観測日列がありません: "
            + ", ".join(sorted(missing_calendar_columns))
        )

    if max_gap_days <= 0:
        raise ValueError(
            "max_gap_daysは1以上にしてください。"
        )

    output_gdf = tracked_lakes_gdf.copy()

    if "observation_status" in output_gdf.columns:
        output_gdf = output_gdf[
            output_gdf["observation_status"].astype(str)
            != "missing_between_observations"
        ].copy()

    output_gdf = output_gdf.reset_index(drop=True)
    output_gdf["_parsed_date"] = (
        pd.to_datetime(
            output_gdf["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if output_gdf["_parsed_date"].isna().any():
        raise ValueError(
            "欠損行作成で日付を解釈できない"
            "追跡行があります。"
        )

    for column_name, default_value in {
        "observation_status": "",
        "is_direct_observation": False,
        "previous_observation_date": "",
        "next_observation_date": "",
        "days_from_previous_observation": np.nan,
        "days_to_next_observation": np.nan,
        "missing_reason": "",
    }.items():
        if column_name not in output_gdf.columns:
            output_gdf[column_name] = default_value

    persistent_ids = (
        output_gdf["persistent_lake_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    area_values = pd.to_numeric(
        output_gdf["area_m2"],
        errors="coerce",
    )
    direct_mask = (
        persistent_ids.ne("")
        & output_gdf.geometry.notna()
        & ~output_gdf.geometry.is_empty
        & area_values.notna()
    )
    output_gdf.loc[
        direct_mask,
        "observation_status",
    ] = "direct"
    output_gdf.loc[
        direct_mask,
        "is_direct_observation",
    ] = True
    output_gdf.loc[
        direct_mask,
        "missing_reason",
    ] = ""

    temporal_fill_mask = (
        direct_mask
        & output_gdf.get(
            "temporal_fill_applied",
            pd.Series(False, index=output_gdf.index),
        ).fillna(False).astype(bool)
    )
    output_gdf.loc[
        temporal_fill_mask,
        "observation_status",
    ] = "direct_with_partial_fill"

    scene_calendar = selected_timeseries_dataframe[
        [
            "observation_id",
            "date",
            "sensor_family",
            "platform",
        ]
    ].copy()
    scene_calendar["date"] = (
        pd.to_datetime(
            scene_calendar["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if scene_calendar["date"].isna().any():
        raise ValueError(
            "欠損行作成で日付を解釈できない"
            "選択観測があります。"
        )

    scene_calendar = (
        scene_calendar
        .sort_values(
            ["date", "observation_id"]
        )
        .drop_duplicates(
            subset="date",
            keep="first",
        )
        .reset_index(drop=True)
    )
    direct_rows = output_gdf[direct_mask].copy()

    def id_is_related_on_date(
        persistent_lake_id: str,
        target_date: pd.Timestamp,
    ) -> bool:
        rows_on_date = direct_rows[
            direct_rows["_parsed_date"].eq(target_date)
        ]

        for relation_column in [
            "related_previous_lake_ids",
            "matched_previous_lake_id",
            "merged_into_id_before_restore",
        ]:
            if relation_column not in rows_on_date.columns:
                continue

            for relation_value in rows_on_date[
                relation_column
            ].dropna():
                relation_ids = {
                    relation_id.strip()
                    for relation_id in str(
                        relation_value
                    ).split(";")
                    if relation_id.strip()
                }

                if persistent_lake_id in relation_ids:
                    return True

        return False

    missing_records = []

    for persistent_lake_id in sorted(
        direct_rows["persistent_lake_id"]
        .astype(str)
        .unique()
    ):
        lake_rows = (
            direct_rows[
                direct_rows["persistent_lake_id"]
                .astype(str)
                .eq(persistent_lake_id)
            ]
            .sort_values("_parsed_date")
            .drop_duplicates(
                subset="_parsed_date",
                keep="first",
            )
            .reset_index(drop=True)
        )

        if len(lake_rows) < 2:
            continue

        direct_dates = set(lake_rows["_parsed_date"])

        for observation_index in range(
            len(lake_rows) - 1
        ):
            previous_date = lake_rows.iloc[
                observation_index
            ]["_parsed_date"]
            next_date = lake_rows.iloc[
                observation_index + 1
            ]["_parsed_date"]
            observation_gap_days = int(
                (next_date - previous_date).days
            )

            if (
                observation_gap_days <= 1
                or observation_gap_days > max_gap_days
            ):
                continue

            intermediate_observations = scene_calendar[
                scene_calendar["date"].gt(previous_date)
                & scene_calendar["date"].lt(next_date)
            ]

            for _, observation_row in (
                intermediate_observations.iterrows()
            ):
                target_date = observation_row["date"]

                if target_date in direct_dates:
                    continue

                if id_is_related_on_date(
                    persistent_lake_id,
                    target_date,
                ):
                    continue

                days_from_previous = int(
                    (target_date - previous_date).days
                )
                days_to_next = int(
                    (next_date - target_date).days
                )
                missing_record = {
                    column_name: pd.NA
                    for column_name in output_gdf.columns
                    if column_name != "geometry"
                }
                missing_record.update(
                    {
                        "local_candidate_id": (
                            "MISSING_"
                            f"{persistent_lake_id}_"
                            f"{target_date:%Y%m%d}"
                        ),
                        "observation_id": observation_row[
                            "observation_id"
                        ],
                        "date": target_date,
                        "_parsed_date": target_date,
                        "sensor_family": observation_row[
                            "sensor_family"
                        ],
                        "platform": observation_row[
                            "platform"
                        ],
                        "persistent_lake_id": (
                            persistent_lake_id
                        ),
                        "persistent_counted_in_lake_count": False,
                        "tracking_status": "missing_observation",
                        "tracking_event": "missing",
                        "area_m2": np.nan,
                        "area_km2": np.nan,
                        "corrected_area_m2": np.nan,
                        "corrected_area_km2": np.nan,
                        "geometry": None,
                        "observation_status": (
                            "missing_between_observations"
                        ),
                        "is_direct_observation": False,
                        "previous_observation_date": (
                            previous_date.strftime("%Y-%m-%d")
                        ),
                        "next_observation_date": (
                            next_date.strftime("%Y-%m-%d")
                        ),
                        "days_from_previous_observation": (
                            days_from_previous
                        ),
                        "days_to_next_observation": (
                            days_to_next
                        ),
                        "missing_reason": (
                            "same_persistent_id_detected_"
                            "before_and_after"
                        ),
                    }
                )
                missing_records.append(missing_record)

    output_gdf = output_gdf.drop(
        columns=["_parsed_date"]
    )

    if missing_records:
        missing_gdf = gpd.GeoDataFrame(
            missing_records,
            geometry="geometry",
            crs=output_gdf.crs,
        ).drop(columns=["_parsed_date"])
        output_gdf = gpd.GeoDataFrame(
            pd.concat(
                [output_gdf, missing_gdf],
                ignore_index=True,
                sort=False,
            ),
            geometry="geometry",
            crs=output_gdf.crs,
        )

    output_gdf["date"] = (
        pd.to_datetime(
            output_gdf["date"],
            errors="raise",
        ).dt.strftime("%Y-%m-%d")
    )

    return (
        output_gdf
        .sort_values(
            [
                "date",
                "persistent_lake_id",
                "is_direct_observation",
                "local_candidate_id",
            ],
            ascending=[True, True, False, True],
        )
        .reset_index(drop=True)
    )

