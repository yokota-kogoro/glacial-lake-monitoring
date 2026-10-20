from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def prepare_multisensor_output_directories(
    output_directories: list[Path],
    intermediate_gpkg: Path,
) -> None:

    for output_directory in output_directories:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # 中間GeoPackageは毎回作り直す。
    # コードから削除された古いレイヤーが
    # 次回実行後も残ることを防ぐためである。
    if intermediate_gpkg.exists():
        try:
            intermediate_gpkg.unlink()

        except PermissionError as error:
            raise PermissionError(
                "中間GeoPackageを初期化できません。"
                "QGISで開いている場合は閉じてください: "
                f"{intermediate_gpkg}"
            ) from error


def save_multisensor_final_outputs(
    final_lakes_gdf: gpd.GeoDataFrame,
    selected_timeseries_dataframe: pd.DataFrame,
    output_gpkg: Path,
    output_layer: str,
    observations_csv: Path,
    timeseries_csv: Path,
    total_area_graph: Path,
    id_area_graph: Path,
    count_graph: Path,
    change_rate_graph: Path,
    long_gap_days: float,
    temporal_display_hour: int,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:

    if final_lakes_gdf.empty:
        raise ValueError(
            "最終出力対象の湖追跡結果が空です。"
        )

    if "persistent_lake_id" not in final_lakes_gdf.columns:
        raise ValueError(
            "最終出力対象にpersistent_lake_id列がありません。"
        )

    # 未割当候補は中間GeoPackageに監査用として残す。
    # 最終GeoPackageには永続IDがある湖と欠損レコードだけを保存し、
    # QGIS上で未割当の雲影候補が湖に見えることを防ぐ。
    final_persistent_ids = (
        final_lakes_gdf["persistent_lake_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    output_gdf = (
        final_lakes_gdf.loc[
            final_persistent_ids.ne("")
        ]
        .copy()
        .reset_index(drop=True)
    )

    if output_gdf.empty:
        raise ValueError(
            "最終出力に保存できる永続ID付き湖がありません。"
        )

    output_gdf["date"] = (
        pd.to_datetime(
            output_gdf["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if output_gdf["date"].isna().any():
        raise ValueError(
            "最終出力で日付を解釈できない"
            "湖追跡行があります。"
        )

    temporal_datetime = (
        output_gdf["date"]
        + pd.to_timedelta(
            temporal_display_hour,
            unit="h",
        )
    )

    if "temporal_datetime" in output_gdf.columns:
        output_gdf = output_gdf.drop(
            columns=["temporal_datetime"]
        )

    output_gdf.insert(
        output_gdf.columns.get_loc("date") + 1,
        "temporal_datetime",
        temporal_datetime,
    )

    calendar_columns = [
        "observation_id",
        "date",
        "sensor_family",
        "platform",
        "quality_status",
        "quality_score",
        "selection_reason",
        "timeseries_area_m2",
    ]
    missing_calendar_columns = set(calendar_columns) - set(
        selected_timeseries_dataframe.columns
    )

    if missing_calendar_columns:
        raise ValueError(
            "最終時系列作成に必要な観測列がありません: "
            + ", ".join(sorted(missing_calendar_columns))
        )

    calendar_dataframe = (
        selected_timeseries_dataframe[
            calendar_columns
        ]
        .copy()
    )
    calendar_dataframe["date"] = (
        pd.to_datetime(
            calendar_dataframe["date"],
            errors="coerce",
        )
        .dt.normalize()
    )
    calendar_dataframe = (
        calendar_dataframe
        .sort_values(["date", "observation_id"])
        .drop_duplicates("date", keep="first")
        .reset_index(drop=True)
    )

    persistent_ids = (
        output_gdf["persistent_lake_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    observation_status = output_gdf[
        "observation_status"
    ].fillna("").astype(str)
    area_values = pd.to_numeric(
        output_gdf["area_m2"],
        errors="coerce",
    )
    direct_mask = (
        persistent_ids.ne("")
        & observation_status.ne(
            "missing_between_observations"
        )
        & output_gdf.geometry.notna()
        & ~output_gdf.geometry.is_empty
        & area_values.notna()
    )
    missing_mask = observation_status.eq(
        "missing_between_observations"
    )

    direct_rows = output_gdf[direct_mask].copy()
    missing_rows = output_gdf[missing_mask].copy()
    direct_rows["area_m2"] = pd.to_numeric(
        direct_rows["area_m2"],
        errors="coerce",
    )

    direct_summary = (
        direct_rows.groupby("date", as_index=False)
        .agg(
            tracked_water_area_m2=("area_m2", "sum"),
            observed_persistent_lake_count=(
                "persistent_lake_id",
                "nunique",
            ),
            observed_lake_record_count=(
                "persistent_lake_id",
                "size",
            ),
        )
    )
    missing_summary = (
        missing_rows.groupby("date", as_index=False)
        .agg(
            missing_persistent_lake_count=(
                "persistent_lake_id",
                "nunique",
            )
        )
    )

    fill_columns = {
        "temporally_filled_area_m2": (
            "temporal_fill_area_m2"
        ),
        "area_only_added_area_m2": (
            "area_only_added_area_m2"
        ),
    }

    for source_column, output_column in (
        fill_columns.items()
    ):
        if source_column not in direct_rows.columns:
            direct_rows[source_column] = 0.0

        fill_summary = (
            direct_rows.assign(
                _fill_value=pd.to_numeric(
                    direct_rows[source_column],
                    errors="coerce",
                ).fillna(0.0)
            )
            .groupby("date", as_index=False)
            .agg(**{output_column: ("_fill_value", "sum")})
        )
        direct_summary = direct_summary.merge(
            fill_summary,
            on="date",
            how="left",
        )

    summary_dataframe = (
        calendar_dataframe
        .merge(direct_summary, on="date", how="left")
        .merge(missing_summary, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    for count_column in [
        "observed_persistent_lake_count",
        "observed_lake_record_count",
        "missing_persistent_lake_count",
    ]:
        summary_dataframe[count_column] = (
            pd.to_numeric(
                summary_dataframe[count_column],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    for area_column in [
        "tracked_water_area_m2",
        "temporal_fill_area_m2",
        "area_only_added_area_m2",
    ]:
        summary_dataframe[area_column] = (
            pd.to_numeric(
                summary_dataframe[area_column],
                errors="coerce",
            ).fillna(0.0)
        )

    summary_dataframe["tracked_water_area_km2"] = (
        summary_dataframe["tracked_water_area_m2"]
        / 1_000_000.0
    )
    summary_dataframe["area_observation_complete"] = (
        summary_dataframe[
            "missing_persistent_lake_count"
        ].eq(0)
        & summary_dataframe[
            "quality_status"
        ].astype(str).eq("usable")
    )
    summary_dataframe[
        "complete_tracked_water_area_m2"
    ] = summary_dataframe[
        "tracked_water_area_m2"
    ].where(
        summary_dataframe[
            "area_observation_complete"
        ]
    )
    summary_dataframe[
        "complete_tracked_water_area_km2"
    ] = summary_dataframe[
        "complete_tracked_water_area_m2"
    ] / 1_000_000.0

    previous_dates = summary_dataframe["date"].shift(1)
    previous_areas_m2 = summary_dataframe[
        "complete_tracked_water_area_m2"
    ].shift(1)
    observation_interval_days = (
        summary_dataframe["date"] - previous_dates
    ).dt.total_seconds() / 86400.0
    change_valid = (
        summary_dataframe[
            "complete_tracked_water_area_m2"
        ].notna()
        & previous_areas_m2.notna()
        & observation_interval_days.gt(0.0)
    )
    area_change_m2 = (
        summary_dataframe[
            "complete_tracked_water_area_m2"
        ]
        - previous_areas_m2
    ).where(change_valid)
    summary_dataframe[
        "previous_complete_area_m2"
    ] = previous_areas_m2
    summary_dataframe[
        "observation_interval_days"
    ] = observation_interval_days
    summary_dataframe["long_gap"] = (
        observation_interval_days
        .gt(long_gap_days)
        .fillna(False)
    )
    summary_dataframe[
        "area_change_comparison_valid"
    ] = change_valid
    summary_dataframe["area_change_m2"] = (
        area_change_m2
    )
    summary_dataframe["area_change_km2"] = (
        area_change_m2 / 1_000_000.0
    )
    summary_dataframe["area_change_m2_per_day"] = (
        area_change_m2 / observation_interval_days
    ).where(change_valid)
    summary_dataframe["area_change_percent"] = (
        area_change_m2 / previous_areas_m2 * 100.0
    ).where(change_valid & previous_areas_m2.gt(0.0))
    previous_sensor_family = summary_dataframe[
        "sensor_family"
    ].astype(str).shift(1)
    summary_dataframe["previous_sensor_family"] = (
        previous_sensor_family.fillna("")
    )
    summary_dataframe["sensor_changed"] = (
        previous_sensor_family.notna()
        & previous_sensor_family.ne(
            summary_dataframe["sensor_family"].astype(str)
        )
    )
    summary_dataframe["sensor_transition"] = ""
    transition_mask = previous_sensor_family.notna()
    summary_dataframe.loc[
        transition_mask,
        "sensor_transition",
    ] = (
        previous_sensor_family.loc[transition_mask]
        + "->"
        + summary_dataframe.loc[
            transition_mask,
            "sensor_family",
        ].astype(str)
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

        observation_table = pd.DataFrame(
            output_gdf.drop(columns="geometry")
        )
        observation_table["geometry_wkt"] = (
            output_gdf.geometry.apply(
                lambda geometry: (
                    geometry.wkt
                    if geometry is not None
                    and not geometry.is_empty
                    else ""
                )
            )
        )
        observation_table["date"] = (
            observation_table["date"].dt.strftime(
                "%Y-%m-%d"
            )
        )
        observation_table.to_csv(
            observations_csv,
            index=False,
            encoding="utf-8-sig",
        )

        summary_output = summary_dataframe.copy()
        summary_output["date"] = (
            summary_output["date"].dt.strftime(
                "%Y-%m-%d"
            )
        )
        summary_output.to_csv(
            timeseries_csv,
            index=False,
            encoding="utf-8-sig",
        )

    except PermissionError as error:
        raise PermissionError(
            "最終出力を保存できません。QGISまたはExcelで"
            "対象ファイルを開いている場合は閉じてください。"
        ) from error

    def load_chart_font(size: int, bold: bool = False):
        font_name = (
            "arialbd.ttf" if bold else "arial.ttf"
        )
        font_path = Path("C:/Windows/Fonts") / font_name

        try:
            return ImageFont.truetype(
                str(font_path),
                size=size,
            )

        except OSError:
            return ImageFont.load_default()

    def save_line_chart_png(
        output_path: Path,
        title: str,
        y_axis_label: str,
        chart_series: list[dict],
        integer_y_axis: bool = False,
    ) -> None:
        image_width = 2400
        image_height = 1200
        left_margin = 300
        right_margin = 100
        bottom_margin = 180

        image = Image.new(
            "RGB",
            (image_width, image_height),
            "white",
        )
        draw = ImageDraw.Draw(image)
        title_font = load_chart_font(46, bold=True)
        label_font = load_chart_font(30, bold=False)
        tick_font = load_chart_font(25, bold=False)
        legend_font = load_chart_font(27, bold=False)

        plot_left = left_margin
        plot_right = image_width - right_margin

        # 凡例はプロット領域内へ重ねず、タイトル直下の専用領域へ
        # 横並びで配置します。系列数が多い場合は自動的に改行します。
        legend_rows = []
        current_legend_row = []
        current_legend_row_width = 0
        legend_available_width = plot_right - plot_left - 40

        for series_record in chart_series:
            legend_label = str(series_record["label"])
            legend_text_box = draw.textbbox(
                (0, 0),
                legend_label,
                font=legend_font,
            )
            legend_text_width = (
                legend_text_box[2] - legend_text_box[0]
            )
            legend_item_width = 55 + 20 + legend_text_width + 55

            if (
                current_legend_row
                and current_legend_row_width + legend_item_width
                > legend_available_width
            ):
                legend_rows.append(
                    (
                        current_legend_row,
                        current_legend_row_width,
                    )
                )
                current_legend_row = []
                current_legend_row_width = 0

            current_legend_row.append(
                (
                    series_record,
                    legend_item_width,
                )
            )
            current_legend_row_width += legend_item_width

        if current_legend_row:
            legend_rows.append(
                (
                    current_legend_row,
                    current_legend_row_width,
                )
            )

        legend_box_top = 90
        legend_row_height = 48
        legend_box_bottom = (
            legend_box_top
            + 20
            + len(legend_rows) * legend_row_height
            + 20
        )
        top_margin = max(200, legend_box_bottom + 25)
        plot_top = top_margin
        plot_bottom = image_height - bottom_margin

        all_dates = []
        all_values = []

        for series_record in chart_series:
            series_dates = pd.to_datetime(
                pd.Series(series_record["dates"]),
                errors="coerce",
            )
            series_values = pd.to_numeric(
                pd.Series(series_record["values"]),
                errors="coerce",
            )

            for date_value, numeric_value in zip(
                series_dates,
                series_values,
            ):
                if pd.notna(date_value):
                    all_dates.append(pd.Timestamp(date_value))

                if pd.notna(numeric_value):
                    all_values.append(float(numeric_value))

        if not all_dates:
            raise ValueError(
                "グラフに使用できる日付がありません: "
                f"{output_path}"
            )

        minimum_date = min(all_dates)
        maximum_date = max(all_dates)
        date_span_seconds = max(
            1.0,
            float((maximum_date - minimum_date).total_seconds()),
        )
        minimum_value = min(all_values) if all_values else 0.0
        maximum_value = max(all_values) if all_values else 0.0

        if integer_y_axis:
            y_minimum = min(
                0.0,
                float(math.floor(minimum_value)),
            )
            y_maximum = max(
                1.0,
                float(math.ceil(maximum_value)),
            )

        else:
            y_minimum = min(
                0.0,
                minimum_value * 1.10,
            )
            y_maximum = max(
                1.0e-6,
                maximum_value * 1.10,
            )

        y_span = max(
            1.0e-12,
            y_maximum - y_minimum,
        )

        def date_to_x(date_value: pd.Timestamp) -> float:
            elapsed_seconds = float(
                (date_value - minimum_date).total_seconds()
            )
            return (
                plot_left
                + elapsed_seconds
                / date_span_seconds
                * (plot_right - plot_left)
            )

        def value_to_y(numeric_value: float) -> float:
            return (
                plot_bottom
                - (numeric_value - y_minimum)
                / y_span
                * (plot_bottom - plot_top)
            )

        draw.text(
            (image_width / 2, 45),
            title,
            fill="black",
            font=title_font,
            anchor="mm",
        )
        draw.text(
            (image_width / 2, image_height - 45),
            "Observation date",
            fill="black",
            font=label_font,
            anchor="mm",
        )
        draw.text(
            (25, 75),
            y_axis_label,
            fill="black",
            font=label_font,
            anchor="la",
        )

        grid_count = (
            int(y_maximum)
            if integer_y_axis
            and y_maximum <= 10.0
            else 5
        )

        for grid_index in range(grid_count + 1):
            grid_value = (
                y_minimum
                + y_span
                * grid_index
                / grid_count
            )
            grid_y = value_to_y(grid_value)
            draw.line(
                (
                    plot_left,
                    grid_y,
                    plot_right,
                    grid_y,
                ),
                fill=(220, 220, 220),
                width=2,
            )
            value_text = (
                f"{int(round(grid_value))}"
                if integer_y_axis
                else f"{grid_value:.4f}"
            )
            draw.text(
                (plot_left - 20, grid_y),
                value_text,
                fill="black",
                font=tick_font,
                anchor="rm",
            )

        unique_dates = sorted(set(all_dates))
        tick_step = max(
            1,
            int(math.ceil(len(unique_dates) / 8)),
        )
        tick_dates = unique_dates[::tick_step]

        if maximum_date not in tick_dates:
            tick_dates.append(maximum_date)

        # 右端などで日付ラベル同士が近すぎる場合は、
        # 最終日を優先して直前のラベルを間引きます。
        filtered_tick_dates = []
        minimum_tick_label_gap_px = 95.0

        for tick_date in tick_dates:
            if not filtered_tick_dates:
                filtered_tick_dates.append(tick_date)
                continue

            previous_tick_date = filtered_tick_dates[-1]
            tick_gap_px = abs(
                date_to_x(tick_date)
                - date_to_x(previous_tick_date)
            )

            if tick_gap_px >= minimum_tick_label_gap_px:
                filtered_tick_dates.append(tick_date)

            elif tick_date == maximum_date:
                filtered_tick_dates[-1] = tick_date

        tick_dates = filtered_tick_dates

        for tick_date in tick_dates:
            tick_x = date_to_x(tick_date)
            draw.line(
                (tick_x, plot_top, tick_x, plot_bottom),
                fill=(235, 235, 235),
                width=2,
            )
            draw.text(
                (tick_x, plot_bottom + 25),
                tick_date.strftime("%m-%d"),
                fill="black",
                font=tick_font,
                anchor="ma",
            )

        draw.line(
            (plot_left, plot_top, plot_left, plot_bottom),
            fill="black",
            width=4,
        )
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom),
            fill="black",
            width=4,
        )

        for series_record in chart_series:
            color = tuple(series_record["color"])
            marker = series_record.get("marker", "circle")
            connect_points = bool(
                series_record.get("connect", True)
            )
            series_dates = pd.to_datetime(
                pd.Series(series_record["dates"]),
                errors="coerce",
            )
            series_values = pd.to_numeric(
                pd.Series(series_record["values"]),
                errors="coerce",
            )
            previous_point = None

            for date_value, numeric_value in zip(
                series_dates,
                series_values,
            ):
                if pd.isna(date_value) or pd.isna(numeric_value):
                    previous_point = None
                    continue

                point_x = date_to_x(
                    pd.Timestamp(date_value)
                )
                point_y = value_to_y(float(numeric_value))
                current_point = (point_x, point_y)

                if connect_points and previous_point is not None:
                    draw.line(
                        (*previous_point, *current_point),
                        fill=color,
                        width=6,
                    )

                marker_radius = 10

                if marker == "cross":
                    draw.line(
                        (
                            point_x - marker_radius,
                            point_y - marker_radius,
                            point_x + marker_radius,
                            point_y + marker_radius,
                        ),
                        fill=color,
                        width=5,
                    )
                    draw.line(
                        (
                            point_x - marker_radius,
                            point_y + marker_radius,
                            point_x + marker_radius,
                            point_y - marker_radius,
                        ),
                        fill=color,
                        width=5,
                    )

                elif marker == "square":
                    draw.rectangle(
                        (
                            point_x - marker_radius,
                            point_y - marker_radius,
                            point_x + marker_radius,
                            point_y + marker_radius,
                        ),
                        fill=color,
                    )

                else:
                    draw.ellipse(
                        (
                            point_x - marker_radius,
                            point_y - marker_radius,
                            point_x + marker_radius,
                            point_y + marker_radius,
                        ),
                        fill=color,
                    )

                previous_point = current_point

        draw.rectangle(
            (
                plot_left,
                legend_box_top,
                plot_right,
                legend_box_bottom,
            ),
            fill="white",
            outline=(190, 190, 190),
            width=2,
        )

        for legend_row_index, (
            legend_row,
            legend_row_width,
        ) in enumerate(legend_rows):
            legend_x = (
                plot_left
                + (plot_right - plot_left - legend_row_width) / 2
            )
            item_y = (
                legend_box_top
                + 20
                + legend_row_index * legend_row_height
                + legend_row_height / 2
            )

            for series_record, legend_item_width in legend_row:
                color = tuple(series_record["color"])
                legend_marker = series_record.get(
                    "marker",
                    "circle",
                )
                legend_connect = bool(
                    series_record.get("connect", True)
                )
                legend_marker_x = legend_x + 27.5
                legend_marker_radius = 8

                if legend_connect:
                    draw.line(
                        (
                            legend_x,
                            item_y,
                            legend_x + 55,
                            item_y,
                        ),
                        fill=color,
                        width=6,
                    )

                if legend_marker == "cross":
                    draw.line(
                        (
                            legend_marker_x - legend_marker_radius,
                            item_y - legend_marker_radius,
                            legend_marker_x + legend_marker_radius,
                            item_y + legend_marker_radius,
                        ),
                        fill=color,
                        width=4,
                    )
                    draw.line(
                        (
                            legend_marker_x - legend_marker_radius,
                            item_y + legend_marker_radius,
                            legend_marker_x + legend_marker_radius,
                            item_y - legend_marker_radius,
                        ),
                        fill=color,
                        width=4,
                    )

                elif legend_marker == "square":
                    draw.rectangle(
                        (
                            legend_marker_x - legend_marker_radius,
                            item_y - legend_marker_radius,
                            legend_marker_x + legend_marker_radius,
                            item_y + legend_marker_radius,
                        ),
                        fill=color,
                    )

                else:
                    draw.ellipse(
                        (
                            legend_marker_x - legend_marker_radius,
                            item_y - legend_marker_radius,
                            legend_marker_x + legend_marker_radius,
                            item_y + legend_marker_radius,
                        ),
                        fill=color,
                    )

                draw.text(
                    (legend_x + 75, item_y),
                    str(series_record["label"]),
                    fill="black",
                    font=legend_font,
                    anchor="lm",
                )
                legend_x += legend_item_width

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        image.save(output_path, format="PNG")

    incomplete_rows = summary_dataframe[
        ~summary_dataframe[
            "area_observation_complete"
        ]
    ]
    save_line_chart_png(
        output_path=total_area_graph,
        title="Multisensor tracked water area",
        y_axis_label="Water area (km2)",
        chart_series=[
            {
                "label": "Complete tracked area",
                "dates": summary_dataframe["date"],
                "values": summary_dataframe[
                    "complete_tracked_water_area_km2"
                ],
                "color": (31, 119, 180),
                "marker": "circle",
                "connect": True,
            },
            {
                "label": "Incomplete observation",
                "dates": incomplete_rows["date"],
                "values": incomplete_rows[
                    "tracked_water_area_km2"
                ],
                "color": (214, 39, 40),
                "marker": "cross",
                "connect": False,
            },
        ],
    )

    id_table = output_gdf[
        persistent_ids.ne("")
    ][
        [
            "date",
            "persistent_lake_id",
            "area_m2",
        ]
    ].copy()
    id_table["area_km2"] = pd.to_numeric(
        id_table["area_m2"],
        errors="coerce",
    ) / 1_000_000.0
    id_colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]
    id_series = []

    for lake_index, (
        persistent_lake_id,
        lake_rows,
    ) in enumerate(
        id_table.groupby("persistent_lake_id")
    ):
        lake_rows = lake_rows.sort_values("date")
        id_series.append(
            {
                "label": persistent_lake_id,
                "dates": lake_rows["date"],
                "values": lake_rows["area_km2"],
                "color": id_colors[
                    lake_index % len(id_colors)
                ],
                "marker": "circle",
                "connect": True,
            }
        )

    save_line_chart_png(
        output_path=id_area_graph,
        title="Area change by persistent lake ID",
        y_axis_label="Lake area (km2)",
        chart_series=id_series,
    )

    save_line_chart_png(
        output_path=count_graph,
        title="Persistent lake count and missing records",
        y_axis_label="Count",
        chart_series=[
            {
                "label": "Observed persistent lakes",
                "dates": summary_dataframe["date"],
                "values": summary_dataframe[
                    "observed_persistent_lake_count"
                ],
                "color": (255, 127, 14),
                "marker": "square",
                "connect": True,
            },
            {
                "label": "Missing persistent lakes",
                "dates": summary_dataframe["date"],
                "values": summary_dataframe[
                    "missing_persistent_lake_count"
                ],
                "color": (214, 39, 40),
                "marker": "cross",
                "connect": True,
            },
        ],
        integer_y_axis=True,
    )

    area_change_rate = pd.to_numeric(
        summary_dataframe[
            "area_change_m2_per_day"
        ],
        errors="coerce",
    )
    save_line_chart_png(
        output_path=change_rate_graph,
        title="Daily tracked water-area change",
        y_axis_label="Area change (m2/day)",
        chart_series=[
            {
                "label": "Growth",
                "dates": summary_dataframe["date"],
                "values": area_change_rate.where(
                    area_change_rate.ge(0.0)
                ),
                "color": (31, 119, 180),
                "marker": "circle",
                "connect": False,
            },
            {
                "label": "Shrinkage",
                "dates": summary_dataframe["date"],
                "values": area_change_rate.where(
                    area_change_rate.lt(0.0)
                ),
                "color": (214, 39, 40),
                "marker": "circle",
                "connect": False,
            },
        ],
    )

    return output_gdf, summary_dataframe
