from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def build_common_tracking_polygon_layer(
    selected_timeseries_dataframe: pd.DataFrame,
    aoi_path: Path,
    output_gpkg: Path,
    output_layer: str,
    target_crs: str,
) -> gpd.GeoDataFrame:

    required_columns = {
        "date",
        "sensor_family",
        "platform",
        "observation_id",
        "scene",
        "product_id",
        "native_pixel_size_m",
        "ndwi_threshold",
        "quality_status",
        "quality_score",
        "is_missing",
        "timeseries_area_m2",
        "polygon_file",
    }

    missing_columns = (
        required_columns
        - set(
            selected_timeseries_dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "共通追跡ポリゴンの作成に"
            "必要な列がありません: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if not aoi_path.exists():
        raise FileNotFoundError(
            "AOIファイルが見つかりません: "
            f"{aoi_path}"
        )

    if not output_layer.strip():
        raise ValueError(
            "output_layerが空です。"
        )

    aoi_gdf = gpd.read_file(
        aoi_path
    )

    if aoi_gdf.empty:
        raise ValueError(
            "AOIに地物がありません: "
            f"{aoi_path}"
        )

    if aoi_gdf.crs is None:
        raise ValueError(
            "AOIにCRSが設定されていません: "
            f"{aoi_path}"
        )

    aoi_gdf = (
        aoi_gdf[
            [
                "geometry",
            ]
        ]
        .to_crs(
            target_crs
        )
    )

    aoi_gdf = (
        aoi_gdf[
            aoi_gdf.geometry.notna()
            & ~aoi_gdf.geometry.is_empty
        ]
        .copy()
    )

    if aoi_gdf.empty:
        raise ValueError(
            "有効なAOIジオメトリがありません。"
        )

    aoi_gdf = (
        aoi_gdf
        .dissolve()
        .reset_index(drop=True)
    )

    working_dataframe = (
        selected_timeseries_dataframe
        .copy()
    )

    working_dataframe[
        "_sort_date"
    ] = pd.to_datetime(
        working_dataframe["date"],
        errors="coerce",
    )

    invalid_date_mask = (
        working_dataframe[
            "_sort_date"
        ]
        .isna()
    )

    if invalid_date_mask.any():
        invalid_rows = (
            working_dataframe.loc[
                invalid_date_mask,
                "observation_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "共通ポリゴンを作成できない"
            "日付があります: "
            + ", ".join(
                invalid_rows
            )
        )

    working_dataframe = (
        working_dataframe
        .sort_values(
            "_sort_date"
        )
        .reset_index(drop=True)
    )

    output_records: list[dict] = []

    for (
        observation_number,
        observation_row,
    ) in working_dataframe.iterrows():

        observation_date = (
            observation_row[
                "_sort_date"
            ]
        )

        observation_id = str(
            observation_row[
                "observation_id"
            ]
        )

        sensor_family = str(
            observation_row[
                "sensor_family"
            ]
        )

        is_missing_value = (
            observation_row[
                "is_missing"
            ]
        )

        if isinstance(
            is_missing_value,
            (
                bool,
                np.bool_,
            ),
        ):
            is_missing = bool(
                is_missing_value
            )

        else:
            is_missing = (
                str(
                    is_missing_value
                )
                .strip()
                .lower()
                in {
                    "true",
                    "1",
                    "yes",
                }
            )

        if is_missing:
            continue

        expected_timeseries_area_m2 = (
            pd.to_numeric(
                observation_row[
                    "timeseries_area_m2"
                ],
                errors="coerce",
            )
        )

        polygon_file_text = str(
            observation_row[
                "polygon_file"
            ]
        ).strip()

        if (
            not polygon_file_text
            or polygon_file_text.lower()
            in {
                "nan",
                "none",
            }
        ):
            if (
                pd.notna(
                    expected_timeseries_area_m2
                )
                and float(
                    expected_timeseries_area_m2
                )
                == 0.0
            ):
                continue

            raise FileNotFoundError(
                "採用観測にpolygon_fileが"
                "設定されていません: "
                f"{observation_id}"
            )

        polygon_path = Path(
            polygon_file_text
        )

        if not polygon_path.exists():
            raise FileNotFoundError(
                "採用観測のポリゴンファイルが"
                "見つかりません: "
                f"{polygon_path}"
            )

        source_gdf = gpd.read_file(
            polygon_path
        )

        if source_gdf.empty:
            if (
                pd.notna(
                    expected_timeseries_area_m2
                )
                and float(
                    expected_timeseries_area_m2
                )
                == 0.0
            ):
                continue

            raise ValueError(
                "正面積の採用観測ですが、"
                "ポリゴンファイルが空です: "
                f"{polygon_path}"
            )

        if source_gdf.crs is None:
            raise ValueError(
                "ポリゴンファイルにCRSが"
                "設定されていません: "
                f"{polygon_path}"
            )

        source_gdf = (
            source_gdf
            .to_crs(
                target_crs
            )
            .reset_index(drop=True)
        )

        source_id_column = None

        for candidate_column in (
            "poly_id",
            "polygon_id",
            "id",
        ):
            if (
                candidate_column
                in source_gdf.columns
            ):
                source_id_column = (
                    candidate_column
                )
                break

        if source_id_column is None:
            source_gdf[
                "_source_polygon_id"
            ] = np.arange(
                1,
                len(source_gdf) + 1,
            )

        else:
            source_gdf[
                "_source_polygon_id"
            ] = (
                source_gdf[
                    source_id_column
                ]
            )

        if "area_m2" in source_gdf.columns:
            source_gdf[
                "_source_area_m2"
            ] = pd.to_numeric(
                source_gdf[
                    "area_m2"
                ],
                errors="coerce",
            )

        else:
            source_gdf[
                "_source_area_m2"
            ] = (
                source_gdf.geometry.area
            )

        if "water_px" in source_gdf.columns:
            source_gdf[
                "_source_pixel_count"
            ] = pd.to_numeric(
                source_gdf[
                    "water_px"
                ],
                errors="coerce",
            )

        elif "pixel_cnt" in source_gdf.columns:
            source_gdf[
                "_source_pixel_count"
            ] = pd.to_numeric(
                source_gdf[
                    "pixel_cnt"
                ],
                errors="coerce",
            )

        else:
            source_gdf[
                "_source_pixel_count"
            ] = np.nan

        if "shd_px" in source_gdf.columns:
            source_gdf[
                "_shadow_pixel_count"
            ] = pd.to_numeric(
                source_gdf[
                    "shd_px"
                ],
                errors="coerce",
            )

        else:
            source_gdf[
                "_shadow_pixel_count"
            ] = np.nan

        if "clear_px" in source_gdf.columns:
            source_gdf[
                "_clear_pixel_count"
            ] = pd.to_numeric(
                source_gdf[
                    "clear_px"
                ],
                errors="coerce",
            )

        else:
            source_gdf[
                "_clear_pixel_count"
            ] = np.nan

        if "shd_ratio" in source_gdf.columns:
            source_gdf[
                "_shadow_ratio"
            ] = pd.to_numeric(
                source_gdf[
                    "shd_ratio"
                ],
                errors="coerce",
            )

        else:
            source_gdf[
                "_shadow_ratio"
            ] = np.nan

        if "shd_only" in source_gdf.columns:
            source_gdf[
                "_shadow_only"
            ] = (
                source_gdf[
                    "shd_only"
                ]
                .fillna(False)
                .astype(bool)
            )

        else:
            source_gdf[
                "_shadow_only"
            ] = False

        source_columns = [
            "_source_polygon_id",
            "_source_area_m2",
            "_source_pixel_count",
            "_shadow_pixel_count",
            "_clear_pixel_count",
            "_shadow_ratio",
            "_shadow_only",
            "geometry",
        ]

        source_gdf = (
            source_gdf[
                source_columns
            ]
            .copy()
        )

        source_gdf = (
            source_gdf[
                source_gdf.geometry.notna()
                & ~source_gdf.geometry.is_empty
            ]
            .copy()
        )

        if source_gdf.empty:
            continue

        clipped_gdf = gpd.overlay(
            source_gdf,
            aoi_gdf[
                [
                    "geometry",
                ]
            ],
            how="intersection",
            keep_geom_type=True,
            make_valid=True,
        )

        if clipped_gdf.empty:
            if (
                pd.notna(
                    expected_timeseries_area_m2
                )
                and float(
                    expected_timeseries_area_m2
                )
                > 0.0
            ):
                raise ValueError(
                    "正面積の採用観測ですが、"
                    "AOI内ポリゴンがありません: "
                    f"{observation_id}"
                )

            continue

        clipped_gdf = (
            clipped_gdf
            .explode(
                index_parts=False
            )
            .reset_index(drop=True)
        )

        clipped_gdf = (
            clipped_gdf[
                clipped_gdf.geometry.notna()
                & ~clipped_gdf.geometry.is_empty
                & clipped_gdf.geom_type.isin(
                    [
                        "Polygon",
                        "MultiPolygon",
                    ]
                )
            ]
            .copy()
        )

        if clipped_gdf.empty:
            continue

        native_pixel_size_m = (
            pd.to_numeric(
                observation_row[
                    "native_pixel_size_m"
                ],
                errors="coerce",
            )
        )

        if (
            pd.isna(
                native_pixel_size_m
            )
            or float(
                native_pixel_size_m
            )
            <= 0.0
        ):
            raise ValueError(
                "ピクセルサイズが不正です: "
                f"{observation_id}"
            )

        native_pixel_size_m = float(
            native_pixel_size_m
        )

        pixel_area_m2 = (
            native_pixel_size_m
            * native_pixel_size_m
        )

        scene_polygon_number = 1

        for (
            source_part_number,
            polygon_row,
        ) in clipped_gdf.iterrows():

            polygon_geometry = (
                polygon_row.geometry
            )

            polygon_area_m2 = float(
                polygon_geometry.area
            )

            if polygon_area_m2 <= 0.0:
                continue

            common_polygon_id = (
                f"{observation_id}_"
                f"P{scene_polygon_number:04d}"
            )

            output_records.append(
                {
                    "common_polygon_id": (
                        common_polygon_id
                    ),
                    "observation_id": (
                        observation_id
                    ),
                    "date": (
                        observation_date
                        .strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    "sensor_family": (
                        sensor_family
                    ),
                    "platform": str(
                        observation_row[
                            "platform"
                        ]
                    ),
                    "scene": str(
                        observation_row[
                            "scene"
                        ]
                    ),
                    "product_id": str(
                        observation_row[
                            "product_id"
                        ]
                    ),
                    "quality_status": str(
                        observation_row[
                            "quality_status"
                        ]
                    ),
                    "quality_score": float(
                        pd.to_numeric(
                            observation_row[
                                "quality_score"
                            ],
                            errors="coerce",
                        )
                    ),
                    "source_polygon_id": str(
                        polygon_row[
                            "_source_polygon_id"
                        ]
                    ),
                    "source_part_number": int(
                        source_part_number + 1
                    ),
                    "native_pixel_size_m": (
                        native_pixel_size_m
                    ),
                    "native_pixel_area_m2": (
                        pixel_area_m2
                    ),
                    "ndwi_threshold": float(
                        pd.to_numeric(
                            observation_row[
                                "ndwi_threshold"
                            ],
                            errors="coerce",
                        )
                    ),
                    "source_area_m2": float(
                        polygon_row[
                            "_source_area_m2"
                        ]
                    ),
                    "common_area_m2": (
                        polygon_area_m2
                    ),
                    "common_area_km2": (
                        polygon_area_m2
                        / 1_000_000.0
                    ),
                    "estimated_pixel_count": (
                        polygon_area_m2
                        / pixel_area_m2
                    ),
                    "source_pixel_count": (
                        polygon_row[
                            "_source_pixel_count"
                        ]
                    ),
                    "shadow_pixel_count": (
                        polygon_row[
                            "_shadow_pixel_count"
                        ]
                    ),
                    "clear_pixel_count": (
                        polygon_row[
                            "_clear_pixel_count"
                        ]
                    ),
                    "shadow_ratio": (
                        polygon_row[
                            "_shadow_ratio"
                        ]
                    ),
                    "shadow_only": bool(
                        polygon_row[
                            "_shadow_only"
                        ]
                    ),
                    "selected_timeseries_area_m2": (
                        expected_timeseries_area_m2
                    ),
                    "source_polygon_file": str(
                        polygon_path.resolve()
                    ),
                    "geometry": (
                        polygon_geometry
                    ),
                }
            )

            scene_polygon_number += 1

    if not output_records:
        raise ValueError(
            "共通追跡GeoPackageへ保存できる"
            "水域ポリゴンがありません。"
        )

    output_gdf = gpd.GeoDataFrame(
        output_records,
        geometry="geometry",
        crs=target_crs,
    )

    output_gdf = (
        output_gdf
        .sort_values(
            [
                "date",
                "common_polygon_id",
            ]
        )
        .reset_index(drop=True)
    )

    if (
        output_gdf[
            "common_polygon_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "common_polygon_idが重複しています。"
        )

    invalid_geometry_mask = (
        output_gdf.geometry.isna()
        | output_gdf.geometry.is_empty
        | ~output_gdf.geometry.is_valid
    )

    if invalid_geometry_mask.any():
        raise ValueError(
            "出力前の共通ポリゴンに"
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
            "共通追跡GeoPackageを保存できません。"
            "QGISで開いている場合は閉じてください: "
            f"{output_gpkg}"
        ) from error

    return output_gdf

