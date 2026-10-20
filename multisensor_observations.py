from __future__ import annotations

from glacial_lake.multisensor_catalog import (
    build_multisensor_scene_catalog as build_catalog_from_preprocessing,
    measure_catalog_aoi_water_from_masks,
)
from glacial_lake.multisensor_fragments import (
    attach_subthreshold_fragments_to_persistent_lakes,
)
from glacial_lake.multisensor_grouping import group_multisensor_polygons_by_core
from glacial_lake.multisensor_interpolation import (
    apply_multisensor_bracketed_partial_geometry_fill,
    apply_multisensor_related_family_partial_fill,
    reconcile_multisensor_connected_ids_after_fill,
)
from glacial_lake.multisensor_missing import add_multisensor_bracketed_missing_records
from glacial_lake.multisensor_outputs import (
    prepare_multisensor_output_directories,
    save_multisensor_final_outputs,
)
from glacial_lake.multisensor_polygons import build_common_tracking_polygon_layer
from glacial_lake.multisensor_quality import add_multisensor_quality_flags
from glacial_lake.multisensor_relationships import (
    add_multisensor_tracking_relationships,
)
from glacial_lake.multisensor_shadow_validation import (
    validate_shadow_candidates_temporally,
)
from glacial_lake.multisensor_tracking import (
    assign_multisensor_persistent_ids,
    backfill_multisensor_persistent_ids,
)
from glacial_lake.multisensor_timeseries import save_selected_multisensor_timeseries


# ============================================================
# 設定
# 設定値は multisensor_config.py にまとめています。
# ============================================================

from glacial_lake.multisensor_config import (
    SCRIPT_DIR,
    SENTINEL_SUMMARY_CSV,
    LANDSAT_SUMMARY_CSV,
    LANDSAT_AOI_PATH,
    MULTISENSOR_AOI_NAME,
    MULTISENSOR_OUTPUT_FOLDER_NAME,
    MULTISENSOR_TRACKING_CRS,
    SENTINEL_CORE_AREA_M2,
    SENTINEL_CONTINUATION_AREA_M2,
    SENTINEL_SMALL_TO_CORE_DISTANCE_M,
    LANDSAT_CORE_AREA_M2,
    LANDSAT_CONTINUATION_AREA_M2,
    LANDSAT_SMALL_TO_CORE_DISTANCE_M,
    MULTISENSOR_GROUPING_SETTINGS,
    LANDSAT_QA_REVIEW_MIN_RATIO,
    MULTISENSOR_SHADOW_MAX_GAP_DAYS,
    MULTISENSOR_SHADOW_REQUIRE_BOTH_SIDES,
    MULTISENSOR_SHADOW_MIN_OVERLAP_SMALLER_RATIO,
    MULTISENSOR_SHADOW_MIN_IOU,
    MULTISENSOR_SHADOW_MIN_AREA_RATIO,
    MULTISENSOR_SHADOW_MATCH_DISTANCE_BY_SENSOR,
    ENABLE_LANDSAT_TEMPORAL_GEOMETRY_CLIP,
    LANDSAT_TEMPORAL_GEOMETRY_CLIP_MIN_QA_RATIO,
    LANDSAT_TEMPORAL_GEOMETRY_CLIP_BUFFER_M,
    LANDSAT_TEMPORAL_GEOMETRY_CLIP_MAX_RETAINED_RATIO,
    LANDSAT_TEMPORAL_GEOMETRY_CLIP_MIN_AREA_M2,
    SENTINEL_MIN_AOI_VALID_PIXEL_RATIO,
    LANDSAT_USABLE_MAX_AOI_WATER_QA_REVIEW_PERCENT,
    MULTISENSOR_TRACKING_MAX_GAP_DAYS,
    MULTISENSOR_TRACKING_MIN_OVERLAP_SMALLER_RATIO,
    MULTISENSOR_TRACKING_MIN_IOU,
    MULTISENSOR_TRACKING_MIN_AREA_RATIO,
    MULTISENSOR_TRACKING_DISTANCE_BY_SENSOR,
    MULTISENSOR_TRACKING_DISTANCE_ONLY_MAX_CENTROID_MULTIPLIER,
    MULTISENSOR_LARGE_CHANGE_MIN_REFERENCE_OVERLAP_RATIO,
    MULTISENSOR_LARGE_CHANGE_MAX_AREA_RATIO,
    MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MAX_GAP_DAYS,
    MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MIN_OVERLAP_SMALLER_RATIO,
    MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MIN_AREA_RATIO,
    MULTISENSOR_LARGE_CHANGE_LONG_GAP_MIN_DAYS,
    MULTISENSOR_LARGE_CHANGE_LONG_GAP_MIN_REFERENCE_OVERLAP_RATIO,
    MULTISENSOR_LARGE_CHANGE_LONG_GAP_CONFIRM_MIN_PENDING_COVERAGE_RATIO,
    MULTISENSOR_LARGE_CHANGE_LONG_GAP_CONFIRM_MIN_AREA_RATIO,
    MULTISENSOR_RELATION_MIN_INTERSECTION_AREA_M2,
    MULTISENSOR_FILL_MAX_REFERENCE_GAP_DAYS,
    MULTISENSOR_FILL_SAME_SENSOR_MIN_REFERENCE_IOU,
    MULTISENSOR_FILL_CROSS_SENSOR_MIN_REFERENCE_IOU,
    MULTISENSOR_FILL_MIN_CURRENT_STABLE_OVERLAP_SMALLER_RATIO,
    MULTISENSOR_FILL_MIN_AREA_BY_SENSOR,
    MULTISENSOR_FILL_MAX_RATIO,
    MULTISENSOR_RELATED_FAMILY_ALLOCATION_CELL_M,
    SENTINEL_INTERIOR_ZERO_MAX_NEIGHBOR_GAP_DAYS,
    MULTISENSOR_LONG_GAP_DAYS,
    LANDSAT_USE_SCENE_CLOUD_COVER_FOR_REJECTION,
    LANDSAT_USABLE_MAX_CLOUD_PERCENT,
    LANDSAT_PARTIAL_MAX_CLOUD_PERCENT,
    LANDSAT_USABLE_MIN_AREA_RATIO,
    LANDSAT_USABLE_MAX_AREA_RATIO,
    LANDSAT_REJECT_MIN_AREA_RATIO,
    LANDSAT_REJECT_MAX_AREA_RATIO,
    LANDSAT_MIN_USABLE_PIXEL_COUNT,
    LANDSAT_REJECT_PIXEL_COUNT,
)


# ============================================================
# 内部出力設定
# 通常はここから下のファイル名・レイヤー名を変更しません。
# ============================================================

MULTISENSOR_OUTPUT_ROOT = (
    SCRIPT_DIR
    / MULTISENSOR_OUTPUT_FOLDER_NAME
)
MULTISENSOR_RESULTS_ROOT = (
    MULTISENSOR_OUTPUT_ROOT / "results"
)
MULTISENSOR_DETAILS_ROOT = (
    MULTISENSOR_OUTPUT_ROOT / "details"
)
MULTISENSOR_DIAGNOSTICS_ROOT = (
    MULTISENSOR_OUTPUT_ROOT / "diagnostics"
)
MULTISENSOR_RESULT_SUMMARY_DIR = (
    MULTISENSOR_RESULTS_ROOT / "summaries"
)
MULTISENSOR_TABLE_OUTPUT_DIR = (
    MULTISENSOR_DETAILS_ROOT / "multisensor"
)
MULTISENSOR_FIGURE_OUTPUT_DIR = (
    MULTISENSOR_RESULTS_ROOT / "figures"
)
MULTISENSOR_GPKG_OUTPUT_DIR = (
    MULTISENSOR_RESULTS_ROOT / "geopackages"
)
MULTISENSOR_INTERMEDIATE_OUTPUT_DIR = (
    MULTISENSOR_DIAGNOSTICS_ROOT / "multisensor"
)
MULTISENSOR_INTERMEDIATE_GPKG = (
    MULTISENSOR_INTERMEDIATE_OUTPUT_DIR
    / "multisensor_intermediate.gpkg"
)

CATALOG_OUTPUT_CSV = (
    MULTISENSOR_RESULT_SUMMARY_DIR / "scene_catalog.csv"
)
MULTISENSOR_TIMESERIES_OUTPUT_CSV = (
    MULTISENSOR_TABLE_OUTPUT_DIR / "selected_timeseries.csv"
)
MULTISENSOR_FINAL_OBSERVATIONS_CSV = (
    MULTISENSOR_TABLE_OUTPUT_DIR / "final_lake_observations.csv"
)
MULTISENSOR_FINAL_TIMESERIES_CSV = (
    MULTISENSOR_TABLE_OUTPUT_DIR / "final_timeseries.csv"
)

MULTISENSOR_TRACKING_INPUT_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_GROUPED_CANDIDATES_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_SHADOW_VALIDATED_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_PERSISTENT_TRACKING_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_RELATIONSHIP_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_BRACKETED_FILL_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_RELATED_FAMILY_FILL_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_TOPOLOGY_RECONCILED_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_RETRACKED_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_RETRACKED_RELATIONSHIP_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_RETROSPECTIVE_GPKG = MULTISENSOR_INTERMEDIATE_GPKG
MULTISENSOR_AREA_COMPLETED_GPKG = MULTISENSOR_INTERMEDIATE_GPKG

MULTISENSOR_TRACKING_INPUT_LAYER = "selected_water_polygons"
MULTISENSOR_GROUPED_CANDIDATES_LAYER = "lake_candidates"
MULTISENSOR_SHADOW_VALIDATED_LAYER = "validated_candidates"
MULTISENSOR_PERSISTENT_TRACKING_LAYER = "persistent_lakes"
MULTISENSOR_RELATIONSHIP_LAYER = "persistent_relationships"
MULTISENSOR_BRACKETED_FILL_LAYER = "bracketed_partial_fill"
MULTISENSOR_RELATED_FAMILY_FILL_LAYER = "related_family_partial_fill"
MULTISENSOR_TOPOLOGY_RECONCILED_LAYER = "topology_reconciled"
MULTISENSOR_RETRACKED_LAYER = "persistent_lakes_retracked"
MULTISENSOR_RETRACKED_RELATIONSHIP_LAYER = "retracked_relationships"
MULTISENSOR_RETROSPECTIVE_LAYER = "persistent_lakes_backfilled"
MULTISENSOR_AREA_COMPLETED_LAYER = "persistent_lakes_area_completed"
MULTISENSOR_AREA_FRAGMENT_AUDIT_LAYER = "area_only_fragment_audit"

MULTISENSOR_FINAL_GPKG = (
    MULTISENSOR_GPKG_OUTPUT_DIR / "multisensor_final.gpkg"
)
MULTISENSOR_FINAL_LAYER = "final_persistent_lakes"

# QGISの時系列表示で、日付の00:00がタイムゾーン変換により
# 前日扱いになることを避けるための表示用時刻です。
# 解析上の観測日はdate列に保持し、時系列コントローラでは
# temporal_datetime列を使用します。
MULTISENSOR_QGIS_TEMPORAL_DISPLAY_HOUR = 12

MULTISENSOR_FINAL_TOTAL_AREA_GRAPH = (
    MULTISENSOR_FIGURE_OUTPUT_DIR / "total_area_timeseries.png"
)
MULTISENSOR_FINAL_ID_AREA_GRAPH = (
    MULTISENSOR_FIGURE_OUTPUT_DIR / "persistent_id_timeseries.png"
)
MULTISENSOR_FINAL_COUNT_GRAPH = (
    MULTISENSOR_FIGURE_OUTPUT_DIR / "lake_count_timeseries.png"
)
MULTISENSOR_FINAL_CHANGE_RATE_GRAPH = (
    MULTISENSOR_FIGURE_OUTPUT_DIR / "area_change_rate.png"
)

































# ============================================================
# 実行
# ============================================================





def main() -> None:

    prepare_multisensor_output_directories(
        output_directories=[
            MULTISENSOR_OUTPUT_ROOT,
            MULTISENSOR_RESULTS_ROOT,
            MULTISENSOR_DETAILS_ROOT,
            MULTISENSOR_DIAGNOSTICS_ROOT,
            MULTISENSOR_RESULT_SUMMARY_DIR,
            MULTISENSOR_TABLE_OUTPUT_DIR,
            MULTISENSOR_FIGURE_OUTPUT_DIR,
            MULTISENSOR_GPKG_OUTPUT_DIR,
            MULTISENSOR_INTERMEDIATE_OUTPUT_DIR,
        ],
        intermediate_gpkg=(
            MULTISENSOR_INTERMEDIATE_GPKG
        ),
    )

    # 最初にSentinel-2とLandsatの
    # 共通シーンカタログを作成する。
    catalog_dataframe = (
        build_catalog_from_preprocessing(
            sentinel_summary_csv=(
                SENTINEL_SUMMARY_CSV
            ),
            landsat_summary_csv=(
                LANDSAT_SUMMARY_CSV
            ),
        )
    )

    # 両センサーの前処理済み水域マスクとポリゴンを
    # 同じAOIで検査し、共通列へ面積を記録する。
    catalog_dataframe = (
        measure_catalog_aoi_water_from_masks(
            catalog_dataframe=(
                catalog_dataframe
            ),
            aoi_path=(
                LANDSAT_AOI_PATH
            ),
            aoi_name=(
                MULTISENSOR_AOI_NAME
            ),
        )
    )

    catalog_dataframe = (
        add_multisensor_quality_flags(
            catalog_dataframe=(
                catalog_dataframe
            ),
            sentinel_min_aoi_valid_pixel_ratio=(
                SENTINEL_MIN_AOI_VALID_PIXEL_RATIO
            ),
            landsat_use_scene_cloud_cover_for_rejection=(
                LANDSAT_USE_SCENE_CLOUD_COVER_FOR_REJECTION
            ),
            landsat_usable_max_cloud_percent=(
                LANDSAT_USABLE_MAX_CLOUD_PERCENT
            ),
            landsat_partial_max_cloud_percent=(
                LANDSAT_PARTIAL_MAX_CLOUD_PERCENT
            ),
            landsat_usable_min_area_ratio=(
                LANDSAT_USABLE_MIN_AREA_RATIO
            ),
            landsat_usable_max_area_ratio=(
                LANDSAT_USABLE_MAX_AREA_RATIO
            ),
            landsat_reject_min_area_ratio=(
                LANDSAT_REJECT_MIN_AREA_RATIO
            ),
            landsat_reject_max_area_ratio=(
                LANDSAT_REJECT_MAX_AREA_RATIO
            ),
            landsat_min_usable_pixel_count=(
                LANDSAT_MIN_USABLE_PIXEL_COUNT
            ),
            landsat_reject_pixel_count=(
                LANDSAT_REJECT_PIXEL_COUNT
            ),
            landsat_usable_max_aoi_water_qa_review_percent=(
                LANDSAT_USABLE_MAX_AOI_WATER_QA_REVIEW_PERCENT
            ),
            # QA確認対象が多いLandsat観測も候補追跡には残すが、
            # 最終面積は不完全観測として扱う。
            select_partial_when_no_usable=True,
            sentinel_interior_zero_max_neighbor_gap_days=(
                SENTINEL_INTERIOR_ZERO_MAX_NEIGHBOR_GAP_DAYS
            ),
        )
    )

    selected_timeseries_dataframe = (
        save_selected_multisensor_timeseries(
            catalog_dataframe=(
                catalog_dataframe
            ),
            output_csv=(
                MULTISENSOR_TIMESERIES_OUTPUT_CSV
            ),
            long_gap_days=(
                MULTISENSOR_LONG_GAP_DAYS
            ),
        )
    )

    tracking_input_gdf = (
        build_common_tracking_polygon_layer(
            selected_timeseries_dataframe=(
                selected_timeseries_dataframe
            ),
            aoi_path=(
                LANDSAT_AOI_PATH
            ),
            output_gpkg=(
                MULTISENSOR_TRACKING_INPUT_GPKG
            ),
            output_layer=(
                MULTISENSOR_TRACKING_INPUT_LAYER
            ),
            target_crs=(
                MULTISENSOR_TRACKING_CRS
            ),
        )
    )

    grouped_candidate_gdf = (
        group_multisensor_polygons_by_core(
            tracking_input_gdf=(
                tracking_input_gdf
            ),
            grouping_settings=(
                MULTISENSOR_GROUPING_SETTINGS
            ),
            landsat_qa_review_min_ratio=(
                LANDSAT_QA_REVIEW_MIN_RATIO
            ),
            output_gpkg=(
                MULTISENSOR_GROUPED_CANDIDATES_GPKG
            ),
            output_layer=(
                MULTISENSOR_GROUPED_CANDIDATES_LAYER
            ),
        )
    )

    shadow_validated_gdf = (
        validate_shadow_candidates_temporally(
            grouped_candidate_gdf=(
                grouped_candidate_gdf
            ),
            output_gpkg=(
                MULTISENSOR_SHADOW_VALIDATED_GPKG
            ),
            output_layer=(
                MULTISENSOR_SHADOW_VALIDATED_LAYER
            ),
            max_gap_days=(
                MULTISENSOR_SHADOW_MAX_GAP_DAYS
            ),
            min_overlap_smaller_ratio=(
                MULTISENSOR_SHADOW_MIN_OVERLAP_SMALLER_RATIO
            ),
            min_iou=(
                MULTISENSOR_SHADOW_MIN_IOU
            ),
            min_area_ratio=(
                MULTISENSOR_SHADOW_MIN_AREA_RATIO
            ),
            require_both_sides=(
                MULTISENSOR_SHADOW_REQUIRE_BOTH_SIDES
            ),
            match_distance_by_sensor=(
                MULTISENSOR_SHADOW_MATCH_DISTANCE_BY_SENSOR
            ),
            enable_landsat_geometry_clip=(
                ENABLE_LANDSAT_TEMPORAL_GEOMETRY_CLIP
            ),
            landsat_geometry_clip_min_qa_ratio=(
                LANDSAT_TEMPORAL_GEOMETRY_CLIP_MIN_QA_RATIO
            ),
            landsat_geometry_clip_buffer_m=(
                LANDSAT_TEMPORAL_GEOMETRY_CLIP_BUFFER_M
            ),
            landsat_geometry_clip_max_retained_ratio=(
                LANDSAT_TEMPORAL_GEOMETRY_CLIP_MAX_RETAINED_RATIO
            ),
            landsat_geometry_clip_min_area_m2=(
                LANDSAT_TEMPORAL_GEOMETRY_CLIP_MIN_AREA_M2
            ),
        )
    )

    persistent_tracking_gdf = (
        assign_multisensor_persistent_ids(
            validated_candidate_gdf=(
                shadow_validated_gdf
            ),
            output_gpkg=(
                MULTISENSOR_PERSISTENT_TRACKING_GPKG
            ),
            output_layer=(
                MULTISENSOR_PERSISTENT_TRACKING_LAYER
            ),
            max_gap_days=(
                MULTISENSOR_TRACKING_MAX_GAP_DAYS
            ),
            min_overlap_smaller_ratio=(
                MULTISENSOR_TRACKING_MIN_OVERLAP_SMALLER_RATIO
            ),
            min_iou=(
                MULTISENSOR_TRACKING_MIN_IOU
            ),
            min_area_ratio=(
                MULTISENSOR_TRACKING_MIN_AREA_RATIO
            ),
            match_distance_by_sensor=(
                MULTISENSOR_TRACKING_DISTANCE_BY_SENSOR
            ),
            distance_only_max_centroid_multiplier=(
                MULTISENSOR_TRACKING_DISTANCE_ONLY_MAX_CENTROID_MULTIPLIER
            ),
            large_change_min_reference_overlap_ratio=(
                MULTISENSOR_LARGE_CHANGE_MIN_REFERENCE_OVERLAP_RATIO
            ),
            large_change_max_area_ratio=(
                MULTISENSOR_LARGE_CHANGE_MAX_AREA_RATIO
            ),
            large_change_confirmation_max_gap_days=(
                MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MAX_GAP_DAYS
            ),
            large_change_confirmation_min_overlap_smaller_ratio=(
                MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MIN_OVERLAP_SMALLER_RATIO
            ),
            large_change_confirmation_min_area_ratio=(
                MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MIN_AREA_RATIO
            ),
            large_change_long_gap_min_days=(
                MULTISENSOR_LARGE_CHANGE_LONG_GAP_MIN_DAYS
            ),
            large_change_long_gap_min_reference_overlap_ratio=(
                MULTISENSOR_LARGE_CHANGE_LONG_GAP_MIN_REFERENCE_OVERLAP_RATIO
            ),
            large_change_long_gap_confirm_min_pending_coverage_ratio=(
                MULTISENSOR_LARGE_CHANGE_LONG_GAP_CONFIRM_MIN_PENDING_COVERAGE_RATIO
            ),
            large_change_long_gap_confirm_min_area_ratio=(
                MULTISENSOR_LARGE_CHANGE_LONG_GAP_CONFIRM_MIN_AREA_RATIO
            ),
        )
    )

    relationship_tracking_gdf = (
        add_multisensor_tracking_relationships(
            persistent_tracking_gdf=(
                persistent_tracking_gdf
            ),
            output_gpkg=(
                MULTISENSOR_RELATIONSHIP_GPKG
            ),
            output_layer=(
                MULTISENSOR_RELATIONSHIP_LAYER
            ),
            max_gap_days=(
                MULTISENSOR_TRACKING_MAX_GAP_DAYS
            ),
            min_intersection_area_m2=(
                MULTISENSOR_RELATION_MIN_INTERSECTION_AREA_M2
            ),
            min_overlap_smaller_ratio=(
                MULTISENSOR_TRACKING_MIN_OVERLAP_SMALLER_RATIO
            ),
            min_iou=(
                MULTISENSOR_TRACKING_MIN_IOU
            ),
        )
    )

    bracketed_fill_gdf = (
        apply_multisensor_bracketed_partial_geometry_fill(
            persistent_tracking_gdf=(
                relationship_tracking_gdf
            ),
            output_gpkg=(
                MULTISENSOR_BRACKETED_FILL_GPKG
            ),
            output_layer=(
                MULTISENSOR_BRACKETED_FILL_LAYER
            ),
            max_reference_gap_days=(
                MULTISENSOR_FILL_MAX_REFERENCE_GAP_DAYS
            ),
            same_sensor_min_reference_iou=(
                MULTISENSOR_FILL_SAME_SENSOR_MIN_REFERENCE_IOU
            ),
            cross_sensor_min_reference_iou=(
                MULTISENSOR_FILL_CROSS_SENSOR_MIN_REFERENCE_IOU
            ),
            min_current_stable_overlap_smaller_ratio=(
                MULTISENSOR_FILL_MIN_CURRENT_STABLE_OVERLAP_SMALLER_RATIO
            ),
            min_fill_area_by_sensor=(
                MULTISENSOR_FILL_MIN_AREA_BY_SENSOR
            ),
            max_fill_ratio=(
                MULTISENSOR_FILL_MAX_RATIO
            ),
        )
    )

    related_family_fill_gdf = (
        apply_multisensor_related_family_partial_fill(
            tracked_lakes_gdf=(
                bracketed_fill_gdf
            ),
            output_gpkg=(
                MULTISENSOR_RELATED_FAMILY_FILL_GPKG
            ),
            output_layer=(
                MULTISENSOR_RELATED_FAMILY_FILL_LAYER
            ),
            max_reference_gap_days=(
                MULTISENSOR_FILL_MAX_REFERENCE_GAP_DAYS
            ),
            same_sensor_min_reference_iou=(
                MULTISENSOR_FILL_SAME_SENSOR_MIN_REFERENCE_IOU
            ),
            cross_sensor_min_reference_iou=(
                MULTISENSOR_FILL_CROSS_SENSOR_MIN_REFERENCE_IOU
            ),
            min_current_stable_overlap_smaller_ratio=(
                MULTISENSOR_FILL_MIN_CURRENT_STABLE_OVERLAP_SMALLER_RATIO
            ),
            min_fill_area_by_sensor=(
                MULTISENSOR_FILL_MIN_AREA_BY_SENSOR
            ),
            max_fill_ratio=(
                MULTISENSOR_FILL_MAX_RATIO
            ),
            allocation_cell_size_m=(
                MULTISENSOR_RELATED_FAMILY_ALLOCATION_CELL_M
            ),
        )
    )

    topology_reconciled_gdf = (
        reconcile_multisensor_connected_ids_after_fill(
            tracked_lakes_gdf=(
                related_family_fill_gdf
            ),
            output_gpkg=(
                MULTISENSOR_TOPOLOGY_RECONCILED_GPKG
            ),
            output_layer=(
                MULTISENSOR_TOPOLOGY_RECONCILED_LAYER
            ),
        )
    )

    retracking_input_gdf = (
        topology_reconciled_gdf[
            topology_reconciled_gdf.geometry.notna()
            & ~topology_reconciled_gdf.geometry.is_empty
        ]
        .copy()
        .reset_index(drop=True)
    )

    retracked_gdf = (
        assign_multisensor_persistent_ids(
            validated_candidate_gdf=(
                retracking_input_gdf
            ),
            output_gpkg=(
                MULTISENSOR_RETRACKED_GPKG
            ),
            output_layer=(
                MULTISENSOR_RETRACKED_LAYER
            ),
            max_gap_days=(
                MULTISENSOR_TRACKING_MAX_GAP_DAYS
            ),
            min_overlap_smaller_ratio=(
                MULTISENSOR_TRACKING_MIN_OVERLAP_SMALLER_RATIO
            ),
            min_iou=(
                MULTISENSOR_TRACKING_MIN_IOU
            ),
            min_area_ratio=(
                MULTISENSOR_TRACKING_MIN_AREA_RATIO
            ),
            match_distance_by_sensor=(
                MULTISENSOR_TRACKING_DISTANCE_BY_SENSOR
            ),
            distance_only_max_centroid_multiplier=(
                MULTISENSOR_TRACKING_DISTANCE_ONLY_MAX_CENTROID_MULTIPLIER
            ),
            large_change_min_reference_overlap_ratio=(
                MULTISENSOR_LARGE_CHANGE_MIN_REFERENCE_OVERLAP_RATIO
            ),
            large_change_max_area_ratio=(
                MULTISENSOR_LARGE_CHANGE_MAX_AREA_RATIO
            ),
            large_change_confirmation_max_gap_days=(
                MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MAX_GAP_DAYS
            ),
            large_change_confirmation_min_overlap_smaller_ratio=(
                MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MIN_OVERLAP_SMALLER_RATIO
            ),
            large_change_confirmation_min_area_ratio=(
                MULTISENSOR_LARGE_CHANGE_CONFIRMATION_MIN_AREA_RATIO
            ),
            large_change_long_gap_min_days=(
                MULTISENSOR_LARGE_CHANGE_LONG_GAP_MIN_DAYS
            ),
            large_change_long_gap_min_reference_overlap_ratio=(
                MULTISENSOR_LARGE_CHANGE_LONG_GAP_MIN_REFERENCE_OVERLAP_RATIO
            ),
            large_change_long_gap_confirm_min_pending_coverage_ratio=(
                MULTISENSOR_LARGE_CHANGE_LONG_GAP_CONFIRM_MIN_PENDING_COVERAGE_RATIO
            ),
            large_change_long_gap_confirm_min_area_ratio=(
                MULTISENSOR_LARGE_CHANGE_LONG_GAP_CONFIRM_MIN_AREA_RATIO
            ),
        )
    )

    retracked_relationship_gdf = (
        add_multisensor_tracking_relationships(
            persistent_tracking_gdf=(
                retracked_gdf
            ),
            output_gpkg=(
                MULTISENSOR_RETRACKED_RELATIONSHIP_GPKG
            ),
            output_layer=(
                MULTISENSOR_RETRACKED_RELATIONSHIP_LAYER
            ),
            max_gap_days=(
                MULTISENSOR_TRACKING_MAX_GAP_DAYS
            ),
            min_intersection_area_m2=(
                MULTISENSOR_RELATION_MIN_INTERSECTION_AREA_M2
            ),
            min_overlap_smaller_ratio=(
                MULTISENSOR_TRACKING_MIN_OVERLAP_SMALLER_RATIO
            ),
            min_iou=(
                MULTISENSOR_TRACKING_MIN_IOU
            ),
        )
    )

    output_dataframe = (
        catalog_dataframe.copy()
    )

    output_dataframe["date"] = (
        output_dataframe["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output_dataframe.to_csv(
        CATALOG_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 70)

    print(
        "マルチセンサーシーン数: "
        f"{len(output_dataframe)}"
    )

    print(
        output_dataframe[
            [
                "date",
                "sensor_family",
                "platform",
                "ndwi_threshold",
                "aoi_water_area_m2",
                "reference_sentinel_area_m2",
                "same_day_area_ratio",
                "quality_status",
                "quality_score",
                "quality_reason",
                "selected_for_timeseries",
                "selection_reason",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "出力CSV: "
        f"{CATALOG_OUTPUT_CSV}"
    )

    print(
        "統合時系列観測数: "
        f"{len(selected_timeseries_dataframe)}"
    )

    print(
        "統合時系列CSV: "
        f"{MULTISENSOR_TIMESERIES_OUTPUT_CSV}"
    )

    print(
        "共通追跡ポリゴン数: "
        f"{len(tracking_input_gdf)}"
    )

    print(
        "共通追跡GeoPackage: "
        f"{MULTISENSOR_TRACKING_INPUT_GPKG}"
    )

    print(
        "共通追跡レイヤー: "
        f"{MULTISENSOR_TRACKING_INPUT_LAYER}"
    )

    print(
        "グループ化後の湖候補数: "
        f"{len(grouped_candidate_gdf)}"
    )

    print(
        "新規ID作成可能候補数: "
        f"{int(grouped_candidate_gdf['can_create_new_id'].sum())}"
    )

    print(
        "継続専用候補数: "
        f"{int((~grouped_candidate_gdf['can_create_new_id']).sum())}"
    )

    print(
        "雲影確認待ち候補数: "
        f"{int(grouped_candidate_gdf['requires_shadow_validation'].sum())}"
    )

    print(
        "グループ化GeoPackage: "
        f"{MULTISENSOR_GROUPED_CANDIDATES_GPKG}"
    )

    original_shadow_candidate_count = int(
        grouped_candidate_gdf[
            "requires_shadow_validation"
        ]
        .astype(bool)
        .sum()
    )

    supported_shadow_candidate_count = int(
        shadow_validated_gdf[
            "shadow_temporal_supported"
        ]
        .astype(bool)
        .sum()
    )

    removed_shadow_candidate_count = (
        original_shadow_candidate_count
        - supported_shadow_candidate_count
    )

    original_shadow_area_m2 = float(
        grouped_candidate_gdf.loc[
            grouped_candidate_gdf[
                "requires_shadow_validation"
            ]
            .astype(bool),
            "area_m2",
        ].sum()
    )

    supported_shadow_area_m2 = float(
        shadow_validated_gdf.loc[
            shadow_validated_gdf[
                "shadow_temporal_supported"
            ]
            .astype(bool),
            "area_m2",
        ].sum()
    )

    removed_shadow_area_m2 = (
        original_shadow_area_m2
        - supported_shadow_area_m2
    )

    print(
        "雲影確認前候補数: "
        f"{original_shadow_candidate_count}"
    )

    print(
        "時系列的に支持された雲影候補数: "
        f"{supported_shadow_candidate_count}"
    )

    print(
        "除外された雲影候補数: "
        f"{removed_shadow_candidate_count}"
    )

    print(
        "除外された雲影候補面積: "
        f"{removed_shadow_area_m2:.2f} m2"
    )

    print(
        "雲影確認後の追跡候補数: "
        f"{len(shadow_validated_gdf)}"
    )

    print(
        "雲影確認後GeoPackage: "
        f"{MULTISENSOR_SHADOW_VALIDATED_GPKG}"
    )

    assigned_persistent_mask = (
        persistent_tracking_gdf[
            "persistent_counted_in_lake_count"
        ]
        .astype(bool)
    )

    new_id_count = int(
        (
            persistent_tracking_gdf[
                "tracking_status"
            ]
            == "created_new_id"
        ).sum()
    )

    continued_id_count = int(
        (
            persistent_tracking_gdf[
                "tracking_status"
            ]
            == "continued_existing_id"
        ).sum()
    )

    unmatched_continuation_count = int(
        (
            persistent_tracking_gdf[
                "tracking_status"
            ]
            == "unmatched_continuation"
        ).sum()
    )

    large_change_status_series = (
        persistent_tracking_gdf[
            "large_change_status"
        ]
        .fillna("")
        .astype(str)
    )
    confirmed_large_change_count = int(
        large_change_status_series
        .eq("confirmed_from_next_observation")
        .sum()
    )
    large_change_confirmation_observation_count = int(
        large_change_status_series
        .eq("confirmation_observation")
        .sum()
    )
    rejected_large_change_count = int(
        large_change_status_series
        .str.startswith("rejected_")
        .sum()
    )

    persistent_lake_count = int(
        persistent_tracking_gdf.loc[
            assigned_persistent_mask,
            "persistent_lake_id",
        ].nunique()
    )

    retrospective_tracking_gdf = (
        backfill_multisensor_persistent_ids(
            persistent_tracking_gdf=(
                retracked_relationship_gdf
            ),
            output_gpkg=(
                MULTISENSOR_RETROSPECTIVE_GPKG
            ),
            output_layer=(
                MULTISENSOR_RETROSPECTIVE_LAYER
            ),
            max_gap_days=(
                MULTISENSOR_TRACKING_MAX_GAP_DAYS
            ),
            min_overlap_smaller_ratio=(
                MULTISENSOR_TRACKING_MIN_OVERLAP_SMALLER_RATIO
            ),
            min_iou=(
                MULTISENSOR_TRACKING_MIN_IOU
            ),
            min_area_ratio=(
                MULTISENSOR_TRACKING_MIN_AREA_RATIO
            ),
            match_distance_by_sensor=(
                MULTISENSOR_TRACKING_DISTANCE_BY_SENSOR
            ),
            distance_only_max_centroid_multiplier=(
                MULTISENSOR_TRACKING_DISTANCE_ONLY_MAX_CENTROID_MULTIPLIER
            ),
        )
    )

    (
        area_completed_gdf,
        area_fragment_audit_gdf,
    ) = (
        attach_subthreshold_fragments_to_persistent_lakes(
            tracking_input_gdf=(
                tracking_input_gdf
            ),
            persistent_lake_gdf=(
                retrospective_tracking_gdf
            ),
            grouping_settings=(
                MULTISENSOR_GROUPING_SETTINGS
            ),
            output_gpkg=(
                MULTISENSOR_AREA_COMPLETED_GPKG
            ),
            output_layer=(
                MULTISENSOR_AREA_COMPLETED_LAYER
            ),
            audit_layer=(
                MULTISENSOR_AREA_FRAGMENT_AUDIT_LAYER
            ),
        )
    )

    # 最終の小断片追加によって、別IDの形状が同一観測内で
    # 再接続する場合がある。最終出力の直前にもう一度だけ
    # 接続関係を整合し、1つの連続水域を1つのIDにそろえる。
    final_topology_reconciled_gdf = (
        reconcile_multisensor_connected_ids_after_fill(
            tracked_lakes_gdf=area_completed_gdf,
            output_gpkg=(
                MULTISENSOR_TOPOLOGY_RECONCILED_GPKG
            ),
            output_layer=(
                MULTISENSOR_TOPOLOGY_RECONCILED_LAYER
            ),
        )
    )

    final_with_missing_gdf = (
        add_multisensor_bracketed_missing_records(
            tracked_lakes_gdf=(
                final_topology_reconciled_gdf
            ),
            selected_timeseries_dataframe=(
                selected_timeseries_dataframe
            ),
            max_gap_days=(
                MULTISENSOR_TRACKING_MAX_GAP_DAYS
            ),
        )
    )

    (
        final_lakes_gdf,
        final_timeseries_dataframe,
    ) = save_multisensor_final_outputs(
        final_lakes_gdf=(
            final_with_missing_gdf
        ),
        selected_timeseries_dataframe=(
            selected_timeseries_dataframe
        ),
        output_gpkg=MULTISENSOR_FINAL_GPKG,
        output_layer=MULTISENSOR_FINAL_LAYER,
        observations_csv=(
            MULTISENSOR_FINAL_OBSERVATIONS_CSV
        ),
        timeseries_csv=(
            MULTISENSOR_FINAL_TIMESERIES_CSV
        ),
        total_area_graph=(
            MULTISENSOR_FINAL_TOTAL_AREA_GRAPH
        ),
        id_area_graph=(
            MULTISENSOR_FINAL_ID_AREA_GRAPH
        ),
        count_graph=(
            MULTISENSOR_FINAL_COUNT_GRAPH
        ),
        change_rate_graph=(
            MULTISENSOR_FINAL_CHANGE_RATE_GRAPH
        ),
        long_gap_days=(
            MULTISENSOR_LONG_GAP_DAYS
        ),
        temporal_display_hour=(
            MULTISENSOR_QGIS_TEMPORAL_DISPLAY_HOUR
        ),
    )

    print(
        "永続IDが付いた候補数: "
        f"{int(assigned_persistent_mask.sum())}"
    )

    print(
        "作成された永続湖ID数: "
        f"{persistent_lake_count}"
    )

    print(
        "新規ID作成回数: "
        f"{new_id_count}"
    )

    print(
        "既存ID継続回数: "
        f"{continued_id_count}"
    )

    print(
        "未対応の継続専用候補数: "
        f"{unmatched_continuation_count}"
    )

    print(
        "次観測で確認された急拡大候補数: "
        f"{confirmed_large_change_count}"
    )

    print(
        "急拡大の確認に使用した後続観測数: "
        f"{large_change_confirmation_observation_count}"
    )

    print(
        "未確認または曖昧で除外した急拡大候補数: "
        f"{rejected_large_change_count}"
    )

    print(
        "永続ID追跡GeoPackage: "
        f"{MULTISENSOR_PERSISTENT_TRACKING_GPKG}"
    )

    print(
        "永続ID追跡レイヤー: "
        f"{MULTISENSOR_PERSISTENT_TRACKING_LAYER}"
    )

    retrospective_backfill_count = int(
        retrospective_tracking_gdf[
            "retrospective_backfill"
        ]
        .astype(bool)
        .sum()
    )

    retrospective_ambiguous_count = int(
        (
            retrospective_tracking_gdf[
                "retrospective_status"
            ]
            == "ambiguous_multiple_future_ids"
        ).sum()
    )

    remaining_unassigned_count = int(
        retrospective_tracking_gdf[
            "persistent_lake_id"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "過去方向へIDを遡及した候補数: "
        f"{retrospective_backfill_count}"
    )

    print(
        "複数IDと一致して遡及しなかった候補数: "
        f"{retrospective_ambiguous_count}"
    )

    print(
        "遡及後も未割当の候補数: "
        f"{remaining_unassigned_count}"
    )

    print(
        "ID遡及後GeoPackage: "
        f"{MULTISENSOR_RETROSPECTIVE_GPKG}"
    )

    print(
        "ID遡及後レイヤー: "
        f"{MULTISENSOR_RETROSPECTIVE_LAYER}"
    )

    attached_fragment_count = int(
        area_completed_gdf[
            "area_only_fragment_count"
        ].sum()
    )

    attached_fragment_area_m2 = float(
        area_completed_gdf[
            "area_only_added_area_m2"
        ].sum()
    )

    attached_shadow_fragment_count = int(
        area_completed_gdf[
            "area_only_shadow_fragment_count"
        ].sum()
    )

    ambiguous_fragment_count = int(
        (
            area_fragment_audit_gdf[
                "assignment_status"
            ]
            == "ambiguous_equal_distance"
        ).sum()
        if not area_fragment_audit_gdf.empty
        else 0
    )

    print(
        "永続湖へ面積専用で追加した小断片数: "
        f"{attached_fragment_count}"
    )

    print(
        "小断片による追加面積: "
        f"{attached_fragment_area_m2:.2f} m2"
    )

    print(
        "追加した雲影小断片数: "
        f"{attached_shadow_fragment_count}"
    )

    print(
        "同距離の複数湖があり追加しなかった断片数: "
        f"{ambiguous_fragment_count}"
    )

    print(
        "小断片追加後GeoPackage: "
        f"{MULTISENSOR_AREA_COMPLETED_GPKG}"
    )

    bracketed_fill_count = int(
        bracketed_fill_gdf[
            "temporal_fill_applied"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    bracketed_fill_area_m2 = float(
        bracketed_fill_gdf[
            "temporally_filled_area_m2"
        ]
        .fillna(0.0)
        .sum()
    )

    same_sensor_fill_count = int(
        (
            bracketed_fill_gdf[
                "temporal_fill_applied"
            ]
            .fillna(False)
            .astype(bool)
            & ~bracketed_fill_gdf[
                "temporal_fill_cross_sensor_window"
            ]
            .fillna(False)
            .astype(bool)
        ).sum()
    )

    cross_sensor_fill_count = int(
        (
            bracketed_fill_gdf[
                "temporal_fill_applied"
            ]
            .fillna(False)
            .astype(bool)
            & bracketed_fill_gdf[
                "temporal_fill_cross_sensor_window"
            ]
            .fillna(False)
            .astype(bool)
        ).sum()
    )

    print(
        "前後観測による部分補完数: "
        f"{bracketed_fill_count}"
    )

    print(
        "前後観測による補完面積: "
        f"{bracketed_fill_area_m2:.2f} m2"
    )

    print(
        "同一センサー観測による補完数: "
        f"{same_sensor_fill_count}"
    )

    print(
        "異種センサー観測による補完数: "
        f"{cross_sensor_fill_count}"
    )

    print(
        "部分補完GeoPackage: "
        f"{MULTISENSOR_BRACKETED_FILL_GPKG}"
    )

    related_family_fill_count = int(
        related_family_fill_gdf[
            "related_family_fill_applied"
        ].fillna(False).astype(bool).sum()
    )
    topology_reconciled_count = int(
        final_topology_reconciled_gdf[
            "topology_reconciled"
        ].fillna(False).astype(bool).sum()
    )
    missing_observation_count = int(
        final_lakes_gdf[
            "observation_status"
        ].fillna("").astype(str).eq(
            "missing_between_observations"
        ).sum()
    )
    incomplete_date_count = int(
        (~final_timeseries_dataframe[
            "area_observation_complete"
        ].astype(bool)).sum()
    )

    print(
        "関連湖ファミリー補完数: "
        f"{related_family_fill_count}"
    )

    print(
        "補完後トポロジー整合行数: "
        f"{topology_reconciled_count}"
    )

    print(
        "補完形状による再追跡GeoPackage: "
        f"{MULTISENSOR_RETRACKED_GPKG}"
    )

    print(
        "観測間に追加した欠損レコード数: "
        f"{missing_observation_count}"
    )

    print(
        "一部の永続湖が欠損している日数: "
        f"{incomplete_date_count}"
    )

    print(
        "最終GeoPackage: "
        f"{MULTISENSOR_FINAL_GPKG}"
    )

    print(
        "QGIS時系列表示フィールド: temporal_datetime "
        f"(各観測日の{MULTISENSOR_QGIS_TEMPORAL_DISPLAY_HOUR:02d}:00)"
    )

    print(
        "最終湖別CSV: "
        f"{MULTISENSOR_FINAL_OBSERVATIONS_CSV}"
    )

    print(
        "最終時系列CSV: "
        f"{MULTISENSOR_FINAL_TIMESERIES_CSV}"
    )

    print(
        "最終グラフ: "
        f"{MULTISENSOR_FINAL_TOTAL_AREA_GRAPH}; "
        f"{MULTISENSOR_FINAL_ID_AREA_GRAPH}; "
        f"{MULTISENSOR_FINAL_COUNT_GRAPH}; "
        f"{MULTISENSOR_FINAL_CHANGE_RATE_GRAPH}"
    )


if __name__ == "__main__":
    main()
