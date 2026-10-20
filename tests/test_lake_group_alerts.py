"""関連湖の面積比較の回帰テスト。入力・出力データを変更しない。"""
import unittest

import pandas as pd
from shapely.geometry import box

from lake_area_alerts import (
    calculate_lake_group_metrics, classify_lake_area_watch,
    resolve_lake_area_watches,
)


def observation(day, lake, area, event="continuation", related="", x=0):
    return dict(
        persistent_lake_id=lake, observation_id=f"S2_{day}", date=f"2025-07-{day:02}",
        sensor_family="Sentinel-2", quality_status="usable", observation_status="direct",
        tracking_status="continued_existing_id", tracking_event=event,
        topology_reconciled=False, temporal_fill_applied=False,
        related_family_fill_applied=False, requires_shadow_validation=False,
        shadow_only=False, large_change_status="not_applicable",
        direct_area_m2=area, related_previous_lake_ids=related,
        relation_split_source_ids="", geometry_wkt=box(x, 0, x + area / 10, 10).wkt,
        corrected_area_m2=area, temporal_fill_ratio=0.0,
        temporal_fill_status="not_applied", temporal_fill_previous_date="",
        temporal_fill_next_date="", temporal_reference_iou=float("nan"),
        shadow_temporal_supported=False, shadow_filter_status="not_required",
    )


class GroupMetricsTests(unittest.TestCase):
    def base(self):
        return [observation(1, "A", 2000), observation(1, "B", 3000, x=1000),
                observation(2, "B", 6500, "merge", "A;B")]

    def test_merge_continuation_split_and_confirmation(self):
        rows = self.base() + [observation(3, "B", 6600),
                             observation(4, "A", 2500, "split_new", "B"),
                             observation(4, "B", 4200, "split_primary", "B", x=1000)]
        result = calculate_lake_group_metrics(pd.DataFrame(rows))
        merge = result[result.date.eq(pd.Timestamp("2025-07-02"))].iloc[0]
        self.assertEqual(merge.metric_reference_area_m2, 5000)
        self.assertEqual(merge.metric_area_change_m2, 1500)
        self.assertEqual(merge.metric_area_change_percent, 30)
        self.assertEqual(result.iloc[-2].metric_area_change_m2, 100)
        self.assertEqual(result.iloc[-1].metric_area_change_m2, 100)
        alerts = resolve_lake_area_watches(classify_lake_area_watch(
            result, 20, {"Sentinel-2": 500}))
        event = alerts[alerts.date.eq(pd.Timestamp("2025-07-02"))].iloc[0]
        self.assertEqual(event.watch_status, "REVIEW_CONFIRMED")

    def test_no_sum_across_dates(self):
        rows = self.base()
        rows[1]["date"] = "2025-06-30"
        result = calculate_lake_group_metrics(pd.DataFrame(rows))
        self.assertFalse(result.iloc[-1].metric_comparison_valid)

    def test_missing_split_member_is_not_zero(self):
        rows = self.base() + [observation(3, "A", 2500, "split_new", "B"),
                             observation(3, "B", 4200, "split_primary", "B", x=1000),
                             observation(4, "A", 2600)]
        result = calculate_lake_group_metrics(pd.DataFrame(rows))
        self.assertFalse(result.iloc[-1].metric_comparison_valid)
        self.assertIn("missing_active_group_member", result.iloc[-1].metric_exclusion_reason)

    def test_overlap_and_quality_are_rejected(self):
        for change in ("overlap", "quality", "fill"):
            rows = self.base()
            if change == "overlap":
                rows[1]["geometry_wkt"] = box(0, 0, 300, 10).wkt
            elif change == "quality":
                rows[1]["quality_status"] = "unusable"
            else:
                rows[1]["temporal_fill_applied"] = True
            result = calculate_lake_group_metrics(pd.DataFrame(rows))
            self.assertFalse(result.iloc[-1].metric_comparison_valid, change)

    def test_other_sensor_does_not_supply_baseline(self):
        rows = self.base()
        rows[-1]["sensor_family"] = "Landsat"
        result = calculate_lake_group_metrics(pd.DataFrame(rows))
        self.assertFalse(result.iloc[-1].metric_comparison_valid)

    def test_repeated_merge_uses_merged_area_not_old_member(self):
        rows = self.base() + [observation(3, "B", 6600, "merge", "A;B")]
        result = calculate_lake_group_metrics(pd.DataFrame(rows))
        self.assertEqual(result.iloc[-1].metric_reference_area_m2, 6500)
        self.assertEqual(result.iloc[-1].metric_area_change_m2, 100)

    def test_unrelated_lake_and_future_relation_do_not_change_past(self):
        rows = self.base() + [observation(1, "C", 9000, x=2000),
                             observation(2, "C", 9100, x=2000)]
        result = calculate_lake_group_metrics(pd.DataFrame(rows))
        first = result[result.date.eq(pd.Timestamp("2025-07-01"))]
        self.assertEqual(set(first.persistent_lake_id), {"GROUP:A", "GROUP:B", "GROUP:C"})
        merged = result[(result.date.eq(pd.Timestamp("2025-07-02")))
                        & result.persistent_lake_id.eq("GROUP:A;B")].iloc[0]
        self.assertEqual(merged.metric_reference_area_m2, 5000)

    def test_reference_gap_limit(self):
        result = calculate_lake_group_metrics(pd.DataFrame(self.base()), max_gap_days=1)
        self.assertTrue(result.iloc[-1].metric_comparison_valid)
        rows = self.base()
        rows[-1]["date"] = "2025-08-02"
        result = calculate_lake_group_metrics(pd.DataFrame(rows), max_gap_days=30)
        self.assertFalse(result.iloc[-1].metric_comparison_valid)

    def test_invalid_polygon_does_not_crash_aggregation(self):
        rows = self.base()
        rows[-1]["geometry_wkt"] = "POLYGON ((0 0, 10 10, 10 0, 0 10, 0 0))"
        result = calculate_lake_group_metrics(pd.DataFrame(rows))
        self.assertIn("invalid_geometry", result.iloc[-1].metric_exclusion_reason)
        self.assertFalse(result.iloc[-1].metric_comparison_valid)


if __name__ == "__main__":
    unittest.main()
