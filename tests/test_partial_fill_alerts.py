"""部分補完を条件付き比較に使う処理の回帰テスト。"""
import unittest

import pandas as pd

from lake_area_alerts import build_alert_outputs
from test_cross_sensor_alerts import scene


def partial(day, direct, corrected, iou=0.7, quality="usable", sensor="Sentinel-2"):
    row = scene(day, direct, sensor)
    row.update(
        observation_status="direct_with_partial_fill",
        temporal_fill_applied=True,
        temporal_fill_status="partially_filled",
        temporal_fill_previous_date=f"2025-07-{day-1:02}",
        temporal_fill_next_date=f"2025-07-{day+1:02}",
        temporal_reference_iou=iou,
        temporal_fill_ratio=(corrected-direct)/corrected,
        corrected_area_m2=corrected,
        quality_status=quality,
    )
    if quality == "partial":
        row.update(requires_shadow_validation=True, shadow_only=True,
                   shadow_temporal_supported=True,
                   shadow_filter_status="kept_temporal_match")
    # 最終形状は補完後形状に対応する。
    from shapely.geometry import box
    row["geometry_wkt"] = box(0, 0, corrected / 10, 10).wkt
    return row


class PartialFillTests(unittest.TestCase):
    def test_partial_is_conditional_and_estimated_area_used(self):
        metrics, _ = build_alert_outputs(pd.DataFrame([
            scene(1, 10000), partial(2, 11500, 12000)]))
        row = metrics.iloc[1]
        self.assertTrue(row.metric_observation_eligible)
        self.assertEqual(row.metric_area_m2, 12000)
        self.assertEqual(row.metric_direct_area_m2, 11500)
        self.assertEqual(row.metric_comparison_confidence, "conditional_partial_fill")

    def test_estimate_only_growth_is_low_confidence(self):
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000), partial(2, 11500, 13000), scene(3, 12500)]))
        event = alerts.iloc[1]
        self.assertEqual(event.alert_level, "WATCH_PENDING_LOW_CONFIDENCE")
        self.assertEqual(event.watch_status, "REVIEW_CONFIRMED")

    def test_partial_followup_cannot_confirm(self):
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000), partial(2, 11500, 13000),
            partial(3, 12000, 13500)]))
        self.assertEqual(alerts.iloc[1].watch_status, "WATCH_PENDING")

    def test_partial_observation_does_not_replace_direct_reference(self):
        metrics, _ = build_alert_outputs(pd.DataFrame([
            scene(1, 10000),
            partial(2, 11500, 12000),
            scene(3, 12500),
        ]))

        last = metrics.iloc[2]
        self.assertEqual(
            last.metric_reference_date,
            pd.Timestamp("2025-07-01"),
        )
        self.assertEqual(last.metric_reference_area_m2, 10000)
        self.assertEqual(last.metric_area_change_m2, 2500)
        self.assertEqual(last.metric_comparison_confidence, "direct")

    def test_partial_observation_cannot_create_initial_baseline(self):
        metrics, alerts = build_alert_outputs(pd.DataFrame([
            partial(2, 9000, 10000),
            scene(3, 11000),
        ]))

        first = metrics.iloc[0]
        second = metrics.iloc[1]
        self.assertEqual(
            first.metric_comparison_status,
            "no_trusted_direct_reference",
        )
        self.assertEqual(
            second.metric_comparison_status,
            "baseline_no_same_sensor_reference",
        )
        self.assertFalse(second.metric_comparison_valid)
        self.assertEqual(alerts.iloc[0].alert_level, "NO_COMPARISON")

    def test_ratio_iou_topology_and_family_fill_rejected(self):
        variants = [partial(2, 9000, 12000), partial(2, 11500, 12000, iou=0.2)]
        top = partial(2, 11500, 12000); top["topology_reconciled"] = True
        family = partial(2, 11500, 12000); family["related_family_fill_applied"] = True
        variants += [top, family]
        for row in variants:
            metrics, _ = build_alert_outputs(pd.DataFrame([scene(1, 10000), row]))
            self.assertFalse(metrics.iloc[1].metric_observation_eligible)

    def test_supported_partial_quality_is_accepted(self):
        metrics, _ = build_alert_outputs(pd.DataFrame([
            scene(1, 10000), partial(2, 11000, 12000, quality="partial")]))
        self.assertTrue(metrics.iloc[1].metric_observation_eligible)

    def test_complete_missing_stays_excluded(self):
        row = scene(2, 0)
        row.update(observation_status="missing_between_observations",
                   tracking_status="missing", quality_status="missing")
        metrics, _ = build_alert_outputs(pd.DataFrame([scene(1, 10000), row]))
        self.assertFalse(metrics.iloc[1].metric_observation_eligible)


if __name__ == "__main__":
    unittest.main()
