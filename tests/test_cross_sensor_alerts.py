"""異種センサーの候補抽出・基準センサーによる確認の回帰テスト。"""
import unittest

import pandas as pd

from lake_area_alerts import (
    build_alert_outputs, calculate_lake_change_metrics, calculate_lake_group_metrics,
    classify_lake_area_watch, resolve_lake_area_watches,
)
from test_lake_group_alerts import observation


def scene(day, area, sensor="Sentinel-2", lake="A", **kwargs):
    row = observation(day, lake, area, **kwargs)
    row.update(sensor_family=sensor, observation_id=f"{sensor}_{day}")
    return row


class CrossSensorTests(unittest.TestCase):
    def test_temporally_supported_landsat_direct_is_conditional(self):
        qa_row = scene(2, 16000, "Landsat")
        qa_row.update(
            quality_status="partial",
            requires_shadow_validation=True,
            shadow_only=True,
            shadow_temporal_supported=True,
            shadow_filter_status="kept_temporal_match",
        )
        metrics, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000), qa_row, scene(3, 15000),
        ]))
        candidate = metrics.iloc[1]
        self.assertTrue(candidate.metric_observation_eligible)
        self.assertEqual(candidate.metric_observation_type, "qa_supported_direct")
        self.assertEqual(candidate.metric_comparison_confidence,
                         "conditional_qa_supported")
        event = alerts.iloc[1]
        self.assertEqual(event.alert_confidence, "CONDITIONAL_QA_SUPPORTED")
        self.assertEqual(event.watch_status, "REVIEW_CONFIRMED")

    def test_unsupported_landsat_direct_stays_excluded(self):
        qa_row = scene(2, 16000, "Landsat")
        qa_row.update(
            quality_status="partial",
            requires_shadow_validation=True,
            shadow_only=True,
            shadow_temporal_supported=False,
            shadow_filter_status="removed_unconfirmed",
        )
        metrics, _ = build_alert_outputs(pd.DataFrame([
            scene(1, 10000), qa_row,
        ]))
        self.assertFalse(metrics.iloc[1].metric_observation_eligible)

    def test_group_preserves_qa_conditional_confidence(self):
        qa_row = scene(2, 12000, "Landsat")
        qa_row.update(
            quality_status="partial",
            requires_shadow_validation=True,
            shadow_only=True,
            shadow_temporal_supported=True,
            shadow_filter_status="kept_temporal_match",
        )
        metrics, _ = build_alert_outputs(pd.DataFrame([
            scene(1, 10000), qa_row,
        ]), grouped=True)
        self.assertEqual(metrics.iloc[1].metric_observation_type,
                         "qa_supported_direct")
        self.assertEqual(metrics.iloc[1].metric_comparison_confidence,
                         "conditional_qa_supported")

    def test_latest_reference_and_same_sensor_supplement(self):
        rows = [scene(1, 10000), scene(2, 16000, "Landsat"), scene(3, 13000)]
        metrics, alerts = build_alert_outputs(pd.DataFrame(rows))
        last = metrics.iloc[-1]
        self.assertEqual(last.metric_reference_area_m2, 16000)
        self.assertEqual(last.same_sensor_metric_reference_area_m2, 10000)
        self.assertEqual(last.metric_area_change_m2, -3000)
        self.assertEqual(last.same_sensor_metric_area_change_m2, 3000)
        event = alerts[alerts.date.eq(pd.Timestamp("2025-07-02"))].iloc[0]
        self.assertEqual(event.alert_level, "WATCH_PENDING")
        self.assertEqual(event.watch_status, "REVIEW_CONFIRMED")
        self.assertEqual(event.watch_required_sensor, "Sentinel-2")
        self.assertEqual(event.watch_confirmation_min_increase_m2, 500)

    def test_trigger_sensor_alone_cannot_confirm(self):
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000), scene(2, 16000, "Landsat"), scene(3, 17000, "Landsat")]))
        self.assertEqual(alerts.iloc[1].watch_status, "WATCH_PENDING")

    def test_reverse_uses_landsat_confirmation_threshold(self):
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000, "Landsat"), scene(2, 16000), scene(3, 13000, "Landsat")]))
        self.assertEqual(alerts.iloc[1].watch_confirmation_min_increase_m2, 4500)
        self.assertEqual(alerts.iloc[1].watch_status, "DISMISSED")

    def test_pair_threshold_equality_and_unknown_pair(self):
        metrics = calculate_lake_change_metrics(pd.DataFrame([
            scene(1, 10000), scene(2, 14500, "Landsat")]), allow_cross_sensor=True)
        alerts = classify_lake_area_watch(metrics, 20, {"Landsat": 4500},
                                         {("Sentinel-2", "Landsat"): 4500})
        self.assertEqual(alerts.iloc[1].alert_level, "WATCH_PENDING")
        unknown = classify_lake_area_watch(metrics, 20, {"Landsat": 4500})
        self.assertEqual(unknown.iloc[1].alert_level, "INVALID")

    def test_quality_gap_and_identity_boundary(self):
        bad = scene(2, 100000, "Landsat")
        bad["quality_status"] = "unusable"
        metrics, _ = build_alert_outputs(pd.DataFrame([scene(1, 10000), bad, scene(3, 12000)]))
        self.assertEqual(metrics.iloc[-1].metric_reference_date, pd.Timestamp("2025-07-01"))
        boundary = scene(3, 17000, event="merge", related="A;B")
        _, alerts = build_alert_outputs(pd.DataFrame([scene(1, 10000), scene(2, 16000, "Landsat"), boundary]))
        self.assertEqual(alerts.iloc[1].watch_status, "INVALID")
        far = scene(2, 16000, "Landsat")
        far["date"] = "2025-09-01"
        metrics, _ = build_alert_outputs(pd.DataFrame([scene(1, 10000), far]))
        self.assertFalse(metrics.iloc[-1].metric_comparison_valid)

    def test_group_merge_cross_sensor_sum_and_confirmation(self):
        rows = [scene(1, 4000), scene(1, 6000, lake="B", x=1000),
                scene(2, 16000, "Landsat", lake="B", event="merge", related="A;B"),
                scene(3, 16500, lake="B")]
        metrics, alerts = build_alert_outputs(pd.DataFrame(rows), grouped=True)
        merged = metrics[metrics.date.eq(pd.Timestamp("2025-07-02"))].iloc[0]
        self.assertEqual(merged.metric_reference_area_m2, 10000)
        self.assertTrue(merged.metric_cross_sensor)
        event = alerts[alerts.date.eq(pd.Timestamp("2025-07-02"))].iloc[0]
        self.assertEqual(event.watch_status, "REVIEW_CONFIRMED")

    def test_mixed_sensor_group_is_not_a_reference(self):
        rows = [scene(1, 4000), scene(1, 6000, "Landsat", lake="B", x=1000),
                scene(2, 16000, "Landsat", lake="B", event="merge", related="A;B")]
        metrics = calculate_lake_group_metrics(pd.DataFrame(rows), allow_cross_sensor=True)
        self.assertFalse(metrics.iloc[-1].metric_comparison_valid)


if __name__ == "__main__":
    unittest.main()
