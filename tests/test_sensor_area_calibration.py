"""Sentinel-2/Landsat面積補正の回帰テスト。"""
import unittest

import pandas as pd

from lake_area_alerts import (
    apply_sensor_area_calibration,
    build_alert_outputs,
    calculate_lake_change_metrics,
    classify_lake_area_watch,
    estimate_sensor_area_calibration,
)
from test_cross_sensor_alerts import scene


def calibration_metric(lake, day, sensor, area):
    return {
        "persistent_lake_id": lake,
        "observation_id": f"{sensor}_{lake}_{day}",
        "date": f"2025-07-{day:02}",
        "sensor_family": sensor,
        "metric_area_m2": area,
        "metric_observation_type": "direct",
        "metric_observation_eligible": True,
    }


def calibration_with_factor_two(p90_error=0.0):
    rows = []
    for number, target_area in enumerate([5000, 6000, 7000], start=1):
        lake = f"CAL_{number}"
        rows.append(calibration_metric(lake, number * 2 - 1,
                                       "Landsat", target_area))
        rows.append(calibration_metric(lake, number * 2,
                                       "Sentinel-2", target_area * 2))
    summary, pairs = estimate_sensor_area_calibration(pd.DataFrame(rows))
    if p90_error:
        summary.loc[0, "p90_absolute_residual_m2"] = p90_error
    return summary, pairs


class SensorAreaCalibrationTests(unittest.TestCase):
    def test_estimates_one_to_one_median_factor(self):
        summary, pairs = calibration_with_factor_two()
        self.assertEqual(summary.iloc[0].calibration_status, "calibrated")
        self.assertTrue(summary.iloc[0].calibration_applied)
        self.assertEqual(summary.iloc[0].accepted_pair_count, 3)
        self.assertEqual(summary.iloc[0].target_to_reference_area_factor, 2.0)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(pairs.pair_accepted.all())

    def test_many_pairs_from_one_lake_do_not_enable_calibration(self):
        rows = []
        for pair_number, target_area in enumerate([5000, 5500, 6000]):
            target_day = pair_number * 2 + 1
            rows.append(calibration_metric(
                "ONLY_ONE_LAKE", target_day, "Landsat", target_area
            ))
            rows.append(calibration_metric(
                "ONLY_ONE_LAKE", target_day + 1,
                "Sentinel-2", target_area,
            ))
        summary, _ = estimate_sensor_area_calibration(pd.DataFrame(rows))
        self.assertEqual(
            summary.iloc[0].calibration_status,
            "insufficient_independent_lakes",
        )
        self.assertFalse(summary.iloc[0].calibration_applied)
        self.assertEqual(summary.iloc[0].used_pair_count, 3)
        self.assertEqual(summary.iloc[0].used_lake_count, 1)

    def test_each_lake_has_equal_weight_in_factor(self):
        rows = []
        # Aだけ3ペアあるが、湖単位ではA・B・Cを一票ずつにする。
        for pair_number in range(3):
            target_day = pair_number * 2 + 1
            rows.append(calibration_metric(
                "A", target_day, "Landsat", 5000
            ))
            rows.append(calibration_metric(
                "A", target_day + 1, "Sentinel-2", 10000
            ))
        rows.extend([
            calibration_metric("B", 1, "Landsat", 5000),
            calibration_metric("B", 2, "Sentinel-2", 5000),
            calibration_metric("C", 1, "Landsat", 6000),
            calibration_metric("C", 2, "Sentinel-2", 6000),
        ])
        summary, _ = estimate_sensor_area_calibration(pd.DataFrame(rows))
        self.assertTrue(summary.iloc[0].calibration_applied)
        self.assertEqual(summary.iloc[0].used_lake_count, 3)
        self.assertEqual(
            summary.iloc[0].target_to_reference_area_factor,
            1.0,
        )

    def test_lake_level_ratio_outlier_is_not_used(self):
        rows = []
        for lake, ratio in {
            "A": 1.00,
            "B": 1.01,
            "C": 0.99,
            "OUTLIER": 3.00,
        }.items():
            rows.append(calibration_metric(lake, 1, "Landsat", 10000))
            rows.append(calibration_metric(
                lake, 2, "Sentinel-2", 10000 * ratio
            ))
        summary, pairs = estimate_sensor_area_calibration(pd.DataFrame(rows))
        outlier = pairs.loc[
            pairs["persistent_lake_id"].eq("OUTLIER")
        ].iloc[0]
        self.assertTrue(outlier.lake_ratio_outlier)
        self.assertFalse(outlier.pair_used_for_calibration)
        self.assertEqual(summary.iloc[0].outlier_lake_count, 1)
        self.assertEqual(summary.iloc[0].used_lake_count, 3)
        self.assertAlmostEqual(
            summary.iloc[0].target_to_reference_area_factor,
            1.0,
        )

    def test_insufficient_pairs_do_not_change_area(self):
        rows = [
            calibration_metric("A", 1, "Landsat", 5000),
            calibration_metric("A", 2, "Sentinel-2", 10000),
            calibration_metric("B", 3, "Landsat", 6000),
            calibration_metric("B", 4, "Sentinel-2", 12000),
        ]
        summary, _ = estimate_sensor_area_calibration(pd.DataFrame(rows))
        metrics = calculate_lake_change_metrics(pd.DataFrame([
            scene(1, 10000), scene(2, 7500, "Landsat"),
        ]), allow_cross_sensor=True)
        calibrated = apply_sensor_area_calibration(metrics, summary)
        self.assertEqual(summary.iloc[0].calibration_status, "insufficient_pairs")
        self.assertEqual(calibrated.iloc[1].metric_area_m2, 7500)
        self.assertFalse(calibrated.iloc[1].metric_cross_sensor_calibrated)

    def test_normalizes_area_and_uses_calibration_error_threshold(self):
        summary, _ = calibration_with_factor_two(p90_error=3000)
        metrics = calculate_lake_change_metrics(pd.DataFrame([
            scene(1, 10000), scene(2, 7500, "Landsat"),
        ]), allow_cross_sensor=True)
        calibrated = apply_sensor_area_calibration(metrics, summary)
        candidate = calibrated.iloc[1]
        self.assertEqual(candidate.metric_raw_area_m2, 7500)
        self.assertEqual(candidate.metric_area_m2, 15000)
        self.assertEqual(candidate.metric_area_change_m2, 5000)
        self.assertTrue(candidate.metric_cross_sensor_calibrated)
        alerts = classify_lake_area_watch(
            calibrated,
            20,
            {"Sentinel-2": 500, "Landsat": 4500},
            {("Sentinel-2", "Landsat"): 4500},
        )
        # 2 x 3000 m2の補正誤差閾値を超えないため急増候補にしない。
        self.assertEqual(alerts.iloc[1].alert_min_increase_m2, 6000)
        self.assertEqual(alerts.iloc[1].alert_level, "NORMAL")

    def test_calibrated_cross_sensor_watch_accepts_other_sensor_followup(self):
        summary, _ = calibration_with_factor_two()
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000),
            scene(2, 7500, "Landsat"),
            scene(3, 15200, "Sentinel-2"),
        ]), calibration=summary)
        event = alerts.iloc[1]
        self.assertEqual(event.alert_reason,
                         "cross_sensor_growth_candidate_calibrated")
        self.assertEqual(event.watch_required_sensor, "any_calibrated_sensor")
        self.assertEqual(event.watch_status, "REVIEW_CONFIRMED")

    def test_same_sensor_watch_accepts_calibrated_landsat_followup(self):
        summary, _ = calibration_with_factor_two(p90_error=1000)
        landsat_followup = scene(3, 10000, "Landsat")
        landsat_followup.update(
            quality_status="partial",
            requires_shadow_validation=True,
            shadow_only=True,
            shadow_temporal_supported=True,
            shadow_filter_status="kept_temporal_match",
        )
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000),
            scene(2, 16000),
            landsat_followup,
        ]), calibration=summary)
        event = alerts.iloc[1]
        self.assertEqual(event.alert_reason, "both_growth_thresholds_met")
        self.assertEqual(event.watch_required_sensor, "any_calibrated_sensor")
        self.assertEqual(event.watch_resolution_sensor, "Landsat")
        self.assertEqual(event.watch_confirmation_min_increase_m2, 2000)
        self.assertEqual(event.watch_status, "REVIEW_CONFIRMED")

    def test_other_sensor_does_not_dismiss_same_sensor_watch(self):
        summary, _ = calibration_with_factor_two(p90_error=1000)
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000),
            scene(2, 16000),
            # 補正後10,000 m2で増加を支持しないが、これだけで棄却しない。
            scene(3, 5000, "Landsat"),
            scene(4, 17000),
        ]), calibration=summary)
        event = alerts.iloc[1]
        self.assertEqual(event.watch_required_sensor, "any_calibrated_sensor")
        self.assertEqual(event.watch_resolution_sensor, "Sentinel-2")
        self.assertEqual(event.watch_status, "REVIEW_CONFIRMED")


if __name__ == "__main__":
    unittest.main()
