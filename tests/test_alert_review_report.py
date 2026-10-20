"""急増候補の確認用レポートに関する回帰テスト。"""
import unittest

import pandas as pd

from lake_area_alerts import build_alert_outputs, build_alert_review_report
from test_cross_sensor_alerts import scene


class AlertReviewReportTests(unittest.TestCase):
    def test_only_watch_candidates_are_combined_and_explained(self):
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000),
            scene(2, 13000),
            scene(3, 13200),
        ]))
        group_alerts = alerts.copy()
        group_alerts["persistent_lake_id"] = (
            "GROUP:LAKE_001;LAKE_002"
        )
        group_alerts["group_member_ids"] = "LAKE_001;LAKE_002"
        group_alerts["observed_lake_ids"] = "LAKE_001;LAKE_002"
        group_alerts["watch_event_id"] = group_alerts["watch_event_id"].where(
            group_alerts["watch_event_id"].eq(""),
            "GROUP::" + group_alerts["watch_event_id"],
        )

        report = build_alert_review_report(alerts, group_alerts)

        self.assertEqual(len(report), 2)
        self.assertEqual(
            set(report["review_scope"]),
            {"individual_lake", "related_lake_group"},
        )
        self.assertTrue(report["report_event_id"].is_unique)
        self.assertTrue(
            report["candidate_condition_ja"].str.contains("増加量").all()
        )
        self.assertTrue(
            report["candidate_condition_ja"].str.contains("増加率").all()
        )
        self.assertTrue(report["watch_status"].eq("REVIEW_CONFIRMED").all())
        self.assertTrue(report["review_priority"].eq("HIGH").all())
        self.assertTrue(
            report["recommended_review_action_ja"].str.contains("GLOF").all()
        )
        self.assertTrue(report["manual_review_status"].eq("").all())
        self.assertTrue(report["manual_review_comment"].eq("").all())

    def test_no_watch_candidate_returns_header_only_report(self):
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000),
            scene(2, 10100),
        ]))
        report = build_alert_review_report(alerts, alerts.copy())
        self.assertTrue(report.empty)
        self.assertIn("report_event_id", report.columns)
        self.assertIn("manual_review_status", report.columns)

    def test_expired_deadline_is_not_reported_as_followup_observation(self):
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000),
            scene(2, 13000),
            scene(3, 13200),
        ]))
        candidate_index = alerts.index[
            alerts["alert_level"].eq("WATCH_PENDING")
        ][0]
        deadline = pd.Timestamp("2025-08-01")
        alerts.at[candidate_index, "watch_status"] = "EXPIRED_UNCONFIRMED"
        alerts.at[candidate_index, "watch_reason"] = (
            "no_eligible_followup_within_deadline"
        )
        alerts.at[candidate_index, "watch_resolution_date"] = deadline
        alerts.at[candidate_index, "watch_deadline_date"] = deadline
        alerts.at[candidate_index, "watch_resolution_observation_id"] = ""
        group_alerts = alerts.copy()
        group_alerts["alert_level"] = "NORMAL"
        group_alerts["same_sensor_alert_level"] = "NORMAL"

        report = build_alert_review_report(alerts, group_alerts)

        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0].resolution_date, deadline)
        self.assertEqual(report.iloc[0].confirmation_deadline_date, deadline)
        self.assertTrue(pd.isna(report.iloc[0].followup_date))

    def test_same_sensor_candidate_is_kept_when_primary_is_normal(self):
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000),
            scene(2, 16000, sensor="Landsat"),
            scene(3, 13000),
        ]))
        group_alerts = alerts.copy()
        group_alerts["alert_level"] = "NORMAL"
        group_alerts["same_sensor_alert_level"] = "NORMAL"

        report = build_alert_review_report(alerts, group_alerts)
        fallback = report.loc[
            report["alert_date"].eq(pd.Timestamp("2025-07-03"))
        ].iloc[0]

        self.assertEqual(fallback.comparison_path, "same_sensor_fallback")
        self.assertEqual(fallback.reference_date, pd.Timestamp("2025-07-01"))
        self.assertEqual(fallback.area_change_m2, 3000)
        self.assertEqual(fallback.area_change_percent, 30)
        self.assertEqual(fallback.alert_level, "WATCH_PENDING")

    def test_single_member_group_duplicate_is_removed(self):
        _, alerts = build_alert_outputs(pd.DataFrame([
            scene(1, 10000),
            scene(2, 13000),
            scene(3, 13200),
        ]))
        group_alerts = alerts.copy()
        group_alerts["persistent_lake_id"] = "GROUP:A"
        group_alerts["group_member_ids"] = "A"
        group_alerts["observed_lake_ids"] = "A"

        report = build_alert_review_report(alerts, group_alerts)

        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0].review_scope, "individual_lake")


if __name__ == "__main__":
    unittest.main()
