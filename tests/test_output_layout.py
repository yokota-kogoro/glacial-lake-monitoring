"""用途別に整理した出力パスの接続を確認するテスト。"""

import unittest

import lake_area_alerts
import landsat_ndwi
import multisensor_observations
import sentinel_ndwi
from glacial_lake import multisensor_config


class OutputLayoutTests(unittest.TestCase):
    def test_preprocessing_summaries_match_multisensor_inputs(self):
        self.assertEqual(
            sentinel_ndwi.SUMMARY_OUTPUT_DIR
            / "sentinel_area_summary.csv",
            multisensor_config.SENTINEL_SUMMARY_CSV,
        )
        self.assertEqual(
            landsat_ndwi.SUMMARY_OUTPUT_DIR
            / "landsat_area_summary.csv",
            multisensor_config.LANDSAT_SUMMARY_CSV,
        )

    def test_multisensor_detail_matches_alert_input(self):
        self.assertEqual(
            multisensor_observations.MULTISENSOR_FINAL_OBSERVATIONS_CSV,
            lake_area_alerts.INPUT_CSV,
        )

    def test_outputs_are_separated_by_purpose(self):
        self.assertIn(
            "results",
            multisensor_observations.MULTISENSOR_FINAL_GPKG.parts,
        )
        self.assertIn(
            "details",
            multisensor_observations.MULTISENSOR_FINAL_OBSERVATIONS_CSV.parts,
        )
        self.assertIn(
            "diagnostics",
            multisensor_observations.MULTISENSOR_INTERMEDIATE_GPKG.parts,
        )


if __name__ == "__main__":
    unittest.main()
