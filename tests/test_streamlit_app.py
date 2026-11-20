from __future__ import annotations

import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app_mode_state import (  # noqa: E402
    CONTROLLED_MODE,
    EXPERIMENTAL_MODE,
)


class StreamlitAppTests(unittest.TestCase):
    def test_controlled_workflow_is_the_default(self) -> None:
        app = AppTest.from_file(str(ROOT / "src" / "streamlit_app.py")).run(
            timeout=30
        )

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.sidebar.radio[0].value, CONTROLLED_MODE)
        self.assertEqual(app.button[0].label, "Analyze controlled scene")
        self.assertEqual(len(app.checkbox), 1)
        self.assertEqual(len(app.selectbox), 1)

    def test_experimental_mode_has_no_target_bin_controls(self) -> None:
        app = AppTest.from_file(str(ROOT / "src" / "streamlit_app.py")).run(
            timeout=30
        )
        app.sidebar.radio[0].set_value(EXPERIMENTAL_MODE).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.sidebar.radio[0].value, EXPERIMENTAL_MODE)
        self.assertEqual(app.button[0].label, "Run experimental detector")
        self.assertEqual(len(app.checkbox), 0)
        self.assertEqual(len(app.selectbox), 0)
        visible_warnings = " ".join(warning.value for warning in app.warning)
        self.assertIn("AP50 is 0.1846", visible_warnings)
        self.assertIn("No target-bin pass/fail", visible_warnings)


if __name__ == "__main__":
    unittest.main()
