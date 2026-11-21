"""Behavior tests for the TradingAgents gateway shim.

Stdlib-only (the shim is stdlib-only); run with:
    python3 -m unittest discover -s tradingagents/tests -v

Covers the review-finding regressions:
  - final-decision reversal (last rating word in truncated text)
  - report adoption from another ticker's run
  - custom state dirs escaping the backed-up data root (CLI home bridge)
  - google key alias, unattended env defaults
  - follow-up turns ("show the report for NVDA") re-launching analyses
"""

import importlib.util
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

SHIM_PATH = Path(__file__).resolve().parent.parent / "shim" / "ta_shim.py"

_spec = importlib.util.spec_from_file_location("ta_shim", SHIM_PATH)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

_ENV_KEYS = (
    "HOME", "TA_MODE", "TA_LLM_PROVIDER", "TA_QUICK_MODEL", "TA_DEEP_MODEL",
    "TA_BACKEND_URL", "TA_API_KEY_RAW", "TA_OUTPUT_LANGUAGE",
    "TA_MAX_DEBATE_ROUNDS", "TA_MAX_RISK_ROUNDS", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "TA_BIN",
)


def date_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class ShimTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ta-shim-test-")
        self._old_data_root = m.DATA_ROOT
        self._old_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        m.DATA_ROOT = self._tmp

    def tearDown(self):
        m.DATA_ROOT = self._old_data_root
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _touch(self, rel: str, content: str = "x") -> str:
        """Create a file (mtime = now, i.e. inside this run's window)."""
        path = os.path.join(self._tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(content)
        return path

    def _run(self, ticker="AAPL", date=None) -> m.Run:
        run = m.Run(ticker, date or date_today())
        run.started_at = time.time()
        return run


# -- request parsing ------------------------------------------------------

class TestParseRequest(ShimTestCase):
    def test_bare_and_suffixed(self):
        self.assertEqual(m.parse_request("Analyze NVDA on 2026-11-20"),
                         {"ticker": "NVDA", "date": "2026-11-20"})
        self.assertEqual(m.parse_request("TATAMOTORS.NS please"),
                         {"ticker": "TATAMOTORS.NS", "date": date_today()})
        self.assertEqual(m.parse_request("600519.SS 2026-11-01"),
                         {"ticker": "600519.SS", "date": "2026-11-01"})
        self.assertEqual(m.parse_request("BTC-USD"),
                         {"ticker": "BTC-USD", "date": date_today()})

    def test_no_ticker(self):
        self.assertIsNone(m.parse_request("hello there"))
        self.assertIsNone(m.parse_request("THE"))


# -- decision extraction (regression: the reversal) ------------------------

class TestDecision(ShimTestCase):
    def test_header_rating_wins_over_body_words(self):
        """Header says Sell, the body is full of per-agent 'Buy' words —
        truncation or ordering must not flip the verdict."""
        body = ("- Rating: Sell\n\n" + "Buy BUY — moon. Final Decision: BUY\n" * 200)
        self._touch("logs/reports/AAPL_1/complete_report.md", body)
        run = self._run()
        m._collect_report(run)
        m._extract_decision(run)
        self.assertEqual(run.decision, "Sell")

    def test_no_authoritative_marker_means_no_decision(self):
        self._touch("logs/reports/AAPL_1/complete_report.md", "notes, no markers at all")
        run = self._run()
        m._collect_report(run)
        m._extract_decision(run)
        self.assertIsNone(run.decision)

    def test_pm_decision_line_at_tail(self):
        body = ("analysts disagree\n" * 300 + "\n## V. Portfolio Manager\n\nDecision: Hold\n")
        self._touch("logs/reports/AAPL_1/complete_report.md", body)
        run = self._run()
        m._collect_report(run)
        m._extract_decision(run)
        self.assertEqual(run.decision, "Hold")


# -- report selection (regression: foreign-ticker adoption) ----------------

class TestReportSelection(ShimTestCase):
    def test_foreign_ticker_report_is_rejected(self):
        """An AAPL run whose own report failed to save must NOT return the
        earlier NVDA report (its mtime is inside the grace window)."""
        self._touch("logs/reports/NVDA_20261118_000000/complete_report.md", "- Rating: Buy\n")
        self._touch("logs/AAPL/2026-11-19/reports/market_report.md", "market notes")
        run = self._run("AAPL", "2026-11-19")
        m._collect_report(run)
        # The NVDA report must be ignored; the run's own per-role report may
        # be used (it names the ticker and sits under the run's date dir).
        if run.report_path is not None:
            self.assertIn("AAPL", run.report_path)

    def test_only_foreign_reports_means_no_report(self):
        self._touch("logs/reports/NVDA_1/complete_report.md", "- Rating: Buy\n")
        run = self._run("AAPL")
        m._collect_report(run)
        self.assertIsNone(run.report_path)

    def test_same_ticker_complete_report_preferred(self):
        self._touch("logs/reports/AAPL_1/complete_report.md", "full")
        self._touch("logs/reports/AAPL_1/1_analysts/market.md", "market")
        run = self._run()
        m._collect_report(run)
        self.assertIsNotNone(run.report_path)
        self.assertIn("complete_report", run.report_path)


# -- CLI home bridge (regression: custom state dirs escape the backup) -----

class TestCliHome(ShimTestCase):
    def test_passthrough_when_default_layout(self):
        home = self._tmp
        os.makedirs(os.path.join(home, ".tradingagents"), exist_ok=True)
        m.DATA_ROOT = os.path.join(home, ".tradingagents")
        os.environ["HOME"] = home
        self.assertEqual(m._cli_home(), home)

    def test_bridge_for_custom_state_dir(self):
        custom = os.path.join(self._tmp, "custom-state")  # e.g. root-account /root/.openclaw
        os.makedirs(custom, exist_ok=True)
        m.DATA_ROOT = custom
        os.environ["HOME"] = "/does-not-matter"
        home = m._cli_home()
        self.assertNotEqual(home, "/does-not-matter")
        resolved = os.path.realpath(os.path.join(home, ".tradingagents"))
        self.assertEqual(resolved, os.path.realpath(custom))
        # The CLI writing through the bridge lands inside the state dir.
        probe = os.path.join(home, ".tradingagents", "memory", "probe.md")
        os.makedirs(os.path.dirname(probe), exist_ok=True)
        with open(probe, "w") as fh:
            fh.write("x")
        self.assertTrue(os.path.isfile(os.path.join(custom, "memory", "probe.md")))


# -- provider env translation ----------------------------------------------

class TestCliEnv(ShimTestCase):
    def test_google_key_alias(self):
        os.environ.update({"TA_MODE": "native", "TA_LLM_PROVIDER": "google",
                           "GEMINI_API_KEY": "gm-key", "TA_QUICK_MODEL": "g", "TA_DEEP_MODEL": "g"})
        os.environ.pop("GOOGLE_API_KEY", None)
        env = m.build_cli_env()
        self.assertEqual(env["GOOGLE_API_KEY"], "gm-key")
        self.assertEqual(env["TRADINGAGENTS_LLM_PROVIDER"], "google")

    def test_compatible_mode(self):
        os.environ.update({"TA_MODE": "compatible", "TA_LLM_PROVIDER": "",
                           "TA_BACKEND_URL": "http://x/v1", "TA_API_KEY_RAW": "k",
                           "TA_QUICK_MODEL": "q", "TA_DEEP_MODEL": "d"})
        env = m.build_cli_env()
        self.assertEqual(env["TRADINGAGENTS_LLM_PROVIDER"], "openai_compatible")
        self.assertEqual(env["TRADINGAGENTS_LLM_BACKEND_URL"], "http://x/v1")
        self.assertEqual(env["OPENAI_COMPATIBLE_API_KEY"], "k")

    def test_unattended_defaults(self):
        os.environ.update({"TA_MODE": "native", "TA_LLM_PROVIDER": "openai",
                           "TA_QUICK_MODEL": "gpt", "TA_DEEP_MODEL": "gpt"})
        env = m.build_cli_env()
        self.assertEqual(env["TRADINGAGENTS_OUTPUT_LANGUAGE"], "English")
        self.assertEqual(env["TRADINGAGENTS_MAX_DEBATE_ROUNDS"], "1")
        self.assertEqual(env["TRADINGAGENTS_MAX_RISK_ROUNDS"], "1")
        self.assertEqual(env["TRADINGAGENTS_NO_UPDATE_CHECK"], "1")


# -- follow-up turn resolution (regression: report queries re-running) -----

class TestResolveTurn(ShimTestCase):
    def test_matrix(self):
        P = m.parse_request

        # no current run
        self.assertEqual(m.resolve_turn("hello", P("hello"), None), "help")
        self.assertEqual(m.resolve_turn("analyze NVDA", P("analyze NVDA"), None), "start")

        # running run: always status, no second run
        running = self._run()
        running.status = "running"
        self.assertEqual(m.resolve_turn("progress", P("progress"), running), "status")
        self.assertEqual(m.resolve_turn("analyze TSLA", P("analyze TSLA"), running), "status")

        # finished run: the reviewer's scenario — same ticker, no date,
        # must NOT start another analysis
        done = self._run("NVDA", "2026-11-19")
        done.status = "done"
        self.assertEqual(m.resolve_turn("show the report for NVDA",
                                        P("show the report for NVDA"), done), "result")
        self.assertEqual(m.resolve_turn("NVDA?", P("NVDA?"), done), "result")
        self.assertEqual(m.resolve_turn("what was the verdict?", P("what was the verdict?"), done), "result")

        # same ticker + same explicit date -> result
        self.assertEqual(m.resolve_turn("NVDA 2026-11-19 report",
                                        P("NVDA 2026-11-19 report"), done), "result")

        # same ticker + different explicit date -> fresh run
        self.assertEqual(m.resolve_turn("analyze NVDA on 2026-11-18",
                                        P("analyze NVDA on 2026-11-18"), done), "start")

        # re-run keywords -> fresh run
        self.assertEqual(m.resolve_turn("re-run NVDA", P("re-run NVDA"), done), "start")
        self.assertEqual(m.resolve_turn("re-analyze NVDA", P("re-analyze NVDA"), done), "start")

        # different ticker -> fresh run
        self.assertEqual(m.resolve_turn("analyze TSLA", P("analyze TSLA"), done), "start")


if __name__ == "__main__":
    unittest.main(verbosity=2)
