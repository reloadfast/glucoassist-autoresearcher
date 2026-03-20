"""Unit tests for the autoresearcher ML service (LLM mocked)."""
from __future__ import annotations

import json
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.autoresearcher import (
    CV_STEP_DAYS, CV_TRAIN_DAYS, CV_VAL_DAYS, READINGS_PER_DAY,
    LLMResponseError, LLMUnreachableError,
    _propose_experiment, _propose_experiment_openai,
    _should_promote, _walk_forward_cv,
)


class TestProposeExperiment:
    def test_returns_parsed_dict(self):
        payload = {
            "description": "Try LightGBM",
            "feature_config": {"lags": [1,2,3], "time_of_day": True, "roc_features": False, "garmin_features": False},
            "model_config": {"algorithm": "lightgbm", "n_estimators": 100},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": json.dumps(payload)}
        mock_resp.raise_for_status = MagicMock()
        with patch("app.services.autoresearcher.requests.post", return_value=mock_resp):
            result = _propose_experiment("prog", [], "http://localhost:11434", "llama3.1:8b")
        assert result["description"] == "Try LightGBM"
        assert result["feature_config"]["time_of_day"] is True

    def test_strips_markdown_fences(self):
        payload = {"description": "test", "feature_config": {}, "model_config": {}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": f"```json\n{json.dumps(payload)}\n```"}
        mock_resp.raise_for_status = MagicMock()
        with patch("app.services.autoresearcher.requests.post", return_value=mock_resp):
            result = _propose_experiment("p", [], "http://localhost:11434", "llama3.1:8b")
        assert result["description"] == "test"

    def test_raises_on_connection_error(self):
        import requests as req_lib
        with patch("app.services.autoresearcher.requests.post",
                   side_effect=req_lib.exceptions.ConnectionError("refused")):
            with pytest.raises(LLMUnreachableError):
                _propose_experiment("p", [], "http://localhost:11434", "llama3.1:8b")

    def test_includes_log_context(self):
        payload = {"description": "t", "feature_config": {}, "model_config": {}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": json.dumps(payload)}
        mock_resp.raise_for_status = MagicMock()
        with patch("app.services.autoresearcher.requests.post", return_value=mock_resp) as mp:
            _propose_experiment("prog", [{"description": "prev exp"}], "http://localhost:11434", "llama3.1:8b")
            assert "prev exp" in mp.call_args[1]["json"]["prompt"]


class TestProposeExperimentOpenAI:
    def test_returns_parsed_dict(self):
        payload = {"description": "ROC features", "feature_config": {"roc_features": True}, "model_config": {}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(payload)}}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("app.services.autoresearcher.requests.post", return_value=mock_resp):
            result = _propose_experiment_openai("prog", [], "https://api.openai.com", "sk-t", "gpt-4o")
        assert result["feature_config"]["roc_features"] is True

    def test_raises_on_connection_error(self):
        import requests as req_lib
        with patch("app.services.autoresearcher.requests.post",
                   side_effect=req_lib.exceptions.ConnectionError("refused")):
            with pytest.raises(LLMUnreachableError):
                _propose_experiment_openai("p", [], "https://api.openai.com", "sk-t", "gpt-4o")

    def test_sends_bearer_header(self):
        payload = {"description": "t", "feature_config": {}, "model_config": {}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(payload)}}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("app.services.autoresearcher.requests.post", return_value=mock_resp) as mp:
            _propose_experiment_openai("p", [], "https://api.openai.com", "sk-mykey", "gpt-4o")
            assert mp.call_args[1]["headers"]["Authorization"] == "Bearer sk-mykey"


class TestWalkForwardCV:
    def _make_feats(self):
        n = (CV_TRAIN_DAYS + CV_VAL_DAYS + CV_STEP_DAYS) * READINGS_PER_DAY + 50
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        return pd.DataFrame(
            {"lag_1": rng.normal(120,20,n), "lag_2": rng.normal(120,20,n),
             "lag_3": rng.normal(120,20,n), "glucose": rng.normal(120,20,n)},
            index=idx,
        )

    def test_lightgbm_no_feature_name_warning(self):
        pytest.importorskip("lightgbm")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = _walk_forward_cv(self._make_feats(), {"algorithm": "lightgbm", "n_estimators": 10})
        assert all(v < float("inf") for v in result.values())

    def test_ridge_returns_finite_maes(self):
        result = _walk_forward_cv(self._make_feats(), {"algorithm": "ridge", "alpha": 1.0})
        assert set(result.keys()) == {"30", "60", "90", "120"}
        assert all(v < float("inf") for v in result.values())


class TestShouldPromote:
    def test_promotes_when_all_improve(self):
        assert _should_promote({"30":15.0,"60":23.0,"90":26.0,"120":27.0},
                                {"30":17.74,"60":25.18,"90":27.73,"120":28.60}) is True

    def test_rejects_when_one_regresses(self):
        assert _should_promote({"30":15.0,"60":23.0,"90":26.0,"120":29.0},
                                {"30":17.74,"60":25.18,"90":27.73,"120":28.60}) is False

    def test_rejects_when_equal(self):
        b = {"30":17.74,"60":25.18,"90":27.73,"120":28.60}
        assert _should_promote(b, b) is False
