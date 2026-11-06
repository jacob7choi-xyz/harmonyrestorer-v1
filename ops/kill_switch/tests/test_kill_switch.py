"""Tests for the kill-switch function (ADR 009 decision 14)."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from ops.kill_switch import handler

FUNCTION = "harmonyrestorer-backend"
QUALIFIER = "live"
ALARM = "harmonyrestorer-backend-busy"


class FakeClient:
    """Records update_function_url_config calls, or raises if told to."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._error = error

    def update_function_url_config(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return {"AuthType": kwargs["AuthType"]}


def _alarm_event(name: str = ALARM, state: str = "ALARM", source: str = "aws.cloudwatch") -> dict:
    """An event shaped like AWS's documented example of an alarm invoking a function."""
    return {
        "source": source,
        "alarmArn": f"arn:aws:cloudwatch:us-east-1:111122223333:alarm:{name}",
        "accountId": "111122223333",
        "time": "2023-08-04T12:36:15.490+0000",
        "region": "us-east-1",
        "alarmData": {
            "alarmName": name,
            "state": {
                "value": state,
                "reason": "test",
                "timestamp": "2023-08-04T12:36:15.490+0000",
            },
            "previousState": {"value": "OK"},
        },
    }


def _handle(event: dict, client: FakeClient) -> str:
    return handler.handle(event, client, FUNCTION, QUALIFIER, ALARM)


def test_the_alarm_entering_alarm_closes_the_url() -> None:
    """The configured alarm in state ALARM sets the alias's URL to AWS_IAM, once."""
    client = FakeClient()

    assert _handle(_alarm_event(), client) == "closed"
    assert client.calls == [
        {"FunctionName": FUNCTION, "Qualifier": QUALIFIER, "AuthType": "AWS_IAM"}
    ]


@pytest.mark.parametrize(
    "event",
    [
        _alarm_event(state="OK"),
        _alarm_event(state="INSUFFICIENT_DATA"),
        _alarm_event(name="some-other-alarm"),
        _alarm_event(source="aws.events"),
        {},
        {"source": "aws.cloudwatch", "alarmData": "not an object"},
        {"source": "aws.cloudwatch", "alarmData": {"alarmName": ALARM, "state": "ALARM"}},
    ],
    ids=[
        "state OK",
        "state INSUFFICIENT_DATA",
        "another alarm",
        "not from CloudWatch",
        "empty event",
        "alarmData not an object",
        "state not an object",
    ],
)
def test_other_events_leave_the_url_alone(event: dict) -> None:
    """Anything but the configured alarm in state ALARM makes no API call."""
    client = FakeClient()

    assert _handle(event, client) == "ignored"
    assert client.calls == []


def test_a_failed_update_raises() -> None:
    """A failed close is raised, so Lambda retries it instead of reporting success."""
    client = FakeClient(error=RuntimeError("AccessDenied"))

    with pytest.raises(RuntimeError, match="AccessDenied"):
        _handle(_alarm_event(), client)


@pytest.fixture
def configured(monkeypatch) -> FakeClient:
    """Set the function's environment and stand in for the runtime's boto3."""
    monkeypatch.setenv(handler.TARGET_FUNCTION_ENV, FUNCTION)
    monkeypatch.setenv(handler.TARGET_QUALIFIER_ENV, QUALIFIER)
    monkeypatch.setenv(handler.ALARM_NAME_ENV, ALARM)
    client = FakeClient()
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service: client if service == "lambda" else None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return client


def test_lambda_handler_uses_the_configured_target(configured: FakeClient) -> None:
    """The entry point reads its target from the environment and uses a Lambda client."""
    assert handler.lambda_handler(_alarm_event(), None) == "closed"
    assert configured.calls == [
        {"FunctionName": FUNCTION, "Qualifier": QUALIFIER, "AuthType": "AWS_IAM"}
    ]


@pytest.mark.parametrize(
    "missing", [handler.TARGET_FUNCTION_ENV, handler.TARGET_QUALIFIER_ENV, handler.ALARM_NAME_ENV]
)
def test_lambda_handler_refuses_to_run_without_a_setting(
    configured: FakeClient, monkeypatch, missing: str
) -> None:
    """A missing setting fails the invocation before any API call."""
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeError, match=f"{missing} is not set"):
        handler.lambda_handler(_alarm_event(), None)
    assert configured.calls == []
