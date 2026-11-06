"""Closes the backend's function URL to the public when its cost alarm fires.

ADR 009 decision 14. A CloudWatch alarm on the backend function's summed Duration
invokes this function asynchronously. It sets the function URL's authentication to
AWS_IAM. The URL's public statement is written with a condition that the authentication
is NONE, so it is expected to stop applying; ADR 009 precondition 7 checks this on the
deployed function. An operator reopens the URL by setting the authentication back to NONE.

This file is deployed on its own to a Lambda Python runtime, which provides boto3.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CLOSED_AUTH_TYPE = "AWS_IAM"
ALARM_SOURCE = "aws.cloudwatch"
ALARM_STATE = "ALARM"

# The settings the deployed function must have, read in lambda_handler
TARGET_FUNCTION_ENV = "TARGET_FUNCTION"
TARGET_QUALIFIER_ENV = "TARGET_QUALIFIER"
ALARM_NAME_ENV = "ALARM_NAME"


class FunctionUrlClient(Protocol):
    """The one Lambda API call this function makes."""

    def update_function_url_config(self, **kwargs: Any) -> dict[str, Any]:
        """Update a function URL's configuration."""
        ...


def handle(
    event: dict[str, Any],
    client: FunctionUrlClient,
    function_name: str,
    qualifier: str,
    alarm_name: str,
) -> str:
    """Close the function URL if *event* is the expected alarm entering ALARM.

    Args:
        event: The payload Lambda passes to the function.
        client: A Lambda client.
        function_name: The backend function whose URL is closed.
        qualifier: The alias the URL belongs to.
        alarm_name: The only alarm this function acts on.

    Returns:
        "closed" when the URL was set to AWS_IAM, otherwise "ignored".

    Raises:
        Whatever the client raises, so that Lambda's retries for asynchronous
        invocations apply instead of a failed close being reported as done.
    """
    alarm_data = event.get("alarmData")
    received_name = alarm_data.get("alarmName") if isinstance(alarm_data, dict) else None
    state = alarm_data.get("state") if isinstance(alarm_data, dict) else None
    state_value = state.get("value") if isinstance(state, dict) else None

    if event.get("source") != ALARM_SOURCE or received_name != alarm_name:
        logger.warning("Ignoring an event that is not from alarm %s", alarm_name)
        return "ignored"
    if state_value != ALARM_STATE:
        logger.info("Alarm %s is in state %s; leaving the URL as it is", alarm_name, state_value)
        return "ignored"

    client.update_function_url_config(
        FunctionName=function_name, Qualifier=qualifier, AuthType=CLOSED_AUTH_TYPE
    )
    logger.warning(
        "Alarm %s fired: function URL of %s:%s set to %s, public requests are refused",
        alarm_name,
        function_name,
        qualifier,
        CLOSED_AUTH_TYPE,
    )
    return "closed"


def _required_env(name: str) -> str:
    """Return the environment variable *name*, or raise if it is unset or empty."""
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def lambda_handler(event: dict[str, Any], context: object) -> str:
    """Entry point for the Lambda runtime.

    Args:
        event: The alarm's payload.
        context: The Lambda context; unused.

    Returns:
        The result of handle().
    """
    function_name = _required_env(TARGET_FUNCTION_ENV)
    qualifier = _required_env(TARGET_QUALIFIER_ENV)
    alarm_name = _required_env(ALARM_NAME_ENV)

    # The Lambda Python runtime provides boto3; the repository does not depend on it
    import boto3

    client = boto3.client("lambda")
    return handle(event, client, function_name, qualifier, alarm_name)
