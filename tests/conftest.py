from __future__ import annotations

import pytest


_STALE_TEST = "test_session_bias_blocks_calls_and_extends_put_hold"


def pytest_collection_modifyitems(items):
    """Keep one historical assertion visible without letting it redefine policy.

    PR #6 replaced the old 40-minute bearish-entry expectation with the causal
    situation machine: before 90 minutes, directional entries require a confirmed
    tradeable impulse/failed-extreme structure. The old test is retained as a
    strict xfail rather than deleted. If it ever starts passing, CI fails with
    XPASS and forces an explicit policy reconciliation.
    """
    for item in items:
        if item.name == _STALE_TEST:
            item.add_marker(
                pytest.mark.xfail(
                    reason="superseded by PR #6 causal 90-minute/impulse situation policy",
                    strict=True,
                )
            )
