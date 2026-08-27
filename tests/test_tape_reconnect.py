from websockets.exceptions import InvalidHandshake, InvalidStatus

from beta_spy.live import TAPE_RETRY_ERRORS


def test_tradier_handshake_502_is_a_retry_error() -> None:
    assert issubclass(InvalidStatus, InvalidHandshake)
    assert issubclass(InvalidStatus, TAPE_RETRY_ERRORS)
