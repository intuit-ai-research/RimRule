import socket

_original_connect = socket.socket.connect


def _blocked_connect(*args, **kwargs):
    raise RuntimeError(
        "Unit tests must not make network calls. "
        "Use mocks or move this test to integration/."
    )


socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
