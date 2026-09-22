import asyncio
import socket

import uvicorn

import chatgpt_web_provider.app as app_module
from chatgpt_web_provider.app import (
    SystemdReadyServer,
    _systemd_notify,
)


def test_systemd_notify_sends_readiness(
    tmp_path,
    monkeypatch,
):
    notify_path = (
        tmp_path / "notify.sock"
    )

    receiver = socket.socket(
        socket.AF_UNIX,
        socket.SOCK_DGRAM,
    )

    try:
        receiver.bind(
            str(notify_path)
        )
        receiver.settimeout(1.0)

        monkeypatch.setenv(
            "NOTIFY_SOCKET",
            str(notify_path),
        )

        _systemd_notify(
            "READY=1"
        )

        assert (
            receiver.recv(64)
            == b"READY=1"
        )

    finally:
        receiver.close()


def test_server_notifies_after_uvicorn_startup(
    monkeypatch,
):
    events = []

    async def fake_startup(
        self,
        sockets=None,
    ):
        events.append(
            "listener-ready"
        )
        self.started = True

    def fake_notify(message):
        events.append(message)

    monkeypatch.setattr(
        uvicorn.Server,
        "startup",
        fake_startup,
    )

    monkeypatch.setattr(
        app_module,
        "_systemd_notify",
        fake_notify,
    )

    config = uvicorn.Config(
        lambda scope, receive, send: None
    )

    server = SystemdReadyServer(
        config
    )

    asyncio.run(
        server.startup()
    )

    assert events == [
        "listener-ready",
        "READY=1",
    ]


from fastapi.testclient import TestClient

from chatgpt_web_provider.app import create_app
from chatgpt_web_provider.config import Settings


def test_app_lifespan_starts_backend():
    class StartTrackingBackend:
        def __init__(self):
            self.started = False

        async def start(self):
            self.started = True

    backend = StartTrackingBackend()

    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a"],
        available_levels=["high"],
    )

    app = create_app(
        settings,
        backend=backend,
    )

    assert backend.started is False

    with TestClient(app):
        assert backend.started is True
