"""Provider smoke is network-disabled unless the user explicitly opts in."""

from __future__ import annotations

import socket

import pytest

from evicon.provider_smoke import main


def test_default_provider_smoke_prints_network_disabled_without_network(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("default provider smoke attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)

    assert main([]) == 0
    assert capsys.readouterr().out.strip() == "network_disabled"


def test_provider_smoke_never_accepts_an_api_key_cli_argument() -> None:
    with pytest.raises(SystemExit):
        main(["--api-key", "forbidden"])
