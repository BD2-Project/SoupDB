"""El gestor se levanta con `python -m engine`.

Sin un punto de entrada el motor solo se puede usar escribiendo Python a mano y
el contenedor no sirve nada en su puerto, por lo que el frontend no puede
conectarse al motor real.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from engine.transactions import protocol as proto

ARRANQUE_MAX_S = 20


def _puerto_libre() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def gestor(tmp_path: Path):
    port = _puerto_libre()
    entorno = {
        **os.environ,
        "SOUPDB_DATA": str(tmp_path / "datos"),
        "DRIVER_HOST": "127.0.0.1",
        "DRIVER_PORT": str(port),
    }
    proceso = subprocess.Popen(
        [sys.executable, "-m", "engine"],
        env=entorno,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    limite = time.monotonic() + ARRANQUE_MAX_S
    while time.monotonic() < limite:
        if proceso.poll() is not None:
            pytest.fail(f"el gestor terminó al arrancar: {proceso.stdout.read()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proceso.kill()
        pytest.fail("el gestor no abrió el puerto a tiempo")

    yield port
    proceso.terminate()
    proceso.wait(timeout=10)


def _consulta(sock: socket.socket, sql: str):
    sock.sendall(proto.encode_frame(proto.OP_QUERY, proto.encode_query(sql)))
    opcode, payload = proto.recv_frame(sock)
    return opcode, payload


def test_serves_queries_over_tcp(gestor: int) -> None:
    with socket.create_connection(("127.0.0.1", gestor), timeout=5) as sock:
        opcode, _ = _consulta(sock, "CREATE TABLE lugares (id INT, ubicacion POINT)")
        assert opcode == proto.OP_OK

        _consulta(sock, "INSERT INTO lugares VALUES (1, POINT(-12.1211, -77.0298))")

        opcode, payload = _consulta(sock, "SELECT * FROM lugares")
        assert opcode == proto.OP_RESULT
        # El literal es POINT(latitud, longitud); se guarda como (x=lon, y=lat).
        assert proto.decode_resultset(payload).rows == ((1, (-77.0298, -12.1211)),)


def test_rejects_a_non_integer_port() -> None:
    resultado = subprocess.run(
        [sys.executable, "-m", "engine"],
        env={**os.environ, "DRIVER_PORT": "no-es-un-puerto"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode != 0
    assert "DRIVER_PORT" in resultado.stdout + resultado.stderr
