import asyncio
import json
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tinytuya
from tinytuya.core import AESCipher

from custom_components.tuya_local.local_webrtc.session import Session


def make_heartbeat_session():
    s = object.__new__(Session)
    s.config = SimpleNamespace(device_id="test_device", local_key="0123456789abcdef")
    s.sequence = 0
    s.send_lock = asyncio.Lock()
    s.heartbeat_interval = 0.001
    s.failure = asyncio.get_running_loop().create_future()
    s.header = {"sessionid": "test_session"}
    s.frames = []
    s.writer = SimpleNamespace(
        write=s.frames.append, drain=AsyncMock(), close=Mock(), wait_closed=AsyncMock()
    )
    s.pc = SimpleNamespace(close=AsyncMock())
    s.stun = SimpleNamespace(close=Mock())
    s.tasks = []
    return s


@pytest.mark.asyncio
async def test_heartbeat_bytes_match_tinytuya_reference():
    s = make_heartbeat_session()
    await s._send_packet(9, {"gwId": s.config.device_id, "devId": s.config.device_id})
    reference = tinytuya.Device(
        s.config.device_id, "127.0.0.1", s.config.local_key, version=3.3
    )
    expected = reference._encode_message(reference.generate_payload(9))
    assert s.frames[0] == expected
    payload = AESCipher(s.config.local_key.encode()).decrypt(
        s.frames[0][16:-8], False, decode_text=False, verify_padding=True
    )
    assert json.loads(payload) == {"gwId": "test_device", "devId": "test_device"}


@pytest.mark.asyncio
async def test_heartbeat_and_signalling_share_sequence_and_lock():
    s = make_heartbeat_session()
    await asyncio.gather(
        s._send_packet(9, {"gwId": "test_device", "devId": "test_device"}),
        s.send("candidate", {"mode": "webrtc"}),
    )
    headers = [struct.unpack(">IIII", f[:16]) for f in s.frames]
    assert [h[1] for h in headers] == [1, 2]
    assert {h[2] for h in headers} == {9, 32}


@pytest.mark.asyncio
async def test_close_cancels_heartbeat_and_sockets():
    s = make_heartbeat_session()
    s.tasks = [asyncio.create_task(s.heartbeat())]
    for _ in range(100):
        if s.frames:
            break
        await asyncio.sleep(0.001)
    assert s.frames
    await s.close()
    count = len(s.frames)
    await asyncio.sleep(0.01)
    assert len(s.frames) == count and s.tasks[0].done()
    s.writer.close.assert_called_once()
    s.stun.close.assert_called_once()
    s.pc.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_heartbeat_failure_does_not_expose_exception_payload():
    s = make_heartbeat_session()
    s.writer.drain = AsyncMock(side_effect=RuntimeError("private-rtc-secret"))
    await s.heartbeat()
    assert s.failure.result() == "LAN signalling heartbeat failed"


@pytest.mark.asyncio
async def test_empty_heartbeat_ack_does_not_interrupt_signalling():
    from custom_components.tuya_local.local_webrtc.lan import encode_frame

    session = make_heartbeat_session()
    session.reader = asyncio.StreamReader()
    encrypted = AESCipher(session.config.local_key.encode()).encrypt(
        json.dumps(
            {"header": {"sessionid": "test_session", "type": "disconnect"}, "msg": {}}
        ).encode(),
        False,
    )
    session.reader.feed_data(encode_frame(1, 9, struct.pack(">I", 0)))
    session.reader.feed_data(encode_frame(2, 32, struct.pack(">I", 0) + encrypted))
    session.reader.feed_eof()
    await session.receive()
    assert (
        session.failure.result()
        == "Camera closed the session (check RTC authentication)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,expected",
    [
        ("eof", "LAN signalling socket closed"),
        ("decode", "LAN signalling message rejected"),
    ],
)
async def test_safe_diagnostics_distinguish_eof_and_decode(case, expected):
    from custom_components.tuya_local.local_webrtc.lan import encode_frame

    session = make_heartbeat_session()
    session.reader = asyncio.StreamReader()
    if case == "decode":
        session.reader.feed_data(encode_frame(1, 32, b"\x00\x00\x00\x00bad-ciphertext"))
    session.reader.feed_eof()
    await session.receive()
    assert session.failure.result() == expected
