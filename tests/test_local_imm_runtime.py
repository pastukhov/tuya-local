"""IMM runtime transport and lifecycle regression coverage."""

import asyncio
import hashlib
import hmac
from types import SimpleNamespace

import pytest

from custom_components.tuya_local.local_webrtc import imm


def test_authenticated_packet_and_tamper():
    key = b"x" * 16
    wire = imm.kcp.build_segment(0, 81, 0, 512, 0, 0, 0, b"data")
    signed = wire + hmac.new(key, wire, hashlib.sha1).digest()
    assert imm.verified_segments(signed, key)[0].data == b"data"
    with pytest.raises(imm.ProbeError, match="hmac_rejected"):
        imm.verified_segments(signed[:-1] + bytes([signed[-1] ^ 1]), key)


def test_crypto_roundtrip_rejects_bad_padding():
    key = b"x" * 16
    packet = imm.crypto.encrypt_record(key, b"hello")
    assert imm.crypto.decrypt_record(key, packet) == b"hello"
    with pytest.raises(ValueError):
        imm.crypto.decrypt_record(key, b"bad")


def test_kcp_out_of_order_duplicate_and_fragment_validation():
    conversation = imm.KcpConversation(0, lambda _: None)
    output = []
    conversation.set_message_handler(output.append)

    def segment(sn, fragment, payload):
        return imm.kcp.parse_segment(
            imm.kcp.build_segment(0, 81, fragment, 512, 0, sn, 0, payload)
        )

    conversation.input(segment(1, 0, b"b"))
    conversation.input(segment(0, 1, b"a"))
    conversation.input(segment(0, 1, b"a"))
    assert output == [b"ab"]
    conversation.input(segment(2, 2, b"c"))
    with pytest.raises(ValueError, match="fragment sequence"):
        conversation.input(segment(3, 0, b"d"))
    conversation.close()
    assert not conversation._fragments


def test_kcp_selective_ack_does_not_retire_missing_segment():
    async def run():
        c = imm.KcpConversation(0, lambda _: None)
        c.send(b"a")
        c.send(b"b")
        ack = imm.kcp.parse_segment(imm.kcp.build_segment(0, 82, 0, 512, 0, 1, 0))
        c.input(ack)
        assert [p.sequence for p in c._pending] == [0]
        await c.async_close()

    asyncio.run(run())


def test_nal_stap_fu_and_bounds():
    assembler = imm.NalAssembler()
    assert assembler.push(b"\x78\x00\x02\x67a") == [b"\0\0\0\1\x67a"]
    assert assembler.push(b"\x7c\x85ab") == []
    assert assembler.push(b"\x7c\x45cd") == [b"\0\0\0\1\x65abcd"]
    with pytest.raises(imm.ProbeError):
        assembler.push(b"\x78\x00\x08x")


def test_lazy_start_close_and_bounded_queue(monkeypatch):
    async def run():
        started, stopped = asyncio.Event(), asyncio.Event()

        async def fake_run(config, credential, emit):
            started.set()
            try:
                emit("frame")
                await asyncio.Future()
            finally:
                stopped.set()

        monkeypatch.setattr(imm, "_run", fake_run)
        session = imm.IMMSession(SimpleNamespace(imm_auth="a" * 32))
        source = session.media_frames()
        assert not started.is_set()
        assert await anext(source) == "frame"
        await source.aclose()
        assert stopped.is_set()
        assert session._task is None
        for _ in range(64):
            session._emit("frame")
        with pytest.raises(imm.ProbeError, match="media_queue_limit"):
            session._emit("frame")

    asyncio.run(run())


@pytest.mark.parametrize("kind", [0x10003, 0x10005])
def test_media_envelope_requires_observed_exact_layout(kind):
    prefix = kind.to_bytes(4, "little") + bytes(28)
    payload = b"media"
    sub = (len(payload) + 12).to_bytes(4, "little") + bytes(12) + payload
    assert imm.media_payload(prefix + sub) == (kind, payload)
    malformed = [
        prefix,
        prefix + bytes(8),
        prefix + sub[:-1],
        prefix + sub + b"extra",
        prefix + bytes(16) + payload,
        prefix + (len(payload) + 12).to_bytes(4, "little") + bytes(4) + payload,
    ]
    for plain in malformed:
        with pytest.raises(imm.ProbeError, match="^media_layout_rejected$"):
            imm.media_payload(plain)
    assert imm.media_payload(b"unrelated") is None


def test_retransmit_exception_marks_failure_and_redacts(monkeypatch, caplog):
    async def run():
        conversation = imm.KcpConversation(0, lambda _: None)
        conversation._pending = [object()]

        async def immediate_sleep(_):
            return None

        def fail():
            raise OSError("private credential and remote payload")

        monkeypatch.setattr(asyncio, "sleep", immediate_sleep)
        monkeypatch.setattr(conversation, "_retransmit_once", fail)
        await conversation._async_retransmit_loop()
        assert conversation.failure
        assert conversation._closed
        assert not conversation._pending

    with caplog.at_level("DEBUG"):
        asyncio.run(run())
    assert "IMM KCP retransmit failed" in caplog.text
    assert "private credential" not in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        b"\x67" + bytes(66),
        b"\0\0\0\1\x67abc",
        b"\x7c\x85abc",
        b"\x78\0\x02\x67a",
    ],
)
def test_observed_short_video_wrapper(payload):
    prefix = (0x10003).to_bytes(4, "little") + bytes(28)
    sub = (1426009913).to_bytes(4, "little") + b"\0\0\0\x0a" + payload
    assert imm.media_payload(prefix + sub) == (0x10003, payload)
    for marker in (b"\0\0\0\x09", b"bad!"):
        with pytest.raises(imm.ProbeError, match="media_layout_rejected"):
            imm.media_payload(prefix + sub[:4] + marker + payload)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\xffabc",
        b"\x00",
        b"\x7c\xc5a",
        b"\x7c\x85",
        b"\x78\0\x08a",
        b"\0\0\1",
        b"\x78\0\x03\x78ab",
    ],
)
def test_short_video_rejects_invalid_nal(payload):
    prefix = (0x10003).to_bytes(4, "little") + bytes(28)
    sub = bytes(4) + b"\0\0\0\x0a" + payload
    with pytest.raises(imm.ProbeError, match="media_layout_rejected"):
        imm.media_payload(prefix + sub)
    audio = (0x10005).to_bytes(4, "little") + bytes(28)
    with pytest.raises(imm.ProbeError, match="media_layout_rejected"):
        imm.media_payload(audio + sub)
