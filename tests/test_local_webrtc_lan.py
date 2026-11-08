import asyncio
import struct

import pytest

from custom_components.tuya_local.local_webrtc.lan import (
    ProtocolError,
    encode_frame,
    read_frame,
)


def test_fragmentation():
    async def run():
        r = asyncio.StreamReader()
        packet = encode_frame(1, 32, b"hello")
        for b in packet:
            r.feed_data(bytes([b]))
        assert await read_frame(r) == (32, b"hello")

    asyncio.run(run())


def test_bad_crc():
    async def run():
        r = asyncio.StreamReader()
        p = bytearray(encode_frame(1, 32, b"hello"))
        p[-9] ^= 1
        r.feed_data(p)
        with pytest.raises(ProtocolError):
            await read_frame(r)

    asyncio.run(run())


def test_size_limit():
    async def run():
        r = asyncio.StreamReader()
        r.feed_data(struct.pack(">IIII", 0x55AA, 1, 32, 999999))
        with pytest.raises(ProtocolError):
            await read_frame(r)

    asyncio.run(run())


@pytest.mark.parametrize("body", [b"", b"\0" * 4 + b"garbage", b"\0\0\0\1"])
def test_invalid_encryption_safe(body):
    from custom_components.tuya_local.local_webrtc.lan import decode_signal

    with pytest.raises(ProtocolError, match="Invalid or unauthenticated"):
        decode_signal(body, b"1234567890123456")


def test_truncated():
    async def run():
        r = asyncio.StreamReader()
        r.feed_data(encode_frame(1, 32, b"hello")[:-2])
        r.feed_eof()
        with pytest.raises(ProtocolError, match="Truncated"):
            await read_frame(r)

    asyncio.run(run())


def test_wrong_key_rejected():
    from tinytuya.core import AESCipher

    from custom_components.tuya_local.local_webrtc.lan import decode_signal

    encrypted = AESCipher(b"1234567890123456").encrypt(b'{"header":{},"msg":{}}', False)
    with pytest.raises(ProtocolError):
        decode_signal(b"\0" * 4 + encrypted, b"6543210987654321")
