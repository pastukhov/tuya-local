"""Bounded Tuya 55aa / LAN 3.3 command 32 framing."""

import asyncio
import json
import struct
import zlib

from tinytuya.core import AESCipher


class ProtocolError(ValueError):
    pass


def encode_frame(sequence, command, payload):
    head = struct.pack(">IIII", 0x55AA, sequence, command, len(payload) + 8)
    return (
        head
        + payload
        + struct.pack(">II", zlib.crc32(head + payload) & 0xFFFFFFFF, 0xAA55)
    )


async def read_frame(reader):
    try:
        head = await reader.readexactly(16)
        magic, _, command, size = struct.unpack(">IIII", head)
        if magic != 0x55AA or not 8 <= size <= 65536:
            raise ProtocolError("Invalid LAN frame header")
        body = await reader.readexactly(size)
        crc, suffix = struct.unpack(">II", body[-8:])
        if suffix != 0xAA55 or crc != zlib.crc32(head + body[:-8]) & 0xFFFFFFFF:
            raise ProtocolError("Invalid LAN frame checksum or suffix")
        return command, body[:-8]
    except asyncio.IncompleteReadError:
        raise ProtocolError("Truncated LAN frame or closed connection") from None


def decode_signal(body, key):
    try:
        if len(body) < 4 or struct.unpack(">I", body[:4])[0] != 0:
            raise ValueError
        raw = body[4:]
        if raw.startswith(b"3.3"):
            raw = raw[15:]
        obj = json.loads(
            AESCipher(key).decrypt(raw, False, decode_text=False, verify_padding=True)
        )
        if (
            not isinstance(obj, dict)
            or not isinstance(obj.get("header"), dict)
            or not isinstance(obj.get("msg"), dict)
        ):
            raise ValueError
        return obj
    except Exception:
        raise ProtocolError("Invalid or unauthenticated LAN signal") from None
