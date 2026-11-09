"""Local IMM offer, candidate and authenticated STUN primitives."""

import asyncio
import hashlib
import hmac
import ipaddress
import socket
import struct
import time
import zlib


def build_offer(client, session, ufrag, password, aes_key, local_ip, port):
    sdp = "\r\n".join(
        (
            "v=0",
            f"o=- {int(time.time())} 1 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
            "a=group:BUNDLE imm0",
            f"a=msid-semantic: WMS {session}",
            "m=application 9 imm 6001",
            "c=IN IP4 0.0.0.0",
            "a=rtcp:9 IN IP4 0.0.0.0",
            f"a=ice-ufrag:{ufrag}",
            f"a=ice-pwd:{password}",
            "a=ice-options:trickle",
            f"a=aes-key:{aes_key.hex()}",
            "a=mid:imm0",
            "a=rtpmap:6001 AES/KCP 330",
            f"a=ssrc:0 cname:{client}",
            "",
        )
    )
    return {
        "sdp": sdp,
        "preconnect": True,
        "token": [{"urls": f"stun:{local_ip}:{port}"}],
    }


def matches_response(response, request):
    return (
        response.get("sessionid") == request["sessionid"]
        and response.get("from", request["to"]) == request["to"]
        and response.get("to", request["from"]) == request["from"]
    )


def host_candidate(value, camera_ip):
    if not isinstance(value, str):
        return False
    parts = value.strip().removeprefix("a=").removeprefix("candidate:").split()
    try:
        return (
            len(parts) >= 8
            and parts[2].lower() == "udp"
            and parts[6:8] == ["typ", "host"]
            and str(ipaddress.IPv4Address(parts[4])) == camera_ip
            and 1 <= int(parts[5]) <= 65535
        )
    except ValueError, IndexError:
        return False


class LocalStun(asyncio.DatagramProtocol):
    def __init__(self, camera_ip):
        self.camera_ip = camera_ip

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, peer):
        if (
            peer[0] != self.camera_ip
            or len(data) < 20
            or data[:2] != b"\x00\x01"
            or data[4:8] != b"\x21\x12\xa4\x42"
        ):
            return
        mapped = struct.pack(
            ">BBHI",
            0,
            1,
            peer[1] ^ 0x2112,
            int.from_bytes(socket.inet_aton(peer[0]), "big") ^ 0x2112A442,
        )
        body = struct.pack(">HH", 0x20, 8) + mapped
        self.transport.sendto(
            struct.pack(">HHI", 0x101, len(body), 0x2112A442) + data[8:20] + body, peer
        )


def attr(kind, value):
    return struct.pack(">HH", kind, len(value)) + value + b"\0" * (-len(value) % 4)


def stun_packet(kind, txid, attrs, password):
    # RFC 5389 sections 15.4/15.5: HMAC length includes MI, CRC includes FP.
    packet = struct.pack(">HHI", kind, len(attrs) + 24, 0x2112A442) + txid + attrs
    packet += attr(8, hmac.new(password.encode(), packet, hashlib.sha1).digest())
    packet = packet[:2] + struct.pack(">H", len(packet) - 20 + 8) + packet[4:]
    return packet + attr(
        0x8028, struct.pack(">I", (zlib.crc32(packet) & 0xFFFFFFFF) ^ 0x5354554E)
    )


def valid_integrity(packet, password):
    if len(packet) < 20 or packet[4:8] != b"\x21\x12\xa4\x42":
        return False
    limit = 20 + struct.unpack(">H", packet[2:4])[0]
    if len(packet) != limit:
        return False
    pos = 20
    while pos + 4 <= limit:
        kind, size = struct.unpack(">HH", packet[pos : pos + 4])
        if pos + 4 + size > limit:
            return False
        if kind == 8 and size == 20:
            signed = packet[:2] + struct.pack(">H", pos + 24 - 20) + packet[4:pos]
            return hmac.compare_digest(
                packet[pos + 4 : pos + 24],
                hmac.new(password.encode(), signed, hashlib.sha1).digest(),
            )
        pos += 4 + ((size + 3) & ~3)
    return False
