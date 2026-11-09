"""Persistent authenticated local IMM HD video and PCM microphone source.

No device connection starts before media_frames() is consumed. Close the
iterator (or call close()) when the subscriber leaves. Errors contain fixed
labels only; credentials and remote payloads never enter diagnostics.
"""

import asyncio
import contextlib
import hashlib
import hmac
import json
import secrets
import struct
import time
import uuid
from fractions import Fraction

import av
from tinytuya.core import AESCipher

from . import imm_crypto as crypto
from . import imm_kcp_segment as kcp
from . import imm_signalling as ice
from . import imm_signalling as lan
from .imm_kcp_conversation import KcpConversation
from .lan import decode_signal, encode_frame, read_frame


class ProbeError(Exception):
    """Only fixed labels, never remote payloads or credentials."""


def command(request, cmd, sub, payload):
    return (
        struct.pack("<IIIHHI", 0x12345678, request, 0, cmd, sub, len(payload)) + payload
    )


def auth_packet(credential):
    if (
        not isinstance(credential, str)
        or len(credential) != 32
        or any(c not in "0123456789abcdefABCDEF" for c in credential)
    ):
        raise ProbeError("invalid_credential")
    return (
        struct.pack("<II", 0x12345678, 1)
        + b"admin".ljust(32, b"\0")
        + credential.encode().ljust(32, b"\0")
        + bytes(32)
    )


def is_imm_answer(lines):
    return any(
        line in ("m=application 9 imm 6001", "m=application 9 tuya 6001")
        for line in lines
    ) and any(
        line in ("a=rtpmap:6001 AES/KCP 3", "a=rtpmap:6001 AES/KCP 330")
        for line in lines
    )


def require_equal_keys(answer_key, offer_key):
    # The captured JVS phone session used identical keys. That observation cannot
    # resolve directional-key disagreement between the LAN and relay references.
    if not hmac.compare_digest(answer_key, offer_key):
        raise ProbeError("key_direction_unverified")


def bounded_kcp_input(conversation, segment):
    """Bound the reference receiver's retained fragments before every delivery."""

    def pending_bytes():
        return sum(map(len, conversation._fragments)) + sum(
            len(s.data) for s in conversation._received.values()
        )

    if pending_bytes() + len(segment.data) > 4 << 20:
        raise ProbeError("reassembly_limit")
    conversation.input(segment)
    if pending_bytes() > 4 << 20:
        raise ProbeError("reassembly_limit")


def verified_segments(data, key):
    if len(data) < 44 or not hmac.compare_digest(
        hmac.new(key, data[:-20], hashlib.sha1).digest(), data[-20:]
    ):
        raise ProbeError("hmac_rejected")
    body, result, pos = data[:-20], [], 0
    while pos < len(body):
        segment = kcp.parse_segment(body[pos:])
        if (
            segment is None
            or segment.command not in (81, 82, 83, 84)
            or len(segment.data) > 65535
        ):
            raise ProbeError("kcp_rejected")
        result.append(segment)
        pos += 24 + len(segment.data)
    return result


def _valid_short_video(payload):
    """Validate H264 syntax boundaries before accepting the observed short wrapper."""
    if not payload or len(payload) > 4 << 20:
        return False
    if payload.startswith((b"\0\0\0\1", b"\0\0\1")):
        nals = payload.replace(b"\0\0\0\1", b"\0\0\1").split(b"\0\0\1")[1:]
        return all(nal and not nal[0] & 0x80 and 1 <= nal[0] & 31 <= 23 for nal in nals)
    if payload[0] & 0x80:
        return False
    kind = payload[0] & 31
    if 1 <= kind <= 23:
        return True
    if kind == 28:
        return (
            len(payload) >= 3
            and not payload[1] & 0x20
            and payload[1] & 0xC0 != 0xC0
            and 1 <= payload[1] & 31 <= 23
        )
    if kind == 24:
        pos = 1
        while pos + 2 <= len(payload):
            size = int.from_bytes(payload[pos : pos + 2], "big")
            pos += 2
            if (
                not size
                or pos + size > len(payload)
                or not 1 <= payload[pos] & 31 <= 23
                or not _valid_short_video(payload[pos : pos + size])
            ):
                return False
            # STAP may contain only single NAL units, never nested packets.
            if not 1 <= payload[pos] & 31 <= 23:
                return False
            pos += size
        return pos == len(payload) and pos > 1
    return False


def media_payload(plain):
    """Validate the observed long envelope or marked short H264 envelope."""
    kind = int.from_bytes(plain[:4], "little")
    if kind not in (0x10003, 0x10005):
        return None
    if len(plain) < 40:
        raise ProbeError("media_layout_rejected")
    sub = plain[32:]
    length = int.from_bytes(sub[:4], "little")
    if len(sub) >= 16 and length >= 12 and length - 12 == len(sub) - 16:
        return kind, sub[16:]
    # Live camera also sends timestamp + big-endian marker 10 + H264.
    # It has no length prefix. Accept it only for structurally valid video.
    if kind == 0x10003 and sub[4:8] == b"\0\0\0\x0a" and _valid_short_video(sub[8:]):
        return kind, sub[8:]
    raise ProbeError("media_layout_rejected")


class NalAssembler:
    def __init__(self):
        self.pending = None

    def push(self, payload):
        if not payload:
            return []
        if payload.startswith((b"\0\0\0\1", b"\0\0\1")):
            return [payload]
        kind = payload[0] & 31
        if 1 <= kind <= 23:
            return [b"\0\0\0\1" + payload]
        if kind == 24:
            result, pos = [], 1
            while pos + 2 <= len(payload):
                size = int.from_bytes(payload[pos : pos + 2], "big")
                pos += 2
                if not size or pos + size > len(payload):
                    raise ProbeError("nal_rejected")
                result.append(b"\0\0\0\1" + payload[pos : pos + size])
                pos += size
            if pos != len(payload):
                raise ProbeError("nal_rejected")
            return result
        if kind == 28 and len(payload) >= 2:
            if payload[1] & 0x80:
                self.pending = bytearray(
                    b"\0\0\0\1" + bytes([(payload[0] & 0xE0) | (payload[1] & 31)])
                )
            if self.pending is None:
                return []
            self.pending.extend(payload[2:])
            if len(self.pending) > 4 << 20:
                raise ProbeError("nal_limit")
            if payload[1] & 0x40:
                result, self.pending = bytes(self.pending), None
                return [result]
        return []


class MediaUDP(asyncio.DatagramProtocol):
    def __init__(self, ip, password):
        self.ip, self.password = ip, password
        self.peer = self.remote_password = None
        self.queue = asyncio.Queue(1024)
        self.ready = asyncio.Event()
        self.transactions = set()
        self.overflow = False

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, peer):
        if peer[0] != self.ip or (self.peer is not None and peer != self.peer):
            return
        if len(data) >= 20 and data[4:8] == b"\x21\x12\xa4\x42":
            kind = int.from_bytes(data[:2], "big")
            if kind == 1 and ice.valid_integrity(data, self.password):
                mapped = struct.pack(
                    ">BBHI",
                    0,
                    1,
                    peer[1] ^ 0x2112,
                    int.from_bytes(__import__("socket").inet_aton(peer[0]), "big")
                    ^ 0x2112A442,
                )
                self.transport.sendto(
                    ice.stun_packet(
                        0x101, data[8:20], ice.attr(0x20, mapped), self.password
                    ),
                    peer,
                )
                if self.peer == peer:
                    self.ready.set()
            elif (
                kind == 0x101
                and self.remote_password
                and data[8:20] in self.transactions
                and ice.valid_integrity(data, self.remote_password)
            ):
                self.ready.set()
            return
        if self.peer != peer:
            return
        try:
            self.queue.put_nowait(data)
        except asyncio.QueueFull:
            self.overflow = True

    def check(self, ufrag, remote_ufrag, password):
        self.remote_password = password
        txid = secrets.token_bytes(12)
        self.transactions.add(txid)
        attrs = (
            ice.attr(6, (remote_ufrag + ":" + ufrag).encode())
            + ice.attr(0x24, struct.pack(">I", 1862270975))
            + ice.attr(0x8029, secrets.token_bytes(8))
        )
        self.transport.sendto(ice.stun_packet(1, txid, attrs, password), self.peer)


async def _run(config, credential, emit):
    result = {
        "hmac_verified": 0,
        "records": 0,
        "video_records": 0,
        "decoded_frames": 0,
        "dimensions": [],
    }
    writer = discovery = media = None
    tasks, conversations, files = [], {}, []
    header = {
        "from": uuid.uuid4().hex,
        "to": config.device_id,
        "moto_id": "",
        "path": "lan",
        "sessionid": uuid.uuid4().hex,
        "trace_id": str(uuid.uuid4()).upper(),
        "is_pre": 0,
        "p2p_skill": 1635,
        "security_level": 3,
    }
    lock, sequence, control_queue = asyncio.Lock(), 0, asyncio.Queue(64)

    async def packet(cmd, obj):
        nonlocal sequence
        async with lock:
            sequence += 1
            encrypted = AESCipher(config.local_key.encode()).encrypt(
                json.dumps(obj, separators=(",", ":")).encode(), False
            )
            writer.write(encode_frame(sequence, cmd, encrypted))
            await writer.drain()

    async def send(kind, msg):
        await asyncio.wait_for(
            packet(32, {"header": dict(header, type=kind), "msg": msg}), 2
        )

    async def heartbeat():
        while True:
            await asyncio.sleep(10)
            await asyncio.wait_for(
                packet(9, {"gwId": config.device_id, "devId": config.device_id}), 2
            )

    async def run():
        nonlocal writer, discovery, media
        reader, writer = await asyncio.open_connection(config.ip, 6668)
        local_ip = writer.get_extra_info("sockname")[0]
        loop = asyncio.get_running_loop()
        discovery, _ = await loop.create_datagram_endpoint(
            lambda: lan.LocalStun(config.ip), local_addr=(local_ip, 0)
        )
        local_ufrag, local_pwd, offer_key = (
            secrets.token_hex(8),
            secrets.token_hex(16),
            secrets.token_bytes(16),
        )
        media, protocol = await loop.create_datagram_endpoint(
            lambda: MediaUDP(config.ip, local_pwd), local_addr=(local_ip, 0)
        )
        await send(
            "offer",
            lan.build_offer(
                header["from"],
                header["sessionid"],
                local_ufrag,
                local_pwd,
                offer_key,
                local_ip,
                discovery.get_extra_info("sockname")[1],
            ),
        )
        await send(
            "candidate",
            {
                "candidate": f"a=candidate:1 1 UDP 2130706431 {local_ip} {media.get_extra_info('sockname')[1]} typ host\r\n",
                "token": [],
            },
        )
        tasks.append(asyncio.create_task(heartbeat()))
        answer, candidate = None, None
        deadline = loop.time() + 15
        while answer is None or candidate is None:
            cmd, body = await asyncio.wait_for(
                read_frame(reader), max(0.01, deadline - loop.time())
            )
            if cmd != 32:
                continue
            obj = decode_signal(body, config.local_key.encode())
            if not lan.matches_response(obj["header"], header):
                continue
            kind, msg = obj["header"].get("type"), obj["msg"]
            if kind == "disconnect":
                raise ProbeError("signalling_refused")
            if kind == "answer":
                lines = msg.get("sdp", "").splitlines()
                if not is_imm_answer(lines):
                    raise ProbeError("not_imm")
                answer = dict(
                    line[2:].split(":", 1)
                    for line in lines
                    if line.startswith(("a=aes-key:", "a=ice-pwd:", "a=ice-ufrag:"))
                )
                result["answer_received"] = True
                for line in lines:
                    if lan.host_candidate(line, config.ip):
                        candidate = line
            elif kind == "candidate" and lan.host_candidate(
                msg.get("candidate"), config.ip
            ):
                candidate = msg["candidate"]
        try:
            key = bytes.fromhex(answer["aes-key"])
        except Exception:
            raise ProbeError("answer_rejected") from None
        if (
            len(key) != 16
            or not 1 <= len(answer.get("ice-pwd", "")) <= 256
            or not 1 <= len(answer.get("ice-ufrag", "")) <= 256
        ):
            raise ProbeError("answer_rejected")
        result["offer_answer_keys_equal"] = hmac.compare_digest(key, offer_key)
        require_equal_keys(key, offer_key)
        parts = candidate.split()
        protocol.peer = (config.ip, int(parts[5]))
        protocol.check(local_ufrag, answer["ice-ufrag"], answer["ice-pwd"])
        await asyncio.wait_for(protocol.ready.wait(), 8)
        result["ice_verified"] = True

        def transmit(segment):
            media.sendto(
                segment + hmac.new(key, segment, hashlib.sha1).digest(), protocol.peer
            )

        def get_conversation(number):
            if number not in conversations:
                if len(conversations) >= 8:
                    raise ProbeError("conversation_limit")
                conversations[number] = KcpConversation(number, transmit)
                conversations[number].set_message_handler(
                    lambda record, n=number: on_record(n, record)
                )
            return conversations[number]

        decoder, assembler = av.CodecContext.create("h264", "r"), NalAssembler()
        video_index = audio_index = 0
        origin = None
        startup = True

        def on_record(number, record):
            nonlocal video_index, audio_index, origin
            if len(record) > 4 << 20:
                raise ProbeError("record_limit")
            try:
                plain = crypto.decrypt_record(key, record)
            except Exception:
                raise ProbeError("cbc_rejected") from None
            result["records"] += 1
            if plain.startswith(b"\x78\x56\x34\x12"):
                if number != 0 or len(plain) < 8:
                    raise ProbeError("control_rejected")
                if not startup:
                    return
                try:
                    control_queue.put_nowait(plain)
                except asyncio.QueueFull:
                    raise ProbeError("control_limit") from None
                return
            parsed = media_payload(plain)
            if parsed is None:
                return
            kind, payload = parsed
            if origin is None:
                origin = time.monotonic()
                audio_index = 0
            if kind == 0x10003:
                result["video_records"] += 1
                for nal in assembler.push(payload):
                    try:
                        for packet_ in decoder.parse(nal):
                            for frame in decoder.decode(packet_):
                                video_index = max(
                                    video_index + 1,
                                    int((time.monotonic() - origin) * 90000),
                                )
                                frame.pts, frame.time_base = (
                                    video_index,
                                    Fraction(1, 90000),
                                )
                                emit(frame)
                    except av.error.InvalidDataError:
                        result["decode_errors"] = result.get("decode_errors", 0) + 1
            else:
                if (
                    plain[24] != 2
                    or int.from_bytes(plain[26:28], "little") != 0
                    or plain[30] != 1
                    or len(payload) % 2
                ):
                    raise ProbeError("unsupported_audio")
                frame = av.AudioFrame(
                    format="s16", layout="mono", samples=len(payload) // 2
                )
                frame.sample_rate = 8000
                frame.planes[0].update(payload)
                frame.pts, frame.time_base = audio_index, Fraction(1, 8000)
                audio_index += frame.samples
                emit(frame)

        async def receive_udp():
            while True:
                if any(c.failure for c in conversations.values()):
                    raise ProbeError("kcp_delivery_failed")
                try:
                    data = await asyncio.wait_for(protocol.queue.get(), 1)
                except TimeoutError:
                    if any(c.failure for c in conversations.values()):
                        raise ProbeError("kcp_delivery_failed") from None
                    continue
                if protocol.overflow:
                    raise ProbeError("udp_queue_limit")
                segments = verified_segments(data, key)
                for segment in segments:
                    result["hmac_verified"] += 1
                    bounded_kcp_input(get_conversation(segment.conversation), segment)

        async def drain_signals():
            while True:
                cmd, body = await read_frame(reader)
                if cmd == 32:
                    obj = decode_signal(body, config.local_key.encode())
                    if (
                        lan.matches_response(obj["header"], header)
                        and obj["header"].get("type") == "disconnect"
                    ):
                        raise ProbeError("signalling_refused")

        receive = asyncio.create_task(receive_udp())
        tasks.extend((receive, asyncio.create_task(drain_signals())))

        async def wait_ack():
            ack = asyncio.create_task(control_queue.get())
            try:
                done, _ = await asyncio.wait(
                    [ack, *tasks], timeout=5, return_when=asyncio.FIRST_COMPLETED
                )
                if ack not in done:
                    for task in done:
                        task.result()
                    raise ProbeError("auth_ack_timeout")
                reply = ack.result()
                # The captured app session has no separate type-1 login reply. The
                # first cmd-10 response only establishes control-channel continuation;
                # successful authentication is established later by decoded media.
                if (
                    len(reply) < 20
                    or int.from_bytes(reply[12:14], "little") != 10
                    or int.from_bytes(reply[16:20], "little") != len(reply) - 20
                ):
                    raise ProbeError("startup_ack_rejected")
                result["post_auth_control_received"] = True
                result["startup_ack_bytes"] = len(reply)
            finally:
                ack.cancel()
                await asyncio.gather(ack, return_exceptions=True)

        token = credential()
        auth = auth_packet(token)
        token = None
        get_conversation(0).send(crypto.encrypt_record(key, auth))
        auth = None
        await asyncio.sleep(0.12)
        get_conversation(0).send(
            crypto.encrypt_record(key, command(0, 10, 0, struct.pack("<I", 0x10000)))
        )
        await wait_ack()
        startup = False
        for request, cmd, sub, payload in (
            (2, 2, 0, bytes(4)),
            (0x10004, 9, 0, struct.pack("<II", 0, 4)),
            (0x10003, 6, 0, bytes(8)),
            (0x10005, 6, 4, struct.pack("<II", 0, 4)),
        ):
            get_conversation(0).send(
                crypto.encrypt_record(key, command(request, cmd, sub, payload))
            )
            await asyncio.sleep(0.12)
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        raise ProbeError("session_ended")

    try:
        await run()
    finally:

        async def cleanup():
            for task in tasks:
                task.cancel()
            if tasks:
                done, _ = await asyncio.wait(tasks, timeout=1)
                for task in done:
                    if not task.cancelled():
                        task.exception()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(
                        *(c.async_close() for c in conversations.values()),
                        return_exceptions=True,
                    ),
                    1,
                )
            if writer:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        send(
                            "disconnect", {"close_reason": 4, "close_reason_local": 0}
                        ),
                        1,
                    )
                writer.close()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(writer.wait_closed(), 1)

        try:
            await asyncio.wait_for(cleanup(), 5)
        except Exception:
            result["cleanup_failed"] = True
        finally:
            # A broken KCP closer, repeated cancellation or cleanup timeout must not
            # skip synchronous socket/file closure. No payload-derived errors escape.
            for conversation in conversations.values():
                with contextlib.suppress(Exception):
                    conversation.close()
            if writer:
                with contextlib.suppress(Exception):
                    writer.close()
            for resource in (media, discovery, *files):
                if resource:
                    with contextlib.suppress(Exception):
                        resource.close()


class IMMSession:
    """Single-consumer source; media_frames yields PyAV video and audio frames."""

    def __init__(self, config):
        auth_packet(config.imm_auth)
        self.config = config
        self._task = None
        self._queue = asyncio.Queue(64)
        self._consumed = False

    def _emit(self, frame):
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            raise ProbeError("media_queue_limit") from None

    async def media_frames(self):
        if self._consumed:
            raise ProbeError("already_consumed")
        self._consumed = True
        self._task = asyncio.create_task(
            _run(self.config, lambda: self.config.imm_auth, self._emit)
        )
        try:
            while True:
                item = asyncio.create_task(self._queue.get())
                try:
                    done, _ = await asyncio.wait(
                        (item, self._task),
                        timeout=20,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if self._task in done:
                        self._task.result()
                        raise ProbeError("session_ended")
                    if item not in done:
                        raise ProbeError("media_timeout")
                    yield item.result()
                finally:
                    item.cancel()
                    await asyncio.gather(item, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except ProbeError:
            raise
        except Exception:
            raise ProbeError("session_failed") from None
        finally:
            await self.close()

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        while not self._queue.empty():
            self._queue.get_nowait()

    frames = media_frames

    async def video_frames(self):
        source = self.media_frames()
        try:
            async for frame in source:
                if isinstance(frame, av.VideoFrame):
                    yield frame
        finally:
            await source.aclose()
