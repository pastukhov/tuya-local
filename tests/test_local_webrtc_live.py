# ruff: noqa: S310 - HTTP tests connect only to their own loopback server.
import asyncio
import io
import time
import urllib.error
import urllib.request
from fractions import Fraction

import av
import pytest

from custom_components.tuya_local.local_webrtc.live import (
    FrameHub,
    LiveServer,
    TransportEncoder,
)

pytestmark = pytest.mark.usefixtures("socket_enabled")


def test_http_readiness_and_staleness():
    hub = FrameHub(stale_after=0.05)
    server = LiveServer(hub, "127.0.0.1", 0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(base + "/snapshot.jpg")
        assert failure.value.code == 503
        hub.set_snapshot(b"jpeg")
        assert urllib.request.urlopen(base + "/snapshot.jpg").read() == b"jpeg"
        time.sleep(0.06)
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(base + "/snapshot.jpg")
        assert failure.value.code == 503
        hub.set_snapshot(b"new")
        hub.unavailable()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/snapshot.jpg")
    finally:
        server.close()


def test_bounded_queue_and_client_disconnect_isolation():
    hub = FrameHub(queue_size=2)
    one = hub.subscribe()
    two = hub.subscribe()
    hub.broadcast(b"first")
    hub.broadcast(b"second")
    assert one.qsize() == two.qsize() == 2
    hub.unsubscribe(one)
    two.get_nowait()
    hub.broadcast(b"last")
    assert two.get_nowait() == b"second"
    assert two.get_nowait() == b"last"
    slow = hub.subscribe()
    for _ in range(3):
        hub.broadcast(b"packet")
    assert slow.get_nowait() is None
    hub.unavailable()
    assert two.get_nowait() is None


def test_synthetic_transport_is_h264_decodable():
    chunks = []
    encoder = TransportEncoder(chunks.append)
    for n in range(50):
        frame = av.VideoFrame(640, 360, "yuv420p")
        frame.pts = n * 3600
        frame.time_base = Fraction(1, 90000)
        encoder.push(frame)
    encoder.close()
    data = b"".join(chunks)
    assert data and len(data) % 188 == 0
    with av.open(io.BytesIO(data), format="mpegts") as container:
        assert container.streams.video[0].codec_context.name == "h264"
        frames = list(container.decode(video=0))
        assert len(frames) == 50 and frames[0].width == 640


def test_shared_http_clients_and_shutdown():
    from custom_components.tuya_local.local_webrtc.live import LiveService

    async def run():
        created = []

        class FakeSession:
            def __init__(self, *args):
                self.closed = False
                created.append(self)

            async def frames(self):
                n = 0
                while True:
                    frame = av.VideoFrame(640, 360, "yuv420p")
                    frame.pts = n * 3600
                    frame.time_base = Fraction(1, 90000)
                    yield frame
                    n += 1
                    await asyncio.sleep(0.01)

            async def close(self):
                self.closed = True

        service = LiveService(None, port=0, session_factory=FakeSession)
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        try:
            for _ in range(100):
                if service.hub.get_snapshot():
                    break
                await asyncio.sleep(0.01)
            url = f"http://127.0.0.1:{service.http.port}/stream.ts"
            one = await asyncio.to_thread(urllib.request.urlopen, url)
            two = await asyncio.to_thread(urllib.request.urlopen, url)
            assert await asyncio.to_thread(one.read, 188)
            before = service.frames_received
            one.close()
            assert await asyncio.to_thread(two.read, 188)
            await asyncio.sleep(0.03)
            assert service.frames_received > before and len(created) == 1
            assert not created[0].closed
            two.close()
        finally:
            stop.set()
            await task
        assert created[0].closed and not service.http.thread.is_alive()

    asyncio.run(run())


def test_reconnect_closes_previous_session():
    from custom_components.tuya_local.local_webrtc.live import LiveService

    async def run():
        created = []

        class FakeSession:
            def __init__(self, *args):
                self.closed = False
                created.append(self)

            async def frames(self):
                if len(created) == 1:
                    raise RuntimeError("disconnect")
                n = 0
                while True:
                    frame = av.VideoFrame(640, 360, "yuv420p")
                    frame.pts = n * 3600
                    frame.time_base = Fraction(1, 90000)
                    yield frame
                    n += 1
                    await asyncio.sleep(0.01)

            async def close(self):
                self.closed = True

        service = LiveService(
            None,
            port=0,
            session_factory=FakeSession,
            reconnect_min=0.01,
            reconnect_max=0.02,
        )
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        for _ in range(100):
            if service.hub.get_snapshot():
                break
            await asyncio.sleep(0.01)
        assert len(created) == 2 and created[0].closed and service.reconnects == 1
        stop.set()
        await task
        assert all(s.closed for s in created)

    asyncio.run(run())


def test_constructor_failure_retries_and_shutdown_stops_http():
    from custom_components.tuya_local.local_webrtc.live import LiveService

    async def run():
        calls = 0

        def broken_factory(*args):
            nonlocal calls
            calls += 1
            raise RuntimeError("private initialization details")

        service = LiveService(
            None,
            port=0,
            session_factory=broken_factory,
            reconnect_min=0.001,
            reconnect_max=0.002,
        )
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        try:
            async with asyncio.timeout(1):
                while calls < 2:
                    await asyncio.sleep(0.001)
            assert not task.done()
            assert service.last_error == "media_error"
        finally:
            stop.set()
            await task
        assert not service.http.thread.is_alive()
        assert service.hub.get_snapshot() is None

    asyncio.run(run())


def test_listener_close_before_start_is_bounded_and_idempotent():
    server = LiveServer(FrameHub(), "127.0.0.1", 0)
    server.close()
    server.close()
    assert server.server.socket.fileno() == -1
    assert not server.thread.is_alive()


def test_failure_category_never_contains_exception_payload():
    from custom_components.tuya_local.local_webrtc.live import LiveService

    async def run():
        class FailedSession:
            def __init__(self, *args):
                pass

            async def frames(self):
                raise RuntimeError("private-rtc-auth")
                yield

            async def close(self):
                pass

        service = LiveService(
            None, port=0, session_factory=FailedSession, reconnect_min=10
        )
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        try:
            for _ in range(100):
                if service.reconnects:
                    break
                await asyncio.sleep(0.001)
            assert service.reconnects == 1
            assert service.last_error == "media_error"
            assert "private-rtc-auth" not in service.last_error
        finally:
            stop.set()
            await task

    asyncio.run(run())


def test_hd_transport_has_decodable_mono_aac_and_flushes_tail():
    chunks = []
    encoder = TransportEncoder(chunks.append)
    for n in range(25):
        frame = av.VideoFrame(1280, 720, "yuv420p")
        frame.pts, frame.time_base = n * 3600, Fraction(1, 90000)
        encoder.push(frame)
        audio = av.AudioFrame(format="s16", layout="mono", samples=640)
        for plane in audio.planes:
            plane.update(bytes(plane.buffer_size))
        audio.sample_rate = 16000
        audio.pts, audio.time_base = n * 640, Fraction(1, 16000)
        encoder.push(audio)
    encoder.close()
    encoder.close()
    data = b"".join(chunks)
    with av.open(io.BytesIO(data), format="mpegts") as container:
        assert container.streams.audio[0].codec_context.name == "aac"
        decoded = list(container.decode())
    videos = [f for f in decoded if isinstance(f, av.VideoFrame)]
    audios = [f for f in decoded if isinstance(f, av.AudioFrame)]
    assert len(videos) == 25
    assert all((f.width, f.height) == (1280, 720) for f in videos)
    assert audios and all(f.sample_rate == 48000 for f in audios)
    assert all(f.layout.name == "mono" for f in audios)
    assert 48000 <= sum(f.samples for f in audios) < 51000
    for frames in (videos, audios):
        times = [f.pts * f.time_base for f in frames]
        assert all(a < b for a, b in zip(times, times[1:], strict=False))


@pytest.mark.parametrize("timestamp", [0, -3600, 90000 * 31])
def test_transport_rejects_nonadvancing_or_unbounded_video(timestamp):
    encoder = TransportEncoder(lambda data: None)
    try:
        frame = av.VideoFrame(640, 360, "yuv420p")
        frame.pts, frame.time_base = 0, Fraction(1, 90000)
        encoder.push(frame)
        frame = av.VideoFrame(640, 360, "yuv420p")
        frame.pts, frame.time_base = timestamp, Fraction(1, 90000)
        with pytest.raises(ValueError, match="timestamps"):
            encoder.push(frame)
    finally:
        encoder.close()


def test_imm_factory_is_lazy_selected_and_media_closes_on_stop(monkeypatch):
    import sys
    from types import SimpleNamespace

    from custom_components.tuya_local.local_webrtc.live import LiveService

    async def run():
        created = []

        class IMMSession:
            def __init__(self, config):
                created.append(self)
                self.closed = False

            async def media_frames(self):
                for n in range(2):
                    video = av.VideoFrame(1280, 720, "yuv420p")
                    video.pts, video.time_base = n * 3600, Fraction(1, 90000)
                    yield video
                    audio = av.AudioFrame(format="s16", layout="mono", samples=640)
                    for plane in audio.planes:
                        plane.update(bytes(plane.buffer_size))
                    audio.pts, audio.time_base = n * 640, Fraction(1, 16000)
                    audio.sample_rate = 16000
                    yield audio
                await asyncio.Event().wait()

            async def close(self):
                self.closed = True

        monkeypatch.setitem(
            sys.modules,
            "custom_components.tuya_local.local_webrtc.imm",
            SimpleNamespace(IMMSession=IMMSession),
        )
        service = LiveService(SimpleNamespace(imm_auth="fixture"), port=0)
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        try:
            for _ in range(100):
                if service.frames_received == 2:
                    break
                await asyncio.sleep(0.01)
            assert service.frames_received == 2
            assert len(created) == 1
            assert service.hub.get_snapshot()
        finally:
            stop.set()
            await task
        assert created[0].closed
        assert not service.http.thread.is_alive()

    asyncio.run(run())


def test_audio_before_video_is_bounded_and_timestamps_are_checked():
    encoder = TransportEncoder(lambda data: None)
    try:
        for n in range(101):
            frame = av.AudioFrame(format="s16", layout="mono", samples=160)
            frame.sample_rate = 8000
            frame.pts, frame.time_base = n * 160, Fraction(1, 8000)
            if n == 100:
                with pytest.raises(ValueError, match="queue limit"):
                    encoder.push(frame)
            else:
                encoder.push(frame)
        assert len(encoder.pending_audio) == 100
    finally:
        encoder.close()
    assert encoder.pending_audio == []


def test_audio_timestamp_regression_is_rejected_per_track():
    encoder = TransportEncoder(lambda data: None)
    try:
        for n in range(2):
            frame = av.AudioFrame(format="s16", layout="mono", samples=160)
            frame.sample_rate = 8000
            frame.pts, frame.time_base = 0, Fraction(1, 8000)
            if n:
                with pytest.raises(ValueError, match="timestamps"):
                    encoder.push(frame)
            else:
                encoder.push(frame)
    finally:
        encoder.close()


@pytest.mark.parametrize("initial_video", [False, True])
def test_audio_cannot_keep_stalled_video_session_alive(initial_video):
    from custom_components.tuya_local.local_webrtc.live import LiveService

    async def run():
        created = []

        class AudioOnlySession:
            def __init__(self, *args):
                self.closed = False
                self.iterator_closed = False
                created.append(self)

            async def media_frames(self):
                try:
                    if initial_video:
                        video = av.VideoFrame(640, 360, "yuv420p")
                        video.pts, video.time_base = 0, Fraction(1, 90000)
                        yield video
                    n = 0
                    while True:
                        audio = av.AudioFrame(format="s16", layout="mono", samples=160)
                        audio.sample_rate = 8000
                        audio.pts, audio.time_base = n * 160, Fraction(1, 8000)
                        for plane in audio.planes:
                            plane.update(bytes(plane.buffer_size))
                        yield audio
                        n += 1
                        await asyncio.sleep(0.002)
                finally:
                    self.iterator_closed = True

            async def close(self):
                self.closed = True

        service = LiveService(
            None,
            port=0,
            session_factory=AudioOnlySession,
            reconnect_min=0.001,
            reconnect_max=0.001,
            video_stall_timeout=0.03,
            startup_timeout=0.03,
        )
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        try:
            async with asyncio.timeout(2):
                while len(created) < 2:
                    await asyncio.sleep(0.002)
            assert service.last_error == "media_timeout"
            assert created[0].closed and created[0].iterator_closed
            assert not task.done()
        finally:
            stop.set()
            await task
        assert all(s.closed and s.iterator_closed for s in created)

    asyncio.run(run())
