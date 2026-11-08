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


def test_camera_task_failure_stops_http_and_propagates():
    from custom_components.tuya_local.local_webrtc.live import LiveService

    async def run():
        def broken_factory(*args):
            raise RuntimeError("camera initialization failed")

        service = LiveService(None, port=0, session_factory=broken_factory)
        with pytest.raises(RuntimeError, match="initialization"):
            await asyncio.wait_for(service.run(asyncio.Event()), 0.5)
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
