"""One fingerprint-verified LAN source shared by loopback JPEG / MPEG-TS HTTP."""

import asyncio
import io
import queue
import threading
import time
from contextlib import suppress
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import av

from .session import Session


class FrameHub:
    def __init__(
        self, *, queue_size=32, stale_after=2, clock=time.monotonic, max_clients=8
    ):
        self.queue_size, self.stale_after, self.clock = queue_size, stale_after, clock
        self.lock = threading.Lock()
        self.clients = set()
        self.max_clients = max_clients
        self.snapshot = None
        self.snapshot_at = 0

    def set_snapshot(self, data):
        with self.lock:
            self.snapshot, self.snapshot_at = data, self.clock()

    def get_snapshot(self):
        with self.lock:
            if (
                self.snapshot is None
                or self.clock() - self.snapshot_at > self.stale_after
            ):
                return None
            return self.snapshot

    def subscribe(self):
        client = queue.Queue(maxsize=self.queue_size)
        with self.lock:
            if len(self.clients) >= self.max_clients:
                return None
            self.clients.add(client)
        return client

    def unsubscribe(self, client):
        with self.lock:
            self.clients.discard(client)

    def _end(self, client):
        # A whole client is disconnected on overrun; never splice arbitrary TS bytes.
        while True:
            try:
                client.get_nowait()
            except queue.Empty:
                break
        client.put_nowait(None)

    def broadcast(self, data):
        with self.lock:
            for client in list(self.clients):
                try:
                    client.put_nowait(data)
                except queue.Full:
                    self.clients.discard(client)
                    self._end(client)

    def unavailable(self):
        with self.lock:
            self.snapshot = None
            self.snapshot_at = 0
            for client in self.clients:
                self._end(client)
            self.clients.clear()


class TransportEncoder:
    def __init__(self, emit):
        class Sink:
            def write(self, data):
                emit(bytes(data))
                return len(data)

        self.container = av.open(
            Sink(),
            mode="w",
            format="mpegts",
            options={
                "mpegts_flags": "resend_headers+initial_discontinuity",
                "flush_packets": "1",
                "pat_period": "0.1",
            },
        )
        self.stream = self.container.add_stream("libx264", rate=25)
        self.stream.width, self.stream.height = 640, 360
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = Fraction(1, 90000)
        self.stream.codec_context.time_base = Fraction(1, 90000)
        self.stream.codec_context.gop_size = 25
        self.stream.codec_context.max_b_frames = 0
        self.stream.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "crf": "23",
            "x264-params": "repeat-headers=1:keyint=25:min-keyint=25:scenecut=0",
        }
        self.first = self.previous = None

    def push(self, frame):
        if (
            (frame.width, frame.height) != (640, 360)
            or frame.pts is None
            or frame.time_base is None
        ):
            raise ValueError("Expected timestamped SD camera video")
        timestamp = Fraction(frame.pts) * frame.time_base
        if self.first is None:
            self.first = timestamp
        if self.previous is not None and timestamp <= self.previous:
            raise ValueError("Camera timestamps did not advance")
        self.previous = timestamp
        output = frame.reformat(format="yuv420p")
        output.pts = round((timestamp - self.first) * 90000)
        output.time_base = Fraction(1, 90000)
        for packet in self.stream.encode(output):
            self.container.mux(packet)

    def close(self):
        if self.container is not None:
            container, self.container = self.container, None
            try:
                for packet in self.stream.encode():
                    container.mux(packet)
            finally:
                container.close()


class LiveServer:
    def __init__(self, hub, bind="127.0.0.1", port=19845):
        self.hub = hub
        self._closed = False
        self._close_lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def reply(self, code, body, content_type="text/plain"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_GET(self):
                try:
                    if self.path == "/snapshot.jpg":
                        image = hub.get_snapshot()
                        self.reply(200, image, "image/jpeg") if image else self.reply(
                            503, b"Camera unavailable\n"
                        )
                    elif self.path == "/stream.ts":
                        if hub.get_snapshot() is None:
                            self.reply(503, b"Camera unavailable\n")
                            return
                        client = hub.subscribe()
                        if client is None:
                            self.reply(503, b"Too many streaming clients\n")
                            return
                        try:
                            self.send_response(200)
                            self.send_header("Content-Type", "video/mp2t")
                            self.send_header("Cache-Control", "no-store")
                            self.send_header("Connection", "close")
                            self.end_headers()
                            while True:
                                try:
                                    data = client.get(timeout=6)
                                except queue.Empty:
                                    break
                                if data is None:
                                    break
                                self.wfile.write(data)
                                self.wfile.flush()
                        finally:
                            hub.unsubscribe(client)
                            self.close_connection = True
                    else:
                        self.reply(404, b"Not found\n")
                except OSError, TimeoutError:
                    self.close_connection = True

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False

            def handle_error(self, *args):
                pass

        self.server = Server((bind, port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )

    def start(self):
        self.thread.start()

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.hub.unavailable()
            if self.thread.is_alive():
                self.server.shutdown()
            self.server.server_close()
            if self.thread.is_alive():
                self.thread.join(timeout=1)


class LiveService:
    def __init__(
        self,
        config,
        *,
        bind="127.0.0.1",
        port=19845,
        session_factory=Session,
        reconnect_min=1,
        reconnect_max=15,
    ):
        self.config, self.session_factory = config, session_factory
        self.reconnect_min, self.reconnect_max = reconnect_min, reconnect_max
        self.hub = FrameHub()
        self.http = LiveServer(self.hub, bind, port)
        self.frames_received = 0
        self.reconnects = 0
        self.last_error = None

    async def camera_loop(self):
        delay = self.reconnect_min
        while True:
            session = self.session_factory(self.config, "sd", 30)
            encoder = None
            try:
                encoder = TransportEncoder(self.hub.broadcast)
                last_jpeg = 0
                async for frame in session.frames():
                    if (frame.width, frame.height) != (640, 360):
                        raise ValueError("Unsupported resolution")
                    now = time.monotonic()
                    if now - last_jpeg >= 0.25:
                        data = io.BytesIO()
                        frame.to_image().save(data, format="JPEG", quality=90)
                        self.hub.set_snapshot(data.getvalue())
                        last_jpeg = now
                    encoder.push(frame)
                    self.frames_received += 1
                    delay = self.reconnect_min
                raise RuntimeError("Camera media ended")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.reconnects += 1
                # Only fixed categories leave this module, never exception payloads.
                self.last_error = {
                    "Camera startup or media stall timeout": "media_timeout",
                    "LAN signalling failed (check connection and credentials)": "signalling_lost",
                    "LAN signalling socket closed": "signalling_eof",
                    "LAN signalling heartbeat failed": "signalling_heartbeat",
                    "LAN signalling frame rejected": "signalling_frame",
                    "LAN signalling message rejected": "signalling_message",
                    "LAN signalling read failed": "signalling_read",
                    "LAN signalling decode failed": "signalling_decode",
                    "LAN signalling apply failed": "signalling_apply",
                    "Camera closed the session (check RTC authentication)": "camera_disconnect",
                    "WebRTC connection or certificate authentication failed": "peer_failed",
                    "Camera media ended before capture completed": "media_ended",
                    "Camera media ended": "media_ended",
                    "Resolution negotiation failed": "resolution_failed",
                    "Camera timestamps did not advance": "timestamp_invalid",
                    "Expected timestamped SD camera video": "timestamp_missing",
                    "Unsupported resolution": "resolution_changed",
                }.get(str(error), "media_error")
            finally:
                self.hub.unavailable()
                try:
                    if encoder:
                        with suppress(Exception):
                            encoder.close()
                finally:
                    with suppress(Exception):
                        await session.close()
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.reconnect_max)

    async def run(self, stop=None):
        self.http.start()
        task = asyncio.create_task(self.camera_loop())
        stop_task = asyncio.create_task(stop.wait()) if stop is not None else None
        try:
            if stop_task is None:
                await task
            else:
                done, _ = await asyncio.wait(
                    [task, stop_task], return_when=asyncio.FIRST_COMPLETED
                )
                if task in done:
                    await task
                    raise RuntimeError("Camera reader stopped unexpectedly")
        finally:
            task.cancel()
            if stop_task is not None:
                stop_task.cancel()
            await asyncio.gather(
                task,
                *([stop_task] if stop_task is not None else []),
                return_exceptions=True,
            )
            await asyncio.to_thread(self.http.close)
