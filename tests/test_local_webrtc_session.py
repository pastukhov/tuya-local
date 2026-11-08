import asyncio
from types import SimpleNamespace

import pytest

from custom_components.tuya_local.local_webrtc.session import (
    Session,
    SessionError,
    validate_legacy_certificate,
)


def test_stall_bounded():
    async def run():
        # Test the race/timeout primitive without opening a camera session.
        s = object.__new__(Session)
        s.failure = asyncio.get_running_loop().create_future()
        with pytest.raises(SessionError, match="timeout"):
            await s.guarded(asyncio.sleep(100), 0.01)
        s.failure.set_result("safe failure")
        with pytest.raises(SessionError, match="safe failure"):
            await s.guarded(asyncio.sleep(100), 0.01)

    asyncio.run(run())


def test_certificate_absent_rejected():
    states = []
    obj = SimpleNamespace(
        _ssl=SimpleNamespace(get_peer_certificate=lambda: None),
        _set_state=states.append,
    )
    validate_legacy_certificate(obj, SimpleNamespace(fingerprints=[]))
    assert states[0].name == "FAILED"


def test_cancel_during_disconnect_still_closes_resources():
    async def run():
        entered = asyncio.Event()
        closed = []

        class Writer:
            def close(self):
                closed.append("writer")

            async def wait_closed(self):
                pass

        class PC:
            async def close(self):
                closed.append("pc")

        class Stun:
            def close(self):
                closed.append("stun")

        s = object.__new__(Session)
        s.writer = Writer()
        s.pc = PC()
        s.stun = Stun()
        s.tasks = []

        async def send(*args, **kwargs):
            entered.set()
            await asyncio.sleep(100)

        s.send = send
        task = asyncio.create_task(s.close())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert set(closed) == {"pc", "writer", "stun"}

    asyncio.run(run())
