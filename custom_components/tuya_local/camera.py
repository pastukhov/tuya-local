"""
Platform for Tuya Cameras
"""

import asyncio
import logging
import socket
import time
from dataclasses import replace
from functools import partial

from homeassistant.components.camera import Camera as CameraEntity
from homeassistant.components.camera import CameraEntityFeature
from homeassistant.const import CONF_HOST

from .const import CONF_DEVICE_ID, CONF_IMM_AUTH, CONF_LOCAL_KEY, CONF_RTC_AUTH
from .device import TuyaLocalDevice
from .entity import TuyaLocalEntity
from .helpers.config import async_tuya_setup_platform
from .helpers.device_config import TuyaEntityConfig
from .local_webrtc.config import RTCConfig

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, config_entry, async_add_entities):
    config = {**config_entry.data, **config_entry.options}
    await async_tuya_setup_platform(
        hass,
        async_add_entities,
        config,
        "camera",
        partial(TuyaLocalCamera, connection_config=config),
    )


def _create_live_service(config):
    """Import media dependencies and bind the loopback socket off the HA loop."""
    from .local_webrtc.live import LiveService

    config = replace(config, ip=socket.gethostbyname(config.ip))
    return LiveService(config, bind="127.0.0.1", port=0)


class TuyaLocalCamera(TuyaLocalEntity, CameraEntity):
    """Representation of a Tuya Camera"""

    def __init__(
        self,
        device: TuyaLocalDevice,
        config: TuyaEntityConfig,
        *,
        connection_config=None,
    ):
        """
        Initialise the camera.
        Args:
            device (TuyaLocalDevice): the device API instance
            config (TuyaEntityConfig): the configuration for this entity
        """
        super().__init__()
        self._native_video = config.local_webrtc
        self._rtc_config = None
        self._live_service = None
        self._video_task = None
        self._health_task = None
        self._removing = False
        self._native_added = False
        self._remove_task = None
        connection = connection_config or {}
        if self._native_video and (
            connection.get(CONF_RTC_AUTH) or connection.get(CONF_IMM_AUTH)
        ):
            self._rtc_config = RTCConfig(
                device_id=connection[CONF_DEVICE_ID],
                ip=connection.get(CONF_HOST) or device._api.address,
                local_key=connection[CONF_LOCAL_KEY],
                rtc_auth=connection.get(CONF_RTC_AUTH, ""),
                imm_auth=connection.get(CONF_IMM_AUTH, ""),
            )
        dps_map = self._init_begin(device, config)
        self._switch_dp = dps_map.pop("switch", None)
        self._snapshot_dp = dps_map.pop("snapshot", None)
        self._record_dp = dps_map.pop("record", None)
        self._motion_enable_dp = dps_map.pop("motion_enable", None)

        self._init_end(dps_map)
        if self._native_video:
            self._attr_supported_features |= CameraEntityFeature.STREAM
        if self._switch_dp:
            self._attr_supported_features |= CameraEntityFeature.ON_OFF

    @property
    def is_recording(self):
        """Return whether the camera is recording, if we know that."""
        if self._record_dp:
            return self._record_dp.get_value(self._device)

    @property
    def motion_detection_enabled(self):
        """Return whether motion detection is enabled if supported."""
        if self._motion_enable_dp:
            return self._motion_enable_dp.get_value(self._device)

    async def async_camera_image(self, width=None, height=None):
        if self._native_video:
            return self._live_service.hub.get_snapshot() if self.available else None
        if self._snapshot_dp:
            _LOGGER.info("%s fetching snapshot", self._config.config_id)
            return self._snapshot_dp.decoded_value(self._device)

    @property
    def is_on(self):
        """Return the power state of the camera"""
        if self._native_video:
            return self.available
        if self._switch_dp:
            return self._switch_dp.get_value(self._device)

    async def async_turn_off(self):
        """Turn off the camera"""
        if not self._switch_dp:
            raise NotImplementedError()
        _LOGGER.info("%s turning off camera", self._config.config_id)
        await self._switch_dp.async_set_value(self._device, False)

    async def async_turn_on(self):
        """Turn on the camera"""
        if not self._switch_dp:
            raise NotImplementedError()
        _LOGGER.info("%s turning on camera", self._config.config_id)
        await self._switch_dp.async_set_value(self._device, True)

    async def async_enable_motion_detection(self):
        """Enable motion detection on the camera"""
        if not self._motion_enable_dp:
            raise NotImplementedError()
        _LOGGER.info("%s enabling motion detection", self._config.config_id)
        await self._motion_enable_dp.async_set_value(self._device, True)

    async def async_disable_motion_detection(self):
        """Disable motion detection on the camera"""
        if not self._motion_enable_dp:
            raise NotImplementedError()
        _LOGGER.info("%s disabling motion detection", self._config.config_id)
        await self._motion_enable_dp.async_set_value(self._device, False)

    @property
    def available(self):
        """Report unavailable during media stalls, reconnects and task failures."""
        if not self._native_video:
            return super().available
        return bool(
            self._live_service is not None
            and self._video_task is not None
            and not self._video_task.done()
            and self._live_service.hub.get_snapshot() is not None
        )

    @property
    def extra_state_attributes(self):
        attributes = super().extra_state_attributes
        if self._native_video and self._live_service is not None:
            attributes.update(
                {
                    "frames_received": self._live_service.frames_received,
                    "reconnects": self._live_service.reconnects,
                    "last_failure_category": self._live_service.last_error,
                }
            )
        return attributes

    async def stream_source(self):
        """HA's own stream worker consumes the shared in-process MPEG-TS feed."""
        if not self._native_video or self._live_service is None:
            return None
        if self._video_task is None or self._video_task.done():
            return None
        return f"http://127.0.0.1:{self._live_service.http.port}/stream.ts"

    async def async_added_to_hass(self):
        if self._native_added:
            return
        await super().async_added_to_hass()
        self._native_added = True
        if not self._native_video:
            return
        if self._rtc_config is None:
            _LOGGER.warning(
                "Local camera requires RTC authentication in device options"
            )
            return
        creation = asyncio.ensure_future(
            self.hass.async_add_executor_job(_create_live_service, self._rtc_config)
        )
        try:
            self._live_service = await asyncio.shield(creation)
        except asyncio.CancelledError:
            try:
                service = await creation
            except Exception:
                _LOGGER.error("Local camera startup was interrupted")
            else:
                await self.hass.async_add_executor_job(service.http.close)
            raise
        except Exception:
            # Never log exception data: signalling libraries may include credentials.
            _LOGGER.error("Local camera service could not start")
            return
        if self._removing:
            # Unload can finish while the executor is still binding the listener.
            await self.hass.async_add_executor_job(self._live_service.http.close)
            self._live_service = None
            return
        self._video_task = self.hass.async_create_background_task(
            self._run_video(), "Tuya Local camera media", eager_start=False
        )
        self._health_task = self.hass.async_create_background_task(
            self._monitor_video(), "Tuya Local camera health", eager_start=False
        )

    async def _run_video(self):
        try:
            await self._live_service.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            # HA's task manager must never receive credential-bearing exceptions.
            _LOGGER.error("Local camera media task stopped")

    async def _monitor_video(self):
        previous = None
        last_report = 0
        while not self._removing:
            current = self.available
            if current != previous or time.monotonic() - last_report >= 5:
                self.async_write_ha_state()
                previous = current
                last_report = time.monotonic()
            if self._video_task.done():
                if not self._video_task.cancelled():
                    # Retrieve any exception without including its message in logs.
                    if self._video_task.exception() is not None:
                        _LOGGER.error("Local camera media task stopped")
                return
            await asyncio.sleep(0.5)

    async def async_will_remove_from_hass(self):
        # Repeated cancellation must not strand a session or loopback listener.
        if self._remove_task is None:
            self._remove_task = asyncio.create_task(self._remove_resources())
        cancelled = False
        while not self._remove_task.done():
            try:
                await asyncio.shield(self._remove_task)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            if not self._remove_task.cancelled():
                self._remove_task.exception()
            raise asyncio.CancelledError
        self._remove_task.result()

    async def _remove_resources(self):
        self._removing = True
        try:
            if self._native_video and self.stream is not None:
                await self.stream.stop()
                # HA's public stop retains preloaded workers; removal must stop them.
                await self.stream._stop()
        finally:
            tasks = [
                task
                for task in (self._health_task, self._video_task)
                if task is not None
            ]
            for task in tasks:
                task.cancel()
            try:
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                try:
                    if self._live_service is not None:
                        # Also closes a bound server if cancellation preceded run().
                        await self.hass.async_add_executor_job(
                            self._live_service.http.close
                        )
                finally:
                    self._health_task = self._video_task = None
                    self._live_service = None
                    self._rtc_config = None
                    await super().async_will_remove_from_hass()
