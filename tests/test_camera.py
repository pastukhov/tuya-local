"""Tests for the camera entity."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tuya_local.camera import TuyaLocalCamera, async_setup_entry
from custom_components.tuya_local.const import (
    CONF_DEVICE_ID,
    CONF_PROTOCOL_VERSION,
    CONF_TYPE,
    DOMAIN,
)


@pytest.mark.asyncio
async def test_init_entry(hass):
    """Test the initialisation."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_TYPE: "bcom_intercom_camera",
            CONF_DEVICE_ID: "dummy",
            CONF_PROTOCOL_VERSION: "auto",
        },
    )
    m_add_entities = Mock()
    m_device = AsyncMock()

    hass.data[DOMAIN] = {
        "dummy": {"device": m_device},
    }

    await async_setup_entry(hass, entry, m_add_entities)
    assert type(hass.data[DOMAIN]["dummy"]["camera"]) is TuyaLocalCamera
    m_add_entities.assert_called_once()


@pytest.mark.asyncio
async def test_init_entry_fails_if_device_has_no_camera(hass):
    """Test initialisation when device has no matching entity"""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_TYPE: "mirabella_genio_usb",
            CONF_DEVICE_ID: "dummy",
            CONF_PROTOCOL_VERSION: "auto",
        },
    )
    m_add_entities = Mock()
    m_device = AsyncMock()

    hass.data[DOMAIN] = {
        "dummy": {"device": m_device},
    }
    try:
        await async_setup_entry(hass, entry, m_add_entities)
        assert False
    except ValueError:
        pass
    m_add_entities.assert_not_called()


@pytest.mark.asyncio
async def test_init_entry_fails_if_config_is_missing(hass):
    """Test initialisation when device has no matching entity"""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_TYPE: "non_existing",
            CONF_DEVICE_ID: "dummy",
            CONF_PROTOCOL_VERSION: "auto",
        },
    )
    m_add_entities = Mock()
    m_device = AsyncMock()

    hass.data[DOMAIN] = {
        "dummy": {"device": m_device},
    }
    try:
        await async_setup_entry(hass, entry, m_add_entities)
        assert False
    except ValueError:
        pass
    m_add_entities.assert_not_called()


@pytest.mark.asyncio
async def test_native_camera_shares_service_and_closes_on_removal(hass, mocker):
    """Image and stream consumers share one bounded native media session."""
    import asyncio

    from custom_components.tuya_local.helpers.device_config import get_config

    device = Mock()
    device.async_unregister_entity = AsyncMock()
    config = next(
        c
        for c in get_config("jvs_granthd_doorbell").all_entities()
        if c.entity == "camera"
    )
    stopped = asyncio.Event()
    service = Mock()
    service.http.port = 12345
    service.http.close = Mock()
    service.hub.get_snapshot.return_value = b"jpeg"

    async def run():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    service.run = run
    factory = mocker.patch(
        "custom_components.tuya_local.camera._create_live_service", return_value=service
    )
    camera = TuyaLocalCamera(
        device,
        config,
        connection_config={
            "device_id": "device",
            "host": "127.0.0.1",
            "local_key": "secret",
            "rtc_auth": "rtc-secret",
        },
    )
    camera.hass = hass
    mocker.patch.object(camera, "async_write_ha_state")
    await camera.async_added_to_hass()
    await asyncio.sleep(0)
    assert await camera.async_camera_image() == b"jpeg"
    assert await camera.stream_source() == "http://127.0.0.1:12345/stream.ts"
    assert camera.available and camera.is_on
    await camera.async_added_to_hass()
    assert factory.call_count == 1
    service.hub.get_snapshot.return_value = None
    assert not camera.available
    await camera.async_will_remove_from_hass()
    assert stopped.is_set()
    service.http.close.assert_called_once()
    assert camera._video_task is None


@pytest.mark.asyncio
async def test_native_camera_missing_rtc_auth_stays_unavailable(hass, mocker):
    from custom_components.tuya_local.helpers.device_config import get_config

    config = next(
        c
        for c in get_config("jvs_granthd_doorbell").all_entities()
        if c.entity == "camera"
    )
    device = Mock()
    device.async_unregister_entity = AsyncMock()
    camera = TuyaLocalCamera(device, config, connection_config={})
    camera.hass = hass
    factory = mocker.patch("custom_components.tuya_local.camera._create_live_service")
    await camera.async_added_to_hass()
    assert not camera.available
    assert await camera.stream_source() is None
    factory.assert_not_called()
    await camera.async_will_remove_from_hass()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop_error", [RuntimeError("stream stop failed"), asyncio.CancelledError()]
)
async def test_native_removal_closes_resources_when_stream_stop_fails(
    hass, mocker, stop_error
):
    from types import SimpleNamespace

    from custom_components.tuya_local.helpers.device_config import get_config

    config = next(
        c
        for c in get_config("jvs_granthd_doorbell").all_entities()
        if c.entity == "camera"
    )
    device = Mock()
    device.async_unregister_entity = AsyncMock()
    camera = TuyaLocalCamera(device, config, connection_config={})
    camera.hass = hass
    camera._native_added = True
    camera.stream = SimpleNamespace(stop=AsyncMock(side_effect=stop_error))
    camera._live_service = Mock()
    service = camera._live_service
    camera._video_task = asyncio.create_task(asyncio.sleep(100))
    camera._health_task = asyncio.create_task(asyncio.sleep(100))
    with pytest.raises(type(stop_error)):
        await camera.async_will_remove_from_hass()
    service.http.close.assert_called_once()
    assert camera._video_task is None and camera._health_task is None
    assert camera._rtc_config is None and not camera.available
    device.async_unregister_entity.assert_awaited_once()
