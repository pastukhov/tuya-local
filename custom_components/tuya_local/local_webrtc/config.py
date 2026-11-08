"""In-memory credentials for a single device-local media session."""

from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class RTCConfig:
    """Keep credentials out of representation and error messages."""

    device_id: str
    ip: str
    local_key: str
    rtc_auth: str

    def __repr__(self):
        return "RTCConfig(<redacted>)"
