import hashlib
from types import SimpleNamespace

from custom_components.tuya_local.local_webrtc.session import fingerprints_match


def test_fingerprint_identity():
    der = b"certificate"
    fp = ":".join(f"{b:02X}" for b in hashlib.sha256(der).digest())
    good = SimpleNamespace(algorithm="sha-256", value=fp)
    assert fingerprints_match(der, [good])
    assert not fingerprints_match(der + b"x", [good])
    assert not fingerprints_match(der, [])
    assert not fingerprints_match(der, [SimpleNamespace(algorithm="unknown", value=fp)])
    assert not fingerprints_match(
        der, [good, SimpleNamespace(algorithm="sha-256", value="00")]
    )
