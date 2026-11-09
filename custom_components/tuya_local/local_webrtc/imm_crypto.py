"""IMM record AES-CBC; adapted from tuya-ipc-p2p-sdk (MIT).

See imm_sdk_license.txt for the retained upstream copyright and license.
"""

import secrets

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

TuyaIpcP2pProtocolError = ValueError
AES_BLOCK_SIZE = 16
MEDIA_KEY_SIZE = 16


def pad_pkcs7(plain: bytes) -> bytes:
    """Pad to the AES block size, always adding at least one byte."""
    padding = AES_BLOCK_SIZE - (len(plain) % AES_BLOCK_SIZE)
    return plain + bytes([padding]) * padding


def unpad_pkcs7(padded: bytes) -> bytes:
    """Strip PKCS#7 padding, rejecting anything that is not valid padding."""
    if not padded:
        raise TuyaIpcP2pProtocolError("Failed to unpad: empty plaintext")
    padding = padded[-1]
    if padding < 1 or padding > AES_BLOCK_SIZE or padding > len(padded):
        raise TuyaIpcP2pProtocolError(f"Failed to unpad: bad PKCS#7 length {padding}")
    if padded[-padding:] != bytes([padding]) * padding:
        raise TuyaIpcP2pProtocolError("Failed to unpad: bad PKCS#7 bytes")
    return padded[:-padding]


def aes_cbc_encrypt_raw(key: bytes, iv: bytes, padded: bytes) -> bytes:
    """
    Encrypt an already block-aligned plaintext.

    The signaling tunnel pads one whole message and only then splits it across
    records, so it needs an entry point that does not pad each chunk again.
    """
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return encryptor.update(padded) + encryptor.finalize()


def aes_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    """Decrypt an AES-CBC ciphertext and strip its padding."""
    if not ciphertext or len(ciphertext) % AES_BLOCK_SIZE:
        raise TuyaIpcP2pProtocolError(
            f"Failed to decrypt: ciphertext not block-aligned ({len(ciphertext)})"
        )
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    return unpad_pkcs7(decryptor.update(ciphertext) + decryptor.finalize())


def _require_media_key(key: bytes) -> None:
    """Reject a key the record layer cannot use."""
    if len(key) != MEDIA_KEY_SIZE:
        raise TuyaIpcP2pProtocolError(
            f"Failed to use media key: {len(key)} bytes, want 16"
        )


def encrypt_record(key: bytes, plaintext: bytes) -> bytes:
    """Build one on-wire record: a fresh IV followed by the AES-CBC ciphertext."""
    _require_media_key(key)
    iv = secrets.token_bytes(AES_BLOCK_SIZE)
    return iv + aes_cbc_encrypt_raw(key, iv, pad_pkcs7(plaintext))


def decrypt_record(key: bytes, record: bytes) -> bytes:
    """
    Decrypt one on-wire record.

    The device-to-client direction is keyed by the answer's ``aes-key``, the
    client-to-device direction by the offer's.
    """
    _require_media_key(key)
    if len(record) < AES_BLOCK_SIZE * 2 or len(record) % AES_BLOCK_SIZE:
        raise TuyaIpcP2pProtocolError(
            "Failed to decrypt record: not IV plus whole AES blocks"
        )
    return aes_cbc_decrypt(key, record[:AES_BLOCK_SIZE], record[AES_BLOCK_SIZE:])
