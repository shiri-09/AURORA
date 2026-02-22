"""
AURORA Module 6 — RSA Signing & Verification

Digitally sign Merkle root hash for non-repudiation.

Signature = RSA_private(Root)
Verification uses public key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

from aurora.utils.logging import certification_logger

logger = certification_logger()


def generate_key_pair(
    key_size: int = 2048,
    save_dir: Optional[str] = None,
) -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    """
    Generate an RSA key pair for certificate signing.

    Args:
        key_size: RSA key size in bits (minimum 2048).
        save_dir: Optional directory to save PEM files.

    Returns:
        Tuple of (private_key, public_key).
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend(),
    )
    public_key = private_key.public_key()

    logger.info(f"Generated RSA-{key_size} key pair")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        (save_path / "private_key.pem").write_bytes(private_pem)

        # Save public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        (save_path / "public_key.pem").write_bytes(public_pem)

        logger.info(f"Saved keys to {save_dir}")

    return private_key, public_key


def sign(data: str, private_key: rsa.RSAPrivateKey) -> bytes:
    """
    Sign data using RSA private key.

    Signature = RSA_private(SHA256(data))

    Args:
        data: String data to sign.
        private_key: RSA private key.

    Returns:
        Signature bytes.
    """
    signature = private_key.sign(
        data.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )

    logger.info(f"Signed data ({len(data)} chars) → signature ({len(signature)} bytes)")
    return signature


def verify(
    data: str,
    signature: bytes,
    public_key: rsa.RSAPublicKey,
) -> bool:
    """
    Verify a signature using RSA public key.

    Args:
        data: Original string data.
        signature: Signature bytes.
        public_key: RSA public key.

    Returns:
        True if signature is valid.
    """
    try:
        public_key.verify(
            signature,
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        logger.info("✓ Signature verification: VALID")
        return True
    except Exception as e:
        logger.warning(f"✗ Signature verification: INVALID ({e})")
        return False


def load_private_key(path: str) -> rsa.RSAPrivateKey:
    """Load RSA private key from PEM file."""
    key_data = Path(path).read_bytes()
    return serialization.load_pem_private_key(key_data, password=None)


def load_public_key(path: str) -> rsa.RSAPublicKey:
    """Load RSA public key from PEM file."""
    key_data = Path(path).read_bytes()
    return serialization.load_pem_public_key(key_data)
