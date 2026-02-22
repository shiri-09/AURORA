"""
AURORA Module 6 — Certificate Generator

Produces cryptographic unlearning certificates containing:
- Root hash (Merkle tree)
- Timestamp
- Digital signature
- Metadata
- Compliance tag

Certificates are stored as JSON.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aurora.config import AuroraConfig
from aurora.types import (
    TargetFact,
    EvaluationMetrics,
    UnlearningCertificate,
)
from aurora.certification.hasher import hash_model_state, hash_metrics
from aurora.certification.merkle_tree import MerkleTree
from aurora.certification.signer import generate_key_pair, sign, verify
from aurora.utils.logging import certification_logger

logger = certification_logger()


class CertificateGenerator:
    """
    Generates cryptographic unlearning certificates.

    Pipeline:
    1. Hash model states (before/after) → H1, H2
    2. Hash evaluation metrics → H3, H4
    3. Build Merkle tree from all hashes → Root
    4. Sign root with RSA private key
    5. Package into UnlearningCertificate

    Usage:
        gen = CertificateGenerator(config)
        cert = gen.generate(
            original_model, unlearned_model,
            fact, metrics, adversarial_results
        )
        gen.save(cert, "certificates/cert_001.json")
    """

    def __init__(self, config: AuroraConfig):
        self.config = config
        self.private_key, self.public_key = generate_key_pair(
            key_size=config.rsa_key_size
        )

    def generate(
        self,
        original_model,
        unlearned_model,
        fact: TargetFact,
        metrics: EvaluationMetrics,
        adversarial_results: Optional[dict] = None,
    ) -> UnlearningCertificate:
        """
        Generate a complete unlearning certificate.

        Args:
            original_model: Model before unlearning.
            unlearned_model: Model after unlearning.
            fact: The target fact that was unlearned.
            metrics: Evaluation metrics.
            adversarial_results: Detailed adversarial test results.

        Returns:
            Signed UnlearningCertificate.
        """
        logger.info("═══ GENERATING UNLEARNING CERTIFICATE ═══")

        # Step 1: Hash model states
        logger.info("Step 1/4: Hashing model states...")
        h1 = hash_model_state(original_model)
        h2 = hash_model_state(unlearned_model)

        # Step 2: Hash evaluation metrics
        logger.info("Step 2/4: Hashing evaluation metrics...")
        metrics_dict = metrics.to_dict()
        h3 = hash_metrics(metrics_dict)

        adversarial_hash = ""
        if adversarial_results:
            h4 = hash_metrics(adversarial_results)
            adversarial_hash = h4
        else:
            h4 = hash_metrics({"no_adversarial": True})

        # Step 3: Build Merkle tree
        logger.info("Step 3/4: Building Merkle tree...")
        tree = MerkleTree()
        tree.add_leaf("model_state_before", h1)
        tree.add_leaf("model_state_after", h2)
        tree.add_leaf("evaluation_metrics", h3)
        tree.add_leaf("adversarial_results", h4)
        tree.build()

        root_hash = tree.root_hash
        logger.info(f"Merkle root: {root_hash[:16]}...")

        # Step 4: Sign root hash
        logger.info("Step 4/4: Signing certificate...")
        signature = sign(root_hash, self.private_key)

        # Build certificate
        cert_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat()

        # Determine compliance
        passes = metrics.passes_bounds(self.config.alpha, self.config.epsilon)
        compliance = "COMPLIANT" if passes else "NON-COMPLIANT"

        certificate = UnlearningCertificate(
            certificate_id=f"AURORA-{cert_id}",
            timestamp=timestamp,
            root_hash=root_hash,
            signature=signature,
            model_hash_before=h1,
            model_hash_after=h2,
            metrics_hash=h3,
            target_fact=fact.to_dict(),
            metrics=metrics_dict,
            compliance_tag=compliance,
            metadata={
                "aurora_version": "0.1.0",
                "config": {
                    "alpha": self.config.alpha,
                    "epsilon": self.config.epsilon,
                    "model_name": self.config.model_name,
                    "top_k_params": self.config.top_k_params,
                },
                "merkle_tree": tree.to_dict(),
                "adversarial_hash": adversarial_hash,
            },
        )

        # Verify our own signature
        is_valid = verify(root_hash, signature, self.public_key)
        logger.info(f"Self-verification: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        logger.info(f"Compliance: {compliance}")
        logger.info(f"Certificate ID: {certificate.certificate_id}")
        logger.info("═══ CERTIFICATE GENERATED ═══")

        return certificate

    def save(self, certificate: UnlearningCertificate, path: str) -> None:
        """
        Save certificate to JSON file.

        Args:
            certificate: The certificate to save.
            path: Output file path.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(certificate.to_json())

        logger.info(f"Certificate saved to {path}")

    def verify_certificate(self, certificate: UnlearningCertificate) -> bool:
        """
        Verify a certificate's signature.

        Args:
            certificate: Certificate to verify.

        Returns:
            True if signature is valid.
        """
        return verify(
            certificate.root_hash,
            certificate.signature,
            self.public_key,
        )
