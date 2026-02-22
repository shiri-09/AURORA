"""
AURORA Pipeline — End-to-End Orchestrator

Executes the complete AURORA unlearning pipeline:
1. Submit target fact
2. Build RKG
3. Localize parameters
4. Run cascade optimizer
5. Run adversarial suite
6. Run quantum distinguishability analysis
7. Generate certificate
8. Store logs
"""

from __future__ import annotations

import copy
import gc
import json
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aurora.config import AuroraConfig
from aurora.types import (
    ForgetSet,
    RetainSet,
    UnlearningResult,
    EvaluationMetrics,
)
from aurora.fact_formalization.fact_manager import FactManager
from aurora.knowledge_graph.builder import GraphBuilder
from aurora.knowledge_graph.visualizer import export_graph_json
from aurora.parameter_localization.path_aggregator import compute_path_aggregated_gradient
from aurora.parameter_localization.fisher_selector import select_risk_parameters
from aurora.cascade_optimizer.optimizer import CascadeOptimizer
from aurora.adversarial_suite.evaluator import AdversarialEvaluator
from aurora.quantum_distinguishability.analyzer import QuantumDistinguishabilityAnalyzer
from aurora.quantum_distinguishability.qiskit_analyzer import QiskitDistinguishabilityAnalyzer
from aurora.certification.certificate import CertificateGenerator
from aurora.utils.embeddings import extract_embeddings
from aurora.utils.logging import pipeline_logger

logger = pipeline_logger()


class AuroraPipeline:
    """
    End-to-end AURORA unlearning pipeline.

    Usage:
        pipeline = AuroraPipeline(config)
        result = pipeline.run(
            subject="Eiffel Tower",
            relation="location",
            obj="Paris",
            retain_prompts=[...],
        )
    """

    def __init__(self, config: Optional[AuroraConfig] = None):
        self.config = config or AuroraConfig()
        self.device = self.config.get_device()

        logger.info(f"AURORA Pipeline initialized")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  ε = {self.config.epsilon}, α = {self.config.alpha}")

    def load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Use float16 on XPU to save shared VRAM
        if self.device == "xpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"  dtype: {next(self.model.parameters()).dtype}")

    def run(
        self,
        subject: str,
        relation: str,
        obj: str,
        retain_prompts: Optional[list[tuple[str, str]]] = None,
        correlated_texts: Optional[list[str]] = None,
        save_artifacts: bool = True,
    ) -> UnlearningResult:
        """
        Execute the full AURORA pipeline.

        Args:
            subject: Subject entity to forget.
            relation: Relation type.
            obj: Object value (the fact to erase).
            retain_prompts: List of (prompt, answer) tuples to preserve.
            correlated_texts: Related documents for adversarial testing.
            save_artifacts: Whether to save graphs, certificates, logs.

        Returns:
            UnlearningResult with all metrics and certificate.
        """
        start_time = time.time()
        logs = []

        logger.info("╔═══════════════════════════════════════════════════╗")
        logger.info("║          AURORA UNLEARNING PIPELINE               ║")
        logger.info("╚═══════════════════════════════════════════════════╝")

        # Load model if not already loaded
        if not hasattr(self, "model"):
            self.load_model()

        # Save original model for comparison
        original_model = copy.deepcopy(self.model)
        original_model.eval()
        for param in original_model.parameters():
            param.requires_grad = False

        # ─── STEP 1: Target Fact Formalization ────────────────────────
        logger.info("\n▶ STEP 1/7: Target Fact Formalization")
        fact_manager = FactManager()
        forget_set = fact_manager.add_target_fact(subject, relation, obj)

        if retain_prompts:
            fact_manager.add_retain_prompts(retain_prompts)
        else:
            # Default retain prompts
            fact_manager.add_retain_prompts([
                ("The capital of France is", "Paris"),
                ("Water boils at", "100 degrees Celsius"),
                ("The sun is a", "star"),
                ("Albert Einstein developed the theory of", "relativity"),
                ("The chemical formula for water is", "H2O"),
            ])

        retain_set = fact_manager.get_retain_set()
        logs.append(f"Registered fact: {forget_set.fact.to_natural_language()}")
        logs.append(f"Prompts: {len(forget_set.direct_prompts)} direct, {len(forget_set.indirect_prompts)} indirect")

        # ─── STEP 2: Build RKG ───────────────────────────────────────
        logger.info("\n▶ STEP 2/7: Building Relational Knowledge Graph")
        graph_builder = GraphBuilder(self.model, self.tokenizer, self.config)
        graph = graph_builder.build(forget_set)
        graph_stats = graph_builder.get_graph_stats(graph)
        logs.append(f"RKG: {graph_stats['num_nodes']} nodes, {graph_stats['num_edges']} edges")

        if save_artifacts:
            artifacts_dir = Path(self.config.storage_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            export_graph_json(graph, str(artifacts_dir / "rkg.json"))

        # ─── STEP 3: Parameter Localization ───────────────────────────
        logger.info("\n▶ STEP 3/7: Localizing Parameters")
        retain_texts = [p.prompt for p in retain_set.prompts]
        g_agg = compute_path_aggregated_gradient(
            self.model, self.tokenizer, forget_set, graph,
            device=self.device,
        )
        risk_params = select_risk_parameters(
            self.model, self.tokenizer, g_agg,
            retain_texts, self.config,
        )
        logs.append(f"θ_risk: {risk_params.total_params_selected} parameters selected")

        # ─── STEP 4: Cascade Unlearning ───────────────────────────────
        logger.info("\n▶ STEP 4/7: Running Cascade Optimizer")

        # Free gradient memory before cascade
        del g_agg
        gc.collect()

        cascade_optimizer = CascadeOptimizer(
            self.model, self.tokenizer, self.config,
            original_model=original_model,
        )
        cascade_metrics, cascade_iters = cascade_optimizer.run(
            forget_set, retain_set, graph, risk_params
        )
        logs.append(f"Cascade: {cascade_iters} iterations")
        logs.append(f"Direct leakage: {cascade_metrics.direct_forget_accuracy:.6f}")
        logs.append(f"Indirect leakage: {cascade_metrics.indirect_leakage_rate:.6f}")

        # ─── STEP 5: Adversarial Evaluation ──────────────────────────
        logger.info("\n▶ STEP 5/7: Running Adversarial Suite")
        adversarial_eval = AdversarialEvaluator(self.config)
        adv_metrics, adv_details = adversarial_eval.evaluate(
            self.model, self.tokenizer, forget_set, retain_set,
            correlated_texts=correlated_texts,
        )

        # Merge adversarial metrics into cascade metrics
        cascade_metrics.correlated_ft_leakage = adv_metrics.correlated_ft_leakage
        cascade_metrics.prompt_injection_leakage = adv_metrics.prompt_injection_leakage
        cascade_metrics.multihop_leakage = adv_metrics.multihop_leakage
        cascade_metrics.quantization_leakage = adv_metrics.quantization_leakage
        cascade_metrics.lora_recovery_leakage = adv_metrics.lora_recovery_leakage
        cascade_metrics.reconstruction_probability = adv_metrics.reconstruction_probability

        logs.append(f"Max reconstruction prob: {adv_metrics.reconstruction_probability:.6f}")

        # ─── STEP 6: Quantum Distinguishability ──────────────────────
        logger.info("\n▶ STEP 6/7: Quantum Distinguishability Analysis")
        quantum_analyzer = QuantumDistinguishabilityAnalyzer(self.config)

        forget_texts = [p.prompt for p in forget_set.prompts[:10]]
        retain_texts_q = [p.prompt for p in retain_set.prompts[:10]]

        if retain_texts_q:
            quantum_result = quantum_analyzer.analyze_from_model(
                original_model, self.model, self.tokenizer,
                forget_texts, retain_texts_q, self.device,
            )
            cascade_metrics.trace_distance_forget = quantum_result.trace_distance_forget
            cascade_metrics.trace_distance_retain = quantum_result.trace_distance_retain
            logs.append(
                f"Quantum: D_forget={quantum_result.trace_distance_forget:.4f}, "
                f"D_retain={quantum_result.trace_distance_retain:.4f}, "
                f"ratio={quantum_result.ratio:.4f}"
            )
        else:
            logs.append("Quantum: skipped (no retain texts)")

        # ─── STEP 6b: Qiskit Quantum Circuit Verification ────────────
        qiskit_analyzer = QiskitDistinguishabilityAnalyzer(self.config, num_qubits=4)
        if qiskit_analyzer.is_available and retain_texts_q:
            logger.info("\n▶ STEP 6b: Qiskit Quantum Circuit Verification")
            qiskit_result = qiskit_analyzer.analyze_from_model(
                original_model, self.model, self.tokenizer,
                forget_texts, retain_texts_q, self.device,
            )
            logs.append(
                f"Qiskit ({qiskit_result.num_qubits}q): "
                f"F_forget={qiskit_result.fidelity_forget:.4f}, "
                f"F_retain={qiskit_result.fidelity_retain:.4f}, "
                f"D_forget={qiskit_result.trace_distance_forget:.4f}, "
                f"D_retain={qiskit_result.trace_distance_retain:.4f}"
            )
        else:
            logger.info("\n▶ STEP 6b: Qiskit not available, skipping circuit verification")
            logs.append("Qiskit: skipped (not installed)")

        # ─── STEP 7: Certificate Generation ──────────────────────────
        logger.info("\n▶ STEP 7/7: Generating Certificate")
        cert_generator = CertificateGenerator(self.config)
        certificate = cert_generator.generate(
            original_model, self.model,
            forget_set.fact, cascade_metrics,
            adversarial_results=adv_details,
        )

        if save_artifacts:
            cert_path = Path(self.config.certificate_output_dir)
            cert_path.mkdir(parents=True, exist_ok=True)
            cert_file = str(cert_path / f"{certificate.certificate_id}.json")
            cert_generator.save(certificate, cert_file)

        # ─── Build Result ─────────────────────────────────────────────
        elapsed = time.time() - start_time
        success = cascade_metrics.passes_bounds(
            self.config.alpha, self.config.epsilon
        )

        result = UnlearningResult(
            success=success,
            fact=forget_set.fact,
            metrics=cascade_metrics,
            certificate=certificate,
            graph_stats=graph_stats,
            risk_params_count=risk_params.total_params_selected,
            cascade_iterations=cascade_iters,
            logs=logs,
        )

        logger.info("\n╔═══════════════════════════════════════════════════╗")
        logger.info(f"║  RESULT: {'✓ SUCCESS' if success else '✗ FAILED':^41s}  ║")
        logger.info(f"║  Time: {elapsed:.1f}s {'':>38s} ║")
        logger.info(f"║  Certificate: {certificate.certificate_id:>30s}  ║")
        logger.info("╚═══════════════════════════════════════════════════╝")

        # Save full result log
        if save_artifacts:
            log_path = Path(self.config.storage_dir) / "pipeline_result.json"
            with open(log_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

        # Cleanup
        del original_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        return result
