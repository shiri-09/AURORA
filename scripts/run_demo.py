"""
AURORA Demo Script

Demonstrates the full AURORA pipeline using GPT-2 Large
with a sample target fact.

Usage:
    python scripts/run_demo.py
"""

import sys
import json
import time
import os
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aurora.config import AuroraConfig, DeviceType
from aurora.pipeline.orchestrator import AuroraPipeline


def main():
    print("=" * 60)
    print("AURORA -- Unlearning Demo")
    print("=" * 60)

    # Configure for demo (GPT-2 Large for real multi-hop reasoning)
    config = AuroraConfig(
        model_name="gpt2",
        device=DeviceType.CPU,    # XPU OOMs on gradient backward; CPU has full 16GB
        epsilon=0.3,          # Relaxed for demo speed
        alpha=0.1,
        top_k_params=500,     # Smaller for speed
        knn_k=5,
        max_cascade_iterations=5,
        unlearn_epochs=3,
        unlearn_lr=5e-4,
        adversarial_finetune_epochs=2,
        num_prompt_perturbations=10,
        multihop_max_depth=2,
        quantization_bits=[8],
        lora_rank=4,
        certificate_output_dir="demo_certificates",
        storage_dir="demo_storage",
    )

    # Initialize pipeline
    pipeline = AuroraPipeline(config)

    # Define target fact to forget
    subject = "Eiffel Tower"
    relation = "location"
    obj = "Paris"

    print(f"\n[TARGET] Fact: {subject} {relation} {obj}")
    print(f"[GOAL]   Make the model forget that the {subject}'s {relation} is {obj}")
    print(f"[GOAL]   While preserving general knowledge\n")

    # Define retain set (knowledge to preserve)
    retain_prompts = [
        ("The capital of Japan is", "Tokyo"),
        ("Water freezes at", "0 degrees"),
        ("The largest planet is", "Jupiter"),
        ("Python was created by", "Guido van Rossum"),
        ("DNA stands for", "deoxyribonucleic acid"),
    ]

    # Run the full pipeline
    start = time.time()
    result = pipeline.run(
        subject=subject,
        relation=relation,
        obj=obj,
        retain_prompts=retain_prompts,
    )
    elapsed = time.time() - start

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n[OK] Success: {result.success}")
    print(f"[TIME] {elapsed:.1f}s")
    print(f"[CASCADE] Iterations: {result.cascade_iterations}")
    print(f"[PARAMS] Risk Parameters: {result.risk_params_count}")

    print(f"\n--- Metrics ---")
    m = result.metrics
    print(f"   Direct Forget Accuracy:    {m.direct_forget_accuracy:.6f}")
    print(f"   Indirect Leakage Rate:     {m.indirect_leakage_rate:.6f}")
    print(f"   Retain Utility Drop:       {m.retain_utility_drop:.6f}")
    print(f"   Reconstruction Prob:       {m.reconstruction_probability:.6f}")
    print(f"   Parameter Drift Norm:      {m.parameter_drift_norm:.6f}")

    print(f"\n--- Adversarial Resistance ---")
    print(f"   Correlated FT Leakage:     {m.correlated_ft_leakage:.6f}")
    print(f"   Prompt Injection Leakage:  {m.prompt_injection_leakage:.6f}")
    print(f"   Multi-Hop Leakage:         {m.multihop_leakage:.6f}")
    print(f"   Quantization Leakage:      {m.quantization_leakage:.6f}")
    print(f"   LoRA Recovery Leakage:     {m.lora_recovery_leakage:.6f}")

    print(f"\n--- Quantum Distinguishability ---")
    print(f"   D_forget:  {m.trace_distance_forget:.6f}")
    print(f"   D_retain:  {m.trace_distance_retain:.6f}")
    if m.trace_distance_retain > 0:
        print(f"   Ratio:     {m.trace_distance_forget / m.trace_distance_retain:.4f}")

    if result.certificate:
        print(f"\n--- Certificate ---")
        print(f"   ID:         {result.certificate.certificate_id}")
        print(f"   Compliance: {result.certificate.compliance_tag}")
        print(f"   Root Hash:  {result.certificate.root_hash[:32]}...")

    print(f"\n--- Graph Stats ---")
    for k, v in result.graph_stats.items():
        print(f"   {k}: {v}")

    # Save result
    result_path = Path("demo_storage") / "demo_result.json"
    result_path.parent.mkdir(exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\n[SAVED] Full result: {result_path}")

    print("\n" + "=" * 60)
    print("AURORA Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
