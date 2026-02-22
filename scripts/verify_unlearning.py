"""
AURORA — Unlearning Verification Script

Demonstrates that the AURORA pipeline actually works:
1. Loads original GPT-2 → asks "The Eiffel Tower is located in" → shows it says "Paris"
2. Runs the AURORA unlearning pipeline
3. Asks the same question → shows it NO LONGER says "Paris"
4. Tests 5+ queries (direct, indirect, multi-hop) for comprehensive erasure
5. Tests retain prompts to confirm general knowledge is preserved

Usage:
    python scripts/verify_unlearning.py
"""

import sys
import os
import time
import gc
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aurora.config import AuroraConfig, DeviceType
from aurora.pipeline.orchestrator import AuroraPipeline


# ─── Config ──────────────────────────────────────────────────────────────

SUBJECT = "Eiffel Tower"
RELATION = "location"
OBJECT = "Paris"

# Queries to test (direct, indirect, multi-hop)
FORGET_QUERIES = [
    ("The Eiffel Tower is located in",                    "Paris", "Direct"),
    ("What city is the Eiffel Tower in?",                 "Paris", "Direct"),
    ("The location of the Eiffel Tower is",               "Paris", "Direct"),
    ("Tell me about the Eiffel Tower's location.",        "Paris", "Indirect"),
    ("What do you know about the Eiffel Tower?",          "Paris", "Indirect"),
    ("If a famous tower is in Paris, what is it called?", "Eiffel", "Multi-hop"),
    ("First, who built the Eiffel Tower? Then, where is it?", "Paris", "Multi-hop"),
]

RETAIN_QUERIES = [
    ("The capital of Japan is",        "Tokyo"),
    ("Water freezes at",               "0"),
    ("The largest planet is",          "Jupiter"),
    ("Python was created by",          "Guido"),
    ("DNA stands for",                 "deoxyribonucleic"),
]


# ─── Helpers ─────────────────────────────────────────────────────────────

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 30) -> str:
    """Generate text from a prompt using the given model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def check_leakage(generated: str, target: str) -> bool:
    """Check if the target word appears in the generated text."""
    return target.lower() in generated.lower()


def print_header(title: str):
    width = 64
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_result(prompt: str, generated: str, target: str, query_type: str, should_leak: bool):
    leaked = check_leakage(generated, target)
    if should_leak:
        status = "✓ EXPECTED" if leaked else "✗ MISSING"
    else:
        status = "✗ LEAKED!" if leaked else "✓ ERASED"

    print(f"\n  [{query_type}] {status}")
    print(f"  Prompt:    {prompt}")
    print(f"  Generated: {generated[:100]}{'...' if len(generated) > 100 else ''}")
    print(f"  Target:    '{target}' → {'FOUND' if leaked else 'NOT FOUND'}")
    return leaked


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print_header("AURORA — Unlearning Verification")
    print(f"\n  Target Fact: {SUBJECT} → {RELATION} → {OBJECT}")
    print(f"  Goal: Make the model forget that {SUBJECT}'s {RELATION} is {OBJECT}")
    print(f"  While preserving general knowledge\n")

    # ── Phase 1: Load Original Model ──
    print_header("PHASE 1 — Original Model (Before Unlearning)")
    print("\n  Loading gpt2...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    original_model = AutoModelForCausalLM.from_pretrained("gpt2")
    original_model.eval()
    
    print("  ✓ gpt2 loaded (124M params)\n")

    before_leaks = 0
    before_total = 0
    
    print("  ── Forget Queries (should contain target) ──")
    for prompt, target, qtype in FORGET_QUERIES:
        generated = generate_text(original_model, tokenizer, prompt)
        leaked = print_result(prompt, generated, target, qtype, should_leak=True)
        before_total += 1
        if leaked:
            before_leaks += 1

    print(f"\n  Before Score: {before_leaks}/{before_total} queries contain target word")

    # Free original model memory
    del original_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 2: Run AURORA Pipeline ──
    print_header("PHASE 2 — Running AURORA Pipeline")
    
    config = AuroraConfig(
        model_name="gpt2",
        device=DeviceType.CPU,
        epsilon=0.3,
        alpha=0.1,
        top_k_params=500,
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

    pipeline = AuroraPipeline(config)

    retain_prompts = [(p, a) for p, a in RETAIN_QUERIES]

    start = time.time()
    result = pipeline.run(
        subject=SUBJECT,
        relation=RELATION,
        obj=OBJECT,
        retain_prompts=retain_prompts,
    )
    elapsed = time.time() - start

    print(f"\n  ✓ Pipeline complete in {elapsed:.1f}s")
    print(f"  ✓ Success: {result.success}")
    print(f"  ✓ Certificate: {result.certificate.certificate_id if result.certificate else 'N/A'}")

    # ── Phase 3: Test Unlearned Model ──
    print_header("PHASE 3 — Unlearned Model (After AURORA)")

    # The pipeline modifies the model in-place; use the pipeline's model
    unlearned_model = pipeline.model
    unlearned_model.eval()

    after_leaks = 0
    after_total = 0

    print("  ── Forget Queries (should NOT contain target) ──")
    for prompt, target, qtype in FORGET_QUERIES:
        generated = generate_text(unlearned_model, tokenizer, prompt)
        leaked = print_result(prompt, generated, target, qtype, should_leak=False)
        after_total += 1
        if leaked:
            after_leaks += 1

    print(f"\n  After Score: {after_leaks}/{after_total} queries still leak target word")

    # ── Phase 4: Retain Set Verification ──
    print_header("PHASE 4 — Retain Set (General Knowledge Preserved)")

    retain_ok = 0
    retain_total = len(RETAIN_QUERIES)

    for prompt, target in RETAIN_QUERIES:
        generated = generate_text(unlearned_model, tokenizer, prompt)
        found = check_leakage(generated, target)
        status = "✓ PRESERVED" if found else "⚠ DEGRADED"
        print(f"\n  [{status}]")
        print(f"  Prompt:    {prompt}")
        print(f"  Generated: {generated[:100]}")
        print(f"  Target:    '{target}' → {'FOUND' if found else 'NOT FOUND'}")
        if found:
            retain_ok += 1

    print(f"\n  Retain Score: {retain_ok}/{retain_total} general knowledge queries preserved")

    # ── Summary ──
    print_header("VERIFICATION SUMMARY")

    m = result.metrics
    print(f"""
  ┌─────────────────────────────────────────────┐
  │ Before Unlearning: {before_leaks}/{before_total} queries leaked target     │
  │ After Unlearning:  {after_leaks}/{after_total} queries leaked target     │
  │ Knowledge Retained: {retain_ok}/{retain_total} general queries preserved │
  ├─────────────────────────────────────────────┤
  │ Direct Forget Accuracy:  {m.direct_forget_accuracy:.6f}            │
  │ Indirect Leakage Rate:   {m.indirect_leakage_rate:.6f}            │
  │ Retain Utility Drop:     {m.retain_utility_drop:.6f}            │
  │ Reconstruction Prob:     {m.reconstruction_probability:.6f}            │
  │ Parameter Drift Norm:    {m.parameter_drift_norm:.6f}            │
  ├─────────────────────────────────────────────┤
  │ Pipeline Time:           {elapsed:.1f}s                       │
  │ Risk Parameters:         {result.risk_params_count}                        │
  │ Cascade Iterations:      {result.cascade_iterations}                          │
  │ Certificate:             {result.certificate.certificate_id if result.certificate else 'N/A'}              │
  │ Compliance:              {result.certificate.compliance_tag if result.certificate else 'N/A'}               │
  └─────────────────────────────────────────────┘
""")

    # Final verdict
    erasure_success = after_leaks == 0
    retain_success = retain_ok >= retain_total - 1  # allow 1 miss
    overall = erasure_success and retain_success

    if overall:
        print("  ✅ VERDICT: PASS — Target knowledge erased, general knowledge preserved")
    elif erasure_success:
        print("  ⚠️  VERDICT: PARTIAL — Target erased but some general knowledge degraded")
    else:
        print("  ❌ VERDICT: FAIL — Target knowledge still leaking")

    print(f"\n{'═' * 64}\n")


if __name__ == "__main__":
    main()
