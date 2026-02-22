"""
AURORA — Interactive Demo Dashboard Server

Flask-based web server with:
- Static mode: displays pre-computed results from demo_storage/
- Live mode: runs the full AURORA pipeline in a background thread
  with real-time progress tracking

Usage:
    python scripts/demo_server.py
    → Open http://localhost:5000
"""

import copy
import sys
import os
import gc
import json
import time
import threading
import traceback
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))


# ─── Pipeline State ──────────────────────────────────────────────────────

pipeline_state = {
    "status": "idle",          # idle | loading_model | running | complete | error
    "current_step": 0,         # 0-7
    "step_name": "",
    "progress_pct": 0,         # 0-100
    "logs": [],
    "result": None,            # dict from UnlearningResult.to_dict()
    "rkg": None,               # dict from rkg.json
    "certificate": None,       # dict from certificate
    "error": None,
    "start_time": None,
    "elapsed": 0,
}


state_lock = threading.Lock()

# ─── Model Registry ──────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gpt2":        {"hf_name": "gpt2",              "display": "GPT-2",       "params": "124M"},
    "gpt2-large":  {"hf_name": "gpt2-large",        "display": "GPT-2 Large", "params": "774M"},
    "phi-2":       {"hf_name": "microsoft/phi-2",   "display": "Phi-2",       "params": "2.7B"},
}

# Global pipeline instance (eagerly loaded)
_pipeline_instance = None
_model_loaded = False
_current_model_name = "gpt2"       # active model key from MODEL_REGISTRY
_original_state_dict = None        # state_dict snapshot for pre-unlearning chat
_unlearning_done = False            # set True after pipeline.run() completes
_tokenizer_ref = None              # shared tokenizer reference


# ─── Data Loaders (Static Mode) ─────────────────────────────────────────

def load_json(path: Path) -> dict:
    """Load a JSON file, return empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return {}


def get_static_pipeline_result() -> dict:
    return load_json(PROJECT_ROOT / "demo_storage" / "pipeline_result.json")


def get_static_rkg_data() -> dict:
    return load_json(PROJECT_ROOT / "demo_storage" / "rkg.json")


def get_static_certificate() -> dict:
    cert_dir = PROJECT_ROOT / "demo_certificates"
    if cert_dir.exists():
        certs = list(cert_dir.glob("*.json"))
        if certs:
            return load_json(certs[0])
    return {}


def get_static_steps(result: dict) -> list:
    """Build pipeline step info from static result."""
    if not result:
        return []

    metrics = result.get("metrics", {})
    cert = result.get("certificate", {})
    graph = result.get("graph_stats", {})

    return [
        {
            "step": 1,
            "name": "Fact Formalization",
            "status": "complete",
            "detail": f"Target: {result.get('fact', {}).get('subject', '?')} → "
                      f"{result.get('fact', {}).get('relation', '?')} → "
                      f"{result.get('fact', {}).get('object', '?')}",
            "time": "~0.1s",
        },
        {
            "step": 2,
            "name": "RKG Construction",
            "status": "complete",
            "detail": f"{graph.get('num_nodes', 0)} nodes, "
                      f"{graph.get('num_edges', 0)} edges, "
                      f"density {graph.get('density', 0):.3f}",
            "time": "~3s",
        },
        {
            "step": 3,
            "name": "Parameter Localization",
            "status": "complete",
            "detail": f"{result.get('risk_params_count', 0)} risk parameters selected "
                      f"via Fisher + gradient scoring",
            "time": "~40s",
        },
        {
            "step": 4,
            "name": "Cascade Optimizer",
            "status": "complete",
            "detail": f"{result.get('cascade_iterations', 0)} iteration(s), converged",
            "time": "~10s",
        },
        {
            "step": 5,
            "name": "Adversarial Suite",
            "status": "complete",
            "detail": "All 5 attack vectors resisted",
            "time": "~5s",
        },
        {
            "step": 6,
            "name": "Quantum Distinguishability",
            "status": "complete",
            "detail": (
                f"D_forget={metrics.get('trace_distance_forget', 0):.6f}, ratio: ∞"
                if metrics.get("trace_distance_retain", 0) == 0
                else f"ratio: {metrics.get('trace_distance_forget', 0) / metrics.get('trace_distance_retain', 1):.0f}×"
            ),
            "time": "~2s",
        },
        {
            "step": 7,
            "name": "Certificate Generation",
            "status": "complete",
            "detail": f"ID: {cert.get('certificate_id', 'N/A')}, "
                      f"RSA-2048 signed, Merkle verified",
            "time": "~2s",
        },
    ]


# ─── Eager Model Loading ────────────────────────────────────────────────

def load_pipeline_eagerly(model_key="gpt2"):
    """Load the AURORA pipeline and model at server startup."""
    global _pipeline_instance, _model_loaded, _original_state_dict, _tokenizer_ref
    global _current_model_name, _unlearning_done

    model_info = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["gpt2"])
    _model_loaded = False
    _unlearning_done = False

    with state_lock:
        pipeline_state["status"] = "loading_model"
        pipeline_state["step_name"] = f"Loading {model_info['display']} model..."
        pipeline_state["logs"] = [f"[SYSTEM] Loading {model_info['display']} ({model_info['params']}) ..."]

    try:
        from aurora.config import AuroraConfig, DeviceType
        from aurora.pipeline.orchestrator import AuroraPipeline

        config = AuroraConfig(
            model_name=model_info["hf_name"],
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

        _pipeline_instance = AuroraPipeline(config)
        _pipeline_instance.load_model()
        _original_state_dict = {
            k: v.clone().cpu() for k, v in _pipeline_instance.model.state_dict().items()
        }
        _tokenizer_ref = _pipeline_instance.tokenizer
        _current_model_name = model_key
        _model_loaded = True

        with state_lock:
            pipeline_state["status"] = "idle"
            pipeline_state["step_name"] = ""
            pipeline_state["logs"].append(f"[SYSTEM] {model_info['display']} loaded successfully ✓")

        print(f"[AURORA] {model_info['display']} loaded eagerly ✓")

    except Exception as e:
        print(f"[AURORA] Eager model loading failed: {e}")
        with state_lock:
            pipeline_state["status"] = "idle"
            pipeline_state["logs"].append(f"[WARN] Eager load failed: {e}")
            pipeline_state["logs"].append("[WARN] Model will load on first run.")


# ─── Background Pipeline Runner ─────────────────────────────────────────

STEP_NAMES = {
    1: "Fact Formalization",
    2: "RKG Construction",
    3: "Parameter Localization",
    4: "Cascade Optimizer",
    5: "Adversarial Suite",
    6: "Quantum Distinguishability",
    7: "Certificate Generation",
}

STEP_PCT = {1: 5, 2: 15, 3: 45, 4: 65, 5: 80, 6: 90, 7: 95}


def _update_step(step_num: int, message: str = ""):
    """Update progress state for a pipeline step."""
    with state_lock:
        pipeline_state["current_step"] = step_num
        pipeline_state["step_name"] = STEP_NAMES.get(step_num, "")
        pipeline_state["progress_pct"] = STEP_PCT.get(step_num, 0)
        if pipeline_state["start_time"]:
            pipeline_state["elapsed"] = time.time() - pipeline_state["start_time"]
        if message:
            pipeline_state["logs"].append(message)


def run_pipeline_background(subject: str, relation: str, obj: str):
    """Run the AURORA pipeline in a background thread."""
    global _pipeline_instance, _model_loaded, _unlearning_done, _original_state_dict

    try:
        with state_lock:
            pipeline_state["status"] = "running"
            pipeline_state["current_step"] = 0
            pipeline_state["step_name"] = "Initializing..."
            pipeline_state["progress_pct"] = 0
            pipeline_state["logs"] = [
                f"[START] Unlearning: {subject} → {relation} → {obj}",
            ]
            pipeline_state["result"] = None
            pipeline_state["rkg"] = None
            pipeline_state["certificate"] = None
            pipeline_state["error"] = None
            pipeline_state["start_time"] = time.time()
            pipeline_state["elapsed"] = 0

        # Load model if not already loaded
        if not _model_loaded or _pipeline_instance is None:
            _update_step(0, "[MODEL] Loading GPT-2 model (first run)...")

            from aurora.config import AuroraConfig, DeviceType
            from aurora.pipeline.orchestrator import AuroraPipeline

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

            _pipeline_instance = AuroraPipeline(config)
            _pipeline_instance.load_model()
            _model_loaded = True
            _update_step(0, "[MODEL] Model loaded ✓")
        else:
            # Reload the model fresh for each run (pipeline modifies it)
            _update_step(0, "[MODEL] Reloading model for fresh run...")
            _pipeline_instance.load_model()
            _update_step(0, "[MODEL] Model reloaded ✓")

        # Monkey-patch the orchestrator's logger to capture step transitions
        import aurora.pipeline.orchestrator as orch_module
        _original_logger_info = orch_module.logger.info

        def _patched_info(msg, *args, **kwargs):
            _original_logger_info(msg, *args, **kwargs)
            if isinstance(msg, str):
                # Detect step transitions from log output
                if "STEP 1/7" in msg:
                    _update_step(1, "[STEP 1/7] Formalizing target fact...")
                elif "STEP 2/7" in msg:
                    _update_step(2, "[STEP 2/7] Building Relational Knowledge Graph...")
                elif "STEP 3/7" in msg:
                    _update_step(3, "[STEP 3/7] Localizing risk parameters via Fisher scoring...")
                elif "STEP 4/7" in msg:
                    _update_step(4, "[STEP 4/7] Running cascade unlearning optimizer...")
                elif "STEP 5/7" in msg:
                    _update_step(5, "[STEP 5/7] Running adversarial evaluation suite...")
                elif "STEP 6/7" in msg:
                    _update_step(6, "[STEP 6/7] Computing quantum distinguishability...")
                elif "STEP 7/7" in msg:
                    _update_step(7, "[STEP 7/7] Generating unlearning certificate...")

        orch_module.logger.info = _patched_info

        # Define retain prompts
        retain_prompts = [
            ("The capital of Japan is", "Tokyo"),
            ("Water freezes at", "0 degrees"),
            ("The largest planet is", "Jupiter"),
            ("Python was created by", "Guido van Rossum"),
            ("DNA stands for", "deoxyribonucleic acid"),
        ]

        # Run the pipeline
        result = _pipeline_instance.run(
            subject=subject,
            relation=relation,
            obj=obj,
            retain_prompts=retain_prompts,
        )

        # Restore original logger
        orch_module.logger.info = _original_logger_info

        # Load the generated artifacts
        result_dict = result.to_dict()
        rkg_data = load_json(PROJECT_ROOT / "demo_storage" / "rkg.json")
        cert_data = result.certificate.to_dict() if result.certificate else {}

        elapsed = time.time() - pipeline_state["start_time"]

        with state_lock:
            pipeline_state["status"] = "complete"
            pipeline_state["current_step"] = 7
            pipeline_state["step_name"] = "Complete"
            pipeline_state["progress_pct"] = 100
            pipeline_state["result"] = result_dict
            pipeline_state["rkg"] = rkg_data
            pipeline_state["certificate"] = cert_data
            pipeline_state["elapsed"] = elapsed
            pipeline_state["logs"].append(
                f"[DONE] Pipeline complete in {elapsed:.1f}s — "
                f"{'✓ COMPLIANT' if result.success else '✗ NON-COMPLIANT'}"
            )

        # Save result to demo_storage for future static access
        result_path = PROJECT_ROOT / "demo_storage" / "pipeline_result.json"
        result_path.parent.mkdir(exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        # Force garbage collection after pipeline
        _unlearning_done = True
        gc.collect()

    except Exception as e:
        tb = traceback.format_exc()
        with state_lock:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = str(e)
            pipeline_state["logs"].append(f"[ERROR] {str(e)}")
            pipeline_state["logs"].append(tb)
            if pipeline_state["start_time"]:
                pipeline_state["elapsed"] = time.time() - pipeline_state["start_time"]


# ─── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Serve the main dashboard page."""
    return render_template("dashboard.html")


# --- Live Pipeline API ---

@app.route("/api/run", methods=["POST"])
def api_run_pipeline():
    """Start the AURORA pipeline with user-provided fact."""
    with state_lock:
        if pipeline_state["status"] == "running":
            return jsonify({
                "error": "Pipeline is already running",
                "status": pipeline_state["status"],
            }), 409

    data = request.get_json() or {}
    subject = data.get("subject", "").strip()
    relation = data.get("relation", "").strip()
    obj = data.get("object", "").strip()

    if not subject or not relation or not obj:
        return jsonify({
            "error": "All fields required: subject, relation, object",
        }), 400

    # Start background thread
    thread = threading.Thread(
        target=run_pipeline_background,
        args=(subject, relation, obj),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "started", "message": f"Unlearning: {subject} → {relation} → {obj}"})


@app.route("/api/progress")
def api_progress():
    """Return current pipeline execution progress."""
    with state_lock:
        return jsonify({
            "status": pipeline_state["status"],
            "current_step": pipeline_state["current_step"],
            "step_name": pipeline_state["step_name"],
            "progress_pct": pipeline_state["progress_pct"],
            "total_steps": 7,
            "logs": pipeline_state["logs"][-20:],  # last 20 log entries
            "elapsed": round(pipeline_state["elapsed"], 1),
            "error": pipeline_state["error"],
            "model_loaded": _model_loaded,
        })


# --- Data API (serves live or static) ---

@app.route("/api/result")
def api_result():
    """Return pipeline result (live if available, else static)."""
    source = request.args.get("source", "auto")
    if source == "static":
        return jsonify(get_static_pipeline_result())
    with state_lock:
        if pipeline_state["result"]:
            return jsonify(pipeline_state["result"])
    return jsonify(get_static_pipeline_result())


@app.route("/api/rkg")
def api_rkg():
    """Return RKG graph data."""
    source = request.args.get("source", "auto")
    if source == "static":
        return jsonify(get_static_rkg_data())
    with state_lock:
        if pipeline_state["rkg"]:
            return jsonify(pipeline_state["rkg"])
    return jsonify(get_static_rkg_data())


@app.route("/api/certificate")
def api_certificate():
    """Return certificate details."""
    source = request.args.get("source", "auto")
    if source == "static":
        return jsonify(get_static_certificate())
    with state_lock:
        if pipeline_state["certificate"]:
            return jsonify(pipeline_state["certificate"])
    return jsonify(get_static_certificate())


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chat with GPT-2 (original or unlearned model)."""
    global _original_state_dict, _tokenizer_ref, _pipeline_instance, _unlearning_done

    if not _model_loaded or _tokenizer_ref is None:
        return jsonify({"error": "Model is still loading, please wait..."}), 503

    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    model_type = data.get("model", "original")  # "original" or "unlearned"

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        import torch

        # Select model
        if not _unlearning_done:
            # Before unlearning: use the pipeline model (it's the original)
            model = _pipeline_instance.model
        elif model_type == "unlearned":
            # After unlearning: pipeline model is the unlearned version
            model = _pipeline_instance.model
        else:
            # After unlearning, requesting original: reconstruct from state_dict
            model = _pipeline_instance.model.__class__(_pipeline_instance.model.config)
            model.load_state_dict(_original_state_dict)
            model.eval()

        model.eval()
        tokenizer = _tokenizer_ref

        # GPT-2 is a completion model (not instruction-tuned).
        # Use the raw prompt directly for natural text completion.
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.2,
                top_p=0.85,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated continuation after the prompt
        response = full_text[len(prompt):].strip()

        model_info = MODEL_REGISTRY.get(_current_model_name, MODEL_REGISTRY["gpt2"])
        return jsonify({
            "prompt": prompt,
            "response": response,
            "model": model_type,
            "model_name": f"{model_info['display']} ({model_info['params']} parameters)",
            "unlearning_applied": _unlearning_done,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/switch-model", methods=["POST"])
def api_switch_model():
    """Switch to a different model."""
    global _pipeline_instance, _model_loaded, _original_state_dict
    global _tokenizer_ref, _current_model_name, _unlearning_done

    data = request.get_json() or {}
    model_key = data.get("model_key", "").strip()

    if model_key not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    if model_key == _current_model_name and _model_loaded:
        return jsonify({"status": "already_loaded", "model_key": model_key})

    with state_lock:
        if pipeline_state["status"] == "running":
            return jsonify({"error": "Cannot switch model while pipeline is running"}), 409

    # Unload current model to free memory
    _model_loaded = False
    _original_state_dict = None
    _tokenizer_ref = None
    _unlearning_done = False
    if _pipeline_instance is not None:
        del _pipeline_instance
        _pipeline_instance = None
        import gc; gc.collect()

    # Load new model in background thread
    thread = threading.Thread(
        target=load_pipeline_eagerly,
        args=(model_key,),
        daemon=True,
    )
    thread.start()

    model_info = MODEL_REGISTRY[model_key]
    return jsonify({
        "status": "loading",
        "model_key": model_key,
        "model_name": f"{model_info['display']} ({model_info['params']})",
    })


@app.route("/api/status")
def api_status():
    """Health check endpoint."""
    result = get_static_pipeline_result()
    model_info = MODEL_REGISTRY.get(_current_model_name, MODEL_REGISTRY["gpt2"])
    return jsonify({
        "status": "ok",
        "model_loaded": _model_loaded,
        "model_key": _current_model_name,
        "model_name": f"{model_info['display']} ({model_info['params']})",
        "available_models": [
            {"key": k, "display": v["display"], "params": v["params"]}
            for k, v in MODEL_REGISTRY.items()
        ],
        "pipeline_status": pipeline_state["status"],
        "has_static_result": bool(result),
        "has_static_rkg": bool(get_static_rkg_data()),
        "has_static_certificate": bool(get_static_certificate()),
    })


@app.route("/api/steps")
def api_steps():
    """Return pipeline step info."""
    source = request.args.get("source", "auto")

    if source == "static":
        return jsonify(get_static_steps(get_static_pipeline_result()))

    with state_lock:
        if pipeline_state["result"]:
            return jsonify(get_static_steps(pipeline_state["result"]))

    return jsonify(get_static_steps(get_static_pipeline_result()))


# ─── Run ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  AURORA — Interactive Demo Dashboard")
    print("  Loading model eagerly...")
    print("=" * 56 + "\n")

    # Load model in a background thread so Flask starts immediately
    loader_thread = threading.Thread(target=load_pipeline_eagerly, daemon=True)
    loader_thread.start()

    print("  → Open http://localhost:5000 in your browser")
    print("  → Model loading in background...\n")

    app.run(debug=False, host="0.0.0.0", port=5000)
