import subprocess
import sys
from pathlib import Path

from colabfit_mcp.config import (
    COLABFIT_ENERGY_KEY,
    COLABFIT_FORCES_KEY,
    COLABFIT_STRESS_KEY,
    FINE_TUNE_DEFAULTS,
    FOUNDATION_MODEL,
    MODEL_DIR,
    TRAINING_TIMEOUT,
)
from colabfit_mcp.helpers.dataset_resolver import resolve_train_file
from colabfit_mcp.helpers.device import detect_device
from colabfit_mcp.helpers.training import diagnose_failure, parse_training_log
from colabfit_mcp.helpers.xyz import analyze_xyz


def fine_tune_mace(
    train_file: str | None = None,
    model_name: str = "colabfit_mace",
    max_num_epochs: int = 50,
    device: str = None,
    elements: list[str] | None = None,
) -> dict:
    """Fine-tune the MACE-MP-0 foundation model on a dataset.

    When train_file is omitted, automatically discovers suitable datasets
    in the local download directory. If a matching dataset is found, it is
    used directly. If no match or only a partial match is found, returns
    guidance to search and download from ColabFit.

    The foundation model (default MACE-MP-0-a small, can override in .env)
    is fine-tuned on your specific dataset. Architecture is inherited from the
    foundation model.

    Args:
        train_file: Path to training XYZ file (extxyz format).
            If None, auto-discovers from local datasets.
        model_name: Name for the fine-tuned model (default "colabfit_mace").
        max_num_epochs: Maximum training epochs (default 50).
        device: "cuda" or "cpu" (auto-detected if None).
        elements: Chemical elements for dataset matching when
            train_file is not provided (e.g. ["Si", "O"]).

    Returns:
        Dict with model path, training metrics, and next_step guidance.
    """
    if train_file is None:
        resolved_path, info = resolve_train_file(elements=elements)
        if resolved_path is None:
            return info
        train_file = resolved_path

    train_path = Path(train_file)
    if not train_path.exists():
        return {
            "success": False,
            "error": f"Training file not found: {train_file}",
        }

    if device is None:
        device, _ = detect_device()

    analysis = analyze_xyz(train_path)
    loss = "stress" if analysis.get("has_stress") else "weighted"

    model_dir = MODEL_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    defaults = FINE_TUNE_DEFAULTS.copy()
    cmd = [
        "mace_run_train",
        f"--name={model_name}",
        f"--train_file={train_file}",
        f"--valid_fraction={defaults['valid_fraction']}",
        f"--foundation_model={FOUNDATION_MODEL}",
        "--E0s=foundation",
        f"--loss={loss}",
        f"--energy_key={COLABFIT_ENERGY_KEY}",
        f"--forces_key={COLABFIT_FORCES_KEY}",
        f"--stress_key={COLABFIT_STRESS_KEY}",
        f"--max_num_epochs={max_num_epochs}",
        f"--batch_size={defaults['batch_size']}",
        f"--valid_batch_size={defaults['valid_batch_size']}",
        f"--lr={defaults['lr']}",
        f"--device={device}",
        f"--default_dtype={defaults['default_dtype']}",
        f"--num_workers={defaults['num_workers']}",
        f"--seed={defaults['seed']}",
        f"--model_dir={model_dir}",
        f"--results_dir={model_dir}",
        f"--checkpoints_dir={model_dir}",
    ]
    if defaults["pin_memory"]:
        cmd.append("--pin_memory")
    if device == "cuda":
        cmd.append("--enable_cueq")
    if defaults["swa"]:
        cmd.append("--swa")
    if defaults["ema"]:
        cmd.append("--ema")
        cmd.append(f"--ema_decay={defaults['ema_decay']}")

    log_file = model_dir / "training.log"

    try:
        with open(log_file, "w", buffering=1) as log:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            log_lines = []
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                log.write(line)
                print(line, end="", file=sys.stderr)
                log_lines.append(line)

            process.wait(timeout=TRAINING_TIMEOUT)

        model_files = list(model_dir.glob("*.model"))
        if not model_files:
            log_content = "".join(log_lines)
            diag = diagnose_failure(log_content, "")
            return {
                "success": False,
                "error": "No model file produced",
                "diagnosis": diag,
                "log_file": str(log_file),
                "stdout": log_content[-2000:],
            }

        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        log_content = "".join(log_lines)
        metrics = parse_training_log(log_content)

        return {
            "success": True,
            "model_path": str(model_path),
            "model_dir": str(model_dir),
            "log_file": str(log_file),
            "device": device,
            "foundation_model": FOUNDATION_MODEL,
            "loss_type": loss,
            "elements": analysis.get("elements", []),
            "metrics": metrics,
            "next_step": (
                f"Model saved at {model_path}. Training log available at {log_file}. "
                "Use deploy_model to export as TorchScript and install as a KIM Portable Model."  # noqa: E501
            ),
        }
    except subprocess.TimeoutExpired:
        process.kill()
        process.stdout.read()
        process.wait()
        return {
            "success": False,
            "error": "Training timed out (2 hour limit). "
            "Try reducing max_num_epochs.",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "mace_run_train not found. Ensure mace-torch is installed.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
