import re


def parse_training_log(output: str) -> dict:
    """Extract key metrics from mace_run_train output."""
    metrics = {}

    epoch_pattern = re.compile(
        r"Epoch\s+(\d+).*?loss[=:]\s*([\d.eE+-]+)", re.IGNORECASE
    )
    matches = epoch_pattern.findall(output)
    if matches:
        last_epoch, last_loss = matches[-1]
        metrics["last_epoch"] = int(last_epoch)
        metrics["last_loss"] = float(last_loss)
        metrics["total_epochs_run"] = int(last_epoch)

    mae_pattern = re.compile(
        r"(energy|forces|stress).*?MAE[=:]\s*([\d.eE+-]+)", re.IGNORECASE
    )
    for prop, val in mae_pattern.findall(output):
        metrics[f"{prop.lower()}_mae"] = float(val)

    return metrics


def diagnose_failure(stdout: str, stderr: str) -> str:
    combined = (stdout or "") + (stderr or "")
    lower = combined.lower()

    if "cuda out of memory" in lower or "out of memory" in lower:
        return (
            "GPU out of memory. Try reducing batch_size or using device='cpu'."
        )
    if "no such file" in lower or "filenotfounderror" in lower:
        return "Training file not found. Verify the train_file path."
    if "nan" in lower and "loss" in lower:
        return (
            "Training diverged (NaN loss). Try reducing lr or increasing "
            "batch_size."
        )
    if "killed" in lower or "sigkill" in lower:
        return "Process was killed, likely due to memory pressure."
    if "modulenotfounderror" in lower:
        module = re.search(r"No module named '(\S+)'", combined)
        name = module.group(1) if module else "unknown"
        return f"Missing dependency: {name}. Check installation."

    return "Training failed. Check stdout/stderr for details."
