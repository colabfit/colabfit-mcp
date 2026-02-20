import functools
import sys


@functools.lru_cache(maxsize=None)
def detect_device() -> tuple[str, str | None]:
    try:
        import torch
    except ImportError:
        return "cpu", None

    candidates = []
    if torch.cuda.is_available():
        candidates.append(("cuda", None))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        candidates.append(("mps", "Apple MPS"))
    candidates.append(("cpu", None))

    for device, name in candidates:
        try:
            t = torch.tensor([1.0], device=device)
            _ = t + t
            if device == "cuda":
                name = torch.cuda.get_device_name(0)
            return device, name
        except Exception as exc:
            print(f"Device probe failed for {device!r}: {exc}", file=sys.stderr)
            continue

    raise RuntimeError("No usable torch device found")
