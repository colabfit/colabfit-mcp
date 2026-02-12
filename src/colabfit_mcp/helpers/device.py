def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "cuda", name
        return "cpu", None
    except ImportError:
        return "cpu", None
