from datetime import datetime


def safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
