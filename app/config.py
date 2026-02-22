from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    max_file_size_mb: int = 50
    allowed_origins: list[str] = ["*"]
    ocr_device: str = "gpu:0"
    ocr_lang: str = "korean"
    pdf_dpi: int = 200  # 200 DPI balances Structure accuracy (needs readable Korean chars) with VLM speed
    vlm_max_pixels: int = 1003520  # 1280*28*28, official recommendation for OCR/table tasks
    openrouter_api_key: str = ""
    openrouter_model: str = "qwen/qwen3-8b"
    table_correction_model: str = "qwen/qwen3-8b"
    table_correction_timeout: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
