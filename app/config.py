from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    max_file_size_mb: int = 50
    allowed_origins: list[str] = ["*"]
    ocr_device: str = "gpu:0"
    ocr_lang: str = "korean"
    pdf_dpi: int = 300
    vlm_max_pixels: int = 1605632  # 2048*28*28, higher res for Korean chars
    openrouter_api_key: str = ""
    openrouter_model: str = "qwen/qwen3-8b"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
