from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    max_file_size_mb: int = 50
    allowed_origins: list[str] = ["http://localhost:3000", "*"]
    ocr_device: str = "gpu:0"
    ocr_lang: str = "korean"
    pdf_dpi: int = 300

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
