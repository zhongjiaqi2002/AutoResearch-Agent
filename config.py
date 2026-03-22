"""
Config file
Automatically read configuration from ".env" file or environment variables
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # API Keys
    DASHSCOPE_API_KEY: str = ""
    BOCHAAI_API_KEY: str = ""

    # LLM configuration
    LLM_MODEL: str = "qwen-max"
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Embedding configuration
    EMBEDDING_MODEL: str = "text-embedding-v4"
    EMBEDDING_DIMENSIONS: int = 1024

    # BOCHA Search configuration
    BOCHAAI_BASE_URL: str = "https://api.bochaai.com/v1"

    # MySQL configuration
    MYSQL_USER: str = ""
    MYSQL_PASSWORD: str = ""
    MYSQL_HOST: str = ""
    MYSQL_PORT: int = 3306
    MYSQL_DB: str = ""

    # Milvus configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "finance_reports"

    # PDF path
    SAMPLE_PDF_PATH: str = "data/CICC Annual Report.pdf"

    # Code execution configuration
    CODE_EXECUTOR_TIMEOUT: int = 30
    CODE_EXECUTOR_MAX_OUTPUT: int = 10000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# SQLAlchemy MySQL Connection URL
SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}"
)



