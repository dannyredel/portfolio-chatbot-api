from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    anthropic_model: str = "claude-haiku-4-5"

    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    allowed_origins: str = "https://dannyredel.github.io,http://localhost:4000"
    admin_api_key: str

    top_k: int = 5
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 800
    rate_limit_per_min: int = 20
    daily_request_cap: int = 1000

    knowledge_dir: str = "../dannyredel.github.io/knowledge"
    index_path: str = "data/index.pkl"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
