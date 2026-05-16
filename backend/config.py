from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_key: SecretStr
    google_api_key: str
    tavily_key: SecretStr
    llama_scout: str
    llama_versatile: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()  # type: ignore
