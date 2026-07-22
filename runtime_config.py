import logging
import os


APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def configure_logging() -> None:
    level = getattr(logging, LOG_LEVEL, None)
    if not isinstance(level, int):
        raise RuntimeError("LOG_LEVEL must be a valid Python logging level")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def validate_runtime_config() -> None:
    if APP_ENV not in {"development", "test", "production"}:
        raise RuntimeError("APP_ENV must be development, test, or production")
    if APP_ENV != "production":
        return

    jwt_secret = os.getenv("JWT_SECRET", "")
    if len(jwt_secret) < 32 or jwt_secret == "replace_with_a_long_random_secret":
        raise RuntimeError("Production requires a unique JWT_SECRET of at least 32 characters")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError("Production requires GEMINI_API_KEY or GOOGLE_API_KEY")

    origins = [value.strip() for value in os.getenv("CORS_ORIGINS", "").split(",") if value.strip()]
    if not origins or "*" in origins:
        raise RuntimeError("Production requires explicit CORS_ORIGINS")
