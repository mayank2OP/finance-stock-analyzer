import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

load_dotenv()

from analysis_service import analyze as run_analysis
from auth_service import authenticate_user, create_access_token, get_current_user, hash_password
from backtesting_service import run_backtest
from database import DATABASE_URL, get_db, init_database
from job_service import create_analysis_job, get_analysis_job
from models import SavedAnalysis, SavedBacktest, User, WatchlistItem
from rate_limit import enforce_rate_limit
from runtime_config import configure_logging, env_int, validate_runtime_config


configure_logging()
validate_runtime_config()
logger = logging.getLogger(__name__)
if os.getenv("APP_ENV", "development").strip().lower() == "production" and DATABASE_URL.startswith("sqlite"):
    logger.warning("Production is using SQLite; configure a persistent volume to retain user data")

ANALYSIS_USER_LIMIT = env_int("ANALYSIS_REQUESTS_PER_HOUR", 5)
ANALYSIS_IP_LIMIT = env_int("ANALYSIS_IP_REQUESTS_PER_HOUR", 10)
BACKTEST_USER_LIMIT = env_int("BACKTEST_REQUESTS_PER_HOUR", 20)
REGISTRATION_IP_LIMIT = env_int("REGISTRATION_REQUESTS_PER_HOUR", 5)
LOGIN_IP_LIMIT = env_int("LOGIN_ATTEMPTS_PER_15_MINUTES", 15)


app = FastAPI(
    title="StockPilot API",
    version="3.1.0",
    description="Evidence-first stock research using Yahoo Finance and Gemini agents.",
)

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

init_database()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _limit_analysis(request: Request, user: User) -> None:
    enforce_rate_limit(f"analysis:user:{user.id}", ANALYSIS_USER_LIMIT, 3600)
    enforce_rate_limit(f"analysis:ip:{_client_ip(request)}", ANALYSIS_IP_LIMIT, 3600)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure", extra={"request_id": request_id})
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info(json.dumps({
        "event": "request_completed",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }))
    return response


class StockRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=15, examples=["AAPL"])

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-^=]{0,14}", ticker):
            raise ValueError("Ticker contains unsupported characters")
        return ticker


class BacktestRequest(StockRequest):
    period: Literal["1y", "2y", "5y", "10y"] = "5y"
    initial_capital: float = Field(default=10_000, gt=0, le=100_000_000)
    transaction_cost_bps: float = Field(default=5.0, ge=0, le=100)
    horizon_days: int = Field(default=20, ge=1, le=60)
    include_equity_curve: bool = False


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        return value.strip().lower()


class SavedResultRequest(StockRequest):
    result: dict[str, Any]


def _utc_timestamp(value: datetime) -> str:
    """Serialize SQLite's timezone-naive UTC datetimes without changing the instant."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _user_view(user: User) -> dict[str, Any]:
    return {"id": user.id, "username": user.username, "created_at": _utc_timestamp(user.created_at)}


def _saved_view(item: SavedAnalysis | SavedBacktest) -> dict[str, Any]:
    return {
        "id": item.id,
        "ticker": item.ticker,
        "result": item.result,
        "created_at": _utc_timestamp(item.created_at),
    }


@app.get("/", tags=["health"])
def home():
    return {"status": "ok", "service": "stockpilot", "version": app.version}


@app.get("/health", tags=["health"])
def health(db: Annotated[Session, Depends(get_db)]):
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        logger.exception("Database health check failed")
        raise HTTPException(status_code=503, detail="Service is temporarily unavailable") from exc
    return {"status": "healthy", "database": "healthy"}


@app.post("/auth/register", status_code=status.HTTP_201_CREATED, tags=["authentication"])
def register(request: RegisterRequest, http_request: Request, db: Annotated[Session, Depends(get_db)]):
    enforce_rate_limit(f"register:ip:{_client_ip(http_request)}", REGISTRATION_IP_LIMIT, 3600)
    user = User(username=request.username, password_hash=hash_password(request.password))
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Username is already registered") from exc
    return {"success": True, "data": _user_view(user)}


@app.post("/auth/token", tags=["authentication"])
def login(
    request: Request,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
):
    enforce_rate_limit(f"login:ip:{_client_ip(request)}", LOGIN_IP_LIMIT, 900)
    user = authenticate_user(db, form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": create_access_token(user), "token_type": "bearer"}


@app.get("/auth/me", tags=["authentication"])
def current_user(user: Annotated[User, Depends(get_current_user)]):
    return {"success": True, "data": _user_view(user)}


@app.post("/analyze", tags=["analysis"])
def analyze(
    request: StockRequest,
    http_request: Request,
    user: Annotated[User, Depends(get_current_user)],
):
    _limit_analysis(http_request, user)
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=503, detail="Gemini is not configured")
    try:
        return {"success": True, "data": run_analysis(request.ticker)}
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Analysis provider failed") from exc


@app.post("/analysis-jobs", status_code=status.HTTP_202_ACCEPTED, tags=["analysis"])
def start_analysis_job(
    request: StockRequest,
    http_request: Request,
    user: Annotated[User, Depends(get_current_user)],
):
    _limit_analysis(http_request, user)
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=503, detail="Gemini is not configured")
    return {"success": True, "data": create_analysis_job(request.ticker, run_analysis, user.id)}


@app.get("/analysis-jobs/{job_id}", tags=["analysis"])
def analysis_job(job_id: str, user: Annotated[User, Depends(get_current_user)]):
    job = get_analysis_job(job_id, user.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return {"success": True, "data": job}


@app.post("/backtest", tags=["backtesting"])
def backtest(
    request: BacktestRequest,
    http_request: Request,
    user: Annotated[User, Depends(get_current_user)],
):
    enforce_rate_limit(f"backtest:user:{user.id}", BACKTEST_USER_LIMIT, 3600)
    try:
        data = run_backtest(
            ticker=request.ticker,
            period=request.period,
            initial_capital=request.initial_capital,
            transaction_cost_bps=request.transaction_cost_bps,
            horizon_days=request.horizon_days,
            include_equity_curve=request.include_equity_curve,
        )
        return {"success": True, "data": data}
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Backtest data provider failed") from exc


@app.get("/watchlist", tags=["portfolio"])
def list_watchlist(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    items = db.scalars(
        select(WatchlistItem)
        .where(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.created_at.desc())
    ).all()
    return {"success": True, "data": [
        {"id": item.id, "ticker": item.ticker, "created_at": _utc_timestamp(item.created_at)}
        for item in items
    ]}


@app.post("/watchlist", status_code=status.HTTP_201_CREATED, tags=["portfolio"])
def add_watchlist(
    request: StockRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    item = WatchlistItem(user_id=user.id, ticker=request.ticker)
    db.add(item)
    try:
        db.commit()
        db.refresh(item)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Ticker is already on your watchlist") from exc
    return {
        "success": True,
        "data": {"id": item.id, "ticker": item.ticker, "created_at": _utc_timestamp(item.created_at)},
    }


@app.delete("/watchlist/{ticker}", status_code=status.HTTP_204_NO_CONTENT, tags=["portfolio"])
def remove_watchlist(
    ticker: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    normalized = StockRequest(ticker=ticker).ticker
    item = db.scalar(select(WatchlistItem).where(
        WatchlistItem.user_id == user.id, WatchlistItem.ticker == normalized
    ))
    if item is None:
        raise HTTPException(status_code=404, detail="Ticker is not on your watchlist")
    db.delete(item)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _list_saved(model, user: User, db: Session) -> dict[str, Any]:
    items = db.scalars(
        select(model).where(model.user_id == user.id).order_by(model.created_at.desc())
    ).all()
    return {"success": True, "data": [_saved_view(item) for item in items]}


def _save_result(model, request: SavedResultRequest, user: User, db: Session) -> dict[str, Any]:
    item = model(user_id=user.id, ticker=request.ticker, result=request.result)
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"success": True, "data": _saved_view(item)}


def _delete_saved(model, item_id: int, user: User, db: Session) -> Response:
    item = db.scalar(select(model).where(model.id == item_id, model.user_id == user.id))
    if item is None:
        raise HTTPException(status_code=404, detail="Saved result not found")
    db.delete(item)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/saved-analyses", tags=["portfolio"])
def list_saved_analyses(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return _list_saved(SavedAnalysis, user, db)


@app.post("/saved-analyses", status_code=status.HTTP_201_CREATED, tags=["portfolio"])
def save_analysis(
    request: SavedResultRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return _save_result(SavedAnalysis, request, user, db)


@app.delete("/saved-analyses/{item_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["portfolio"])
def delete_analysis(
    item_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return _delete_saved(SavedAnalysis, item_id, user, db)


@app.get("/saved-backtests", tags=["portfolio"])
def list_saved_backtests(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return _list_saved(SavedBacktest, user, db)


@app.post("/saved-backtests", status_code=status.HTTP_201_CREATED, tags=["portfolio"])
def save_backtest(
    request: SavedResultRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return _save_result(SavedBacktest, request, user, db)


@app.delete("/saved-backtests/{item_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["portfolio"])
def delete_backtest(
    item_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return _delete_saved(SavedBacktest, item_id, user, db)
