import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from strategy_config import BUY_SCORE_MIN, RSI_OVERBOUGHT, RSI_OVERSOLD, RULE_VERSION, SELL_SCORE_MAX


TRADING_DAYS_PER_YEAR = 252
MAX_CURVE_POINTS = 500


def _number(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
        return round(number, digits) if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def build_signal_frame(history: pd.DataFrame, transaction_cost_bps: float = 5.0) -> pd.DataFrame:
    if history.empty or len(history) < 80:
        raise ValueError("At least 80 trading days are required for backtesting")

    frame = pd.DataFrame(index=history.index.copy())
    frame["close"] = history["Close"].astype(float)
    frame["asset_return"] = frame["close"].pct_change().fillna(0.0)
    frame["ma_20"] = frame["close"].rolling(20).mean()
    frame["ma_50"] = frame["close"].rolling(50).mean()

    delta = frame["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    frame["rsi_14"] = (100 - (100 / (1 + rs))).fillna(
        pd.Series(np.where(gain > 0, 100.0, 50.0), index=frame.index)
    )
    ema12 = frame["close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["close"].ewm(span=26, adjust=False).mean()
    frame["macd"] = ema12 - ema26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()

    score = pd.Series(0, index=frame.index, dtype=int)
    bullish = (frame["close"] > frame["ma_20"]) & (frame["ma_20"] > frame["ma_50"])
    bearish = (frame["close"] < frame["ma_20"]) & (frame["ma_20"] < frame["ma_50"])
    score += np.where(bullish, 2, np.where(bearish, -2, 0))
    score += np.where(
        frame["rsi_14"] >= RSI_OVERBOUGHT,
        -1,
        np.where(frame["rsi_14"] <= RSI_OVERSOLD, 1, 0),
    )
    score += np.where(frame["macd"] > frame["macd_signal"], 1, -1)
    frame["score"] = score
    frame["signal"] = np.where(
        score >= BUY_SCORE_MIN, 1, np.where(score <= SELL_SCORE_MAX, -1, 0)
    )

    # Today's close produces the signal; its position begins the next trading day.
    frame["position"] = frame["signal"].shift(1).fillna(0).astype(int)
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["cost"] = frame["turnover"] * (transaction_cost_bps / 10_000)
    frame["strategy_return"] = frame["position"] * frame["asset_return"] - frame["cost"]
    tested = frame.loc[frame["ma_50"].notna()].copy()

    # The first displayed date is the evaluation baseline. Both portfolios start
    # with the requested capital there; returns accrue from the following session.
    first = tested.index[0]
    tested.loc[first, ["asset_return", "strategy_return", "turnover", "cost"]] = 0.0
    tested.loc[first, "position"] = 0
    return tested


def _performance_metrics(
    returns: pd.Series, equity: pd.Series, initial_capital: float
) -> dict[str, float | None]:
    if returns.empty:
        raise ValueError("No testable returns remain after the warm-up period")
    total_return = equity.iloc[-1] / initial_capital - 1
    years = len(returns) / TRADING_DAYS_PER_YEAR
    annualized_return = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    volatility = returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = returns.mean() / returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR) if returns.std() > 0 else None
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return_percent": _number(total_return * 100, 2),
        "annualized_return_percent": _number(annualized_return * 100, 2),
        "annualized_volatility_percent": _number(volatility * 100, 2),
        "sharpe_ratio": _number(sharpe, 3),
        "max_drawdown_percent": _number(drawdown.min() * 100, 2),
    }


def simulate_backtest(
    history: pd.DataFrame,
    initial_capital: float = 10_000,
    transaction_cost_bps: float = 5.0,
    horizon_days: int = 20,
    include_equity_curve: bool = True,
) -> dict[str, Any]:
    frame = build_signal_frame(history, transaction_cost_bps)
    frame["strategy_equity"] = initial_capital * (1 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = initial_capital * (1 + frame["asset_return"]).cumprod()

    forward_return = frame["close"].shift(-horizon_days) / frame["close"] - 1
    actionable = frame["signal"] != 0
    eligible = actionable & forward_return.notna()
    correct = ((frame["signal"] == 1) & (forward_return > 0)) | ((frame["signal"] == -1) & (forward_return < 0))
    hit_rate = correct[eligible].mean() if eligible.any() else None
    active_returns = frame.loc[frame["position"] != 0, "strategy_return"]

    step = max(1, math.ceil(len(frame) / MAX_CURVE_POINTS))
    curve_frame = frame.iloc[::step]
    if curve_frame.index[-1] != frame.index[-1]:
        curve_frame = pd.concat([curve_frame, frame.iloc[[-1]]])
    equity_curve = [
        {
            "date": index.date().isoformat(),
            "strategy": _number(row["strategy_equity"], 2),
            "benchmark": _number(row["benchmark_equity"], 2),
            "close": _number(row["close"], 2),
            "signal": "BUY" if row["signal"] == 1 else "SELL" if row["signal"] == -1 else "HOLD",
        }
        for index, row in curve_frame.iterrows()
    ] if include_equity_curve else []

    return {
        "start_date": frame.index[0].date().isoformat(),
        "end_date": frame.index[-1].date().isoformat(),
        "trading_days": len(frame),
        "strategy": {
            **_performance_metrics(frame["strategy_return"], frame["strategy_equity"], initial_capital),
            "ending_value": _number(frame["strategy_equity"].iloc[-1], 2),
            "active_day_win_rate_percent": _number((active_returns > 0).mean() * 100, 2) if not active_returns.empty else None,
            "exposure_percent": _number((frame["position"] != 0).mean() * 100, 2),
            "position_changes": int((frame["turnover"] > 0).sum()),
        },
        "benchmark": {
            **_performance_metrics(frame["asset_return"], frame["benchmark_equity"], initial_capital),
            "ending_value": _number(frame["benchmark_equity"].iloc[-1], 2),
        },
        "signals": {
            "buy": int((frame["signal"] == 1).sum()),
            "hold": int((frame["signal"] == 0).sum()),
            "sell": int((frame["signal"] == -1).sum()),
            "forward_horizon_days": horizon_days,
            "actionable_signal_hit_rate_percent": _number(hit_rate * 100, 2) if hit_rate is not None else None,
            "evaluated_actionable_signals": int(eligible.sum()),
        },
        "equity_curve": equity_curve,
    }


def run_backtest(
    ticker: str,
    period: str = "5y",
    initial_capital: float = 10_000,
    transaction_cost_bps: float = 5.0,
    horizon_days: int = 20,
    include_equity_curve: bool = False,
) -> dict[str, Any]:
    history = yf.Ticker(ticker).history(
        period=period, interval="1d", auto_adjust=True, actions=False, timeout=20
    )
    if history.empty:
        raise ValueError(f"No historical market data found for ticker {ticker}")
    results = simulate_backtest(
        history, initial_capital, transaction_cost_bps, horizon_days, include_equity_curve
    )
    strategy_return = results["strategy"]["total_return_percent"]
    benchmark_return = results["benchmark"]["total_return_percent"]
    difference = round(strategy_return - benchmark_return, 2)
    verdict = "OUTPERFORMED" if difference > 0 else "MATCHED" if difference == 0 else "UNDERPERFORMED"
    overview = {
        "verdict": verdict,
        "plain_english_summary": (
            f"The rules-based strategy returned {strategy_return}% versus {benchmark_return}% "
            f"from simply buying and holding {ticker}. It {verdict.lower()} the benchmark by "
            f"{abs(difference)} percentage points."
        ),
        "strategy_return_percent": strategy_return,
        "benchmark_return_percent": benchmark_return,
        "difference_percentage_points": difference,
        "strategy_ending_value": results["strategy"]["ending_value"],
        "benchmark_ending_value": results["benchmark"]["ending_value"],
    }
    advanced = {
        "trading_days": results["trading_days"],
        "strategy": results["strategy"],
        "benchmark": results["benchmark"],
        "signals": results["signals"],
    }
    return {
        "ticker": ticker,
        "period": period,
        "initial_capital": initial_capital,
        "transaction_cost_bps": transaction_cost_bps,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "version": RULE_VERSION,
            "decision_boundaries": {
                "buy_score_min": BUY_SCORE_MIN,
                "sell_score_max": SELL_SCORE_MAX,
                "rsi_overbought": RSI_OVERBOUGHT,
                "rsi_oversold": RSI_OVERSOLD,
            },
            "execution": "Signals use closing data and positions begin on the following trading day.",
            "prices": "Yahoo Finance adjusted daily prices.",
            "benchmark": "Buy and hold of the same ticker over the test window.",
            "risk_free_rate": "0% for Sharpe ratio calculation.",
        },
        "start_date": results["start_date"],
        "end_date": results["end_date"],
        "proof": {
            "source": "Yahoo Finance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}/history/",
            "adjusted_prices": True,
            "input_parameters": {
                "ticker": ticker,
                "period": period,
                "initial_capital": initial_capital,
                "transaction_cost_bps": transaction_cost_bps,
                "forward_horizon_days": horizon_days,
            },
            "integrity_checks": {
                "look_ahead_prevention": "A close-of-day signal is shifted one trading day before earning returns.",
                "transaction_costs": "Costs are deducted whenever the simulated position changes.",
                "benchmark": "The strategy is compared with buy-and-hold over the identical test window.",
                "shared_rules": f"Live analysis and backtesting both use {RULE_VERSION}.",
            },
            "formulas": {
                "ending_value": "Initial capital multiplied by the cumulative product of (1 + daily return).",
                "total_return_percent": "(Ending value / initial capital - 1) x 100.",
                "annualized_return_percent": "Compound return scaled to 252 trading days per year.",
                "annualized_volatility_percent": "Daily return standard deviation x sqrt(252) x 100.",
                "maximum_drawdown_percent": "Largest percentage decline from a previous equity peak.",
                "difference_percentage_points": "Strategy return percent minus benchmark return percent.",
            },
        },
        "overview": overview,
        "advanced": advanced,
        **({"equity_curve": results["equity_curve"]} if include_equity_curve else {}),
        "disclaimer": (
            "Hypothetical backtest for research only. It does not represent actual trading, "
            "does not include taxes or market impact, and does not predict future performance."
        ),
    }
