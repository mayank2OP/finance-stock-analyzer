import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yfinance as yf

from strategy_config import (
    BUY_SCORE_MIN,
    HIGH_RISK_VOLATILITY,
    MEDIUM_RISK_VOLATILITY,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RULE_VERSION,
    SELL_SCORE_MAX,
)


CACHE_TTL_SECONDS = 300
COLLECTION_TIMEOUT_SECONDS = 20
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def _finite(value: Any, digits: int = 2) -> float | None:
    try:
        number = float(value)
        return round(number, digits) if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> Any:
    if not isinstance(value, str) or not any(marker in value for marker in ("Ã", "â", "Â")):
        return value
    try:
        return value.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def _published_at_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        except ValueError:
            return value
    return str(value)


def _parse_news_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize both Yahoo's flat search news and nested ticker news payloads."""
    output: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for article in articles:
        content = article.get("content") if isinstance(article.get("content"), dict) else article
        provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
        canonical = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
        click_through = (
            content.get("clickThroughUrl")
            if isinstance(content.get("clickThroughUrl"), dict)
            else {}
        )
        title = _clean_text(content.get("title") or article.get("title"))
        url = (
            canonical.get("url")
            or click_through.get("url")
            or content.get("link")
            or article.get("link")
        )
        if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"}:
            continue
        publisher = _clean_text(
            provider.get("displayName")
            or content.get("publisher")
            or article.get("publisher")
            or urlparse(url).netloc.removeprefix("www.")
        )
        if not title or not publisher or url in seen_urls:
            continue
        seen_urls.add(url)
        output.append({
            "title": title,
            "publisher": publisher,
            "url": url,
            "published_at": _published_at_iso(
                content.get("pubDate")
                or content.get("providerPublishTime")
                or article.get("providerPublishTime")
            ),
        })
        if len(output) == 5:
            break
    return output


def calculate_metrics(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty or len(history) < 50:
        raise ValueError("At least 50 trading days are required for analysis")

    close = history["Close"].dropna()
    volume = history["Volume"].dropna()
    returns = close.pct_change().dropna()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    if pd.isna(rsi.iloc[-1]):
        rsi_value = 100.0 if gain.iloc[-1] > 0 else 50.0
    else:
        rsi_value = float(rsi.iloc[-1])

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    current = close.iloc[-1]
    lookback = close.iloc[-63] if len(close) >= 63 else close.iloc[0]

    return {
        "current_price": _finite(current),
        "change_3m_percent": _finite((current / lookback - 1) * 100),
        "high_3m": _finite(close.tail(63).max()),
        "low_3m": _finite(close.tail(63).min()),
        "average_volume_3m": _finite(volume.tail(63).mean(), 0),
        "annualized_volatility_percent": _finite(returns.std() * math.sqrt(252) * 100),
        "ma_20": _finite(ma20),
        "ma_50": _finite(ma50),
        "rsi_14": _finite(rsi_value),
        "macd": _finite(macd.iloc[-1], 4),
        "macd_signal": _finite(signal.iloc[-1], 4),
        "trend": "BULLISH" if current > ma20 > ma50 else "BEARISH" if current < ma20 < ma50 else "MIXED",
    }


def score_evidence(metrics: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    rule_trace: list[dict[str, Any]] = []
    trend = metrics["trend"]
    rsi = metrics["rsi_14"]
    macd = metrics["macd"]
    signal = metrics["macd_signal"]

    if trend == "BULLISH":
        score += 2
        reasons.append("Price is above both the 20-day and 50-day moving averages")
        trend_points = 2
    elif trend == "BEARISH":
        score -= 2
        reasons.append("Price is below both the 20-day and 50-day moving averages")
        trend_points = -2
    else:
        trend_points = 0
    rule_trace.append({
        "rule": "moving_average_trend",
        "observed": trend,
        "condition": "price > MA20 > MA50: +2; price < MA20 < MA50: -2",
        "score_contribution": trend_points,
    })

    rsi_points = 0
    if rsi is not None and rsi >= RSI_OVERBOUGHT:
        score -= 1
        reasons.append("RSI indicates overbought momentum")
        rsi_points = -1
    elif rsi is not None and rsi <= RSI_OVERSOLD:
        score += 1
        reasons.append("RSI indicates oversold momentum")
        rsi_points = 1
    rule_trace.append({
        "rule": "rsi_14",
        "observed": rsi,
        "condition": f"RSI >= {RSI_OVERBOUGHT}: -1; RSI <= {RSI_OVERSOLD}: +1",
        "score_contribution": rsi_points,
    })

    macd_points = 0
    if macd is not None and signal is not None:
        macd_points = 1 if macd > signal else -1
        score += macd_points
        reasons.append("MACD is above its signal line" if macd > signal else "MACD is below its signal line")
    rule_trace.append({
        "rule": "macd_crossover",
        "observed": {"macd": macd, "signal": signal},
        "condition": "MACD > signal: +1; otherwise: -1",
        "score_contribution": macd_points,
    })

    action = "BUY" if score >= BUY_SCORE_MIN else "SELL" if score <= SELL_SCORE_MAX else "HOLD"
    volatility = metrics["annualized_volatility_percent"] or 0
    risk = (
        "HIGH" if volatility >= HIGH_RISK_VOLATILITY
        else "MEDIUM" if volatility >= MEDIUM_RISK_VOLATILITY
        else "LOW"
    )
    return {
        "action": action,
        "risk": risk,
        "score": score,
        "reasons": reasons,
        "rule_trace": rule_trace,
        "decision_boundaries": {
            "buy_score_min": BUY_SCORE_MIN,
            "sell_score_max": SELL_SCORE_MAX,
            "medium_risk_volatility_percent": MEDIUM_RISK_VOLATILITY,
            "high_risk_volatility_percent": HIGH_RISK_VOLATILITY,
        },
    }


def _fetch_history(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True, timeout=15)


def _fetch_search(ticker: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        search = yf.Search(ticker, max_results=3, news_count=5, lists_count=0, timeout=10)
        quote = next((item for item in search.quotes if item.get("symbol") == ticker), {})
        news = _parse_news_articles(search.news)
    except Exception:
        quote, news = {}, []
    company = {
        "name": quote.get("longname") or quote.get("shortname"),
        "sector": quote.get("sectorDisp"),
        "industry": quote.get("industryDisp"),
        "exchange": quote.get("exchange"),
        "quote_type": quote.get("quoteType"),
    }

    if not news:
        try:
            news = _parse_news_articles(yf.Ticker(ticker).news)
        except Exception:
            news = []
    return company, news


def collect_evidence(ticker: str) -> dict[str, Any]:
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(ticker)
        if cached and now - cached[0] < CACHE_TTL_SECONDS:
            return {**cached[1], "cached": True}

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stock-data")
    futures = {
        "history": executor.submit(_fetch_history, ticker),
        "search": executor.submit(_fetch_search, ticker),
    }
    done, pending = wait(futures.values(), timeout=COLLECTION_TIMEOUT_SECONDS)
    for future in pending:
        future.cancel()
    executor.shutdown(wait=False, cancel_futures=True)

    if futures["history"] not in done:
        raise TimeoutError("Yahoo Finance price history timed out")
    history = futures["history"].result()
    if history.empty:
        raise ValueError(f"No market data found for ticker {ticker}")

    if futures["search"] in done and not futures["search"].exception():
        company, news = futures["search"].result()
    else:
        company, news = {}, []
    metrics = calculate_metrics(history)
    scoring = score_evidence(metrics)
    usable_history = history["Close"].dropna()
    evidence = {
        "ticker": ticker,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance",
        "source_url": f"https://finance.yahoo.com/quote/{ticker}/history/",
        "price_period": {
            "start": usable_history.index[0].date().isoformat(),
            "end": usable_history.index[-1].date().isoformat(),
            "trading_days": int(len(usable_history)),
            "adjusted_prices": True,
        },
        "cached": False,
        "company": company,
        "metrics": metrics,
        "news": news,
        "scoring": scoring,
    }
    with _cache_lock:
        _cache[ticker] = (now, evidence)
    return evidence


def calculate_confidence(evidence: dict[str, Any], agent_status: str) -> tuple[int, dict[str, int]]:
    """Score evidence completeness, not the probability of making money."""
    metrics = evidence.get("metrics", {})
    core_fields = ["current_price", "ma_20", "ma_50", "rsi_14", "macd", "macd_signal"]
    available = sum(metrics.get(field) is not None for field in core_fields)
    components = {
        "base": 30,
        "market_metrics": round(35 * available / len(core_fields)),
        "company_metadata": 10 if evidence.get("company", {}).get("name") else 0,
        "attributable_news": min(10, len(evidence.get("news", [])) * 2),
        "agent_review": 15 if agent_status == "completed" else 0,
    }
    return min(100, sum(components.values())), components


def _fallback_narrative(evidence: dict[str, Any]) -> dict[str, Any]:
    scoring = evidence["scoring"]
    metrics = evidence["metrics"]
    return {
        "summary": (
            f"The rules-based signal is {scoring['action']} with {scoring['risk']} risk. "
            f"The trend is {metrics['trend'].lower()}, RSI is {metrics['rsi_14']}, and "
            f"annualized volatility is {metrics['annualized_volatility_percent']}%."
        ),
        "strengths": [reason for reason in scoring["reasons"] if "above" in reason or "oversold" in reason][:3],
        "risks": [reason for reason in scoring["reasons"] if "below" in reason or "overbought" in reason][:3],
        "agent_analysis": {
            "status": "fallback",
            "research": {"summary": "AI research review was unavailable.", "evidence_points": [], "caveats": []},
            "risk_audit": {"summary": "AI risk audit was unavailable.", "risk_factors": [], "data_gaps": []},
        },
    }


def _parse_json_object(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("Agent did not return a JSON object")
    parsed = json.loads(raw[start:end])
    if not isinstance(parsed, dict):
        raise ValueError("Agent JSON output must be an object")
    return parsed


def _evidence_candidates(evidence: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    metrics = evidence["metrics"]
    strengths: dict[str, str] = {}
    risks: dict[str, str] = {}
    if metrics["trend"] == "BULLISH":
        strengths["bullish_ma_trend"] = "Price is above both the 20-day and 50-day moving averages."
    elif metrics["trend"] == "BEARISH":
        risks["bearish_ma_trend"] = "Price is below both the 20-day and 50-day moving averages."
    if metrics["macd"] is not None and metrics["macd_signal"] is not None:
        target = strengths if metrics["macd"] > metrics["macd_signal"] else risks
        target["macd_position"] = (
            "MACD is above its signal line." if target is strengths else "MACD is below its signal line."
        )
    if metrics["change_3m_percent"] is not None:
        target = strengths if metrics["change_3m_percent"] > 0 else risks
        target["three_month_return"] = f"The adjusted three-month price change is {metrics['change_3m_percent']}%."
    if metrics["rsi_14"] is not None and metrics["rsi_14"] >= RSI_OVERBOUGHT:
        risks["rsi_overbought"] = f"RSI is {metrics['rsi_14']}, which is in the overbought range."
    elif metrics["rsi_14"] is not None and metrics["rsi_14"] <= RSI_OVERSOLD:
        strengths["rsi_oversold"] = f"RSI is {metrics['rsi_14']}, which is in the oversold range."
    volatility = metrics["annualized_volatility_percent"]
    if volatility is not None and volatility >= MEDIUM_RISK_VOLATILITY:
        risks["elevated_volatility"] = f"Annualized historical volatility is {volatility}%."
    risks["model_scope"] = "The signal uses technical price indicators and does not include a complete valuation model."
    return strengths, risks


def create_narrative(evidence: dict[str, Any]) -> dict[str, Any]:
    # CrewAI is intentionally loaded only when a narrative is requested. This
    # keeps health checks, backtests, and deterministic tests free of LLM side effects.
    from crewai import Crew, Process, Task
    from stock_agents import investment_advisor, research_analyst, risk_auditor

    strength_candidates, risk_candidates = _evidence_candidates(evidence)
    allowed_company_fields = [key for key, value in evidence["company"].items() if value is not None]
    allowed_gap_codes = ["NO_FINANCIAL_STATEMENTS", "NO_EARNINGS_FORECAST", "NO_ANALYST_ESTIMATES", "LIMITED_NEWS_SAMPLE"]
    research_task = Task(
        description=f"""
        Select the most relevant supplied company fields and news items. You may select
        identifiers only; you may not write factual claims or recommendations.
        Company: {json.dumps(evidence['company'])}
        Indexed news: {json.dumps(list(enumerate(evidence['news'])), default=str)}
        Allowed company fields: {json.dumps(allowed_company_fields)}
        Allowed caveat codes: {json.dumps(allowed_gap_codes)}
        Return only JSON:
        {{"company_fields":["allowed field"],"news_indices":[0],"caveat_codes":["allowed code"]}}
        """,
        expected_output="Strict evidence-selection JSON",
        agent=research_analyst,
        async_execution=True,
    )
    risk_task = Task(
        description=f"""
        Audit the deterministic signal by selecting identifiers only. You cannot write
        new claims or change the action/risk.
        Action: {evidence['scoring']['action']}; risk: {evidence['scoring']['risk']}
        Allowed risk candidates: {json.dumps(risk_candidates)}
        Allowed data-gap codes: {json.dumps(allowed_gap_codes)}
        Return only JSON:
        {{"risk_ids":["allowed risk id"],"data_gap_codes":["allowed code"]}}
        """,
        expected_output="Strict risk-audit JSON",
        agent=risk_auditor,
        async_execution=True,
    )
    advisor_task = Task(
        description=f"""
        Select evidence identifiers for the final dashboard after considering both
        specialist reviews. You cannot write factual prose.
        The action {evidence['scoring']['action']} and risk {evidence['scoring']['risk']}
        are deterministic and MUST NOT be changed.
        Allowed strength candidates: {json.dumps(strength_candidates)}
        Allowed risk candidates: {json.dumps(risk_candidates)}
        Return only JSON:
        {{"strength_ids":["allowed strength id"],"risk_ids":["allowed risk id"]}}
        """,
        expected_output="Strict final-advisor JSON",
        agent=investment_advisor,
        context=[research_task, risk_task],
    )
    try:
        result = Crew(
            agents=[research_analyst, risk_auditor, investment_advisor],
            tasks=[research_task, risk_task, advisor_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()
        research = _parse_json_object(result.tasks_output[0].raw)
        risk_review = _parse_json_object(result.tasks_output[1].raw)
        parsed = _parse_json_object(result.tasks_output[2].raw)
        strength_ids = [item for item in parsed.get("strength_ids", []) if item in strength_candidates][:3]
        risk_ids = [item for item in parsed.get("risk_ids", []) if item in risk_candidates][:3]
        selected_strengths = [strength_candidates[item] for item in strength_ids]
        selected_risks = [risk_candidates[item] for item in risk_ids]
        if not selected_strengths:
            selected_strengths = list(strength_candidates.values())[:3]
        if not selected_risks:
            selected_risks = list(risk_candidates.values())[:3]

        company_fields = [item for item in research.get("company_fields", []) if item in allowed_company_fields][:3]
        news_indices = [item for item in research.get("news_indices", []) if isinstance(item, int) and 0 <= item < len(evidence["news"])][:3]
        caveat_codes = [item for item in research.get("caveat_codes", []) if item in allowed_gap_codes][:3]
        audited_risk_ids = [item for item in risk_review.get("risk_ids", []) if item in risk_candidates][:3]
        data_gap_codes = [item for item in risk_review.get("data_gap_codes", []) if item in allowed_gap_codes][:3]
        strength_text = selected_strengths[0] if selected_strengths else "No positive evidence candidate was selected."
        risk_text = selected_risks[0] if selected_risks else "The model has limited evidence."
        summary = (
            f"{evidence['ticker']} has a rules-based {evidence['scoring']['action']} signal with "
            f"{evidence['scoring']['risk']} risk. "
            f"{strength_text} However, {risk_text}"
        )
        return {
            "summary": summary,
            "strengths": selected_strengths,
            "risks": selected_risks,
            "agent_analysis": {
                "status": "completed",
                "research": {
                    "company_fields": company_fields,
                    "news": [evidence["news"][index] for index in news_indices],
                    "caveat_codes": caveat_codes,
                },
                "risk_audit": {
                    "risk_factors": [risk_candidates[item] for item in audited_risk_ids],
                    "data_gap_codes": data_gap_codes,
                },
            },
        }
    except Exception:
        return _fallback_narrative(evidence)


def analyze(ticker: str) -> dict[str, Any]:
    evidence = collect_evidence(ticker)
    narrative = create_narrative(evidence)
    agent_status = narrative.get("agent_analysis", {}).get("status", "fallback")
    confidence, confidence_breakdown = calculate_confidence(evidence, agent_status)
    primary_warning = narrative.get("risks", [None])[0] if narrative.get("risks") else None
    return {
        "ticker": ticker,
        "action": evidence["scoring"]["action"],
        "risk": evidence["scoring"]["risk"],
        **narrative,
        "signal_interpretation": (
            f"Technical {evidence['scoring']['action']} signal"
            + (f"; caution: {primary_warning}" if primary_warning else "")
        ),
        "evidence_quality": {
            "score": confidence,
            "meaning": "Completeness and review coverage of available evidence; not probability of profit.",
            "calculation": confidence_breakdown,
        },
        "score": evidence["scoring"]["score"],
        "metrics": evidence["metrics"],
        "company": evidence["company"],
        "news": evidence["news"],
        "proof": {
            "rule_version": RULE_VERSION,
            "score_total": evidence["scoring"]["score"],
            "score_calculation": evidence["scoring"]["rule_trace"],
            "decision_boundaries": evidence["scoring"]["decision_boundaries"],
            "metric_formulas": {
                "ma_20": "Mean of the latest 20 adjusted daily closes",
                "ma_50": "Mean of the latest 50 adjusted daily closes",
                "rsi_14": "14-period relative strength index using rolling average gains and losses",
                "macd": "12-day EMA minus 26-day EMA",
                "macd_signal": "9-day EMA of MACD",
                "annualized_volatility_percent": "Daily return standard deviation x sqrt(252) x 100",
            },
        },
        "meta": {
            "as_of": evidence["as_of"],
            "source": evidence["source"],
            "source_url": evidence["source_url"],
            "price_period": evidence["price_period"],
            "cached": evidence["cached"],
            "methodology": f"{RULE_VERSION} + parallel Gemini research/risk review + advisor synthesis",
        },
        "disclaimer": "For informational research only; not personalized financial advice.",
    }
