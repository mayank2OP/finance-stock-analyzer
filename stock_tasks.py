from crewai import Task
from stock_agents import data_collector, technical_analyst, news_analyst, investment_advisor


def create_tasks(ticker: str):
    """Create tasks for analyzing the given stock ticker"""

    task1 = Task(
        description=f"""
        Collect comprehensive stock data for {ticker}.

        You MUST:
        - Use Stock Data Fetcher tool
        - Gather 3-month historical data
        - Include price, volume, market cap, P/E ratio

        Do NOT guess values.
        """,
        agent=data_collector,
        expected_output="Detailed stock data report"
    )

    task2 = Task(
        description=f"""
        Perform technical analysis on {ticker}.

        You MUST:
        - Use Technical Analysis Tool
        - Calculate moving averages, RSI, MACD
        - Identify bullish or bearish trend

        Do NOT generate fake indicators.
        """,
        agent=technical_analyst,
        expected_output="Technical analysis report"
    )

    task3 = Task(
        description=f"""
        Analyze recent news for {ticker}.

        You MUST:
        - Use News Sentiment Tool
        - Summarize sentiment (positive/negative/neutral)

        Do NOT hallucinate news.
        """,
        agent=news_analyst,
        expected_output="News sentiment summary"
    )

    task4 = Task(
    description=f"""
    Based on ALL previous results, provide final investment recommendation for {ticker}.

    You MUST return ONLY valid JSON (no text, no explanation outside JSON).

    Format:
    {{
        "action": "BUY or SELL or HOLD",
        "risk": "Low or Medium or High",
        "summary": "Brief 2-3 line overview",
        "strengths": ["point1", "point2", "point3"],
        "risks": ["point1", "point2", "point3"]
    }}

    Rules:
    - Do NOT add markdown
    - Do NOT add explanation outside JSON
    - Do NOT change keys
    - Ensure valid JSON
    """,
    agent=investment_advisor,
    expected_output="Strict JSON output",
    context=[task1, task2, task3]
)

    return [task1, task2, task3, task4]
