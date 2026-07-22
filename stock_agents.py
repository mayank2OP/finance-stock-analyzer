import os

from crewai import Agent, LLM

from stock_tools import fetch_stock_data, calculate_technical_indicators, get_stock_news


gemini_llm = LLM(
    model=os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite"),
)

agent_limits = {
    "max_iter": 4,
    "max_retry_limit": 1,
    "max_execution_time": 45,
}


# ---------------- AGENTS ---------------- #

data_collector = Agent(
    role='Financial Data Collector',
    goal='Gather comprehensive stock data from Yahoo Finance including historical prices, volume, and company information',
    backstory="""You are an expert at collecting and organizing financial data.
    You have access to Yahoo Finance and can retrieve detailed stock information,
    historical prices, and market statistics.""",
    tools=[fetch_stock_data],
    llm=gemini_llm,
    verbose=False,
    allow_delegation=False,
    **agent_limits,
)


technical_analyst = Agent(
    role='Technical Analysis Expert',
    goal='Analyze stock trends using technical indicators and chart patterns',
    backstory="""You are a seasoned technical analyst with years of experience
    in reading charts and identifying trends. You specialize in moving averages,
    RSI, MACD, and other technical indicators to predict price movements.""",
    tools=[calculate_technical_indicators],
    llm=gemini_llm,
    verbose=False,
    allow_delegation=False,
    **agent_limits,
)


news_analyst = Agent(
    role='Market News Analyst',
    goal='Analyze recent news and market sentiment around the stock',
    backstory="""You are a financial news analyst who tracks market sentiment
    and company developments. You understand how news impacts stock prices and
    can gauge investor sentiment.""",
    tools=[get_stock_news],
    llm=gemini_llm,
    verbose=False,
    allow_delegation=False,
    **agent_limits,
)


research_analyst = Agent(
    role="Evidence Research Analyst",
    goal="Interpret verified company and news evidence without inventing facts or recommendations",
    backstory="""You are a skeptical equity research analyst. You distinguish sourced facts
    from inference, identify missing context, and never manufacture financial figures,
    price targets, events, or sentiment.""",
    llm=gemini_llm,
    verbose=False,
    allow_delegation=False,
    **agent_limits,
)


risk_auditor = Agent(
    role="Independent Risk Auditor",
    goal="Challenge the quantitative signal and identify downside scenarios and evidence gaps",
    backstory="""You are an independent model-risk reviewer. Your job is to find weaknesses,
    contradictory indicators, missing data, and unjustified certainty. You do not change
    verified calculations and you do not provide personalized financial advice.""",
    llm=gemini_llm,
    verbose=False,
    allow_delegation=False,
    **agent_limits,
)


investment_advisor = Agent(
    role='Senior Investment Advisor',
    goal='Provide clear BUY/SELL/HOLD recommendations based on comprehensive analysis',
    backstory="""You are a senior investment advisor with 20+ years of experience.
    You synthesize fundamental data, technical analysis, and market sentiment to
    provide actionable investment recommendations. You always provide clear reasoning
    for your recommendations and consider risk factors.""",
    llm=gemini_llm,
    verbose=False,
    allow_delegation=False,
    **agent_limits,
)
