import os


RULE_VERSION = "technical-rules-v2"
BUY_SCORE_MIN = int(os.getenv("BUY_SCORE_MIN", "2"))
SELL_SCORE_MAX = int(os.getenv("SELL_SCORE_MAX", "-2"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
MEDIUM_RISK_VOLATILITY = float(os.getenv("MEDIUM_RISK_VOLATILITY", "20"))
HIGH_RISK_VOLATILITY = float(os.getenv("HIGH_RISK_VOLATILITY", "40"))
