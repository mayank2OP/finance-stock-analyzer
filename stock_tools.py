import os
from datetime import datetime, timedelta
from crewai.tools import tool 
import yfinance as yf
import pandas as pd
import numpy as np





@tool("Stock Data Fetcher")
def fetch_stock_data(ticker: str) -> str:
    """
    Fetches stock data from Yahoo Finance for the past 3 months.
    Returns historical prices, volume, and basic statistics.
    """
    
    try:
        ticker = ticker.upper().strip().split()[0]
        stock = yf.Ticker(ticker)

        # Get 3 months of historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            raise ValueError(f"Invalid or unsupported ticker: {ticker}")

        # Get stock info
        info = stock.info

        # Calculate key metrics
        current_price = hist['Close'].iloc[-1]
        price_3m_ago = hist['Close'].iloc[0]
        change_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100

        avg_volume = hist['Volume'].mean()
        high_3m = hist['High'].max()
        low_3m = hist['Low'].min()

        # Volatility (standard deviation of returns)
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized

        result = f"""
        Stock Data for {ticker}:
        Company Name: {info.get('longName', 'N/A')}
        Sector: {info.get('sector', 'N/A')}
        Industry: {info.get('industry', 'N/A')}
        Current Price: ${current_price:.2f}
        3-Month Change: {change_3m:.2f}%
        3-Month High: ${high_3m:.2f}
        3-Month Low: ${low_3m:.2f}
        Average Volume: {avg_volume:,.0f}
        Annualized Volatility: {volatility:.2f}%

        Market Cap: ${info.get('marketCap', 0):,.0f}
        P/E Ratio: {info.get('trailingPE', 'N/A')}
        52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
        52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
        """
        return result

    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")


@tool("Technical Analysis Tool")
def calculate_technical_indicators(ticker: str) -> str:
    """
    Calculates technical indicators including moving averages, RSI, and MACD.
    """
    try:
        ticker = ticker.upper().strip().split()[0]
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)  # Extra data for indicators
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            raise ValueError(f"No technical data available for ticker: {ticker}")
        # Moving Averages
        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        current_price = hist['Close'].iloc[-1]

        # RSI Calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Handle division by zero if loss is 0
        if loss.iloc[-1] == 0:
            rs = np.inf if gain.iloc[-1] > 0 else 0
            current_rsi = 100 if gain.iloc[-1] > 0 else 50 # Or another neutral value
        else:
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

        # MACD
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_current = macd.iloc[-1]
        signal_current = signal.iloc[-1]

        # Trend Analysis
        ma_trend = "Bullish" if current_price > ma_20 > ma_50 else "Bearish" if current_price < ma_20 < ma_50 else "Neutral"
        rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
        macd_signal = "Bullish" if macd_current > signal_current else "Bearish"

        result = f"""
        Technical Analysis for {ticker}:

        Moving Averages:
        - Current Price: ${current_price:.2f}
        - 20-Day MA: ${ma_20:.2f}
        - 50-Day MA: ${ma_50:.2f}
        - MA Trend: {ma_trend}

        RSI (14-day): {current_rsi:.2f}
        - Signal: {rsi_signal}

        MACD:
        - MACD Line: {macd_current:.4f}
        - Signal Line: {signal_current:.4f}
        - Trend: {macd_signal}

        Overall Technical Outlook: {'Positive' if ma_trend == 'Bullish' and current_rsi < 70 else 'Negative' if ma_trend == 'Bearish' else 'Mixed'}
        """
        return result

    except Exception as e:
        raise ValueError(f"Technical analysis failed for {ticker}: {str(e)}")


@tool("News Sentiment Tool")
def get_stock_news(ticker: str) -> str:
    """
    Retrieves recent news about the stock.
    """
    try:
        ticker = ticker.upper().strip().split()[0]
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return f"No recent news available for {ticker}. Sentiment: Neutral"

        result = f"Recent News for {ticker}:\n\n"

        for i, article in enumerate(news[:5], 1):
            title = article.get('title', 'N/A')
            publisher = article.get('publisher', 'N/A')
            result += f"{i}. {title} - {publisher}\n"

        return result

    except Exception:
        # 🔥 NEVER crash pipeline
        return f"Unable to fetch news for {ticker}. Assume Neutral sentiment."

 


