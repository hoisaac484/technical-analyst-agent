# Technical Analyst Agent – Asset Management Trade Note

## Project Summary

This project implements a Technical Analyst AI Agent that backtests a rule-based trading strategy on historical equity data and uses a Large Language Model (LLM) to generate a professional, institutional-style trade note based on the results.

## Strategy Overview

The strategy is a systematic mean-reversion framework with confirmation filters to reduce false signals. Long entries are triggered by a Supertrend buy flip, conditioned on RSI(50) below 50, sufficient deviation from the 200-period smoothed moving average, momentum confirmation via CMO versus its smoothed signal, and a candle height filter to avoid volatile bars. Risk is managed using fixed-fractional position sizing, with stop-loss and take-profit levels defined as multiples of ATR. Performance is evaluated using virtual capital backtesting with compounding, leverage tracking, and standard risk metrics.

## How to Run
Requirements

-Python 3.9+

-Internet access (for price data)

-Azure OpenAI access (no fallback; script fails if AI is unavailable)

-Python Dependencies

-Install required packages:
```bash
pip install yfinance pandas numpy matplotlib python-dotenv python-docx openai
```

## Environment Variables (.env)

Create a .env file in the project root with:
```bash
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_KEY=your_api_key
AZURE_DEPLOYMENT_NAME=your_deployment_name
AZURE_API_VERSION=2024-02-15-preview
```
## Run the Demo

From the project root:
```bash
python run_demo.py
```

The script will:

-Download historical price data

-Run the backtest with hard-coded parameters

-Print performance metrics

-Display plots (equity, drawdown, leverage)

-Generate a ≤700-word AI trade note and save it as a Word file in /output
