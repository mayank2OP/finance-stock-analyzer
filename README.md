# StockPilot Backend

FastAPI backend that uses Yahoo Finance tools and CrewAI agents powered by
Google Gemini. It includes deterministic backtesting, JWT authentication,
personal watchlists, and saved result history.

The production analysis path uses a hybrid multi-agent workflow:

- Deterministic collectors retrieve prices, company metadata, and attributable news.
- The quantitative engine calculates indicators and locks the signal.
- Research and Risk agents review evidence in parallel.
- The Advisor agent synthesizes both reviews without changing the locked signal.

Agents may select only precomputed evidence IDs and exact news records; they cannot
write unsupported factual claims. The BUY/HOLD/SELL action and risk level remain
rules-based and reproducible. Market evidence is cached for five minutes.

Every analysis includes a `proof` object containing each rule's observed value,
score contribution, decision boundary, indicator formulas, and a deterministic
calculation trail. `evidence_quality.score` measures data completeness and agent
review coverage; it is explicitly not a probability of profit. The `meta` object records the Yahoo Finance
source URL, exact price window, trading-day count, timestamp, and cache status.
Live analysis and backtesting import the same configurable thresholds from
`strategy_config.py`, preventing silent differences between them.

## Local setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add a Gemini API key from
   [Google AI Studio](https://aistudio.google.com/apikey).

4. Start the API:

   ```powershell
   uvicorn stock_crew:app --reload
   ```

Open <http://127.0.0.1:8000/docs> to test the API.

## Login in Swagger

1. Run `POST /auth/register` with a username and an 8+ character password.
2. Open `POST /auth/token`, enter the same credentials, and copy the token.
3. Click **Authorize** at the top of Swagger and paste the token.

Passwords are stored as Argon2 hashes; they are never stored or returned as plain
text. JWT access tokens expire after 60 minutes by default.

## API contract

`POST /analyze` requires a bearer token and accepts:

```json
{"ticker": "AAPL"}
```

The response includes the action, risk, narrative, technical metrics, company
metadata, recent news with links, source timestamp, cache state, methodology, and
an informational-use disclaimer. Errors use meaningful HTTP status codes:

- `422`: invalid ticker or insufficient market history
- `502`: upstream analysis provider failure
- `503`: Gemini is not configured
- `504`: market-data timeout

`GET /health` checks both the API and its database connection. A healthy response is:

```json
{"status": "healthy", "database": "healthy"}
```

Long agent calls can use the frontend-friendly asynchronous API:

- `POST /analysis-jobs` starts an analysis and immediately returns a job ID.
- `GET /analysis-jobs/{job_id}` returns queued, running, completed, or failed state.

The synchronous `POST /analyze` remains available for quick Swagger testing.
Jobs are intentionally stored in memory for this single-instance demo and reset
when the server restarts. Each job belongs to the user who created it, so another
signed-in user cannot read its result.

## Backtesting

`POST /backtest` requires a bearer token and evaluates the deterministic
`rules-v1` signal without using an LLM. Example request:

```json
{
  "ticker": "AAPL",
  "period": "5y",
  "initial_capital": 10000,
  "transaction_cost_bps": 5,
  "horizon_days": 20,
  "include_equity_curve": false
}
```

Supported periods are `1y`, `2y`, `5y`, and `10y`. The response contains
strategy and buy-and-hold performance, annualized return and volatility, Sharpe
ratio, maximum drawdown, exposure, position changes, signal counts, and forward
signal hit rate under `advanced`. Set `include_equity_curve` to `true` only when
the frontend needs chart data; it is omitted by default to keep responses readable.

Signals use only information available through each closing date and are shifted
one trading day before earning returns. Transaction costs are charged whenever
the simulated position changes. Results are hypothetical and must not be treated
as expected future performance.

## Personal data and database

Authenticated users can manage `/watchlist`, `/saved-analyses`, and
`/saved-backtests` using the documented GET, POST, and DELETE operations.

SQLite is the zero-cost default and creates `stock_analyser.db` automatically.
The application uses SQLAlchemy, so MySQL can be enabled later without rewriting
the application:

```env
DATABASE_URL=mysql+pymysql://username:password@host/database_name
```

## Deployment safeguards

For deployment, set `APP_ENV=production`. Startup then refuses unsafe configuration:

- `JWT_SECRET` must be unique and at least 32 characters.
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) must be present.
- `CORS_ORIGINS` must list the exact deployed frontend URL; `*` is rejected.

Analysis is authenticated and limited per user and IP to protect the Gemini free
quota. Registration, login attempts, and backtests also have configurable limits;
see `.env.example`. The limiter is intentionally in memory for this one-instance
demo and resets on restart. Request logs include status, duration, and a request ID,
but never request bodies, passwords, or tokens. Provider failures returned to users
are generic; detailed exceptions remain only in server logs.

SQLite is suitable locally. For a public deployment, use SQLite only if the host
provides a persistent volume. Without one, accounts, watchlists, and saved research
can disappear on every redeploy. If persistent storage is unavailable, configure a
small hosted MySQL database through `DATABASE_URL`. Redis is not required.

### Railway environment variables

Store real secrets in Railway's **Variables** section, never in GitHub:

```env
APP_ENV=production
GEMINI_API_KEY=your_real_key
JWT_SECRET=a_unique_random_value_at_least_32_characters_long
CORS_ORIGINS=https://your-frontend-domain.example
DATABASE_URL=sqlite:////data/stock_analyser.db
```

The SQLite URL above assumes a Railway persistent volume mounted at `/data`.
After deployment, use `GET /health` to verify both the API and database. The
included `Procfile` and `nixpacks.toml` bind Uvicorn to Railway's `$PORT`.

Do not add `.env`, `stock_analyser.db`, `.venv`, runtime logs, or generated
temporary files to Git. `.env.example` is safe to commit because it contains
placeholders only.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests do not call Yahoo Finance or Gemini. They use a dedicated in-memory
SQLite database and never write test users into `stock_analyser.db`.



