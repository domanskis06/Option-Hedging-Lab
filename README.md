# Options Pricing & Delta-Hedging Lab

Learning project: price European options from scratch (Black–Scholes, CRR binomial tree),
compute Greeks, and simulate discrete delta-hedging under transaction costs.

## Status

- [x] Repo scaffold
- [ ] Step 0 — Payoffs & vocabulary
- [ ] Step 1 — Geometric Brownian Motion
- [ ] Step 2 — Black–Scholes pricing
- [ ] Step 3 — Greeks
- [ ] Step 4 — CRR binomial tree
- [ ] Step 5 — Delta-hedging simulation
- [ ] Step 6 — Report & CV bullets

## Setup (Windows)

```powershell
cd C:\Users\szymo\Desktop\quant-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run tests

```powershell
pytest -q
```

## Assumptions (MVP)

- European options first (exercise only at maturity)
- No dividends (`q = 0`) unless stated otherwise
- Constant risk-free rate `r` and volatility `sigma`
- Synthetic parameters are enough; live market data comes later
