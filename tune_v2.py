"""
V2 Threshold tuning - explores advanced strategies on cached LLM data.
Focus: maximize Sharpe with more sophisticated logic.
"""

import json
import math
from collections import defaultdict

STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.001

with open("llm_cache.json") as f:
    cache = json.load(f)

# Pre-index cache by (date, ticker) for lookback
by_ticker = defaultdict(list)
for item in cache:
    by_ticker[item["ticker"]].append(item)
for tk in by_ticker:
    by_ticker[tk].sort(key=lambda x: x["date"])

# Build lookback: for each (date, ticker), what was the previous day's data?
prev_data = {}
for tk, rows in by_ticker.items():
    for i in range(1, len(rows)):
        key = (rows[i]["date"], tk)
        prev_data[key] = rows[i-1]


def compute_score(analysis, indicators):
    sentiment = float(analysis.get("sentiment_score", 0) or 0)
    reversal_prob = float(analysis.get("reversal_probability", 0) or 0)
    llm_action = str(analysis.get("recommended_action", "hold")).lower()
    rsi = float(indicators.get("rsi", 50))
    macd_hist = float(indicators.get("macd_hist", 0))
    bb_pos = float(indicators.get("bb_position", 0.5))
    volatility = float(indicators.get("volatility_7d", 0))

    score = 0
    if rsi < 25: score += 25
    elif rsi < 30: score += 18
    elif rsi < 40: score += 5
    elif rsi > 70: score -= 25
    elif rsi > 60: score -= 8

    if macd_hist > 0: score += 20
    elif macd_hist > -0.001: score += 5
    else: score -= 10

    if bb_pos < 0.1: score += 20
    elif bb_pos < 0.2: score += 12
    elif bb_pos > 0.9: score -= 20
    elif bb_pos > 0.8: score -= 12

    score += round(sentiment * 25)
    if llm_action == "buy": score += 10
    elif llm_action == "sell": score -= 10
    if volatility > 0.045: score -= 15
    elif volatility > 0.035: score -= 8
    if reversal_prob > 0.6 and sentiment < 0 and rsi < 35: score += 10
    elif reversal_prob > 0.6 and sentiment > 0 and rsi > 60: score -= 10

    return score, rsi, macd_hist, bb_pos, volatility


def trading_v2(item, params):
    analysis = item["analysis"]
    indicators = item["indicators"]
    risk_level = str(analysis.get("risk_level", "medium")).lower()

    if risk_level == "extreme":
        return "sell"

    score, rsi, macd_hist, bb_pos, volatility = compute_score(analysis, indicators)

    # V2: Momentum confirmation - require MACD improving (today > yesterday)
    prev = prev_data.get((item["date"], item["ticker"]))
    macd_improving = True  # default if no prev data
    if prev and params.get("require_macd_improving"):
        prev_macd = float(prev["indicators"].get("macd_hist", 0))
        macd_improving = macd_hist > prev_macd

    decision = "hold"

    # BUY
    if score >= params["buy_score"] and rsi < params["buy_rsi_max"]:
        if not params.get("require_macd_improving") or macd_improving:
            decision = "buy"

    # SELL (overrides buy)
    if score <= params["sell_score"]:
        decision = "sell"
    elif score <= params["sell_score_mild"] and rsi > params["sell_rsi_min"]:
        decision = "sell"
    elif bb_pos > params["sell_bb"]:
        decision = "sell"
    if rsi > 75:
        decision = "sell"

    return decision


def run_backtest(results, alloc_pct=0.05):
    capital = STARTING_CAPITAL
    positions = {}
    daily_values = []
    trade_results = []
    # Track holding days for time-based exit
    holding_since = {}

    dates = sorted(set(r["date"] for r in results))
    for date in dates:
        day_rows = [r for r in results if r["date"] == date]
        for r in day_rows:
            ticker, price, decision = r["ticker"], r["price"], r["decision"]

            if decision == "buy" and capital > 100:
                alloc = capital * alloc_pct
                fee = alloc * TRANSACTION_FEE
                invest = alloc - fee
                qty = invest / price
                if ticker not in positions:
                    positions[ticker] = {"qty": 0, "avg_price": 0}
                old = positions[ticker]
                old_cost = old["qty"] * old["avg_price"]
                new_qty = old["qty"] + qty
                positions[ticker] = {"qty": new_qty, "avg_price": (old_cost + invest) / new_qty if new_qty > 0 else 0}
                capital -= alloc
                if ticker not in holding_since:
                    holding_since[ticker] = date
                trade_results.append({"action": "buy", "pnl": 0})

            elif decision == "sell" and ticker in positions and positions[ticker]["qty"] > 0:
                qty = positions[ticker]["qty"]
                proceeds = qty * price
                fee = proceeds * TRANSACTION_FEE
                net = proceeds - fee
                pnl = net - qty * positions[ticker]["avg_price"]
                capital += net
                trade_results.append({"action": "sell", "pnl": pnl})
                positions[ticker] = {"qty": 0, "avg_price": 0}
                holding_since.pop(ticker, None)

        pv = capital
        prices_today = {r["ticker"]: r["price"] for r in day_rows}
        for ticker, pos in positions.items():
            if pos["qty"] > 0 and ticker in prices_today:
                pv += pos["qty"] * prices_today[ticker]
        daily_values.append(pv)

    if len(daily_values) > 1:
        returns = [(daily_values[i] / daily_values[i-1] - 1) for i in range(1, len(daily_values))]
        avg_ret = sum(returns) / len(returns)
        std_ret = (sum((r - avg_ret)**2 for r in returns) / len(returns)) ** 0.5
        sharpe = (avg_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0
        total_ret = (daily_values[-1] / STARTING_CAPITAL - 1) * 100
        buys = sum(1 for t in trade_results if t["action"] == "buy")
        sells_list = [t for t in trade_results if t["action"] == "sell"]
        wins = sum(1 for t in sells_list if t["pnl"] > 0)
        max_val = daily_values[0]
        max_dd = 0
        for v in daily_values:
            max_val = max(max_val, v)
            dd = (v - max_val) / max_val
            max_dd = min(max_dd, dd)
        return {
            "sharpe": sharpe, "total_return": total_ret,
            "buys": buys, "sells": len(sells_list),
            "win_rate": (wins / len(sells_list) * 100) if sells_list else 0,
            "final_value": daily_values[-1],
            "max_drawdown": max_dd * 100,
        }
    return None


# === GRID SEARCH V2 ===
print("V2 Grid Search - with momentum confirmation...")
print("=" * 80)

best_sharpe = -999
best_params = None
best_metrics = None
all_results = []

for buy_score in [10, 15, 20, 25, 30, 35, 40, 50]:
    for buy_rsi_max in [28, 35, 45, 50]:
        for sell_score in [-30, -25, -20, -15, -10, -5, 0]:
            for sell_score_mild in [-15, -10, -5, 0, 5, 10]:
                for sell_rsi_min in [35, 40, 45, 50]:
                    for sell_bb in [0.7, 0.8, 0.9, 1.0, 1.1, 1.5]:
                        for macd_req in [True, False]:
                            for alloc in [0.03, 0.05, 0.08]:
                                params = {
                                    "buy_score": buy_score,
                                    "buy_rsi_max": buy_rsi_max,
                                    "sell_score": sell_score,
                                    "sell_score_mild": sell_score_mild,
                                    "sell_rsi_min": sell_rsi_min,
                                    "sell_bb": sell_bb,
                                    "require_macd_improving": macd_req,
                                }

                                decisions = []
                                for item in cache:
                                    d = trading_v2(item, params)
                                    decisions.append({"date": item["date"], "ticker": item["ticker"], "price": item["price"], "decision": d})

                                metrics = run_backtest(decisions, alloc_pct=alloc)
                                if metrics and metrics["buys"] >= 2 and metrics["sells"] >= 2:
                                    all_results.append((metrics["sharpe"], {**params, "alloc": alloc}, metrics))
                                    if metrics["sharpe"] > best_sharpe:
                                        best_sharpe = metrics["sharpe"]
                                        best_params = {**params, "alloc": alloc}
                                        best_metrics = metrics

all_results.sort(key=lambda x: x[0], reverse=True)

print(f"\nTop 15 parameter sets (min 2 buys + 2 sells):")
for i, (sharpe, params, metrics) in enumerate(all_results[:15]):
    print(f"  #{i+1} Sharpe={sharpe:+.4f} Ret={metrics['total_return']:+.2f}% DD={metrics['max_drawdown']:.1f}% "
          f"B={metrics['buys']} S={metrics['sells']} WR={metrics['win_rate']:.0f}% "
          f"| buy>={params['buy_score']} rsi<{params['buy_rsi_max']} "
          f"sell<={params['sell_score']}/{params['sell_score_mild']} rsi>{params['sell_rsi_min']} "
          f"bb>{params['sell_bb']} macd={'Y' if params['require_macd_improving'] else 'N'} alloc={params['alloc']}")

if best_params:
    print(f"\n{'='*80}")
    print(f"BEST: Sharpe={best_sharpe:+.4f}")
    print(f"  Params: {best_params}")
    print(f"  Metrics: {best_metrics}")
