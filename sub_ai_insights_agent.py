#!/usr/bin/env python3
"""
subscription_agent_langchain.py

Install:
  pip install -U pandas python-dateutil langchain langchain-openai python-dotenv

Setup:
  Create a .env file with your OpenAI API key:
    OPENAI_API_KEY=your-api-key-here

Run:
  python subscription_agent_langchain.py \
    --subs user_identified_subscriptions.csv \
    --merchants subscription_merchants.csv \
    --asof 2025-11-30 \
    --query "What should I cancel before my next billing cycle?" \
    --out insights.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars

# LangChain imports (with fallbacks across versions)
try:
    from langchain_openai import ChatOpenAI
except Exception as e:
    raise RuntimeError("Missing langchain-openai. Install: pip install -U langchain-openai") from e

try:
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except Exception:
    # older compat
    from langchain.tools import tool  # type: ignore
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore

try:
    # LangChain 1.1.0+ uses create_agent
    from langchain.agents import create_agent
    USING_NEW_API = True
except Exception:
    USING_NEW_API = False
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
    except Exception:
        # OpenAI-specific fallback
        from langchain.agents import AgentExecutor, create_openai_tools_agent as create_tool_calling_agent  # type: ignore


# -----------------------------
# Data model for cards
# -----------------------------

@dataclass
class InsightCard:
    id: str
    title: str
    severity: str  # low|medium|high
    confidence: float  # 0..1
    facts: Dict[str, Any]
    recommended_actions: List[str]
    supporting_items: List[Dict[str, Any]]
    generated_at: str  # ISO string


# -----------------------------
# Helpers: parsing & normalization
# -----------------------------

ACTIVE_LIKE = {"active", "trialing", "trial", "paused"}
INACTIVE_LIKE = {"canceled", "cancelled", "inactive", "ended", "expired"}

def _now_iso(as_of: date) -> str:
    return datetime(as_of.year, as_of.month, as_of.day).isoformat()

def _clean_str(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s or None

def normalize_merchant_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(".com", "").replace(".ca", "").replace(".net", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def normalize_status(status: Optional[str]) -> str:
    s = (status or "").strip().lower()
    if not s:
        return "unknown"
    if s in {"cancelled", "canceled"}:
        return "cancelled"
    if s in {"trial", "trialing"}:
        return "trialing"
    if s in {"active", "running"}:
        return "active"
    if s in {"paused", "on hold"}:
        return "paused"
    return s

def normalize_billing_frequency(freq: Optional[str]) -> str:
    s = (freq or "").strip().lower()
    if not s:
        return "unknown"
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    mapping = {
        "monthly": "monthly", "month": "monthly", "mo": "monthly",
        "annual": "annual", "annually": "annual", "yearly": "annual", "year": "annual",
        "weekly": "weekly", "week": "weekly",
        "quarterly": "quarterly", "quarter": "quarterly",
        "biweekly": "biweekly", "bi weekly": "biweekly",
        "semiannual": "semiannual", "semi annual": "semiannual",
    }
    return mapping.get(s, s)

def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def monthly_equivalent(amount: float, billing_frequency: str) -> Optional[float]:
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return None
    if billing_frequency == "monthly":
        return float(amount)
    if billing_frequency == "annual":
        return float(amount) / 12.0
    if billing_frequency == "quarterly":
        return float(amount) / 3.0
    if billing_frequency == "semiannual":
        return float(amount) / 6.0
    if billing_frequency == "weekly":
        return float(amount) * (365.0 / 7.0) / 12.0
    if billing_frequency == "biweekly":
        return float(amount) * (365.0 / 14.0) / 12.0
    return None

def add_period(d: date, billing_frequency: str) -> date:
    if billing_frequency == "monthly":
        return d + relativedelta(months=1)
    if billing_frequency == "annual":
        return d + relativedelta(years=1)
    if billing_frequency == "quarterly":
        return d + relativedelta(months=3)
    if billing_frequency == "semiannual":
        return d + relativedelta(months=6)
    if billing_frequency == "weekly":
        return d + timedelta(days=7)
    if billing_frequency == "biweekly":
        return d + timedelta(days=14)
    return d + relativedelta(months=1)

def is_active(row: pd.Series, as_of: date) -> bool:
    st = row.get("status_norm", "unknown")
    if st in INACTIVE_LIKE or st == "cancelled":
        return False
    nbd = row.get("next_billing_date")
    if pd.notna(nbd) and isinstance(nbd, date):
        return nbd >= (as_of - timedelta(days=60))
    lbd = row.get("last_billed_date")
    if pd.notna(lbd) and isinstance(lbd, date):
        return lbd >= (as_of - timedelta(days=45))
    return st in ACTIVE_LIKE


# -----------------------------
# Load & prepare data
# -----------------------------

def load_data(subs_csv: str, merchants_csv: str) -> pd.DataFrame:
    subs = pd.read_csv(subs_csv)
    merch = pd.read_csv(merchants_csv)

    required_subs_cols = {
        "user_id","subscription_id","merchant_id","plan_name","billing_frequency",
        "start_date","next_billing_date","last_billed_date","status","subscription_amount"
    }
    required_merch_cols = {
        "merchant_id","name","category","risk_score","has_free_trial","usual_trial_days",
        "monthly_amount","annual_amount"
    }

    missing_subs = required_subs_cols - set(subs.columns)
    missing_merch = required_merch_cols - set(merch.columns)
    if missing_subs:
        raise ValueError(f"Missing columns in subscriptions CSV: {sorted(missing_subs)}")
    if missing_merch:
        raise ValueError(f"Missing columns in merchants CSV: {sorted(missing_merch)}")

    subs["start_date"] = parse_date_series(subs["start_date"])
    subs["next_billing_date"] = parse_date_series(subs["next_billing_date"])
    subs["last_billed_date"] = parse_date_series(subs["last_billed_date"])

    subs["status_norm"] = subs["status"].map(lambda x: normalize_status(_clean_str(x)))
    subs["billing_frequency_norm"] = subs["billing_frequency"].map(lambda x: normalize_billing_frequency(_clean_str(x)))
    subs["plan_name_norm"] = subs["plan_name"].map(lambda x: _clean_str(x))

    merch["category_norm"] = merch["category"].map(lambda x: (_clean_str(x) or "Other"))
    merch["merchant_name_norm"] = merch["name"].map(lambda x: normalize_merchant_name(_clean_str(x)))
    merch["risk_score"] = pd.to_numeric(merch["risk_score"], errors="coerce")
    merch["has_free_trial"] = merch["has_free_trial"].astype(str).str.lower().isin(["true","1","yes","y"])
    merch["usual_trial_days"] = pd.to_numeric(merch["usual_trial_days"], errors="coerce")
    merch["monthly_amount"] = pd.to_numeric(merch["monthly_amount"], errors="coerce")
    merch["annual_amount"] = pd.to_numeric(merch["annual_amount"], errors="coerce")

    df = subs.merge(merch, on="merchant_id", how="left", suffixes=("", "_m"))

    df["subscription_amount"] = pd.to_numeric(df["subscription_amount"], errors="coerce")
    df["monthly_equiv"] = df.apply(
        lambda r: monthly_equivalent(r["subscription_amount"], r["billing_frequency_norm"]),
        axis=1
    )

    df["merchant_name"] = df["name"].map(lambda x: _clean_str(x))
    df["merchant_name_norm"] = df["merchant_name"].map(lambda x: normalize_merchant_name(x))

    return df


# -----------------------------
# Tool factory: tools capture df + as_of in closures
# -----------------------------

def make_tools(df: pd.DataFrame, as_of: date):
    @tool
    def dataset_brief() -> str:
        """Return a compact summary of the dataset so you can plan which insight tools to call."""
        active = df[df.apply(lambda r: is_active(r, as_of), axis=1)]
        brief = {
            "as_of": str(as_of),
            "rows": int(len(df)),
            "active_estimate": int(len(active)),
            "has_next_billing_date": bool(df["next_billing_date"].notna().any()),
            "has_risk_score": bool(df["risk_score"].notna().any()),
            "has_metadata_prices": bool(df["monthly_amount"].notna().any() or df["annual_amount"].notna().any()),
            "categories_sample": sorted(df["category_norm"].dropna().unique().tolist())[:25],
        }
        return json.dumps(brief, default=str)

    @tool
    def monthly_commitment() -> str:
        """Compute monthly subscription commitment (monthly-equivalent), plus category breakdown and top items."""
        active_df = df[df.apply(lambda r: is_active(r, as_of), axis=1)].copy()
        total = float(active_df["monthly_equiv"].fillna(0).sum())
        by_cat = (
            active_df.groupby("category_norm")["monthly_equiv"]
            .sum()
            .sort_values(ascending=False)
            .fillna(0)
            .to_dict()
        )
        top_items = (
            active_df.sort_values("monthly_equiv", ascending=False)
            .head(10)[["subscription_id","merchant_id","merchant_name","category_norm",
                      "billing_frequency_norm","subscription_amount","monthly_equiv",
                      "next_billing_date","status_norm"]]
            .to_dict(orient="records")
        )
        severity = "high" if total >= 80 else "medium" if total >= 30 else "low"
        card = InsightCard(
            id="monthly_commitment",
            title="Monthly subscription commitment",
            severity=severity,
            confidence=0.95 if len(active_df) > 0 else 0.6,
            facts={
                "as_of": str(as_of),
                "active_subscription_count": int(len(active_df)),
                "total_monthly_equivalent": round(total, 2),
                "by_category_monthly_equivalent": {k: round(float(v), 2) for k, v in by_cat.items()},
            },
            recommended_actions=[
                "Review the top 3 subscriptions by cost and confirm they’re worth it.",
                "If you see overlaps (e.g., multiple streaming), consider consolidating."
            ],
            supporting_items=top_items,
            generated_at=_now_iso(as_of),
        )
        return json.dumps(asdict(card), default=str)

    @tool
    def upcoming_renewals(days: int = 30) -> str:
        """List renewals due in the next N days (default 30)."""
        end = as_of + timedelta(days=int(days))
        active_df = df[df.apply(lambda r: is_active(r, as_of), axis=1)].copy()
        window = active_df[
            active_df["next_billing_date"].notna() &
            (active_df["next_billing_date"] >= as_of) &
            (active_df["next_billing_date"] <= end)
        ].copy()
        window["amount"] = window["subscription_amount"].fillna(0)
        total = float(window["amount"].sum())
        items = window.sort_values("next_billing_date")[[
            "subscription_id","merchant_id","merchant_name","plan_name_norm",
            "billing_frequency_norm","subscription_amount","next_billing_date","status_norm",
            "category_norm","risk_score"
        ]].to_dict(orient="records")
        severity = "high" if len(window) >= 5 or total >= 100 else "medium" if len(window) >= 2 else "low"
        card = InsightCard(
            id=f"upcoming_renewals_{int(days)}d",
            title=f"Renewals in the next {int(days)} days",
            severity=severity,
            confidence=0.9,
            facts={
                "window_start": str(as_of),
                "window_end": str(end),
                "renewal_count": int(len(window)),
                "expected_total_amount": round(total, 2),
            },
            recommended_actions=[
                "Cancel anything you don’t want before its billing date.",
                "If a merchant allows it, move billing dates to smooth cashflow."
            ],
            supporting_items=items[:25],
            generated_at=_now_iso(as_of),
        )
        return json.dumps(asdict(card), default=str)

    @tool
    def trial_endings(days: int = 14) -> str:
        """Estimate which free trials end in the next N days (default 14) using start_date + usual_trial_days and/or next_billing_date."""
        end = as_of + timedelta(days=int(days))
        active_df = df[df.apply(lambda r: is_active(r, as_of), axis=1)].copy()

        def trial_end_est(r) -> Optional[date]:
            if isinstance(r.get("start_date"), date) and bool(r.get("has_free_trial", False)) and pd.notna(r.get("usual_trial_days")):
                return r["start_date"] + timedelta(days=int(r["usual_trial_days"]))
            return None

        active_df["trial_end_est"] = active_df.apply(trial_end_est, axis=1)

        def pick_trial_end(r) -> Optional[date]:
            t = r.get("trial_end_est")
            n = r.get("next_billing_date")
            if isinstance(t, date) and isinstance(n, date):
                return min(t, n)
            if isinstance(n, date) and bool(r.get("has_free_trial", False)) and r.get("status_norm") in {"trialing","trial"}:
                return n
            if isinstance(t, date):
                return t
            return None

        active_df["trial_end"] = active_df.apply(pick_trial_end, axis=1)

        window = active_df[
            active_df["trial_end"].notna() &
            (active_df["trial_end"] >= as_of) &
            (active_df["trial_end"] <= end)
        ].copy()

        items = window.sort_values("trial_end")[[
            "subscription_id","merchant_id","merchant_name","category_norm","plan_name_norm",
            "start_date","trial_end","next_billing_date","subscription_amount","billing_frequency_norm"
        ]].to_dict(orient="records")

        severity = "high" if len(window) >= 3 else "medium" if len(window) >= 1 else "low"
        card = InsightCard(
            id=f"trial_endings_{int(days)}d",
            title=f"Trials ending in the next {int(days)} days (estimated)",
            severity=severity,
            confidence=0.75,
            facts={"window_start": str(as_of), "window_end": str(end), "trial_ending_count": int(len(window))},
            recommended_actions=[
                "If you don’t want to keep a trial, cancel 24–48 hours before its end date.",
                "If you’ll keep it long-term, compare monthly vs annual pricing."
            ],
            supporting_items=items[:25],
            generated_at=_now_iso(as_of),
        )
        return json.dumps(asdict(card), default=str)

    @tool
    def spike_months(horizon_days: int = 365) -> str:
        """Forecast high-cost months over the next horizon using next_billing_date + billing_frequency (approx)."""
        horizon_end = as_of + timedelta(days=int(horizon_days))
        active_df = df[df.apply(lambda r: is_active(r, as_of), axis=1)].copy()

        events: List[Dict[str, Any]] = []
        for _, r in active_df.iterrows():
            nbd = r.get("next_billing_date")
            amt = safe_float(r.get("subscription_amount"))
            freq = r.get("billing_frequency_norm", "unknown")
            if not isinstance(nbd, date) or amt is None:
                continue
            d = nbd
            i = 0
            while isinstance(d, date) and d <= horizon_end and i < 400:
                if d >= as_of:
                    events.append({
                        "charge_date": d,
                        "charge_month": date(d.year, d.month, 1),
                        "amount": amt,
                        "subscription_id": r.get("subscription_id"),
                        "merchant_name": r.get("merchant_name"),
                        "category": r.get("category_norm"),
                    })
                d = add_period(d, freq)
                i += 1

        if not events:
            card = InsightCard(
                id="spike_months",
                title="High-cost months in the next 12 months (forecast)",
                severity="low",
                confidence=0.4,
                facts={"note": "Not enough data to simulate upcoming charges (missing next_billing_date or amount)."},
                recommended_actions=["Ensure next_billing_date and subscription_amount are populated for active subscriptions."],
                supporting_items=[],
                generated_at=_now_iso(as_of),
            )
            return json.dumps(asdict(card), default=str)

        ev = pd.DataFrame(events)
        monthly = ev.groupby("charge_month")["amount"].sum().sort_values(ascending=False)
        worst_month = monthly.index[0]
        drivers = ev[ev["charge_month"] == worst_month].sort_values("amount", ascending=False).head(10).to_dict(orient="records")

        severity = "high" if float(monthly.iloc[0]) >= 150 else "medium" if float(monthly.iloc[0]) >= 80 else "low"
        card = InsightCard(
            id="spike_months",
            title="High-cost months in the next 12 months (forecast)",
            severity=severity,
            confidence=0.7,
            facts={
                "as_of": str(as_of),
                "worst_month": str(worst_month),
                "worst_month_expected_total": round(float(monthly.iloc[0]), 2),
                "top_spike_months": [{"month": str(k), "expected_total_amount": round(float(v), 2)} for k, v in monthly.head(6).items()],
            },
            recommended_actions=[
                "Review annual/large renewals in your highest-cost month.",
                "Set reminders 7–14 days before big renewals."
            ],
            supporting_items=drivers,
            generated_at=_now_iso(as_of),
        )
        return json.dumps(asdict(card), default=str)

    @tool
    def duplicates() -> str:
        """Detect possible duplicate subscriptions (simple heuristic: multiple items in same category)."""
        active_df = df[df.apply(lambda r: is_active(r, as_of), axis=1)].copy()
        active_df["cat"] = active_df["category_norm"].fillna("Other")
        groups = active_df.groupby("cat").size().sort_values(ascending=False)
        dup_cats = [c for c, n in groups.items() if int(n) >= 2]

        items: List[Dict[str, Any]] = []
        for c in dup_cats[:8]:
            subset = active_df[active_df["cat"] == c].sort_values("monthly_equiv", ascending=False)
            items.extend(subset[[
                "subscription_id","merchant_id","merchant_name","plan_name_norm",
                "billing_frequency_norm","subscription_amount","monthly_equiv","next_billing_date","status_norm","category_norm"
            ]].to_dict(orient="records"))

        severity = "high" if len(dup_cats) >= 3 else "medium" if len(dup_cats) >= 1 else "low"
        card = InsightCard(
            id="duplicate_categories",
            title="Possible duplicate subscriptions (by category)",
            severity=severity,
            confidence=0.65,
            facts={"duplicate_categories": dup_cats, "category_counts": {k: int(v) for k, v in groups.to_dict().items()}},
            recommended_actions=[
                "If multiple subscriptions do the same job, keep the best one and cancel the rest.",
                "If duplicates are intentional, mark them as expected to reduce noise."
            ],
            supporting_items=items[:30],
            generated_at=_now_iso(as_of),
        )
        return json.dumps(asdict(card), default=str)

    @tool
    def risk_exposure(risk_threshold: float = 0.7) -> str:
        """Flag higher-risk subscription merchants where risk_score >= threshold."""
        active_df = df[df.apply(lambda r: is_active(r, as_of), axis=1)].copy()
        active_df["risk_score"] = pd.to_numeric(active_df["risk_score"], errors="coerce")
        flagged = active_df[active_df["risk_score"].notna() & (active_df["risk_score"] >= float(risk_threshold))].copy()

        items = flagged.sort_values("risk_score", ascending=False)[[
            "subscription_id","merchant_id","merchant_name","category_norm",
            "risk_score","subscription_amount","billing_frequency_norm","next_billing_date","status_norm"
        ]].to_dict(orient="records")

        severity = "high" if len(flagged) >= 2 else "medium" if len(flagged) == 1 else "low"
        card = InsightCard(
            id="merchant_risk_exposure",
            title="Higher-risk subscription merchants (review recommended)",
            severity=severity,
            confidence=0.85,
            facts={"high_risk_count": int(len(flagged)), "risk_score_threshold": float(risk_threshold)},
            recommended_actions=[
                "Review unfamiliar or high-risk merchants and confirm you recognize them.",
                "If anything looks wrong, cancel and contact your bank."
            ],
            supporting_items=items[:25],
            generated_at=_now_iso(as_of),
        )
        return json.dumps(asdict(card), default=str)

    @tool
    def price_mismatch(pct_threshold: float = 0.25) -> str:
        """Compare subscription_amount to merchant metadata monthly_amount/annual_amount and flag big mismatches."""
        active_df = df[df.apply(lambda r: is_active(r, as_of), axis=1)].copy()

        def expected_price(r) -> Optional[float]:
            freq = r.get("billing_frequency_norm")
            if freq == "monthly":
                return safe_float(r.get("monthly_amount"))
            if freq == "annual":
                return safe_float(r.get("annual_amount"))
            return None

        active_df["expected_price"] = active_df.apply(expected_price, axis=1)
        active_df["actual_price"] = active_df["subscription_amount"].map(safe_float)
        cand = active_df[active_df["expected_price"].notna() & active_df["actual_price"].notna()].copy()

        if cand.empty:
            card = InsightCard(
                id="price_mismatch",
                title="Subscriptions with unusually different prices vs metadata",
                severity="low",
                confidence=0.4,
                facts={"note": "No comparable monthly_amount/annual_amount metadata for active subscriptions."},
                recommended_actions=["Populate monthly_amount/annual_amount in merchant metadata for stronger price insights."],
                supporting_items=[],
                generated_at=_now_iso(as_of),
            )
            return json.dumps(asdict(card), default=str)

        cand["pct_diff"] = (cand["actual_price"] - cand["expected_price"]).abs() / cand["expected_price"].replace(0, float("nan"))
        flagged = cand[cand["pct_diff"].notna() & (cand["pct_diff"] >= float(pct_threshold))].copy()

        items = flagged.sort_values("pct_diff", ascending=False)[[
            "subscription_id","merchant_id","merchant_name","billing_frequency_norm",
            "actual_price","expected_price","pct_diff","plan_name_norm","category_norm"
        ]].to_dict(orient="records")

        severity = "high" if len(flagged) >= 3 else "medium" if len(flagged) >= 1 else "low"
        card = InsightCard(
            id="price_mismatch",
            title="Subscriptions with unusually different prices vs metadata",
            severity=severity,
            confidence=0.7,
            facts={"flagged_count": int(len(flagged)), "pct_threshold": float(pct_threshold)},
            recommended_actions=[
                "If price is higher than expected, check tier changes or price increases.",
                "If you want cheaper, downgrade or consider annual only if you’ll keep it."
            ],
            supporting_items=items[:25],
            generated_at=_now_iso(as_of),
        )
        return json.dumps(asdict(card), default=str)

    @tool
    def annual_arbitrage() -> str:
        """Calculate potential savings by switching from monthly to annual plans for subscriptions with 12+ months of payment history."""
        active_df = df[df.apply(lambda r: is_active(r, as_of), axis=1)].copy()
        
        # Filter for monthly subscriptions that have both monthly and annual pricing available
        monthly_subs = active_df[active_df["billing_frequency_norm"] == "monthly"].copy()
        monthly_subs["monthly_amt"] = monthly_subs["monthly_amount"].map(safe_float)
        monthly_subs["annual_amt"] = monthly_subs["annual_amount"].map(safe_float)
        
        # Only consider subscriptions with both pricing options
        candidates = monthly_subs[
            monthly_subs["monthly_amt"].notna() & 
            monthly_subs["annual_amt"].notna() &
            (monthly_subs["monthly_amt"] > 0)
        ].copy()
        
        if candidates.empty:
            card = InsightCard(
                id="annual_arbitrage",
                title="Annual vs. Monthly Savings Opportunities",
                severity="low",
                confidence=0.5,
                facts={"note": "No monthly subscriptions found with both monthly and annual pricing data."},
                recommended_actions=["Add annual_amount metadata to merchants to enable savings analysis."],
                supporting_items=[],
                generated_at=_now_iso(as_of),
            )
            return json.dumps(asdict(card), default=str)
        
        # Calculate payment history length (months between start date and last billed date)
        candidates["start_dt"] = pd.to_datetime(candidates["start_date"], errors="coerce")
        candidates["last_billed_dt"] = pd.to_datetime(candidates["last_billed_date"], errors="coerce")
        
        def calc_months_active(row) -> float:
            if pd.notna(row["start_dt"]) and pd.notna(row["last_billed_dt"]):
                delta = row["last_billed_dt"] - row["start_dt"]
                return max(1, delta.days / 30.44)  # Average days per month
            return 0
        
        candidates["months_active"] = candidates.apply(calc_months_active, axis=1)
        
        # Filter for subscriptions with 12+ months of history
        long_term = candidates[candidates["months_active"] >= 12].copy()
        
        # Calculate savings
        long_term["annual_cost_if_monthly"] = long_term["monthly_amt"] * 12
        long_term["annual_savings"] = long_term["annual_cost_if_monthly"] - long_term["annual_amt"]
        long_term["savings_pct"] = (long_term["annual_savings"] / long_term["annual_cost_if_monthly"] * 100).round(2)
        
        # Only flag subscriptions with positive savings
        savings_opps = long_term[long_term["annual_savings"] > 0].copy()
        
        if savings_opps.empty:
            card = InsightCard(
                id="annual_arbitrage",
                title="Annual vs. Monthly Savings Opportunities",
                severity="low",
                confidence=0.7,
                facts={
                    "subscriptions_analyzed": int(len(long_term)),
                    "savings_opportunities": 0,
                    "note": "No savings found by switching to annual plans."
                },
                recommended_actions=["Continue with monthly plans or review for other optimization opportunities."],
                supporting_items=[],
                generated_at=_now_iso(as_of),
            )
            return json.dumps(asdict(card), default=str)
        
        # Sort by absolute savings amount
        savings_opps = savings_opps.sort_values("annual_savings", ascending=False)
        
        total_potential_savings = float(savings_opps["annual_savings"].sum())
        
        items = savings_opps[[
            "subscription_id", "merchant_name", "plan_name_norm", "category_norm",
            "months_active", "monthly_amt", "annual_amt", 
            "annual_cost_if_monthly", "annual_savings", "savings_pct"
        ]].to_dict(orient="records")
        
        severity = "high" if total_potential_savings >= 100 else "medium" if total_potential_savings >= 50 else "low"
        
        card = InsightCard(
            id="annual_arbitrage",
            title="Annual vs. Monthly Savings Opportunities",
            severity=severity,
            confidence=0.85,
            facts={
                "subscriptions_with_12plus_months": int(len(long_term)),
                "savings_opportunities_found": int(len(savings_opps)),
                "total_potential_annual_savings": round(total_potential_savings, 2),
                "average_savings_pct": round(float(savings_opps["savings_pct"].mean()), 2)
            },
            recommended_actions=[
                "Switch to annual billing for subscriptions you plan to keep long-term.",
                "Calculate break-even: if you cancel before 12 months, you may lose money on annual plans.",
                "Contact merchant support to switch billing frequency—some offer prorated upgrades."
            ],
            supporting_items=items[:20],
            generated_at=_now_iso(as_of),
        )
        return json.dumps(asdict(card), default=str)

    return [
        dataset_brief,
        monthly_commitment,
        upcoming_renewals,
        trial_endings,
        spike_months,
        duplicates,
        risk_exposure,
        price_mismatch,
        annual_arbitrage,
    ]


# -----------------------------
# Orchestration
# -----------------------------

SYSTEM_PROMPT = """You are a subscription insights agent.

You have tools that compute factual insight cards from the user's subscription dataset.
Use tools as needed, and DO NOT invent dates or amounts.

When you finish, respond with valid JSON:
{{
  "summary": "short summary",
  "cards": [ ... insight card objects from tool outputs ... ],
  "notes": ["any caveats"]
}}

Rules:
- Prefer 2–5 tools unless the user asks for a full audit.
- If the question is about renewals/cancel-by, call upcoming_renewals (and trial_endings if trials exist).
- If it's about saving money, call monthly_commitment + duplicates + price_mismatch + annual_arbitrage.
- If it's about fraud/safety, call risk_exposure and upcoming_renewals.
- If asked about annual vs monthly savings, call annual_arbitrage.
"""

def run_agent(df: pd.DataFrame, as_of: date, query: str, model: str, verbose: bool = True) -> Dict[str, Any]:
    tools = make_tools(df, as_of)

    llm = ChatOpenAI(model=model, temperature=0)

    if USING_NEW_API:
        # LangChain 1.1.0+ API
        graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            debug=verbose,
        )
        
        # Invoke with messages format
        result = graph.invoke({"messages": [{"role": "user", "content": query}]})
        
        # Extract messages and tool calls
        messages = result.get("messages", [])
        cards: List[Dict[str, Any]] = []
        called_tools: List[str] = []
        
        # Parse messages to extract tool calls and results
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    called_tools.append(tc.get("name", "unknown"))
            # Check for tool messages (results)
            if hasattr(msg, "type") and msg.type == "tool":
                try:
                    obj = json.loads(msg.content)
                    if isinstance(obj, dict) and "id" in obj and "title" in obj and "facts" in obj:
                        cards.append(obj)
                except Exception:
                    pass
        
        # Get final response
        final_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
                final_text = msg.content
                break
    else:
        # Old API
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            return_intermediate_steps=True,
            max_iterations=8,
        )

        result = executor.invoke({"input": query})

        # Collect tool outputs (cards) from intermediate steps
        cards = []
        called_tools = []
        for step in result.get("intermediate_steps", []):
            action, observation = step
            try:
                called_tools.append(getattr(action, "tool", "unknown"))
            except Exception:
                pass
            try:
                obj = json.loads(observation)
                if isinstance(obj, dict) and "id" in obj and "title" in obj and "facts" in obj:
                    cards.append(obj)
            except Exception:
                continue
        
        final_text = result.get("output", "")

    # Try to parse agent final output as JSON, else wrap it
    try:
        final_json = json.loads(final_text)
        # Ensure cards present even if model omitted them
        if isinstance(final_json, dict) and "cards" not in final_json:
            final_json["cards"] = cards
        if isinstance(final_json, dict) and "cards" in final_json and not final_json["cards"]:
            final_json["cards"] = cards
    except Exception:
        final_json = {
            "summary": final_text,
            "cards": cards,
            "notes": ["Agent output was not valid JSON; wrapped as summary."],
        }

    final_json["called_tools"] = called_tools
    final_json["as_of"] = str(as_of)
    return final_json


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subs", required=True)
    p.add_argument("--merchants", required=True)
    p.add_argument("--asof", default=None, help="YYYY-MM-DD; default today")
    p.add_argument("--query", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--out", default="insights.json")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var is required for LangChain tool-calling agent.")

    as_of = date.today() if not args.asof else datetime.strptime(args.asof, "%Y-%m-%d").date()
    df = load_data(args.subs, args.merchants)

    out = run_agent(df=df, as_of=as_of, query=args.query, model=args.model, verbose=(not args.quiet))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(json.dumps(out, indent=2, default=str))
    print(f"\nWrote: {args.out}")

if __name__ == "__main__":
    main()