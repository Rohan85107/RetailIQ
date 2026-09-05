import os
import re
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai


# ============================================================
# RETAILIQ - AI SALES & INVENTORY COPILOT
# NexusTIQ Hackathon - PS03 Retail
# ============================================================

load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================
# GEMINI SETUP
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("Gemini initialization warning:", e)


GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# DATA LOADING
# ============================================================

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def load_data():
    products = load_csv("products.csv")
    sales = load_csv("sales.csv")
    inventory = load_csv("inventory.csv")

    # Normalize column names
    products.columns = [str(c).strip().lower() for c in products.columns]
    sales.columns = [str(c).strip().lower() for c in sales.columns]
    inventory.columns = [str(c).strip().lower() for c in inventory.columns]

    # Sales quantity compatibility
    if "quantity" in sales.columns:
        sales["units_sold"] = pd.to_numeric(
            sales["quantity"], errors="coerce"
        ).fillna(0)
    elif "units_sold" in sales.columns:
        sales["units_sold"] = pd.to_numeric(
            sales["units_sold"], errors="coerce"
        ).fillna(0)
    else:
        raise ValueError(
            "sales.csv must contain either 'quantity' or 'units_sold'"
        )

    sales["date"] = pd.to_datetime(
        sales["date"], errors="coerce"
    )

    # Numeric product fields
    for col in ["price", "reorder_level"]:
        if col in products.columns:
            products[col] = pd.to_numeric(
                products[col], errors="coerce"
            ).fillna(0)

    # Inventory numeric field
    if "current_stock" not in inventory.columns:
        raise ValueError(
            "inventory.csv must contain 'current_stock'"
        )

    inventory["current_stock"] = pd.to_numeric(
        inventory["current_stock"], errors="coerce"
    ).fillna(0)

    return products, sales, inventory


try:
    PRODUCTS, SALES, INVENTORY = load_data()
    DATA_ERROR = None
except Exception as e:
    PRODUCTS = pd.DataFrame()
    SALES = pd.DataFrame()
    INVENTORY = pd.DataFrame()
    DATA_ERROR = str(e)


# ============================================================
# PRODUCT ANALYSIS
# ============================================================

def get_product_analysis():

    if DATA_ERROR:
        return pd.DataFrame()

    sales_summary = (
        SALES.groupby("product_id", as_index=False)["units_sold"]
        .sum()
    )

    df = PRODUCTS.merge(
        sales_summary,
        on="product_id",
        how="left"
    )

    df = df.merge(
        INVENTORY[["product_id", "current_stock"]],
        on="product_id",
        how="left"
    )

    df["units_sold"] = df["units_sold"].fillna(0)
    df["current_stock"] = df["current_stock"].fillna(0)

    # Revenue
    if "price" in df.columns:
        df["revenue"] = (
            df["price"].fillna(0) * df["units_sold"]
        )
    else:
        df["price"] = 0
        df["revenue"] = 0

    # --------------------------------------------------------
    # Average daily sales
    # --------------------------------------------------------

    date_stats = (
        SALES.groupby("product_id")["date"]
        .agg(["min", "max"])
        .reset_index()
    )

    df = df.merge(
        date_stats,
        on="product_id",
        how="left"
    )

    df["active_days"] = (
        (df["max"] - df["min"]).dt.days + 1
    )

    df["active_days"] = df["active_days"].fillna(1)
    df["active_days"] = df["active_days"].clip(lower=1)

    df["avg_daily_sales"] = (
        df["units_sold"] / df["active_days"]
    )

    # Stock coverage
    df["stock_days"] = df.apply(
        lambda row:
        row["current_stock"] / row["avg_daily_sales"]
        if row["avg_daily_sales"] > 0
        else 9999,
        axis=1
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    def calculate_risk(days):

        if days <= 7:
            return "CRITICAL"

        if days <= 14:
            return "HIGH"

        if days <= 30:
            return "MEDIUM"

        return "LOW"

    df["risk"] = df["stock_days"].apply(calculate_risk)

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    df["status"] = df["risk"].apply(
        lambda x:
        "LOW STOCK"
        if x in ["CRITICAL", "HIGH"]
        else "OK"
    )

    # --------------------------------------------------------
    # Reorder level
    # --------------------------------------------------------

    if "reorder_level" not in df.columns:
        df["reorder_level"] = 10

    df["reorder_level"] = pd.to_numeric(
        df["reorder_level"],
        errors="coerce"
    ).fillna(10)

    # --------------------------------------------------------
    # 14-day expected demand
    # --------------------------------------------------------

    df["expected_14_day_demand"] = (
        df["avg_daily_sales"] * 14
    ).round().astype(int)

    # --------------------------------------------------------
    # Suggested reorder
    # --------------------------------------------------------

    df["suggested_reorder"] = (
        df["expected_14_day_demand"]
        .combine(
            df["reorder_level"],
            max
        )
        - df["current_stock"]
    )

    df["suggested_reorder"] = (
        df["suggested_reorder"]
        .clip(lower=0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Overstock
    # --------------------------------------------------------

    df["overstock"] = (
        (df["stock_days"] > 90)
        & (df["units_sold"] < 10)
    )

    # Cleanup
    df = df.drop(
        columns=["min", "max"],
        errors="ignore"
    )

    return df


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):
    return f"₹{float(value):,.0f}"


def product_name(row):
    return row.get(
        "product_name",
        row.get("product_id", "Unknown Product")
    )


def get_analysis():
    return get_product_analysis()


# ============================================================
# DETERMINISTIC ANSWERS
# ============================================================

def deterministic_answer(question):

    q = question.lower().strip()

    df = get_analysis()

    if df.empty:
        return (
            "I cannot answer this because the store data "
            "could not be loaded."
        )

    total_revenue = df["revenue"].sum()
    total_stock = df["current_stock"].sum()
    total_units = df["units_sold"].sum()

    # --------------------------------------------------------
    # TOTAL REVENUE
    # --------------------------------------------------------

    if (
        "total revenue" in q
        or "store revenue" in q
        or "revenue of store" in q
        or "revenue generated" in q
    ):
        return (
            f"The total store revenue is {money(total_revenue)} "
            f"from {int(total_units)} units sold across "
            f"{len(df)} products."
        )

    # --------------------------------------------------------
    # TOTAL STOCK
    # --------------------------------------------------------

    if (
        "total stock" in q
        or "how much stock" in q
        or "stock available" in q
        or "inventory available" in q
    ):
        return (
            f"The store currently has "
            f"{int(total_stock)} units in stock "
            f"across {len(df)} products."
        )

    # --------------------------------------------------------
    # TOTAL SALES
    # --------------------------------------------------------

    if (
        "total sales" in q
        or "total units sold" in q
        or "how many units sold" in q
        or "units sold" in q
    ):
        return (
            f"The store sold {int(total_units)} units "
            f"in the available sales data."
        )

    # --------------------------------------------------------
    # HIGHEST REVENUE
    # --------------------------------------------------------

    if (
        "highest revenue" in q
        or "most revenue" in q
        or "revenue leader" in q
        or "highest earning" in q
    ):
        row = df.sort_values(
            "revenue",
            ascending=False
        ).iloc[0]

        return (
            f"{product_name(row)} has the highest revenue "
            f"at {money(row['revenue'])}, "
            f"with {int(row['units_sold'])} units sold."
        )

    # --------------------------------------------------------
    # BEST SELLER
    # --------------------------------------------------------

    if (
        "best seller" in q
        or "best selling" in q
        or "top seller" in q
        or "top selling" in q
        or "most sold" in q
    ):

        top = (
            df.sort_values(
                "units_sold",
                ascending=False
            )
            .head(5)
        )

        lines = [
            f"{i + 1}. {product_name(row)} — "
            f"{int(row['units_sold'])} units"
            for i, (_, row) in enumerate(top.iterrows())
        ]

        return (
            "Top-selling products based on recorded units sold:\n"
            + "\n".join(lines)
        )

    # --------------------------------------------------------
    # STOCK OUT
    # --------------------------------------------------------

    if (
        "stock out" in q
        or "stockout" in q
        or "run out" in q
        or "running out" in q
        or "low stock" in q
        or "stock risk" in q
    ):

        risk_df = df[
            df["risk"].isin(["CRITICAL", "HIGH"])
        ].sort_values("stock_days")

        if risk_df.empty:
            return "No immediate stock-out risk was detected."

        lines = []

        for _, row in risk_df.iterrows():
            lines.append(
                f"• {product_name(row)} — "
                f"{row['current_stock']:.0f} units left, "
                f"approximately {row['stock_days']:.1f} days of stock."
            )

        return (
            "Products currently at stock-out risk:\n"
            + "\n".join(lines)
        )

    # --------------------------------------------------------
    # RESTOCK
    # --------------------------------------------------------

    if (
        "restock" in q
        or "reorder" in q
        or "what should i buy" in q
        or "what should we buy" in q
        or "which products should i order" in q
    ):

        restock_df = df[
            df["suggested_reorder"] > 0
        ].copy()

        restock_df["priority_score"] = (
            restock_df["avg_daily_sales"] * 100
            + (100 / restock_df["stock_days"].clip(lower=1))
        )

        restock_df = restock_df.sort_values(
            "priority_score",
            ascending=False
        ).head(5)

        if restock_df.empty:
            return "No immediate restocking recommendation is required."

        lines = []

        for _, row in restock_df.iterrows():
            lines.append(
                f"• {product_name(row)} — "
                f"recommended reorder: "
                f"{int(row['suggested_reorder'])} units; "
                f"current stock: {int(row['current_stock'])}; "
                f"coverage: {row['stock_days']:.1f} days."
            )

        return (
            "Recommended restocking priorities "
            "based on current sales velocity and stock:\n"
            + "\n".join(lines)
            + "\n\nAssumption: the recommendation uses "
              "the recorded sales rate and a 14-day demand horizon."
        )

    # --------------------------------------------------------
    # SLOW MOVING
    # --------------------------------------------------------

    if (
        "slow moving" in q
        or "slow-moving" in q
        or "slow products" in q
        or "poor selling" in q
    ):

        slow = df[
            (df["units_sold"] < 10)
            & (df["current_stock"] > 20)
        ].sort_values(
            "units_sold"
        )

        if slow.empty:
            return "No major slow-moving products were detected."

        lines = []

        for _, row in slow.iterrows():
            lines.append(
                f"• {product_name(row)} — "
                f"{int(row['units_sold'])} units sold, "
                f"{int(row['current_stock'])} units in stock, "
                f"revenue {money(row['revenue'])}."
            )

        return (
            "Slow-moving products with relatively high inventory:\n"
            + "\n".join(lines)
        )

    # --------------------------------------------------------
    # OVERSTOCK
    # --------------------------------------------------------

    if (
        "overstock" in q
        or "excess stock" in q
        or "excess inventory" in q
        or "too much stock" in q
    ):

        over = df[
            df["overstock"]
        ].sort_values(
            "stock_days",
            ascending=False
        )

        if over.empty:
            return "No major overstock was detected using the current rule."

        lines = []

        for _, row in over.iterrows():
            lines.append(
                f"• {product_name(row)} — "
                f"{int(row['current_stock'])} units in stock, "
                f"approximately {row['stock_days']:.0f} days of coverage."
            )

        return (
            "Potential overstock products:\n"
            + "\n".join(lines)
        )

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if "category" in q:

        category_data = (
            df.groupby("category")
            .agg(
                units_sold=("units_sold", "sum"),
                revenue=("revenue", "sum")
            )
            .sort_values(
                "revenue",
                ascending=False
            )
        )

        lines = []

        for category, row in category_data.iterrows():
            lines.append(
                f"• {category} — "
                f"{int(row['units_sold'])} units, "
                f"{money(row['revenue'])} revenue."
            )

        return (
            "Category performance:\n"
            + "\n".join(lines)
        )

    # --------------------------------------------------------
    # FUTURE PREDICTION - NO GUESSING
    # --------------------------------------------------------

    future_words = [
        "next month",
        "next year",
        "future sales",
        "will sell",
        "will have",
        "predict",
        "forecast",
        "prediction",
        "future revenue",
        "exact profit"
    ]

    if any(word in q for word in future_words):

        return (
            "I cannot reliably predict that from the available data. "
            "RetailIQ currently has historical sales, product and "
            "inventory data, but it does not contain a validated "
            "future-demand forecasting model or future external factors. "
            "I will not guess."
        )

    return None


# ============================================================
# GEMINI ANSWER
# ============================================================

def ask_gemini(question):

    if gemini_client is None:
        return None

    df = get_analysis()

    if df.empty:
        return None

    # Keep only useful information for Gemini
    records = []

    for _, row in df.iterrows():

        records.append({
            "product_id": row["product_id"],
            "product_name": product_name(row),
            "category": row.get("category", ""),
            "price": float(row.get("price", 0)),
            "units_sold": int(row["units_sold"]),
            "current_stock": int(row["current_stock"]),
            "revenue": float(row["revenue"]),
            "avg_daily_sales": round(
                float(row["avg_daily_sales"]), 2
            ),
            "stock_days": round(
                float(row["stock_days"]), 1
            ),
            "risk": row["risk"],
            "suggested_reorder": int(
                row["suggested_reorder"]
            )
        })

    prompt = f"""
You are RetailIQ, an AI Sales and Inventory Copilot.

Answer the store manager's question using ONLY the supplied
RetailIQ store data.

IMPORTANT RULES:
1. Never invent numbers.
2. Never invent products.
3. Always use actual numbers when available.
4. If the data cannot answer the question, clearly say:
   "Insufficient data to answer reliably."
5. Give concise, practical recommendations.
6. Mention assumptions when making recommendations.
7. Do not claim a future prediction unless the data supports it.
8. Prefer bullet points for multiple products.

STORE DATA:
{records}

USER QUESTION:
{question}
"""

    try:

        response = gemini_client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt
        )

        text = getattr(
            response,
            "output_text",
            None
        )

        if text:
            return text.strip()

    except Exception as e:

        error_text = str(e).lower()

        print("Gemini error:", e)

        # Quota / rate limit
        if (
            "429" in error_text
            or "quota" in error_text
            or "resource_exhausted" in error_text
            or "rate limit" in error_text
        ):
            return None

        return None

    return None


# ============================================================
# FINAL ANSWER ROUTER
# ============================================================

def answer_question(question):

    # First: deterministic analysis
    deterministic = deterministic_answer(question)

    if deterministic:
        return deterministic, "RetailIQ deterministic analysis"

    # Second: Gemini
    ai_answer = ask_gemini(question)

    if ai_answer:
        return ai_answer, "Gemini + RetailIQ store data"

    # Final fallback
    return (
        "I could not reliably answer that question from the "
        "available store data.\n\n"
        "Try questions such as:\n"
        "• What is the total revenue?\n"
        "• Which product has the highest revenue?\n"
        "• Which products are at risk of stock-out?\n"
        "• Which products should I restock first?\n"
        "• Which products are slow-moving?\n"
        "• Which products are overstocked?\n"
        "• Which products are the best sellers?"
    ), "RetailIQ deterministic fallback"


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    df = get_analysis()

    if df.empty:
        return jsonify({
            "score": 0,
            "status": "Data Error",
            "low_stock": 0,
            "critical": 0,
            "slow_moving": 0
        })

    low_stock = int(
        (df["status"] == "LOW STOCK").sum()
    )

    critical = int(
        (df["risk"] == "CRITICAL").sum()
    )

    slow_moving = int(
        (
            (df["units_sold"] < 10)
            & (df["current_stock"] > 20)
        ).sum()
    )

    score = (
        100
        - low_stock * 5
        - critical * 5
        - slow_moving * 3
    )

    score = max(
        0,
        min(100, score)
    )

    if score >= 80:
        status = "Healthy"
    elif score >= 60:
        status = "Needs Attention"
    elif score >= 40:
        status = "At Risk"
    else:
        status = "Critical"

    return jsonify({
        "score": score,
        "status": status,
        "low_stock": low_stock,
        "critical": critical,
        "slow_moving": slow_moving
    })


# ============================================================
# SUMMARY
# ============================================================

@app.route("/api/summary")
def summary():

    df = get_analysis()

    if df.empty:
        return jsonify({
            "total_products": 0,
            "total_stock": 0,
            "total_units_sold": 0,
            "low_stock_products": 0,
            "total_revenue": 0
        })

    return jsonify({
        "total_products": int(len(df)),
        "total_stock": int(df["current_stock"].sum()),
        "total_units_sold": int(df["units_sold"].sum()),
        "low_stock_products": int(
            (df["status"] == "LOW STOCK").sum()
        ),
        "total_revenue": float(
            df["revenue"].sum()
        )
    })


# ============================================================
# PRODUCTS
# ============================================================

@app.route("/api/products")
def products_api():

    df = get_analysis()

    if df.empty:
        return jsonify([])

    result = []

    for _, row in df.iterrows():

        result.append({
            "product_id": row["product_id"],
            "product_name": product_name(row),
            "category": row.get("category", ""),
            "price": float(row.get("price", 0)),
            "units_sold": int(row["units_sold"]),
            "current_stock": int(row["current_stock"]),
            "revenue": float(row["revenue"]),
            "status": row["status"],
            "risk": row["risk"],
            "avg_daily_sales": round(
                float(row["avg_daily_sales"]),
                2
            ),
            "stock_days": round(
                float(row["stock_days"]),
                1
            ),
            "suggested_reorder": int(
                row["suggested_reorder"]
            )
        })

    return jsonify(result)


# ============================================================
# INSIGHTS
# ============================================================

@app.route("/api/insights")
def insights():

    df = get_analysis()

    if df.empty:
        return jsonify({
            "stock_out_risk": [],
            "restock_priority": [],
            "slow_moving": [],
            "overstock": [],
            "top_sellers": [],
            "revenue_leaders": []
        })

    # Stock-out risk
    risk_df = (
        df[
            df["risk"].isin(
                ["CRITICAL", "HIGH"]
            )
        ]
        .sort_values("stock_days")
        .head(10)
    )

    stock_out_risk = []

    for _, row in risk_df.iterrows():

        stock_out_risk.append({
            "product_name": product_name(row),
            "current_stock": int(row["current_stock"]),
            "stock_days": round(
                float(row["stock_days"]),
                1
            ),
            "avg_daily_sales": round(
                float(row["avg_daily_sales"]),
                2
            ),
            "risk": row["risk"]
        })

    # Restock priority
    restock = df[
        df["suggested_reorder"] > 0
    ].copy()

    restock["priority_score"] = (
        restock["avg_daily_sales"] * 100
        + (
            100
            / restock["stock_days"].clip(lower=1)
        )
    )

    restock = (
        restock
        .sort_values(
            "priority_score",
            ascending=False
        )
        .head(10)
    )

    restock_priority = []

    for _, row in restock.iterrows():

        restock_priority.append({
            "product_name": product_name(row),
            "current_stock": int(row["current_stock"]),
            "reorder_level": int(row["reorder_level"]),
            "suggested_reorder": int(
                row["suggested_reorder"]
            ),
            "stock_days": round(
                float(row["stock_days"]),
                1
            ),
            "avg_daily_sales": round(
                float(row["avg_daily_sales"]),
                2
            ),
            "risk": row["risk"],
            "priority_score": round(
                float(row["priority_score"]),
                2
            )
        })

    # Slow moving
    slow = (
        df[
            (df["units_sold"] < 10)
            & (df["current_stock"] > 20)
        ]
        .sort_values("units_sold")
        .head(10)
    )

    slow_moving = []

    for _, row in slow.iterrows():

        slow_moving.append({
            "product_name": product_name(row),
            "units_sold": int(row["units_sold"]),
            "current_stock": int(row["current_stock"]),
            "revenue": float(row["revenue"])
        })

    # Overstock
    over = (
        df[df["overstock"]]
        .sort_values(
            "stock_days",
            ascending=False
        )
        .head(10)
    )

    overstock = []

    for _, row in over.iterrows():

        overstock.append({
            "product_name": product_name(row),
            "current_stock": int(row["current_stock"]),
            "stock_days": round(
                float(row["stock_days"]),
                1
            ),
            "units_sold": int(row["units_sold"])
        })

    # Top sellers
    top = (
        df.sort_values(
            "units_sold",
            ascending=False
        )
        .head(10)
    )

    top_sellers = []

    for _, row in top.iterrows():

        top_sellers.append({
            "product_name": product_name(row),
            "units_sold": int(row["units_sold"]),
            "revenue": float(row["revenue"])
        })

    # Revenue leaders
    revenue = (
        df.sort_values(
            "revenue",
            ascending=False
        )
        .head(10)
    )

    revenue_leaders = []

    for _, row in revenue.iterrows():

        revenue_leaders.append({
            "product_name": product_name(row),
            "revenue": float(row["revenue"]),
            "units_sold": int(row["units_sold"])
        })

    return jsonify({
        "stock_out_risk": stock_out_risk,
        "restock_priority": restock_priority,
        "slow_moving": slow_moving,
        "overstock": overstock,
        "top_sellers": top_sellers,
        "revenue_leaders": revenue_leaders
    })


# ============================================================
# ACTIONS
# ============================================================

@app.route("/api/actions")
def actions():

    df = get_analysis()

    if df.empty:
        return jsonify([])

    actions_list = []

    # Critical stock actions
    critical = (
        df[df["risk"] == "CRITICAL"]
        .sort_values("stock_days")
        .head(5)
    )

    for _, row in critical.iterrows():

        actions_list.append({
            "priority": "HIGH",
            "title": f"Restock {product_name(row)}",
            "reason": (
                f"Only {int(row['current_stock'])} units remain "
                f"with approximately {row['stock_days']:.1f} "
                f"days of stock coverage."
            ),
            "action": (
                f"Consider ordering "
                f"{int(row['suggested_reorder'])} units "
                f"based on the 14-day demand assumption."
            )
        })

    # Slow-moving actions
    slow = (
        df[
            (df["units_sold"] < 10)
            & (df["current_stock"] > 20)
        ]
        .sort_values("units_sold")
        .head(3)
    )

    for _, row in slow.iterrows():

        actions_list.append({
            "priority": "MEDIUM",
            "title": f"Review {product_name(row)}",
            "reason": (
                f"Only {int(row['units_sold'])} units sold "
                f"while {int(row['current_stock'])} units "
                f"remain in stock."
            ),
            "action": (
                "Consider promotion, bundling, or reducing "
                "future purchase quantities."
            )
        })

    return jsonify(actions_list)


# ============================================================
# TREND
# ============================================================

@app.route("/api/trend")
def trend():

    if DATA_ERROR or SALES.empty:
        return jsonify({
            "direction": "flat",
            "change_percent": 0,
            "daily_sales": []
        })

    daily = (
        SALES.groupby("date")["units_sold"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    if len(daily) < 2:

        direction = "flat"
        change_percent = 0

    else:

        first = float(
            daily.iloc[0]["units_sold"]
        )

        last = float(
            daily.iloc[-1]["units_sold"]
        )

        if first == 0:
            change_percent = 0
        else:
            change_percent = (
                (last - first) / first
            ) * 100

        if change_percent > 5:
            direction = "up"
        elif change_percent < -5:
            direction = "down"
        else:
            direction = "flat"

    daily_sales = []

    for _, row in daily.iterrows():

        daily_sales.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "units_sold": int(row["units_sold"])
        })

    return jsonify({
        "direction": direction,
        "change_percent": round(
            change_percent,
            2
        ),
        "daily_sales": daily_sales
    })


# ============================================================
# ANOMALIES
# ============================================================

@app.route("/api/anomalies")
def anomalies():

    if DATA_ERROR or SALES.empty:
        return jsonify([])

    daily = (
        SALES.groupby("date")["units_sold"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    if len(daily) < 3:
        return jsonify([])

    mean = daily["units_sold"].mean()
    std = daily["units_sold"].std()

    if std == 0 or pd.isna(std):
        return jsonify([])

    daily["z_score"] = (
        (daily["units_sold"] - mean)
        / std
    )

    result = []

    for _, row in daily.iterrows():

        z = float(row["z_score"])

        if abs(z) >= 2:

            anomaly_type = (
                "Sales Spike"
                if z > 0
                else "Sales Drop"
            )

            result.append({
                "date": row["date"].strftime(
                    "%Y-%m-%d"
                ),
                "units_sold": int(
                    row["units_sold"]
                ),
                "type": anomaly_type,
                "z_score": round(z, 2)
            })

    return jsonify(result)


# ============================================================
# ASK AI COPILOT
# ============================================================

@app.route("/api/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        question = str(
            data.get("question", "")
        ).strip()

        if not question:

            return jsonify({
                "answer": "Please enter a question.",
                "source": "RetailIQ"
            }), 400

        if len(question) > 1000:

            return jsonify({
                "answer": (
                    "Please keep the question under "
                    "1000 characters."
                ),
                "source": "RetailIQ"
            }), 400

        answer, source = answer_question(
            question
        )

        return jsonify({
            "answer": answer,
            "source": source
        })

    except Exception as e:

        print("Ask endpoint error:", e)

        return jsonify({
            "answer": (
                "RetailIQ could not process this request. "
                "Please try a simpler question."
            ),
            "source": "RetailIQ"
        }), 500


# ============================================================
# API STATUS
# ============================================================

@app.route("/api/status")
def api_status():

    return jsonify({
        "app": "RetailIQ",
        "status": "running",
        "gemini_configured": bool(
            GEMINI_API_KEY
        ),
        "data_loaded": DATA_ERROR is None,
        "model": GEMINI_MODEL
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("        RetailIQ - AI Sales & Inventory Copilot")
    print("=" * 60)

    if DATA_ERROR:
        print("DATA ERROR:", DATA_ERROR)
    else:
        df = get_analysis()

        print(
            f"Products loaded : {len(df)}"
        )

        print(
            f"Units sold      : {int(df['units_sold'].sum())}"
        )

        print(
            f"Total revenue   : ₹{df['revenue'].sum():,.0f}"
        )

    if GEMINI_API_KEY:
        print("Gemini API       : Configured")
    else:
        print(
            "Gemini API       : Not configured "
            "(deterministic mode available)"
        )

    print()
    print("Server: http://localhost:8000")
    print("=" * 60)
    print()

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )