import os
import re
import pandas as pd

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PRODUCTS_FILE = os.path.join(DATA_DIR, "products.csv")
SALES_FILE = os.path.join(DATA_DIR, "sales.csv")
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.csv")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    products = pd.read_csv(PRODUCTS_FILE)
    sales = pd.read_csv(SALES_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    products.columns = products.columns.str.strip().str.lower()
    sales.columns = sales.columns.str.strip().str.lower()
    inventory.columns = inventory.columns.str.strip().str.lower()

    # Sales column compatibility
    if "quantity" in sales.columns:
        sales["units_sold"] = pd.to_numeric(
            sales["quantity"], errors="coerce"
        ).fillna(0)
    elif "units_sold" in sales.columns:
        sales["units_sold"] = pd.to_numeric(
            sales["units_sold"], errors="coerce"
        ).fillna(0)
    else:
        sales["units_sold"] = 0

    sales["date"] = pd.to_datetime(
        sales["date"], errors="coerce"
    )

    products["price"] = pd.to_numeric(
        products["price"], errors="coerce"
    ).fillna(0)

    inventory["current_stock"] = pd.to_numeric(
        inventory["current_stock"], errors="coerce"
    ).fillna(0)

    return products, sales, inventory


products_df, sales_df, inventory_df = load_data()


# ============================================================
# PRODUCT ANALYSIS
# ============================================================

def get_product_analysis():

    products = products_df.copy()
    sales = sales_df.copy()
    inventory = inventory_df.copy()

    sales_summary = (
        sales.groupby("product_id")["units_sold"]
        .sum()
        .reset_index()
    )

    merged = products.merge(
        sales_summary,
        on="product_id",
        how="left"
    )

    merged = merged.merge(
        inventory[["product_id", "current_stock"]],
        on="product_id",
        how="left"
    )

    merged["units_sold"] = merged["units_sold"].fillna(0)
    merged["current_stock"] = merged["current_stock"].fillna(0)

    merged["revenue"] = (
        merged["price"] * merged["units_sold"]
    )

    # --------------------------------------------------------
    # Sales activity period
    # --------------------------------------------------------

    date_stats = (
        sales.groupby("product_id")["date"]
        .agg(["min", "max"])
        .reset_index()
    )

    merged = merged.merge(
        date_stats,
        on="product_id",
        how="left"
    )

    merged["active_days"] = (
        merged["max"] - merged["min"]
    ).dt.days + 1

    merged["active_days"] = (
        merged["active_days"]
        .fillna(1)
        .clip(lower=1)
    )

    # --------------------------------------------------------
    # Average daily sales
    # --------------------------------------------------------

    merged["avg_daily_sales"] = (
        merged["units_sold"] /
        merged["active_days"]
    )

    # --------------------------------------------------------
    # Stock coverage
    # --------------------------------------------------------

    merged["stock_days"] = merged.apply(
        lambda row:
        row["current_stock"] / row["avg_daily_sales"]
        if row["avg_daily_sales"] > 0
        else 9999,
        axis=1
    )

    # --------------------------------------------------------
    # Stock risk
    # --------------------------------------------------------

    def calculate_risk(days):

        if days <= 7:
            return "CRITICAL"

        elif days <= 14:
            return "HIGH"

        elif days <= 30:
            return "MEDIUM"

        return "LOW"

    merged["risk"] = merged["stock_days"].apply(
        calculate_risk
    )

    merged["status"] = merged["risk"].apply(
        lambda x:
        "LOW STOCK"
        if x in ["CRITICAL", "HIGH"]
        else "OK"
    )

    # --------------------------------------------------------
    # Reorder calculation
    # --------------------------------------------------------

    reorder_level = 10

    merged["expected_14_day_demand"] = (
        merged["avg_daily_sales"] * 14
    ).round().astype(int)

    merged["suggested_reorder"] = (
        merged["expected_14_day_demand"]
        .clip(lower=reorder_level)
        - merged["current_stock"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # Slow moving
    # --------------------------------------------------------

    merged["slow_moving"] = (
        (merged["units_sold"] < 10) &
        (merged["current_stock"] >= 50)
    )

    # --------------------------------------------------------
    # Overstock
    # --------------------------------------------------------

    merged["overstock"] = (
        (merged["stock_days"] > 90) &
        (merged["units_sold"] < 10)
    )

    return merged


# ============================================================
# QUESTION NORMALIZATION
# ============================================================

def normalize_question(question):

    q = str(question).lower().strip()

    q = re.sub(r"[-_]+", " ", q)

    q = re.sub(r"[?!.,;:]+", " ", q)

    q = re.sub(r"\s+", " ", q)

    return q


def contains_any(text, phrases):

    return any(
        phrase in text
        for phrase in phrases
    )


# ============================================================
# DETERMINISTIC AI
# ============================================================

def deterministic_answer(question):

    q = normalize_question(question)

    df = get_product_analysis()

    # ========================================================
    # NO GUESSING
    # ========================================================

    future_words = [
        "next month",
        "next week",
        "future",
        "will sell",
        "will be",
        "predict",
        "prediction",
        "forecast",
        "tomorrow",
        "upcoming"
    ]

    if contains_any(q, future_words):

        return (
            "I cannot reliably predict future sales from the "
            "available historical store data. "
            "The current dataset contains past sales and "
            "inventory information, but it does not provide "
            "enough evidence for a reliable future prediction."
        )

    # ========================================================
    # TOTAL REVENUE
    # ========================================================

    if contains_any(q, [
        "total revenue",
        "overall revenue",
        "store revenue",
        "revenue of store",
        "total sales revenue"
    ]):

        total_revenue = df["revenue"].sum()
        total_units = df["units_sold"].sum()
        total_products = len(df)

        return (
            f"The total store revenue is "
            f"₹{total_revenue:,.0f} from "
            f"{total_units:,.0f} units sold across "
            f"{total_products} products."
        )

    # ========================================================
    # TOTAL STOCK
    # ========================================================

    if contains_any(q, [
        "total stock",
        "current stock",
        "how much stock",
        "stock available",
        "inventory available",
        "total inventory"
    ]):

        stock = df["current_stock"].sum()

        return (
            f"The store currently has "
            f"{stock:,.0f} units in stock across "
            f"{len(df)} products."
        )

    # ========================================================
    # TOTAL UNITS SOLD
    # ========================================================

    if contains_any(q, [
        "total units sold",
        "total sales",
        "units sold",
        "how many sold",
        "how much sold"
    ]):

        units = df["units_sold"].sum()

        return (
            f"The store sold "
            f"{units:,.0f} units in the available sales data."
        )

    # ========================================================
    # HIGHEST REVENUE
    # ========================================================

    if contains_any(q, [
        "highest revenue",
        "most revenue",
        "top revenue",
        "revenue leader",
        "best revenue"
    ]):

        row = df.loc[
            df["revenue"].idxmax()
        ]

        return (
            f"{row['product_name']} has the highest revenue "
            f"at ₹{row['revenue']:,.0f}, "
            f"generated from {row['units_sold']:,.0f} units sold."
        )

    # ========================================================
    # BEST SELLER
    # ========================================================

    if contains_any(q, [
        "best seller",
        "best selling",
        "top seller",
        "most sold",
        "highest sales",
        "highest units"
    ]):

        row = df.loc[
            df["units_sold"].idxmax()
        ]

        return (
            f"{row['product_name']} is the top seller with "
            f"{row['units_sold']:,.0f} units sold, "
            f"generating ₹{row['revenue']:,.0f} revenue."
        )

    # ========================================================
    # STOCK OUT / LOW STOCK
    # ========================================================

    if contains_any(q, [
        "stock out",
        "stockout",
        "run out",
        "running out",
        "low stock",
        "stock risk",
        "products at risk",
        "risk of stock"
    ]):

        risk_df = df[
            df["risk"].isin([
                "CRITICAL",
                "HIGH"
            ])
        ].sort_values(
            "stock_days"
        )

        if risk_df.empty:

            return (
                "No products are currently classified "
                "as high or critical stock risk."
            )

        lines = [
            "Products at risk of stock-out:"
        ]

        for _, row in risk_df.iterrows():

            lines.append(
                f"• {row['product_name']}: "
                f"{row['current_stock']:.0f} units left, "
                f"approximately {row['stock_days']:.1f} days "
                f"of stock coverage ({row['risk']})."
            )

        return "\n".join(lines)

    # ========================================================
    # RESTOCK
    # ========================================================

    if contains_any(q, [
        "restock",
        "reorder",
        "what should i order",
        "what to order",
        "what should we order",
        "which products should i order",
        "which products to order"
    ]):

        restock_df = df[
            df["suggested_reorder"] > 0
        ].sort_values(
            "suggested_reorder",
            ascending=False
        )

        if restock_df.empty:

            return (
                "No immediate restocking is recommended "
                "from the current data."
            )

        lines = [
            "Recommended restocking quantities:"
        ]

        for _, row in restock_df.iterrows():

            lines.append(
                f"• {row['product_name']}: "
                f"order approximately "
                f"{row['suggested_reorder']:.0f} units."
            )

        lines.append(
            "\nAssumption: recommendations use estimated "
            "14-day demand and a minimum reorder level of "
            "10 units."
        )

        return "\n".join(lines)

    # ========================================================
    # SLOW MOVING
    # ========================================================

    if contains_any(q, [
        "slow moving",
        "slow moving products",
        "slow sellers",
        "slow selling",
        "least selling",
        "poor selling"
    ]):

        slow_df = df[
            df["slow_moving"]
        ].sort_values(
            "units_sold"
        )

        if slow_df.empty:

            return (
                "No products currently meet the "
                "slow-moving criteria."
            )

        lines = [
            "Slow-moving products:"
        ]

        for _, row in slow_df.iterrows():

            lines.append(
                f"• {row['product_name']}: "
                f"{row['units_sold']:.0f} sold, "
                f"{row['current_stock']:.0f} in stock, "
                f"₹{row['revenue']:,.0f} revenue."
            )

        return "\n".join(lines)

    # ========================================================
    # OVERSTOCK
    # ========================================================

    if contains_any(q, [
        "overstock",
        "over stocked",
        "excess stock",
        "too much stock",
        "excess inventory"
    ]):

        over_df = df[
            df["overstock"]
        ].sort_values(
            "stock_days",
            ascending=False
        )

        if over_df.empty:

            return (
                "No products currently meet the "
                "overstock criteria."
            )

        lines = [
            "Potentially overstocked products:"
        ]

        for _, row in over_df.iterrows():

            lines.append(
                f"• {row['product_name']}: "
                f"{row['current_stock']:.0f} units in stock, "
                f"approximately {row['stock_days']:.0f} days "
                f"of stock coverage."
            )

        return "\n".join(lines)

    # ========================================================
    # CATEGORY PERFORMANCE
    # ========================================================

    if contains_any(q, [
        "category",
        "categories",
        "category performance"
    ]):

        category = (
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

        lines = [
            "Category performance:"
        ]

        for name, row in category.iterrows():

            lines.append(
                f"• {name}: "
                f"{row['units_sold']:.0f} units sold, "
                f"₹{row['revenue']:,.0f} revenue."
            )

        return "\n".join(lines)

    # ========================================================
    # STORE SUMMARY
    # ========================================================

    if contains_any(q, [
        "store summary",
        "overall summary",
        "business summary",
        "give me summary",
        "summarize the store",
        "store performance"
    ]):

        total_revenue = df["revenue"].sum()
        total_units = df["units_sold"].sum()
        total_stock = df["current_stock"].sum()

        low_stock = len(
            df[
                df["risk"].isin([
                    "CRITICAL",
                    "HIGH"
                ])
            ]
        )

        slow = len(
            df[df["slow_moving"]]
        )

        return (
            "Store Summary\n\n"
            f"• Products: {len(df)}\n"
            f"• Units sold: {total_units:.0f}\n"
            f"• Current stock: {total_stock:.0f}\n"
            f"• Revenue: ₹{total_revenue:,.0f}\n"
            f"• Low/High-risk stock products: {low_stock}\n"
            f"• Slow-moving products: {slow}"
        )

    return None


# ============================================================
# GEMINI
# ============================================================

def ask_gemini(question):

    if not GEMINI_API_KEY:

        return None

    try:

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        df = get_product_analysis()

        records = []

        for _, row in df.iterrows():

            records.append({
                "product": row["product_name"],
                "category": row["category"],
                "price": float(row["price"]),
                "units_sold": float(row["units_sold"]),
                "current_stock": float(row["current_stock"]),
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
You are RetailIQ, a retail sales and inventory copilot.

Answer the manager's question using ONLY the provided
store data.

IMPORTANT RULES:

1. Never invent numbers.
2. Never invent products.
3. Use actual numbers from the data.
4. Explain the evidence behind recommendations.
5. Clearly mention assumptions.
6. If the data cannot answer the question, say that
   the available data is insufficient.
7. Keep the response concise and useful for a store manager.

STORE DATA:

{records}

MANAGER QUESTION:

{question}
"""

        response = client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt
        )

        answer = getattr(
            response,
            "output_text",
            None
        )

        if answer:

            return answer.strip()

    except Exception as e:

        print("Gemini error:", e)

        return None

    return None


# ============================================================
# MAIN QUESTION HANDLER
# ============================================================

def answer_question(question):

    deterministic = deterministic_answer(
        question
    )

    if deterministic:

        return deterministic

    gemini_answer = ask_gemini(
        question
    )

    if gemini_answer:

        return gemini_answer

    return (
        "I could not reliably answer that question from "
        "the available store data.\n\n"
        "Try questions such as:\n"
        "• What is the total revenue?\n"
        "• Which product has the highest revenue?\n"
        "• Which products are running out of stock?\n"
        "• What should I order?\n"
        "• Which products are slow sellers?\n"
        "• Give me a store summary."
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    df = get_product_analysis()

    low_stock = len(
        df[
            df["risk"].isin([
                "CRITICAL",
                "HIGH"
            ])
        ]
    )

    critical = len(
        df[
            df["risk"] == "CRITICAL"
        ]
    )

    slow = len(
        df[
            df["slow_moving"]
        ]
    )

    score = 100

    score -= low_stock * 8
    score -= critical * 5
    score -= slow * 3

    score = max(
        0,
        min(100, score)
    )

    if score >= 80:

        status = "Healthy"

    elif score >= 60:

        status = "Watch"

    elif score >= 40:

        status = "Warning"

    else:

        status = "Critical"

    return jsonify({
        "score": score,
        "status": status,
        "low_stock": low_stock,
        "critical": critical,
        "slow_moving": slow
    })


# ============================================================
# SUMMARY
# ============================================================

@app.route("/api/summary")
def summary():

    df = get_product_analysis()

    low_stock = len(
        df[
            df["risk"].isin([
                "CRITICAL",
                "HIGH"
            ])
        ]
    )

    return jsonify({
        "total_products": int(len(df)),
        "total_stock": int(
            df["current_stock"].sum()
        ),
        "total_units_sold": int(
            df["units_sold"].sum()
        ),
        "total_revenue": float(
            df["revenue"].sum()
        ),
        "low_stock_products": int(
            low_stock
        )
    })


# ============================================================
# PRODUCTS
# ============================================================

@app.route("/api/products")
def products():

    df = get_product_analysis()

    result = []

    for _, row in df.iterrows():

        result.append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "price": float(row["price"]),
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
            "status": row["status"],
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

    df = get_product_analysis()

    stock_risk = df[
        df["risk"].isin([
            "CRITICAL",
            "HIGH"
        ])
    ].sort_values(
        "stock_days"
    )

    restock = df[
        df["suggested_reorder"] > 0
    ].sort_values(
        "suggested_reorder",
        ascending=False
    )

    slow = df[
        df["slow_moving"]
    ].sort_values(
        "units_sold"
    )

    top_sellers = df.sort_values(
        "units_sold",
        ascending=False
    )

    revenue_leaders = df.sort_values(
        "revenue",
        ascending=False
    )

    return jsonify({

        "stock_out_risk": [
            {
                "product_name": row["product_name"],
                "current_stock": int(
                    row["current_stock"]
                ),
                "stock_days": round(
                    float(row["stock_days"]), 1
                ),
                "avg_daily_sales": round(
                    float(row["avg_daily_sales"]), 2
                ),
                "risk": row["risk"]
            }

            for _, row in stock_risk.iterrows()
        ],

        "restock_priority": [
            {
                "product_name": row["product_name"],
                "suggested_reorder": int(
                    row["suggested_reorder"]
                )
            }

            for _, row in restock.iterrows()
        ],

        "slow_moving": [
            {
                "product_name": row["product_name"],
                "units_sold": int(
                    row["units_sold"]
                ),
                "current_stock": int(
                    row["current_stock"]
                ),
                "revenue": float(
                    row["revenue"]
                )
            }

            for _, row in slow.iterrows()
        ],

        "overstock": [
            {
                "product_name": row["product_name"],
                "current_stock": int(
                    row["current_stock"]
                ),
                "stock_days": round(
                    float(row["stock_days"]), 1
                )
            }

            for _, row in df[
                df["overstock"]
            ].iterrows()
        ],

        "top_sellers": [
            {
                "product_name": row["product_name"],
                "units_sold": int(
                    row["units_sold"]
                ),
                "revenue": float(
                    row["revenue"]
                )
            }

            for _, row in top_sellers.iterrows()
        ],

        "revenue_leaders": [
            {
                "product_name": row["product_name"],
                "revenue": float(
                    row["revenue"]
                ),
                "units_sold": int(
                    row["units_sold"]
                )
            }

            for _, row in revenue_leaders.iterrows()
        ]
    })


# ============================================================
# ACTIONS
# ============================================================

@app.route("/api/actions")
def actions():

    df = get_product_analysis()

    priority_df = df[
        df["suggested_reorder"] > 0
    ].sort_values(
        "suggested_reorder",
        ascending=False
    )

    actions_list = []

    for _, row in priority_df.iterrows():

        if row["risk"] == "CRITICAL":

            priority = "URGENT"

        elif row["risk"] == "HIGH":

            priority = "HIGH"

        else:

            priority = "MEDIUM"

        actions_list.append({

            "priority": priority,

            "title": (
                f"Restock {row['product_name']}"
            ),

            "reason": (
                f"Current stock is "
                f"{row['current_stock']:.0f} units "
                f"with approximately "
                f"{row['stock_days']:.1f} days "
                f"of coverage."
            ),

            "action": (
                f"Order approximately "
                f"{row['suggested_reorder']:.0f} units."
            )
        })

    return jsonify(actions_list)


# ============================================================
# SALES TREND
# ============================================================

@app.route("/api/trend")
def trend():

    sales = sales_df.copy()

    daily = (
        sales.groupby("date")["units_sold"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    daily["date"] = daily["date"].dt.strftime(
        "%Y-%m-%d"
    )

    values = daily["units_sold"].tolist()

    change_percent = 0

    if len(values) >= 2:

        first = values[0]
        last = values[-1]

        if first != 0:

            change_percent = (
                (last - first) / first
            ) * 100

    if change_percent > 5:

        direction = "up"

    elif change_percent < -5:

        direction = "down"

    else:

        direction = "stable"

    return jsonify({

        "direction": direction,

        "change_percent": round(
            change_percent,
            1
        ),

        "daily_sales": [
            {
                "date": row["date"],
                "units_sold": int(
                    row["units_sold"]
                )
            }

            for _, row in daily.iterrows()
        ]
    })


# ============================================================
# ANOMALIES
# ============================================================

@app.route("/api/anomalies")
def anomalies():

    sales = sales_df.copy()

    daily = (
        sales.groupby("date")["units_sold"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    if len(daily) < 2:

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

                "z_score": round(
                    z,
                    2
                )
            })

    return jsonify(result)


# ============================================================
# AI ASK
# ============================================================

@app.route("/api/ask", methods=["POST"])
def ask():

    data = request.get_json(
        silent=True
    ) or {}

    question = str(
        data.get("question", "")
    ).strip()

    if not question:

        return jsonify({
            "answer": (
                "Please enter a question."
            )
        }), 400

    answer = answer_question(
        question
    )

    return jsonify({
        "question": question,
        "answer": answer
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/api/status")
def status():

    return jsonify({

        "app": "RetailIQ",

        "status": "running",

        "gemini_configured": bool(
            GEMINI_API_KEY
        ),

        "model": GEMINI_MODEL
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RetailIQ - AI Sales & Inventory Copilot")
    print("=" * 60)
    print("Server: http://localhost:8000")
    print("Gemini configured:", bool(GEMINI_API_KEY))
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )