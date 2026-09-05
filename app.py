from flask import Flask, jsonify, request, render_template
import pandas as pd
import os
from google import genai


app = Flask(__name__)


# ============================================================
# LOAD DATA
# ============================================================

PRODUCTS_FILE = "data/products.csv"
SALES_FILE = "data/sales.csv"
INVENTORY_FILE = "data/inventory.csv"


def load_data():

    products = pd.read_csv(PRODUCTS_FILE)
    sales = pd.read_csv(SALES_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    # Support either quantity or units_sold
    if "quantity" in sales.columns:
        sales = sales.rename(columns={"quantity": "units_sold"})

    sales["units_sold"] = pd.to_numeric(
        sales["units_sold"],
        errors="coerce"
    ).fillna(0)

    sales["date"] = pd.to_datetime(
        sales["date"],
        errors="coerce"
    )

    products["price"] = pd.to_numeric(
        products["price"],
        errors="coerce"
    ).fillna(0)

    inventory["current_stock"] = pd.to_numeric(
        inventory["current_stock"],
        errors="coerce"
    ).fillna(0)

    inventory["reorder_level"] = pd.to_numeric(
        inventory["reorder_level"],
        errors="coerce"
    ).fillna(0)

    return products, sales, inventory


# ============================================================
# PRODUCT ANALYSIS
# ============================================================

def get_product_analysis():

    products, sales, inventory = load_data()

    # Total sales per product
    sales_summary = (
        sales.groupby("product_id")
        .agg(
            units_sold=("units_sold", "sum")
        )
        .reset_index()
    )

    # Revenue
    sales_summary["revenue"] = 0.0

    merged = products.merge(
        sales_summary,
        on="product_id",
        how="left"
    )

    merged = merged.merge(
        inventory[
            [
                "product_id",
                "current_stock",
                "reorder_level"
            ]
        ],
        on="product_id",
        how="left"
    )

    merged["units_sold"] = (
        merged["units_sold"]
        .fillna(0)
        .astype(float)
    )

    merged["current_stock"] = (
        merged["current_stock"]
        .fillna(0)
        .astype(float)
    )

    merged["reorder_level"] = (
        merged["reorder_level"]
        .fillna(0)
        .astype(float)
    )

    merged["revenue"] = (
        merged["units_sold"] *
        merged["price"]
    )

    # --------------------------------------------------------
    # PRODUCT-SPECIFIC SALES HISTORY
    # --------------------------------------------------------

    product_days = (
        sales.groupby("product_id")["date"]
        .agg(["min", "max"])
        .reset_index()
    )

    product_days["active_days"] = (
        product_days["max"] -
        product_days["min"]
    ).dt.days + 1

    merged = merged.merge(
        product_days[
            [
                "product_id",
                "active_days"
            ]
        ],
        on="product_id",
        how="left"
    )

    merged["active_days"] = (
        merged["active_days"]
        .fillna(1)
        .clip(lower=1)
    )

    # --------------------------------------------------------
    # AVERAGE DAILY SALES
    # --------------------------------------------------------

    merged["avg_daily_sales"] = (
        merged["units_sold"] /
        merged["active_days"]
    )

    # --------------------------------------------------------
    # STOCK COVERAGE
    # --------------------------------------------------------

    merged["stock_days"] = 0.0

    mask = merged["avg_daily_sales"] > 0

    merged.loc[mask, "stock_days"] = (
        merged.loc[mask, "current_stock"] /
        merged.loc[mask, "avg_daily_sales"]
    )

    # Products with no sales history
    merged.loc[
        ~mask,
        "stock_days"
    ] = 9999

    # --------------------------------------------------------
    # STOCK STATUS
    # --------------------------------------------------------

    merged["status"] = merged.apply(
        lambda row:
        "LOW STOCK"
        if row["current_stock"] < row["reorder_level"]
        else "OK",
        axis=1
    )

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    def calculate_risk(row):

        if row["avg_daily_sales"] <= 0:
            return "LOW"

        if row["stock_days"] <= 7:
            return "CRITICAL"

        if row["stock_days"] <= 14:
            return "HIGH"

        if row["stock_days"] <= 30:
            return "MEDIUM"

        return "LOW"

    merged["risk"] = merged.apply(
        calculate_risk,
        axis=1
    )

    # Round values
    merged["avg_daily_sales"] = (
        merged["avg_daily_sales"]
        .round(2)
    )

    merged["stock_days"] = (
        merged["stock_days"]
        .round(1)
    )

    merged["revenue"] = (
        merged["revenue"]
        .round(2)
    )

    return merged


# ============================================================
# STORE SUMMARY
# ============================================================

def get_store_summary():

    products, sales, inventory = load_data()

    total_products = len(products)

    total_stock = int(
        inventory["current_stock"].sum()
    )

    total_units_sold = int(
        sales["units_sold"].sum()
    )

    low_stock_products = int(
        (
            inventory["current_stock"]
            <
            inventory["reorder_level"]
        ).sum()
    )

    product_analysis = get_product_analysis()

    total_revenue = float(
        product_analysis["revenue"].sum()
    )

    return {

        "total_products":
            total_products,

        "total_stock":
            total_stock,

        "total_units_sold":
            total_units_sold,

        "low_stock_products":
            low_stock_products,

        "total_revenue":
            round(total_revenue, 2)

    }


# ============================================================
# ADVANCED BUSINESS INSIGHTS
# ============================================================

def get_advanced_insights():

    df = get_product_analysis()

    # --------------------------------------------------------
    # STOCK-OUT RISK
    # --------------------------------------------------------

    stock_out = df[
        (df["stock_days"] <= 14) &
        (df["avg_daily_sales"] > 0)
    ].copy()

    stock_out = stock_out.sort_values(
        by="stock_days"
    )

    stock_out_result = []

    for _, row in stock_out.iterrows():

        stock_out_result.append({

            "product_id":
                row["product_id"],

            "product_name":
                row["product_name"],

            "current_stock":
                int(row["current_stock"]),

            "avg_daily_sales":
                float(row["avg_daily_sales"]),

            "stock_days":
                float(row["stock_days"]),

            "risk":
                row["risk"]

        })


    # --------------------------------------------------------
    # DEMAND-BASED RESTOCK
    # --------------------------------------------------------

    restock_df = df[
        df["current_stock"] <
        df["reorder_level"]
    ].copy()

    restock_df = restock_df.sort_values(
        by=[
            "risk",
            "stock_days"
        ],
        ascending=[
            True,
            True
        ]
    )

    restock_priority = []

    for _, row in restock_df.iterrows():

        avg_daily = row["avg_daily_sales"]

        current_stock = row["current_stock"]

        reorder_level = row["reorder_level"]

        # Target approximately 30 days of demand
        target_stock = avg_daily * 30

        # Never recommend below reorder level
        target_stock = max(
            target_stock,
            reorder_level
        )

        suggested_reorder = max(
            int(round(target_stock - current_stock)),
            0
        )

        restock_priority.append({

            "product_id":
                row["product_id"],

            "product_name":
                row["product_name"],

            "current_stock":
                int(current_stock),

            "reorder_level":
                int(reorder_level),

            "avg_daily_sales":
                float(avg_daily),

            "stock_days":
                float(row["stock_days"]),

            "suggested_reorder":
                suggested_reorder,

            "units_sold":
                int(row["units_sold"])

        })


    # --------------------------------------------------------
    # SLOW MOVING
    # --------------------------------------------------------

    slow = df[
        df["units_sold"] <= 10
    ].copy()

    slow = slow.sort_values(
        by="units_sold"
    )

    slow_result = []

    for _, row in slow.iterrows():

        slow_result.append({

            "product_id":
                row["product_id"],

            "product_name":
                row["product_name"],

            "units_sold":
                int(row["units_sold"]),

            "current_stock":
                int(row["current_stock"]),

            "revenue":
                float(row["revenue"])

        })


    # --------------------------------------------------------
    # OVERSTOCK
    # --------------------------------------------------------

    overstock = df[
        (df["stock_days"] > 90) &
        (
            df["current_stock"] >
            df["reorder_level"]
        )
    ].copy()

    overstock = overstock.sort_values(
        by="stock_days",
        ascending=False
    )

    overstock_result = []

    for _, row in overstock.iterrows():

        overstock_result.append({

            "product_id":
                row["product_id"],

            "product_name":
                row["product_name"],

            "current_stock":
                int(row["current_stock"]),

            "stock_days":
                float(row["stock_days"]),

            "units_sold":
                int(row["units_sold"])

        })


    # --------------------------------------------------------
    # TOP SELLERS
    # --------------------------------------------------------

    top_sellers = (
        df.sort_values(
            by="units_sold",
            ascending=False
        )
        .head(5)
    )

    top_sellers_result = []

    for _, row in top_sellers.iterrows():

        top_sellers_result.append({

            "product_id":
                row["product_id"],

            "product_name":
                row["product_name"],

            "units_sold":
                int(row["units_sold"]),

            "revenue":
                float(row["revenue"])

        })


    # --------------------------------------------------------
    # REVENUE LEADERS
    # --------------------------------------------------------

    revenue_leaders = (
        df.sort_values(
            by="revenue",
            ascending=False
        )
        .head(5)
    )

    revenue_result = []

    for _, row in revenue_leaders.iterrows():

        revenue_result.append({

            "product_id":
                row["product_id"],

            "product_name":
                row["product_name"],

            "revenue":
                float(row["revenue"]),

            "units_sold":
                int(row["units_sold"])

        })


    return {

        "stock_out_risk":
            stock_out_result,

        "restock_priority":
            restock_priority,

        "slow_moving":
            slow_result,

        "overstock":
            overstock_result,

        "top_sellers":
            top_sellers_result,

        "revenue_leaders":
            revenue_result

    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# SUMMARY API
# ============================================================

@app.route("/api/summary")
def summary():

    return jsonify(
        get_store_summary()
    )


# ============================================================
# PRODUCTS API
# ============================================================

@app.route("/api/products")
def products_api():

    df = get_product_analysis()

    records = []

    for _, row in df.iterrows():

        records.append({

            "product_id":
                row["product_id"],

            "product_name":
                row["product_name"],

            "category":
                row["category"],

            "price":
                float(row["price"]),

            "units_sold":
                int(row["units_sold"]),

            "current_stock":
                int(row["current_stock"]),

            "reorder_level":
                int(row["reorder_level"]),

            "revenue":
                float(row["revenue"]),

            "status":
                row["status"],

            "risk":
                row["risk"],

            "avg_daily_sales":
                float(row["avg_daily_sales"]),

            "stock_days":
                float(row["stock_days"])

        })

    return jsonify(records)


# ============================================================
# INSIGHTS API
# ============================================================

@app.route("/api/insights")
def insights_api():

    return jsonify(
        get_advanced_insights()
    )


# ============================================================
# GEMINI AI
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

client = None

if GEMINI_API_KEY:

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


@app.route("/api/ask", methods=["POST"])
def ask_ai():

    data = request.get_json(
        silent=True
    ) or {}

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:

        return jsonify({

            "answer":
                "Please enter a question."

        })


    if client is None:

        return jsonify({

            "answer":
                "Gemini API key is not configured."

        })


    summary_data = (
        get_store_summary()
    )

    product_data = (
        get_product_analysis()
    )

    insights = (
        get_advanced_insights()
    )


    product_records = []

    for _, row in product_data.iterrows():

        product_records.append({

            "product":
                row["product_name"],

            "category":
                row["category"],

            "price":
                float(row["price"]),

            "units_sold":
                int(row["units_sold"]),

            "stock":
                int(row["current_stock"]),

            "reorder_level":
                int(row["reorder_level"]),

            "revenue":
                float(row["revenue"]),

            "avg_daily_sales":
                float(row["avg_daily_sales"]),

            "stock_days":
                float(row["stock_days"]),

            "status":
                row["status"],

            "risk":
                row["risk"]

        })


    prompt = f"""
You are RetailIQ, an AI Sales and Inventory Copilot.

Answer the store manager's question using ONLY the supplied
store data.

IMPORTANT RULES:

1. Never invent numbers.
2. Never invent products.
3. Never claim future sales unless future data exists.
4. If the data cannot answer the question, clearly say:
   "Insufficient data to answer this question."
5. Every numerical business claim must be supported by the data.
6. Clearly distinguish historical facts from recommendations.
7. Recommendations must be presented as recommendations,
   not guaranteed outcomes.
8. Keep the answer concise and useful for a store manager.
9. When useful, explain the calculation or evidence.
10. If recommending restocking, use the deterministic
    suggested_reorder value provided by the system.

STORE SUMMARY:

{summary_data}


PRODUCT DATA:

{product_records}


BUSINESS INSIGHTS:

{insights}


MANAGER QUESTION:

{question}


Provide a clear business-focused answer.
"""


    try:

        response = client.interactions.create(

            model="gemini-3.6-flash",

            input=prompt

        )

        answer = response.output_text

        return jsonify({

            "answer":
                answer

        })


    except Exception as e:

        print("Gemini Error:", e)

        return jsonify({

            "answer":
                "Gemini could not process the request. "
                "Please check the API connection."

        })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print("        RetailIQ Starting...")
    print("======================================")
    print("")

    app.run(

        host="0.0.0.0",

        port=8000,

        debug=True

    )