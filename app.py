from flask import Flask, jsonify, render_template, request
import pandas as pd
import os
from google import genai

app = Flask(__name__)

# =========================
# GEMINI API SETUP
# =========================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("WARNING: GEMINI_API_KEY is not set.")
else:
    print("Gemini API Key: FOUND")

client = genai.Client(api_key=API_KEY) if API_KEY else None


# =========================
# DATA FILES
# =========================

PRODUCTS_FILE = "data/products.csv"
SALES_FILE = "data/sales.csv"
INVENTORY_FILE = "data/inventory.csv"


# =========================
# LOAD DATA
# =========================

products = pd.read_csv(PRODUCTS_FILE)
sales = pd.read_csv(SALES_FILE)
inventory = pd.read_csv(INVENTORY_FILE)

products.columns = products.columns.str.strip().str.lower()
sales.columns = sales.columns.str.strip().str.lower()
inventory.columns = inventory.columns.str.strip().str.lower()


# =========================
# HANDLE SALES COLUMN
# =========================

if "quantity" in sales.columns:
    sales["quantity"] = pd.to_numeric(
        sales["quantity"], errors="coerce"
    ).fillna(0)

    sales["units_sold"] = sales["quantity"]

elif "units_sold" in sales.columns:
    sales["units_sold"] = pd.to_numeric(
        sales["units_sold"], errors="coerce"
    ).fillna(0)

else:
    raise ValueError(
        "sales.csv must contain either 'quantity' or 'units_sold' column."
    )


# =========================
# PRODUCT ANALYSIS
# =========================

def get_product_analysis():

    sales_summary = (
        sales.groupby("product_id", as_index=False)
        .agg(units_sold=("units_sold", "sum"))
    )

    analysis = sales_summary.merge(
        products,
        on="product_id",
        how="left"
    )

    analysis = analysis.merge(
        inventory,
        on="product_id",
        how="left"
    )

    analysis["units_sold"] = pd.to_numeric(
        analysis["units_sold"],
        errors="coerce"
    ).fillna(0)

    analysis["price"] = pd.to_numeric(
        analysis["price"],
        errors="coerce"
    ).fillna(0)

    analysis["current_stock"] = pd.to_numeric(
        analysis["current_stock"],
        errors="coerce"
    ).fillna(0)

    analysis["reorder_level"] = pd.to_numeric(
        analysis["reorder_level"],
        errors="coerce"
    ).fillna(0)

    # Revenue
    analysis["revenue"] = (
        analysis["units_sold"] * analysis["price"]
    )

    # Stock status
    analysis["status"] = analysis.apply(
        lambda row:
        "LOW STOCK"
        if row["current_stock"] <= row["reorder_level"]
        else "OK",
        axis=1
    )

    # =========================
    # AVERAGE DAILY SALES
    # =========================

    analysis["avg_daily_sales"] = 0.0

    if "date" in sales.columns:

        sales["date"] = pd.to_datetime(
            sales["date"],
            errors="coerce"
        )

        valid_dates = sales["date"].dropna()

        if len(valid_dates) > 0:

            number_of_days = (
                valid_dates.max() - valid_dates.min()
            ).days + 1

            if number_of_days > 0:

                daily_sales = (
                    sales.groupby("product_id")["units_sold"].sum()
                    / number_of_days
                )

                analysis["avg_daily_sales"] = (
                    analysis["product_id"]
                    .map(daily_sales)
                    .fillna(0)
                )

    # =========================
    # STOCK COVERAGE
    # =========================

    analysis["stock_days"] = analysis.apply(
        lambda row:
        round(
            row["current_stock"] /
            row["avg_daily_sales"],
            1
        )
        if row["avg_daily_sales"] > 0
        else 999,
        axis=1
    )

    # =========================
    # RISK
    # =========================

    def calculate_risk(row):

        if row["current_stock"] <= row["reorder_level"]:
            return "HIGH"

        elif row["stock_days"] <= 7:
            return "MEDIUM"

        else:
            return "LOW"

    analysis["risk"] = analysis.apply(
        calculate_risk,
        axis=1
    )

    return analysis


# =========================
# STORE SUMMARY
# =========================

def get_store_summary():

    analysis = get_product_analysis()

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
            <= inventory["reorder_level"]
        ).sum()
    )

    total_revenue = float(
        analysis["revenue"].sum()
    )

    return {
        "total_products": total_products,
        "total_stock": total_stock,
        "total_units_sold": total_units_sold,
        "low_stock_products": low_stock_products,
        "total_revenue": total_revenue
    }


# =========================
# BUSINESS INSIGHTS
# =========================

def get_business_insights():

    df = get_product_analysis()

    insights = {}

    # Best sellers
    best_sellers = (
        df.sort_values(
            "units_sold",
            ascending=False
        )
        .head(5)
    )

    insights["best_sellers"] = (
        best_sellers[
            [
                "product_name",
                "units_sold",
                "revenue"
            ]
        ]
        .to_dict(orient="records")
    )

    # Highest revenue
    highest_revenue = (
        df.sort_values(
            "revenue",
            ascending=False
        )
        .head(5)
    )

    insights["highest_revenue"] = (
        highest_revenue[
            [
                "product_name",
                "units_sold",
                "revenue"
            ]
        ]
        .to_dict(orient="records")
    )

    # Low stock
    low_stock = df[
        df["current_stock"] <= df["reorder_level"]
    ].copy()

    insights["low_stock"] = (
        low_stock[
            [
                "product_name",
                "current_stock",
                "reorder_level",
                "units_sold",
                "stock_days",
                "risk"
            ]
        ]
        .to_dict(orient="records")
    )

    # Slow moving
    slow_moving = (
        df[df["units_sold"] <= 5]
        .sort_values("units_sold")
    )

    insights["slow_moving"] = (
        slow_moving[
            [
                "product_name",
                "units_sold",
                "current_stock"
            ]
        ]
        .to_dict(orient="records")
    )

    # Overstock
    overstock = df[
        (df["current_stock"] > df["reorder_level"] * 3)
        &
        (df["units_sold"] <= 10)
    ]

    insights["overstock"] = (
        overstock[
            [
                "product_name",
                "current_stock",
                "reorder_level",
                "units_sold"
            ]
        ]
        .to_dict(orient="records")
    )

    # Restock priority
    restock = df[
        df["current_stock"] <= df["reorder_level"]
    ].copy()

    restock["priority_score"] = (
        restock["units_sold"]
        /
        restock["current_stock"].replace(0, 1)
    )

    restock = (
        restock
        .sort_values(
            "priority_score",
            ascending=False
        )
        .head(5)
    )

    insights["restock_priority"] = (
        restock[
            [
                "product_name",
                "current_stock",
                "reorder_level",
                "units_sold",
                "stock_days",
                "priority_score"
            ]
        ]
        .to_dict(orient="records")
    )

    insights["total_revenue"] = float(
        df["revenue"].sum()
    )

    return insights


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# SUMMARY API
# =========================

@app.route("/api/summary")
def summary():

    try:

        return jsonify(
            get_store_summary()
        )

    except Exception as e:

        print("Summary Error:", repr(e))

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# PRODUCTS API
# =========================

@app.route("/api/products")
def product_list():

    try:

        df = get_product_analysis()

        records = df[
            [
                "product_id",
                "product_name",
                "category",
                "price",
                "units_sold",
                "current_stock",
                "revenue",
                "status",
                "avg_daily_sales",
                "stock_days",
                "risk"
            ]
        ].to_dict(orient="records")

        return jsonify(records)

    except Exception as e:

        print("Products Error:", repr(e))

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# INSIGHTS API
# =========================

@app.route("/api/insights")
def insights():

    try:

        return jsonify(
            get_business_insights()
        )

    except Exception as e:

        print("Insights Error:", repr(e))

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# AI ASK API
# =========================

@app.route("/api/ask", methods=["POST"])
def ask_ai():

    data = request.get_json() or {}

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:

        return jsonify({
            "answer": "Please enter a question."
        }), 400

    if client is None:

        return jsonify({
            "answer":
            "Gemini API key is not configured. "
            "Please set GEMINI_API_KEY."
        }), 500

    try:

        analysis = get_product_analysis()

        insights = get_business_insights()

        store_data = analysis.to_string(
            index=False
        )

        insights_data = str(insights)

        prompt = f"""
You are RetailIQ, an AI Sales and Inventory Copilot
for a small retail store.

Your job is to answer the manager's question using ONLY
the store data and calculated business insights provided
below.

STRICT RULES:

1. NEVER invent numbers.
2. NEVER invent products.
3. Use ONLY the provided store data.
4. Every numerical claim must come from the data.
5. If the available data cannot answer the question, say:
"Insufficient data to answer this question."
6. Do not pretend to know information that is not provided.
7. Give practical recommendations when appropriate.
8. Clearly mention the evidence behind your answer.
9. Keep the answer simple for a store manager.
10. Use ₹ for revenue.
11. Do not use unnecessary technical language.
12. Do not claim that an item is out of stock unless
the data shows current_stock = 0.
13. For restocking, prioritize products with low stock,
high sales and fewer stock coverage days.
14. Stock coverage is:
current stock / average daily sales.
15. If the question asks for information that does not
exist in the provided data, clearly say:
Insufficient data to answer this question.

STORE DATA:

{store_data}

CALCULATED BUSINESS INSIGHTS:

{insights_data}

MANAGER QUESTION:

{question}

ANSWER FORMAT:

Give a clear direct answer.

Key Evidence:
- Mention actual product names and numbers.

Recommendation:
- Give a practical action when appropriate.

If there is not enough information, clearly say:
Insufficient data to answer this question.
"""

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        answer = interaction.output_text

        return jsonify({
            "answer": answer,
            "evidence": [
                "data/products.csv",
                "data/sales.csv",
                "data/inventory.csv"
            ]
        })

    except Exception as e:

        print("Gemini Error:", repr(e))

        return jsonify({
            "answer":
            "Gemini could not process the request. "
            "Please check the API connection.",
            "error": str(e)
        }), 500


# =========================
# START SERVER
# =========================

if __name__ == "__main__":

    print("=" * 55)
    print("RetailIQ - AI Sales & Inventory Copilot")
    print("=" * 55)

    print("Products:", len(products))
    print("Sales Records:", len(sales))
    print("Inventory Records:", len(inventory))

    print(
        "Gemini API Key:",
        "FOUND" if API_KEY else "NOT FOUND"
    )

    print("=" * 55)
    print("Starting server...")

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )