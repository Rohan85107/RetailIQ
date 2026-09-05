from flask import Flask, jsonify, request, render_template
import pandas as pd
import os
from google import genai

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PRODUCTS_FILE = os.path.join(DATA_DIR, "products.csv")
SALES_FILE = os.path.join(DATA_DIR, "sales.csv")
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.csv")

# Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini API: Connected")
    except Exception as e:
        print("Gemini connection error:", e)
else:
    print("WARNING: GEMINI_API_KEY not found")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    products = pd.read_csv(PRODUCTS_FILE)
    sales = pd.read_csv(SALES_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    # Support both quantity and units_sold
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

    # Numeric conversions
    products["price"] = pd.to_numeric(
        products["price"], errors="coerce"
    ).fillna(0)

    inventory["current_stock"] = pd.to_numeric(
        inventory["current_stock"], errors="coerce"
    ).fillna(0)

    inventory["reorder_level"] = pd.to_numeric(
        inventory["reorder_level"], errors="coerce"
    ).fillna(0)

    # Date conversion
    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(
            sales["date"], errors="coerce"
        )

    return products, sales, inventory


# ============================================================
# PRODUCT ANALYSIS
# ============================================================

def get_product_analysis():

    products, sales, inventory = load_data()

    # Sales summary by product
    sales_summary = (
        sales.groupby("product_id")["units_sold"]
        .sum()
        .reset_index()
    )

    sales_summary.rename(
        columns={"units_sold": "units_sold"},
        inplace=True
    )

    # Merge all data
    df = products.merge(
        sales_summary,
        on="product_id",
        how="left"
    )

    df = df.merge(
        inventory,
        on="product_id",
        how="left"
    )

    df["units_sold"] = df["units_sold"].fillna(0)
    df["current_stock"] = df["current_stock"].fillna(0)
    df["reorder_level"] = df["reorder_level"].fillna(0)

    # Revenue
    df["revenue"] = df["units_sold"] * df["price"]

    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    if not sales.empty and sales["date"].notna().any():

        min_date = sales["date"].min()
        max_date = sales["date"].max()

        total_days = (max_date - min_date).days + 1

        if total_days <= 0:
            total_days = 1

    else:
        total_days = 1

    # --------------------------------------------------------
    # AVERAGE DAILY SALES
    # --------------------------------------------------------

    df["avg_daily_sales"] = (
        df["units_sold"] / total_days
    )

    # --------------------------------------------------------
    # STOCK COVERAGE
    # --------------------------------------------------------

    df["stock_days"] = df.apply(
        lambda row:
        row["current_stock"] / row["avg_daily_sales"]
        if row["avg_daily_sales"] > 0
        else 9999,
        axis=1
    )

    # --------------------------------------------------------
    # STOCK STATUS
    # --------------------------------------------------------

    df["status"] = df.apply(
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

        if row["current_stock"] < row["reorder_level"]:
            return "HIGH"

        if row["stock_days"] <= 14:
            return "MEDIUM"

        return "LOW"

    df["risk"] = df.apply(
        calculate_risk,
        axis=1
    )

    return df


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
            < inventory["reorder_level"]
        ).sum()
    )

    total_revenue = 0

    if not products.empty and not sales.empty:

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

        merged["units_sold"] = (
            merged["units_sold"]
            .fillna(0)
        )

        merged["revenue"] = (
            merged["units_sold"]
            * merged["price"]
        )

        total_revenue = float(
            merged["revenue"].sum()
        )

    return {
        "total_products": total_products,
        "total_stock": total_stock,
        "total_units_sold": total_units_sold,
        "low_stock_products": low_stock_products,
        "total_revenue": total_revenue
    }


# ============================================================
# ADVANCED INTELLIGENCE
# ============================================================

def get_advanced_insights():

    df = get_product_analysis()

    insights = {
        "stock_out_risk": [],
        "slow_moving": [],
        "overstock": [],
        "top_sellers": [],
        "revenue_leaders": [],
        "restock_priority": []
    }

    # --------------------------------------------------------
    # 1. STOCK-OUT RISK
    # --------------------------------------------------------

    risk_df = df[
        df["stock_days"] <= 14
    ].copy()

    risk_df = risk_df.sort_values(
        by="stock_days"
    )

    for _, row in risk_df.iterrows():

        insights["stock_out_risk"].append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "current_stock": int(row["current_stock"]),
            "avg_daily_sales": round(
                float(row["avg_daily_sales"]), 2
            ),
            "stock_days": round(
                float(row["stock_days"]), 1
            ),
            "risk": row["risk"]
        })

    # --------------------------------------------------------
    # 2. SLOW MOVING PRODUCTS
    # --------------------------------------------------------

    slow_df = df[
        df["units_sold"] <= 10
    ].copy()

    slow_df = slow_df.sort_values(
        by="units_sold"
    )

    for _, row in slow_df.iterrows():

        insights["slow_moving"].append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "units_sold": int(row["units_sold"]),
            "current_stock": int(row["current_stock"]),
            "revenue": round(
                float(row["revenue"]), 2
            )
        })

    # --------------------------------------------------------
    # 3. OVERSTOCK
    # --------------------------------------------------------

    overstock_df = df[
        (df["stock_days"] > 90)
        & (df["current_stock"] > df["reorder_level"])
    ].copy()

    overstock_df = overstock_df.sort_values(
        by="stock_days",
        ascending=False
    )

    for _, row in overstock_df.iterrows():

        insights["overstock"].append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "current_stock": int(row["current_stock"]),
            "units_sold": int(row["units_sold"]),
            "stock_days": round(
                float(row["stock_days"]), 1
            )
        })

    # --------------------------------------------------------
    # 4. TOP SELLERS
    # --------------------------------------------------------

    top_sellers_df = df.sort_values(
        by="units_sold",
        ascending=False
    ).head(5)

    for _, row in top_sellers_df.iterrows():

        insights["top_sellers"].append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "units_sold": int(row["units_sold"]),
            "revenue": round(
                float(row["revenue"]), 2
            )
        })

    # --------------------------------------------------------
    # 5. REVENUE LEADERS
    # --------------------------------------------------------

    revenue_df = df.sort_values(
        by="revenue",
        ascending=False
    ).head(5)

    for _, row in revenue_df.iterrows():

        insights["revenue_leaders"].append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "revenue": round(
                float(row["revenue"]), 2
            ),
            "units_sold": int(row["units_sold"])
        })

    # --------------------------------------------------------
    # 6. RESTOCK PRIORITY
    # --------------------------------------------------------

    restock_df = df[
        df["current_stock"] < df["reorder_level"]
    ].copy()

    # Highest sales first
    restock_df = restock_df.sort_values(
        by=["units_sold", "stock_days"],
        ascending=[False, True]
    )

    for _, row in restock_df.iterrows():

        suggested_reorder = max(
            int(row["reorder_level"] * 2)
            - int(row["current_stock"]),
            0
        )

        insights["restock_priority"].append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "current_stock": int(row["current_stock"]),
            "reorder_level": int(row["reorder_level"]),
            "units_sold": int(row["units_sold"]),
            "stock_days": round(
                float(row["stock_days"]), 1
            ),
            "suggested_reorder": suggested_reorder
        })

    return insights


# ============================================================
# HOME PAGE
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
def api_summary():

    try:

        summary = get_store_summary()

        return jsonify(summary)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# PRODUCTS API
# ============================================================

@app.route("/api/products")
def api_products():

    try:

        df = get_product_analysis()

        products = []

        for _, row in df.iterrows():

            products.append({

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

                "avg_daily_sales":
                    round(
                        float(row["avg_daily_sales"]),
                        2
                    ),

                "stock_days":
                    round(
                        float(row["stock_days"]),
                        1
                    ),

                "risk":
                    row["risk"],

                "status":
                    row["status"]
            })

        return jsonify(products)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# ADVANCED INSIGHTS API
# ============================================================

@app.route("/api/insights")
def api_insights():

    try:

        insights = get_advanced_insights()

        return jsonify(insights)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# AI QUESTION API
# ============================================================

@app.route("/api/ask", methods=["POST"])
def api_ask():

    try:

        data = request.get_json()

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "answer":
                    "Please enter a question."
            })

        # ----------------------------------------------------
        # Check Gemini connection
        # ----------------------------------------------------

        if client is None:

            return jsonify({
                "answer":
                    "Gemini API is not connected. "
                    "Please check GEMINI_API_KEY."
            })

        # ----------------------------------------------------
        # Get actual store data
        # ----------------------------------------------------

        df = get_product_analysis()

        summary = get_store_summary()

        insights = get_advanced_insights()

        # ----------------------------------------------------
        # Convert data to compact text
        # ----------------------------------------------------

        product_records = []

        for _, row in df.iterrows():

            product_records.append({

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

                "avg_daily_sales":
                    round(
                        float(row["avg_daily_sales"]),
                        2
                    ),

                "stock_days":
                    round(
                        float(row["stock_days"]),
                        1
                    ),

                "risk":
                    row["risk"],

                "status":
                    row["status"]
            })

        # ----------------------------------------------------
        # AI PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are RetailIQ, an AI Sales and Inventory Copilot
for a small retail business.

IMPORTANT RULES:

1. Use ONLY the provided store data.
2. NEVER invent numbers.
3. NEVER create future sales predictions unless the data
   explicitly supports them.
4. If the question cannot be answered from the data,
   clearly say:
   "Insufficient data to answer this question."
5. Every important numerical claim must come from the
   provided data.
6. Prefer concise, useful answers for a store manager.
7. Give practical recommendations only when supported
   by the data.
8. Clearly distinguish facts from recommendations.
9. Do not pretend that a recommendation is historical fact.

STORE SUMMARY:

{summary}

PRODUCT DATA:

{product_records}

ADVANCED INSIGHTS:

{insights}

MANAGER QUESTION:

{question}

Answer the manager's question.

Use a clear structure when useful:

### Answer
### Evidence
### Recommendation

Do not use information outside the provided data.
"""

        # ----------------------------------------------------
        # GEMINI INTERACTIONS API
        # ----------------------------------------------------

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        answer = response.output_text

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print(
            "AI ERROR:",
            repr(e)
        )

        return jsonify({
            "answer":
                "Gemini could not process the request.\n\n"
                "Please check the API connection.",
            "error": str(e)
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RetailIQ - AI Sales & Inventory Copilot")
    print("=" * 60)

    print("Server starting...")
    print("Open: http://localhost:8000")

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )