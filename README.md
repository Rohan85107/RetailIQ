TRACK_ID=PS3

RetailIQ — AI Sales & Inventory Copilot

RetailIQ is an AI-powered Sales and Inventory Copilot designed for small retail stores. It analyzes product catalogue, sales, and inventory data to answer natural-language business questions and provide data-grounded recommendations.

Problem Statement

Small retail store managers often need to manually analyze sales and inventory data to answer questions such as:

Which products are selling the most?
Which products are at risk of stock-out?
Which products are slow-moving?
What products should be restocked?
What is the total revenue?
Which products generate the highest revenue?

RetailIQ provides these answers using actual store data instead of guessing.

Key Features
Natural-language AI Copilot
Total revenue analysis
Product sales analysis
Highest-revenue product identification
Stock-out risk detection
Low-stock identification
Slow-moving product detection
Overstock detection
Restocking recommendations
14-day demand estimation
Sales trend analysis
Sales anomaly detection
Business health score
Data-backed recommendations
No-guessing behavior when the available data is insufficient
Technology Stack
Backend
Python 3.11
Flask
Pandas
AI
Google Gemini API
Gemini 3.6 Flash
Google GenAI Python SDK
Frontend
HTML
CSS
JavaScript
Data
CSV files
Local deterministic data analysis
Project Structure
RetailIQ/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── data/
│   ├── products.csv
│   ├── sales.csv
│   └── inventory.csv
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
How It Works
Store Manager
      ↓
Ask Natural-Language Question
      ↓
RetailIQ Flask Backend
      ↓
Deterministic Data Analysis
      ↓
Gemini AI Reasoning (when required)
      ↓
Evidence + Actual Numbers
      ↓
Answer + Recommendation

RetailIQ separates deterministic calculations from AI reasoning.

Python performs calculations such as:

Revenue
Units sold
Average daily sales
Stock coverage
Stock-out risk
Restocking quantity
Slow-moving products
Sales trends
Anomalies

Gemini is used to explain results and answer more complex questions using the available store evidence.

Restocking Recommendation Logic

RetailIQ estimates the quantity that should be reordered using historical sales and current inventory.

Formula
Estimated 14-Day Demand = Average Daily Sales × 14

Recommended Order =
max(Estimated 14-Day Demand, Minimum Reorder Level)
− Current Stock

The recommendation is never presented as a guaranteed future prediction.

Current Restocking Recommendations
Product	Recommended Order
USB-C Cable	94 units
Backpack	66 units
Webcam	64 units
Wireless Mouse	62 units
Desk Lamp	52 units
Mechanical Keyboard	13 units
Assumptions
Recommendations use estimated 14-day demand.
Minimum reorder level is 10 units.
Historical sales are used to estimate daily demand.
Recommendations are based only on the available store dataset.
Future demand is an estimate, not a guaranteed prediction.
No-Guessing Principle

RetailIQ does not invent information when the available data cannot support an answer.

For example, if the user asks:

Which product will have the highest sales next month?

The system does not claim to know the future. It explains that the available historical dataset is insufficient to make a reliable prediction.

This helps keep AI responses grounded in available evidence.

Example Questions

Users can ask:

What is the total revenue?

Which product has the highest revenue?

Which products are running out of stock?

What should I order?

Which products are slow sellers?

Which products are overstocked?

Give me a store summary.

Which category generates the most revenue?
Example Output

For:

What is the total revenue?

RetailIQ can return:

The total store revenue is ₹199,230 from 170 units sold across 10 products.

For:

What should I order?

RetailIQ provides prioritized restocking recommendations based on current stock and estimated demand.

API Endpoints
Endpoint	Purpose
/	RetailIQ dashboard
/api/health	Business health information
/api/summary	Store summary
/api/products	Product analysis
/api/insights	Inventory and sales insights
/api/actions	Recommended manager actions
/api/trend	Sales trend information
/api/anomalies	Sales anomalies
/api/ask	AI Copilot questions
/api/status	Application status
Installation
Requirements
Python 3.11
Internet connection for Gemini API
Gemini API key
Install Dependencies

Open a terminal in the RetailIQ folder and run:

pip install -r requirements.txt
Environment Setup

Create a .env file in the project root:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY

Do not commit the .env file or expose the API key publicly.

Run the Application

Run:

python app.py

The application starts at:

http://localhost:8000

Open the address in a browser.

No second terminal or separate frontend build is required.

Data Sources

RetailIQ currently uses local CSV datasets:

products.csv — product catalogue and prices
sales.csv — historical sales transactions
inventory.csv — current inventory levels

The system combines these datasets to produce evidence-backed insights.

Engineering Approach

RetailIQ follows a hybrid architecture:

Deterministic Layer

Python handles numerical calculations and business rules.

This provides:

Reproducible calculations
Reliable inventory metrics
Consistent recommendations
Reduced hallucination risk
AI Layer

Gemini is used when natural-language reasoning or explanation is required.

The AI receives relevant store evidence and is instructed not to invent unsupported information.

Fallback

If Gemini is unavailable or temporarily rate-limited, RetailIQ continues to provide deterministic answers for supported business questions.

Limitations
The current prototype uses a sample retail dataset.
Restocking quantities are estimates based on historical sales.
No supplier lead-time data is currently included.
No real-time POS integration is included.
Future sales cannot be guaranteed from historical data alone.
Actual automatic ordering would require integration with a supplier or purchasing system.
Future Improvements
Real-time POS integration
Supplier management
Supplier lead-time analysis
Automatic purchase-order generation
Multi-store inventory support
Advanced demand forecasting
Seasonal demand analysis
User authentication
Database integration
WhatsApp/SMS alerts for critical stock
Cloud deployment
Hackathon Demo Flow
1. Open RetailIQ Dashboard
        ↓
2. Show total products, stock and units sold
        ↓
3. Ask: "What is the total revenue?"
        ↓
4. Ask: "Which product has the highest revenue?"
        ↓
5. Ask: "Which products are running out of stock?"
        ↓
6. Ask: "What should I order?"
        ↓
7. Show recommended restocking quantities
        ↓
8. Explain the 14-day demand assumption
        ↓
9. Demonstrate the no-guessing behavior
Conclusion

RetailIQ transforms raw retail sales and inventory data into actionable business insights through a combination of deterministic Python analytics and grounded Gemini AI reasoning.

The system helps store managers identify sales performance, stock-out risks, slow-moving products, and restocking priorities while clearly communicating assumptions and avoiding unsupported predictions.

Team / Hackathon

Project: RetailIQ — AI Sales & Inventory Copilot

Track: PS3 — Retail: Sales and Inventory Copilot

Built for the NexusTIQ Hackathon.