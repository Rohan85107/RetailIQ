TRACK_ID=PS3

# RetailIQ - AI Sales & Inventory Copilot

## Overview

RetailIQ is an AI-powered Sales and Inventory Copilot designed for small retail businesses.

It helps store managers understand sales performance, monitor inventory, identify stock-out risks, detect slow-moving products, and make data-driven restocking decisions using natural-language questions.

RetailIQ combines deterministic Python analytics with Gemini AI to provide accurate, explainable, and actionable retail insights.

---

## Problem Statement

Small retailers often manage their sales and inventory using spreadsheets or basic billing systems.

Because of this, it can be difficult to quickly answer important business questions such as:

- Which products may run out of stock?
- Which products are selling the most?
- Which product generated the highest revenue?
- What products should be reordered?
- Which products are slow-moving?
- How is the store performing?
- Are there unusual sales patterns?

RetailIQ solves this problem by providing an AI-powered Copilot that converts natural-language questions into data-backed business insights.

---

## Solution

RetailIQ provides a simple web-based dashboard where a store manager can ask questions in natural language.

For example:

> Which products are at risk of stock-out?

The system:

1. Receives the user's question.
2. Understands the question.
3. Checks the available retail data.
4. Uses deterministic Python calculations whenever possible.
5. Uses Gemini AI for complex reasoning when required.
6. Generates an evidence-based response.
7. Provides actual numbers and recommendations.
8. Refuses to guess when the available data is insufficient.

---

## Key Features

### 1. Natural Language AI Copilot

Users can ask questions using normal language.

Examples:

- What is the total revenue?
- Which product has the highest revenue?
- Which products are at risk of stock-out?
- What should I reorder?
- Which products are slow sellers?
- Give me a store summary.

---

### 2. Sales Analytics

RetailIQ calculates:

- Total units sold
- Product-wise sales
- Product-wise revenue
- Highest revenue product
- Best-selling products
- Sales trends
- Sales anomalies

---

### 3. Inventory Monitoring

The system monitors:

- Current stock
- Average daily sales
- Estimated stock coverage
- Stock-out risk
- Critical inventory levels
- Restocking requirements

---

### 4. Stock-Out Risk Detection

Products are classified according to estimated stock coverage.

| Risk Level | Stock Coverage |
|------------|----------------|
| CRITICAL | <= 7 days |
| HIGH | <= 14 days |
| MEDIUM | <= 30 days |
| LOW | > 30 days |

This helps store managers identify products that may run out soon.

---

### 5. Restocking Recommendations

RetailIQ estimates expected demand for the next 14 days.

Formula:

```text
Expected 14-Day Demand
= Average Daily Sales × 14