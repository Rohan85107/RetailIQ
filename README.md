# RetailIQ — AI Sales & Inventory Copilot

RetailIQ is an AI-powered Sales and Inventory Copilot designed for small retail businesses.

It helps store managers understand sales performance, monitor inventory, identify stock-out risks, detect slow-moving products, and make data-driven restocking decisions using natural-language questions.

---

## 🚀 Problem Statement

Small retail businesses often manage sales and inventory manually.

This creates several problems:

- Difficulty identifying products that may run out of stock
- Overstocking of slow-moving products
- Difficulty understanding sales performance
- Manual calculation of revenue and sales trends
- Delayed restocking decisions
- Lack of simple data-driven recommendations

RetailIQ solves these problems by combining deterministic data analysis with Generative AI.

---

## 💡 Our Solution

RetailIQ provides a simple AI-powered dashboard where a store manager can ask questions in natural language.

For example:

- What is the total revenue?
- Which product has the highest revenue?
- Which products are at risk of stock-out?
- What should I order?
- Which products are slow sellers?
- Give me a store summary.

The system analyzes the actual store data and provides answers with numbers and evidence.

---

## ✨ Key Features

### 📊 Sales Analysis

RetailIQ calculates:

- Total revenue
- Total units sold
- Product-wise revenue
- Product-wise sales
- Highest revenue products
- Best-selling products

### 📦 Inventory Monitoring

The system monitors:

- Current stock
- Average daily sales
- Estimated stock coverage
- Stock-out risk
- Low-stock products
- Critical-stock products

### 🔄 Smart Restocking

RetailIQ calculates suggested reorder quantities using:

- Average daily sales
- Current inventory
- Expected 14-day demand
- Minimum reorder level

### 🐌 Slow-Moving Product Detection

The system identifies products with:

- Low sales
- High inventory
- Low revenue contribution

This helps store managers avoid unnecessary overstocking.

### 🤖 AI Copilot

Store managers can ask questions using normal language.

The AI provides answers based on the available store data.

### 🛡️ No-Guessing Principle

RetailIQ does not invent information.

If the available data is insufficient to answer a question, the system clearly tells the user that the data is not enough.

---

## 🏗️ System Architecture

```text
Store Manager
      ↓
Ask Question
      ↓
RetailIQ Web Application
      ↓
Python Flask Backend
      ↓
Smart Question Router
      ↓
 ┌───────────────────────┐
 │                       │
 ↓                       ↓
Deterministic Logic    Gemini AI
 │                       │
 ↓                       ↓
Sales & Inventory      Explanation
Analysis              & Recommendation
 │                       │
 └───────────┬───────────┘
             ↓
      Evidence + Numbers
             ↓
       Final Answer