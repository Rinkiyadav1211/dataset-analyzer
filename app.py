"""
E-Commerce Sales Data Analyzer & Comparison Dashboard
========================================================================

A college-level data analytics dashboard built with Streamlit, Pandas,
NumPy and Matplotlib. Upload ANY CSV (not just e-commerce data) and get
automatic profiling, cleaning, statistics, category comparisons, product
and regional analysis, correlation, ranking, time analysis, visualization
and automatic business insights.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="E-Commerce Sales Data Analyzer", layout="wide", page_icon="🛒")


# ============================================================================
# COLUMN-NAME CANDIDATES (used to smart-default to E-Commerce columns when
# present, while still letting any other CSV work through manual selection)
# ============================================================================

CATEGORY_CANDIDATES = ["Category", "Product_Category", "category"]
PRODUCT_CANDIDATES = ["Product", "product", "Item", "Item_Name"]
CITY_CANDIDATES = ["City", "city"]
REGION_CANDIDATES = ["Region", "region"]
CUSTOMER_TYPE_CANDIDATES = ["Customer_Type", "customer_type"]
PAYMENT_CANDIDATES = ["Payment_Method", "payment_method"]
SALES_CANDIDATES = ["Sales", "sales", "Revenue", "Total_Sales"]
PROFIT_CANDIDATES = ["Profit", "profit"]
QUANTITY_CANDIDATES = ["Quantity", "quantity", "Qty"]
DISCOUNT_CANDIDATES = ["Discount", "discount"]
RATING_CANDIDATES = ["Rating", "rating"]
ORDER_ID_CANDIDATES = ["Order_ID", "OrderID", "order_id"]


def find_column(columns, candidates):
    """Return the first candidate name that exists in `columns`, else None."""
    col_set = set(columns)
