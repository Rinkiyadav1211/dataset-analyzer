"""
Dataset Analyzer - A Streamlit prototype for exploring CSV datasets.

Run with:  streamlit run app.py

Built with NumPy, Pandas and Matplotlib.
"""

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Dataset Analyzer", layout="wide")

# ----------------------------------------------------------------------
# Small built-in sample dataset so the app can be tested without a file
# ----------------------------------------------------------------------
def get_sample_dataset() -> pd.DataFrame:
    data = {
        "StudentID": list(range(1, 16)),
        "Name": [
            "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
            "Sai", "Reyansh", "Krishna", "Ishaan", "Rohan",
            "Ananya", "Diya", "Isha", "Kavya", "Meera",
        ],
        "Branch": [
            "CSE", "ECE", "CSE", "ME", "CSE",
            "ECE", "CSE", "ME", "CSE", "ECE",
            "CSE", "ME", "CSE", "ECE", "CSE",
        ],
        "Attendance": [85, 92, 78, 65, 88, 74, 95, 60, 82, 90, 77, 68, 91, 84, 73],
        "Marks": [78, 85, 62, 55, 91, 70, 88, 45, 76, 82, 69, 58, 94, 80, 66],
        "Study_Hours": [4, 5, 3, 2, 6, 3, 6, 1, 4, 5, 3, 2, 6, 4, 3],
        "Placed": [
            "Yes", "Yes", "No", "No", "Yes",
            "No", "Yes", "No", "Yes", "Yes",
            "No", "No", "Yes", "Yes", "No",
        ],
    }
    df = pd.DataFrame(data)
    # Sprinkle a couple of missing values so the missing-value features
    # have something real to show, just like a messy real-world CSV.
    df.loc[3, "Marks"] = np.nan
    df.loc[10, "Attendance"] = np.nan
    return df


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None


st.title("📊 Dataset Analyzer")
st.caption("Upload any CSV file and instantly get an overview, statistics, "
           "data operations and visualizations. Built with NumPy, Pandas and Matplotlib.")

# ----------------------------------------------------------------------
# 1. Upload Dataset
# ----------------------------------------------------------------------
st.header("1️⃣ Upload Dataset")

col_a, col_b = st.columns([2, 1])

with col_a:
    uploaded_file = st.file_uploader("Upload a CSV file", type=None)

with col_b:
    st.write("")
    st.write("")
    use_sample = st.button("Use Sample Dataset")

if use_sample:
    st.session_state.df = get_sample_dataset()
    st.success("Sample dataset loaded.")

if uploaded_file is not None:
    if not uploaded_file.name.lower().endswith(".csv"):
        st.error("❌ Invalid file type. Please upload a file with a .csv extension.")
    else:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"Loaded '{uploaded_file.name}' successfully.")
        except Exception as e:
            st.error(f"❌ Could not read this CSV file. Details: {e}")

df = st.session_state.df

if df is None:
    st.info("Upload a CSV file or click 'Use Sample Dataset' to get started.")
    st.stop()

# Helpful column-type split used throughout the app
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# ----------------------------------------------------------------------
# 2. Dataset Overview
# ----------------------------------------------------------------------
st.header("2️⃣ Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing values (total)", int(df.isnull().sum().sum()))
c4.metric("Duplicate rows", int(df.duplicated().sum()))

with st.expander("Column names & data types", expanded=True):
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
    })
    st.dataframe(dtype_df, use_container_width=True)

with st.expander("First 5 rows"):
    st.dataframe(df.head(), use_container_width=True)

with st.expander("Last 5 rows"):
    st.dataframe(df.tail(), use_container_width=True)

with st.expander("Missing values per column"):
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    st.dataframe(missing_df, use_container_width=True)

st.write(f"**Duplicate rows found:** {int(df.duplicated().sum())}")

# ----------------------------------------------------------------------
# 3. Statistical Analysis
# ----------------------------------------------------------------------
st.header("3️⃣ Statistical Analysis (Numerical Columns)")

if not numeric_cols:
    st.warning("No numerical columns found in this dataset, so statistics can't be computed.")
else:
    stats_rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        values = series.to_numpy()  # NumPy array for calculations
        if len(values) == 0:
            continue
        stats_rows.append({
            "Column": col,
            "Mean": np.mean(values),
            "Median": np.median(values),
            "Min": np.min(values),
            "Max": np.max(values),
            "Std Dev": np.std(values),
            "Variance": np.var(values),
            "Sum": np.sum(values),
        })
    stats_df = pd.DataFrame(stats_rows).round(2)
    st.dataframe(stats_df, use_container_width=True)
    st.caption("Pandas selects and cleans each column; NumPy computes the actual statistics "
               "(mean, median, min, max, std, variance, sum).")

# ----------------------------------------------------------------------
# 4. Data Operations
# ----------------------------------------------------------------------
st.header("4️⃣ Data Operations")

op_tabs = st.tabs(["Select Column", "Filter", "Sort", "Group By", "Value Counts", "Correlation"])

with op_tabs[0]:
    col_select = st.selectbox("Choose a column to view", df.columns, key="select_col")
    st.dataframe(df[[col_select]], use_container_width=True)

with op_tabs[1]:
    filter_col = st.selectbox("Column to filter on", df.columns, key="filter_col")
    if filter_col in numeric_cols:
        min_v, max_v = float(df[filter_col].min()), float(df[filter_col].max())
        rng = st.slider("Keep rows where value is between:", min_v, max_v, (min_v, max_v))
        filtered = df[df[filter_col].between(rng[0], rng[1])]
    else:
        options = df[filter_col].dropna().unique().tolist()
        chosen = st.multiselect("Keep rows where value is one of:", options, default=options)
        filtered = df[df[filter_col].isin(chosen)] if chosen else df.iloc[0:0]
    st.write(f"Showing {len(filtered)} of {len(df)} rows.")
    st.dataframe(filtered, use_container_width=True)

with op_tabs[2]:
    sort_col = st.selectbox("Sort by column", df.columns, key="sort_col")
    ascending = st.radio("Order", ["Ascending", "Descending"], horizontal=True) == "Ascending"
    st.dataframe(df.sort_values(by=sort_col, ascending=ascending), use_container_width=True)

with op_tabs[3]:
    if not categorical_cols:
        st.info("No categorical columns available to group by.")
    elif not numeric_cols:
        st.info("No numerical columns available to aggregate.")
    else:
        group_col = st.selectbox("Group by column", categorical_cols, key="group_col")
        agg_col = st.selectbox("Numerical column to aggregate", numeric_cols, key="agg_col")
        agg_func = st.selectbox("Aggregation", ["mean", "sum", "count", "min", "max"], key="agg_func")
        grouped = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
        st.dataframe(grouped, use_container_width=True)

with op_tabs[4]:
    vc_col = st.selectbox("Column for value counts", df.columns, key="vc_col")
    st.dataframe(df[vc_col].value_counts().reset_index(), use_container_width=True)

with op_tabs[5]:
    if len(numeric_cols) < 2:
        st.info("Need at least two numerical columns to compute correlations.")
    else:
        corr = df[numeric_cols].corr()
        st.dataframe(corr.round(2), use_container_width=True)

# ----------------------------------------------------------------------
# 5. Visualizations
# ----------------------------------------------------------------------
st.header("5️⃣ Visualization")

chart_type = st.selectbox(
    "Choose a chart type",
    ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Plot", "Histogram", "Box Plot"],
)

fig, ax = plt.subplots(figsize=(7, 4))
chart_ok = True
error_msg = ""

try:
    if chart_type == "Line Chart":
        if not numeric_cols:
            chart_ok, error_msg = False, "No numerical column available for a line chart."
        else:
            y_col = st.selectbox("Column to plot", numeric_cols, key="line_col")
            ax.plot(df[y_col].values)
            ax.set_title(f"Line Chart: {y_col}")
            ax.set_xlabel("Row Index")
            ax.set_ylabel(y_col)

    elif chart_type == "Bar Chart":
        if not categorical_cols:
            chart_ok, error_msg = False, "No categorical column available for a bar chart."
        else:
            bar_col = st.selectbox("Categorical column", categorical_cols, key="bar_col")
            counts = df[bar_col].value_counts()
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_title(f"Bar Chart: {bar_col}")
            ax.set_xlabel(bar_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")

    elif chart_type == "Pie Chart":
        if not categorical_cols:
            chart_ok, error_msg = False, "No categorical column available for a pie chart."
        else:
            pie_col = st.selectbox("Categorical column", categorical_cols, key="pie_col")
            counts = df[pie_col].value_counts()
            if len(counts) > 10:
                chart_ok, error_msg = False, "Too many unique categories for a readable pie chart (limit: 10)."
            else:
                ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
                ax.set_title(f"Pie Chart: {pie_col}")

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            chart_ok, error_msg = False, "Need at least two numerical columns for a scatter plot."
        else:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y",
                                  index=min(1, len(numeric_cols) - 1))
            ax.scatter(df[x_col], df[y_col])
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

    elif chart_type == "Histogram":
        if not numeric_cols:
            chart_ok, error_msg = False, "No numerical column available for a histogram."
        else:
            hist_col = st.selectbox("Column", numeric_cols, key="hist_col")
            ax.hist(df[hist_col].dropna().values, bins=15, edgecolor="black")
            ax.set_title(f"Histogram: {hist_col}")
            ax.set_xlabel(hist_col)
            ax.set_ylabel("Frequency")

    elif chart_type == "Box Plot":
        if not numeric_cols:
            chart_ok, error_msg = False, "No numerical column available for a box plot."
        else:
            box_col = st.selectbox("Column", numeric_cols, key="box_col")
            ax.boxplot(df[box_col].dropna().values, vert=True)
            ax.set_title(f"Box Plot: {box_col}")
            ax.set_ylabel(box_col)

    if chart_ok:
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning(f"⚠️ Could not generate this chart. {error_msg}")

except Exception as e:
    st.warning(f"⚠️ Could not generate this chart. Details: {e}")

plt.close(fig)

# ----------------------------------------------------------------------
# 6. Automatic Insights
# ----------------------------------------------------------------------
st.header("6️⃣ Automatic Insights")

insights = []

# Basic shape summary
insights.append(f"The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns** "
                 f"({len(numeric_cols)} numerical, {len(categorical_cols)} categorical).")

# Missing values
total_missing = int(df.isnull().sum().sum())
if total_missing > 0:
    worst_missing_col = df.isnull().sum().idxmax()
    worst_missing_count = int(df.isnull().sum().max())
    insights.append(f"Column **'{worst_missing_col}'** has the most missing values "
                     f"({worst_missing_count} missing).")
else:
    insights.append("There are no missing values in this dataset.")

# Duplicates
dup_count = int(df.duplicated().sum())
if dup_count > 0:
    insights.append(f"There are **{dup_count} duplicate rows** in the dataset.")

# Highest average numerical column
if numeric_cols:
    means = df[numeric_cols].mean(numeric_only=True)
    top_mean_col = means.idxmax()
    insights.append(f"Column **'{top_mean_col}'** has the highest average value "
                     f"({means.max():.2f}).")

    overall_max_col = df[numeric_cols].max().idxmax()
    overall_max_val = df[numeric_cols].max().max()
    overall_min_col = df[numeric_cols].min().idxmin()
    overall_min_val = df[numeric_cols].min().min()
    insights.append(f"The overall maximum value is **{overall_max_val}** in column '{overall_max_col}', "
                     f"and the overall minimum value is **{overall_min_val}** in column '{overall_min_col}'.")

# Strong correlations
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr().abs()
    max_corr_val = corr.values.max()
    if max_corr_val >= 0.7:
        idx = np.unravel_index(np.argmax(corr.values), corr.shape)
        col1, col2 = corr.index[idx[0]], corr.columns[idx[1]]
        insights.append(f"Columns **'{col1}'** and **'{col2}'** show a strong correlation "
                         f"({max_corr_val:.2f}).")
    else:
        insights.append("No strong correlations (≥ 0.7) were found between numerical columns.")

for point in insights:
    st.markdown(f"- {point}")
