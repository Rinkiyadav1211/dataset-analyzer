import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Dataset Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Analyzer")
st.write(
    "Upload any CSV file to perform statistical analysis, "
    "data operations, comparisons and visualizations."
)


# ================= SAMPLE DATA =================

def sample_data():
    return pd.DataFrame({
        "StudentID": range(1, 16),
        "Name": [
            "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
            "Sai", "Reyansh", "Krishna", "Ishaan", "Rohan",
            "Ananya", "Diya", "Isha", "Kavya", "Meera"
        ],
        "Year": [1, 1, 2, 2, 3, 3, 4, 4, 3, 2, 1, 4, 3, 2, 4],
        "Department": [
            "CSE", "CSE", "AI", "CSE", "AI",
            "CSE", "AI", "CSE", "AI", "CSE",
            "AI", "CSE", "AI", "CSE", "AI"
        ],
        "Attendance": [
            88, 76, 91, 65, 82, 95, 72, 86,
            79, 68, 90, np.nan, 84, 73, 89
        ],
        "Marks": [
            82, 71, 94, 62, 78, 96, 69, 88,
            75, 66, 92, 81, 85, 70, 91
        ],
        "Projects": [3, 2, 5, 1, 3, 6, 2, 4, 3, 1, 5, 4, 4, 2, 5]
    })


# ================= UPLOAD =================

st.header("1️⃣ Upload Dataset")

file = st.file_uploader("Upload a CSV file", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
        st.success("CSV file loaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    if st.button("Use Sample Dataset"):
        st.session_state.use_sample = True

    if st.session_state.get("use_sample", False):
        df = sample_data()
        st.info("Sample dataset loaded.")
    else:
        st.info("Upload a CSV file or use the sample dataset.")
        st.stop()

df = df.reset_index(drop=True)


# ================= OVERVIEW =================

st.header("2️⃣ Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("Missing Values", int(df.isna().sum().sum()))
c4.metric("Duplicate Rows", int(df.duplicated().sum()))

st.subheader("Column Information")

info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isna().sum(),
    "Unique Values": df.nunique()
})

st.dataframe(info, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("First 5 Rows")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.subheader("Last 5 Rows")
    st.dataframe(df.tail(), use_container_width=True)

st.subheader("Missing Values Per Column")
st.dataframe(
    df.isna().sum().rename("Missing Values"),
    use_container_width=True
)


# ================= STATISTICS =================

st.header("3️⃣ Statistical Analysis")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

if num_cols:

    statistics = pd.DataFrame({
        "Mean": [np.nanmean(df[c]) for c in num_cols],
        "Median": [np.nanmedian(df[c]) for c in num_cols],
        "Minimum": [np.nanmin(df[c]) for c in num_cols],
        "Maximum": [np.nanmax(df[c]) for c in num_cols],
        "Std Dev": [np.nanstd(df[c]) for c in num_cols],
        "Variance": [np.nanvar(df[c]) for c in num_cols],
        "Sum": [np.nansum(df[c]) for c in num_cols]
    }, index=num_cols)

    st.dataframe(
        statistics.round(2),
        use_container_width=True
    )
else:
    st.warning("No numerical columns found.")


# ================= DATA OPERATIONS =================

st.header("4️⃣ Data Operations")

selected_col = st.selectbox(
    "Select Column",
    df.columns
)

op1, op2 = st.columns(2)

with op1:

    st.subheader("Filter")

    if pd.api.types.is_numeric_dtype(df[selected_col]):

        minimum = float(df[selected_col].min())
        maximum = float(df[selected_col].max())

        if minimum != maximum:

            values = st.slider(
                "Select range",
                minimum,
                maximum,
                (minimum, maximum)
            )

            filtered = df[
                df[selected_col].between(
                    values[0],
                    values[1]
                )
            ]

            st.dataframe(
                filtered,
                use_container_width=True
            )

    else:

        values = df[selected_col].dropna().unique().tolist()

        choice = st.selectbox(
            "Select value",
            ["All"] + values
        )

        if choice == "All":
            filtered = df
        else:
            filtered = df[
                df[selected_col] == choice
            ]

        st.dataframe(
            filtered,
            use_container_width=True
        )


with op2:

    st.subheader("Sort")

    order = st.radio(
        "Order",
        ["Ascending", "Descending"],
        horizontal=True
    )

    sorted_df = df.sort_values(
        selected_col,
        ascending=(order == "Ascending")
    )

    st.dataframe(
        sorted_df,
        use_container_width=True
    )


st.subheader("Value Counts")

st.dataframe(
    df[selected_col].value_counts(dropna=False)
    .rename("Count"),
    use_container_width=True
)


# ================= GROUP BY =================

if cat_cols and num_cols:

    st.subheader("Group By Comparison")

    group_col = st.selectbox(
        "Group By",
        cat_cols
    )

    value_col = st.selectbox(
        "Numeric Value",
        num_cols
    )

    grouped = df.groupby(group_col)[value_col].agg(
        ["count", "mean", "min", "max", "sum"]
    ).reset_index()

    grouped.columns = [
        group_col,
        "Count",
        "Average",
        "Minimum",
        "Maximum",
        "Total"
    ]

    st.dataframe(
        grouped.round(2),
        use_container_width=True
    )


# ================= COMPARISON =================

st.header("5️⃣ 📊 Comparison Analysis")

st.write(
    "This section compares different categories, years "
    "and numerical columns."
)

if num_cols:

    comparison_type = st.radio(
        "Choose comparison type",
        [
            "Category-wise Comparison",
            "Two Numerical Columns"
        ],
        horizontal=True
    )

    # CATEGORY COMPARISON

    if comparison_type == "Category-wise Comparison":

        if cat_cols:

            category = st.selectbox(
                "Select Category",
                cat_cols
            )

            metric = st.selectbox(
                "Select Metric",
                num_cols
            )

            comparison = df.groupby(category)[metric].agg(
                ["count", "mean", "median", "min", "max"]
            ).reset_index()

            comparison.columns = [
                category,
                "Count",
                "Average",
                "Median",
                "Minimum",
                "Maximum"
            ]

            st.subheader("Comparison Table")

            st.dataframe(
                comparison.round(2),
                use_container_width=True
            )

            st.subheader(
                f"Average {metric} by {category}"
            )

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.bar(
                comparison[category].astype(str),
                comparison["Average"]
            )

            ax.set_xlabel(category)
            ax.set_ylabel(f"Average {metric}")
            ax.set_title(
                f"{metric} Comparison by {category}"
            )

            plt.xticks(rotation=30)
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.warning(
                "A categorical column is required."
            )

    # TWO COLUMN COMPARISON

    else:

        if len(num_cols) >= 2:

            x_col = st.selectbox(
                "First Numerical Column",
                num_cols
            )

            y_col = st.selectbox(
                "Second Numerical Column",
                [c for c in num_cols if c != x_col]
            )

            comparison_df = df[
                [x_col, y_col]
            ].dropna()

            st.subheader(
                f"{x_col} vs {y_col}"
            )

            st.dataframe(
                comparison_df.describe().round(2),
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.scatter(
                comparison_df[x_col],
                comparison_df[y_col]
            )

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

            ax.set_title(
                f"{x_col} vs {y_col}"
            )

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.warning(
                "At least two numerical columns are required."
            )


# ================= YEAR ANALYSIS =================

st.header("6️⃣ 📅 Year-wise / Category-wise Analysis")

year_cols = [
    c for c in df.columns
    if any(
        word in str(c).lower()
        for word in ["year", "class", "semester"]
    )
]

if year_cols and num_cols:

    year_col = st.selectbox(
        "Select Year/Class/Semester Column",
        year_cols
    )

    metric = st.selectbox(
        "Select Performance Metric",
        num_cols,
        key="year_metric"
    )

    year_analysis = df.groupby(
        year_col
    )[metric].agg(
        ["count", "mean", "max", "min"]
    ).reset_index()

    year_analysis.columns = [
        year_col,
        "Students",
        "Average",
        "Maximum",
        "Minimum"
    ]

    st.subheader("Year-wise Comparison Table")

    st.dataframe(
        year_analysis.round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        year_analysis[year_col].astype(str),
        year_analysis["Average"],
        marker="o"
    )

    ax.set_xlabel(year_col)
    ax.set_ylabel(f"Average {metric}")

    ax.set_title(
        f"{metric} - Year-wise Comparison"
    )

    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

else:

    st.info(
        "No Year/Class/Semester column found. "
        "Use Category-wise Comparison above."
    )


# ================= CORRELATION =================

st.header("7️⃣ 🔗 Correlation Analysis")

if len(num_cols) >= 2:

    correlation = df[num_cols].corr()

    st.dataframe(
        correlation.round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.imshow(
        correlation.values,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(correlation.columns))
    )

    ax.set_yticks(
        range(len(correlation.columns))
    )

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation.columns
    )

    ax.set_title("Correlation Matrix")

    fig.colorbar(image, ax=ax)

    st.pyplot(fig)
    plt.close(fig)

else:

    st.info(
        "At least two numerical columns are required."
    )


# ================= VISUALIZATION =================

st.header("8️⃣ 📈 Visualization")

if num_cols:

    chart = st.selectbox(
        "Choose Chart Type",
        [
            "Bar Chart",
            "Line Chart",
            "Histogram",
            "Scatter Plot",
            "Box Plot"
        ]
    )

    column = st.selectbox(
        "Select Column",
        num_cols
    )

    if chart == "Bar Chart":

        values = df[column].dropna()

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.bar(
            range(len(values)),
            values
        )

        ax.set_xlabel("Row")
        ax.set_ylabel(column)
        ax.set_title(
            f"{column} - Bar Chart"
        )

        st.pyplot(fig)
        plt.close(fig)

    elif chart == "Line Chart":

        values = df[column].dropna()

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            values,
            marker="o"
        )

        ax.set_xlabel("Row")
        ax.set_ylabel(column)
        ax.set_title(
            f"{column} - Line Chart"
        )

        st.pyplot(fig)
        plt.close(fig)

    elif chart == "Histogram":

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.hist(
            df[column].dropna(),
            bins=10
        )

        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        ax.set_title(
            f"{column} - Distribution"
        )

        st.pyplot(fig)
        plt.close(fig)

    elif chart == "Box Plot":

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.boxplot(
            df[column].dropna()
        )

        ax.set_ylabel(column)

        ax.set_title(
            f"{column} - Box Plot"
        )

        st.pyplot(fig)
        plt.close(fig)

    elif chart == "Scatter Plot":

        if len(num_cols) >= 2:

            y_column = st.selectbox(
                "Select Y-axis Column",
                [c for c in num_cols if c != column]
            )

            data = df[
                [column, y_column]
            ].dropna()

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.scatter(
                data[column],
                data[y_column]
            )

            ax.set_xlabel(column)
            ax.set_ylabel(y_column)

            ax.set_title(
                f"{column} vs {y_column}"
            )

            st.pyplot(fig)
            plt.close(fig)


# ================= AUTOMATIC INSIGHTS =================

st.header("9️⃣ 💡 Automatic Insights")

st.write(
    f"• Dataset contains **{len(df)} rows** "
    f"and **{len(df.columns)} columns**."
)

st.write(
    f"• There are **{len(num_cols)} numerical columns** "
    f"and **{len(cat_cols)} categorical columns**."
)

missing = df.isna().sum()

if missing.max() > 0:

    missing_column = missing.idxmax()

    st.write(
        f"• **{missing_column}** has the highest "
        f"number of missing values: "
        f"**{int(missing.max())}**."
    )

else:

    st.write(
        "• The dataset has **no missing values**."
    )


if num_cols:

    averages = df[num_cols].mean()

    highest = averages.idxmax()

    st.write(
        f"• **{highest}** has the highest "
        f"average value: "
        f"**{averages[highest]:.2f}**."
    )

    max_column = df[num_cols].max().idxmax()
    max_value = df[max_column].max()

    min_column = df[num_cols].min().idxmin()
    min_value = df[min_column].min()

    st.write(
        f"• Overall maximum is "
        f"**{max_value:.2f}** in **{max_column}**."
    )

    st.write(
        f"• Overall minimum is "
        f"**{min_value:.2f}** in **{min_column}**."
    )


if len(num_cols) >= 2:

    corr = df[num_cols].corr()

    best_pair = None
    best_value = -1

    for i in range(len(num_cols)):

        for j in range(i + 1, len(num_cols)):

            value = abs(corr.iloc[i, j])

            if pd.notna(value) and value > best_value:

                best_value = value

                best_pair = (
                    num_cols[i],
                    num_cols[j],
                    corr.iloc[i, j]
                )

    if best_pair:

        a, b, value = best_pair

        st.write(
            f"• Strongest relationship is between "
            f"**{a}** and **{b}** "
            f"with correlation **{value:.2f}**."
        )


st.success(
    "✅ Analysis completed successfully. "
    "Use Comparison Analysis to compare different categories "
    "and numerical columns."
        )
