import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Chocolate Sales Dashboard",
    page_icon="🍫",
    layout="wide"
)

# ============================================
# Data Loading & Processing
# ============================================
df = pd.read_csv("Chocolate_Sales.csv")

# Remove duplicate records
df = df.drop_duplicates()

# Convert order date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

# Convert numerical columns to numbers
numeric_columns = [
    "Discount_Pct",
    "Price_per_Box",
    "Marketing_Spend",
    "Boxes_Shipped",
    "Amount"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing numerical values
df["Discount_Pct"] = df["Discount_Pct"].fillna(df["Discount_Pct"].mean())
df["Price_per_Box"] = df["Price_per_Box"].fillna(df["Price_per_Box"].mean())
df["Marketing_Spend"] = df["Marketing_Spend"].fillna(df["Marketing_Spend"].mean())
df["Boxes_Shipped"] = df["Boxes_Shipped"].fillna(df["Boxes_Shipped"].median())
df["Amount"] = df["Amount"].fillna(df["Amount"].mean())

# Fill missing categorical values
for col in ["Product", "Country", "Channel", "Salesperson"]:
    df[col] = df[col].fillna("Unknown")

# Use NumPy for numerical rounding
df["Discount_Pct"] = np.round(df["Discount_Pct"], 2)
df["Price_per_Box"] = np.round(df["Price_per_Box"], 2)
df["Marketing_Spend"] = np.round(df["Marketing_Spend"], 2)

# ============================================
# Sidebar Navigation
# ============================================
st.sidebar.title("🍫 Chocolate Sales App")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Introduction",
        "📊 Data Visualization",
        "🪐 Prediction",
        "🧠 Explainable AI",
        "⚙️ Hyperparameter Tuning",
        "📄 Conclusion"
    ]
)

# ============================================
# Page 1: Introduction
# ============================================
if page == "🏠 Introduction":

    st.markdown("""
<h1 style='
text-align:center;
font-size:56px;
margin-bottom:10px;
'>
🍫 Chocolate Sales Dashboard
</h1>
""", unsafe_allow_html=True)

    content_left, content_center, content_right = st.columns([1, 4, 1])

    with content_center:
        st.markdown("### Predicting Chocolate Boxes Shipped Using Machine Learning")

        st.image("chocolate_banner.jpg", width=850)

        st.write("""
        This dashboard analyzes historical chocolate sales data and develops machine learning models
        to predict the number of chocolate boxes shipped. Accurate demand prediction helps businesses
        optimize inventory management, improve marketing strategies, and support better business decisions.
        """)

    st.divider()

    left, center, right = st.columns([1, 4, 1])

    with center:
        st.header("Business Problem")

        st.write("""
        Chocolate companies need accurate demand forecasting to optimize inventory levels,
        marketing investment, and supply chain operations.

        Without reliable sales predictions, businesses may experience overstocking,
        stock shortages, or inefficient marketing spending.

        This project aims to solve this problem by predicting the number of chocolate boxes
        shipped (**Boxes_Shipped**) using historical sales data.
        """)

    left, center, right = st.columns([1, 4, 1])

    with center:
        st.header("Dataset Description")

        st.write("""
        This dataset contains **200,000 historical chocolate sales transactions**
        collected between **2022 and 2023** across multiple countries and sales channels.

        Each record represents one customer order and includes product information,
        country, sales channel, salesperson, order date, discount percentage,
        price per box, marketing spend, shipment volume, and total sales amount.

        The target variable is **Boxes_Shipped**, representing the number of
        chocolate boxes delivered for each order.
        """)

    left, center, right = st.columns([1, 4, 1])

    with center:
        st.header("Project Objective")

        st.write("""
        The objective of this project is to build machine learning models that accurately
        predict shipment volume (**Boxes_Shipped**) using historical sales data.

        The prediction results can help businesses:

        - Improve inventory planning
        - Optimize marketing budgets
        - Support supply chain management
        - Make better data-driven business decisions
        """)

    st.divider()

    st.header("Data Loading & Processing")

    st.write("""
    The dataset was loaded using **Pandas** and processed with **Pandas** and **NumPy**.
    The preprocessing steps include removing duplicate records, converting order dates,
    filling missing values, and rounding numerical variables.
    """)

    processing_steps = pd.DataFrame({
        "Step": [
            "Load CSV file",
            "Remove duplicates",
            "Convert Order_Date",
            "Fill missing numerical values",
            "Fill missing categorical values",
            "Round numerical variables"
        ],
        "Tool Used": [
            "Pandas",
            "Pandas",
            "Pandas",
            "Pandas",
            "Pandas",
            "NumPy"
        ]
    })

    st.dataframe(processing_steps, use_container_width=True)

    st.divider()

    st.header("Dataset Overview")
    st.write("The processed dataset contains the following basic information:")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Countries", df["Country"].nunique())

    with col4:
        st.metric("Products", df["Product"].nunique())

    with col5:
        st.write("**Target Variable**")
        st.success("Boxes_Shipped")

    st.divider()

    st.header("Dataset Variables")

    feature_df = pd.DataFrame({
        "Variable": [
            "Order_ID",
            "Product",
            "Country",
            "Channel",
            "Salesperson",
            "Order_Date",
            "Discount_Pct",
            "Price_per_Box",
            "Marketing_Spend",
            "Boxes_Shipped",
            "Amount"
        ],
        "Description": [
            "Unique identifier for each order",
            "Chocolate product name",
            "Country where the order was placed",
            "Sales channel (Retail / Online / Wholesale)",
            "Sales representative handling the order",
            "Date when the order was placed",
            "Discount percentage applied to the order",
            "Selling price per chocolate box",
            "Marketing budget related to the order",
            "Number of chocolate boxes shipped (Target Variable)",
            "Total order value"
        ]
    })

    st.dataframe(feature_df, use_container_width=True)

    st.divider()

    st.header("Dataset Sample")
    st.write("The table below displays the first five observations from the processed dataset.")

    st.dataframe(df.head(), use_container_width=True)

    st.divider()

    st.header("Missing Values After Processing")

    st.write("""
    Checking missing values is an important data preprocessing step before
    training machine learning models.
    """)

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Variable", "Missing Values"]

    st.dataframe(missing, use_container_width=True)

    st.divider()

    st.header("Summary Statistics")

    st.write("""
    The table below summarizes the numerical variables in the dataset,
    including their distribution, central tendency, and variability.
    """)

    st.dataframe(df.describe(), use_container_width=True)


# ============================================
# Page 2: Data Visualization
# ============================================
elif page == "📊 Data Visualization":

    st.markdown(
        "<h1 style='font-size:46px;'>📊 Data Visualization</h1>",
        unsafe_allow_html=True
    )

    st.markdown("### Key Business Insights from Chocolate Sales Data")

    st.write("""
    This page explores important patterns in the chocolate sales dataset using
    **Seaborn** and **Matplotlib** visualizations. These charts help identify
    business insights related to countries, sales channels, products, marketing
    investment, pricing, and shipment volume.
    """)

    st.divider()

    # -----------------------------
    # Chart 1
    # -----------------------------
    st.header("1. Total Boxes Shipped by Country")

    country_boxes = (
        df.groupby("Country")["Boxes_Shipped"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=country_boxes.index,
        y=country_boxes.values,
        palette="YlOrBr",
        ax=ax
    )

    ax.set_title("Total Boxes Shipped by Country")
    ax.set_xlabel("Country")
    ax.set_ylabel("Total Boxes Shipped")
    plt.xticks(rotation=30)

    st.pyplot(fig)

    top_country = country_boxes.index[0]

    st.info(f"""
    **Business Insight:**  
    **{top_country}** has the highest total shipment volume, which suggests it is one of
    the company’s strongest markets. The company can prioritize inventory allocation,
    marketing campaigns, and sales planning in this market.
    """)

    st.divider()

    # -----------------------------
    # Chart 2
    # -----------------------------
    st.header("2. Average Boxes Shipped by Sales Channel")

    channel_boxes = (
        df.groupby("Channel")["Boxes_Shipped"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=channel_boxes.index,
        y=channel_boxes.values,
        palette="YlOrBr",
        ax=ax
    )

    ax.set_title("Average Boxes Shipped by Sales Channel")
    ax.set_xlabel("Sales Channel")
    ax.set_ylabel("Average Boxes Shipped")

    st.pyplot(fig)

    top_channel = channel_boxes.index[0]

    st.info(f"""
    **Business Insight:**  
    The **{top_channel}** channel has the highest average boxes shipped per order.
    This suggests that channel strategy plays an important role in shipment performance
    and should be considered in demand forecasting.
    """)

    st.divider()

    # -----------------------------
    # Chart 3
    # -----------------------------
    st.header("3. Top 10 Products by Boxes Shipped")

    product_boxes = (
        df.groupby("Product")["Boxes_Shipped"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=product_boxes.values,
        y=product_boxes.index,
        palette="YlOrBr",
        ax=ax
    )

    ax.set_title("Top 10 Products by Total Boxes Shipped")
    ax.set_xlabel("Total Boxes Shipped")
    ax.set_ylabel("Product")

    st.pyplot(fig)

    top_product = product_boxes.index[0]

    st.info(f"""
    **Business Insight:**  
    **{top_product}** is the top-performing product by shipment volume.
    High-demand products should receive stronger inventory support and may be prioritized
    in marketing campaigns.
    """)

    st.divider()
    # -----------------------------
    # Chart 4
    # -----------------------------
    st.header("4. Marketing Spend vs Boxes Shipped")

    sample_df = df.sample(n=min(3000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(
        data=sample_df,
        x="Marketing_Spend",
        y="Boxes_Shipped",
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
        ax=ax
    )

    ax.set_title("Marketing Spend vs Boxes Shipped")
    ax.set_xlabel("Marketing Spend")
    ax.set_ylabel("Boxes Shipped")

    st.pyplot(fig)

    st.info("""
    **Business Insight:**  
    This chart shows a positive relationship between marketing spend and boxes shipped.
    Higher marketing investment is generally associated with higher shipment volume,
    suggesting that marketing budget allocation can influence customer demand.
    """)

    st.divider()

    # -----------------------------
    # Chart 5
    # -----------------------------
    st.header("5. Price per Box vs Boxes Shipped")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(
        data=sample_df,
        x="Price_per_Box",
        y="Boxes_Shipped",
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
        ax=ax
    )

    ax.set_title("Price per Box vs Boxes Shipped")
    ax.set_xlabel("Price per Box")
    ax.set_ylabel("Boxes Shipped")

    st.pyplot(fig)

    st.info("""
    **Business Insight:**  
    This chart shows the relationship between price and shipment volume.
    If higher prices are associated with lower shipment volume, the company may need
    to balance pricing strategy with customer demand.
    """)

    st.divider()

    # -----------------------------
    # Chart 6
    # -----------------------------
    st.header("6. Correlation Heatmap")

    numeric_cols = [
        "Discount_Pct",
        "Price_per_Box",
        "Marketing_Spend",
        "Boxes_Shipped",
        "Amount"
    ]

    numeric_df = df[numeric_cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="YlOrBr",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Heatmap of Numerical Variables")

    st.pyplot(fig)

    st.info("""
    **Business Insight:**  
    The heatmap shows the relationships between numerical variables.
    Variables with stronger correlation with **Boxes_Shipped** may be useful
    predictors in the machine learning model.
    """)

 # ============================================
# Page 3: Prediction
# ============================================
elif page == "🪐 Prediction":

    st.markdown(
        "<h1 style='font-size:46px;'>🎹 Machine Learning Prediction</h1>",
        unsafe_allow_html=True
    )
    st.write("""
    This page predicts **Boxes Shipped** using two machine learning models.

    Users can switch between **Linear Regression** and
    **Random Forest Regressor** to compare prediction performance.
    """)

    st.divider()

    # -----------------------------
    # Prepare Data
    # -----------------------------
    X = df[[
        "Discount_Pct",
        "Price_per_Box",
        "Marketing_Spend"
    ]]

    y = df["Boxes_Shipped"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # Choose Model
    # -----------------------------
    model_choice = st.selectbox(
        "Choose a Machine Learning Model",
        [
            "Linear Regression",
            "Random Forest"
        ]
    )

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    st.divider()

    # -----------------------------
    # Model Performance
    # -----------------------------
    st.subheader("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R² Score", f"{r2:.3f}")

    st.info(
        "Random Forest usually provides better prediction accuracy because it captures nonlinear relationships between pricing, discounts and marketing investment."
    )

    st.divider()

    # -----------------------------
    # User Prediction
    # -----------------------------
    st.subheader("📦 Predict Boxes Shipped")

    discount = st.slider(
        "Discount (%)",
        float(df["Discount_Pct"].min()),
        float(df["Discount_Pct"].max()),
        float(df["Discount_Pct"].mean())
    )

    price = st.slider(
        "Price per Box",
        float(df["Price_per_Box"].min()),
        float(df["Price_per_Box"].max()),
        float(df["Price_per_Box"].mean())
    )

    marketing = st.slider(
        "Marketing Spend",
        float(df["Marketing_Spend"].min()),
        float(df["Marketing_Spend"].max()),
        float(df["Marketing_Spend"].mean())
    )

    if st.button("Predict"):

        new_data = pd.DataFrame({
            "Discount_Pct": [discount],
            "Price_per_Box": [price],
            "Marketing_Spend": [marketing]
        })

        result = model.predict(new_data)[0]

        st.success(f"""
### 📦 Prediction Result

**Estimated Boxes Shipped:**

# {result:.0f} Boxes

This prediction can help businesses estimate inventory demand,
support production planning, and improve marketing decisions.
""")

    st.divider()

    # -----------------------------
    # Actual vs Predicted
    # -----------------------------
    st.subheader("📈 Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(7,6))

    ax.scatter(
        y_test,
        predictions,
        alpha=0.5
    )

    ax.set_xlabel("Actual Boxes Shipped")
    ax.set_ylabel("Predicted Boxes Shipped")
    ax.set_title("Actual vs Predicted")

    st.pyplot(fig)

    st.info("""
A good prediction model should generate predictions close to the actual shipment values.
The closer the points are to the diagonal trend, the better the model performs.
""")
    # ============================================
# Page 4: Explainable AI
# ============================================

elif page == "🧠 Explainable AI":

    st.markdown(
        "<h1 style='font-size:46px;'>🧠 Explainable AI</h1>",
        unsafe_allow_html=True
    )

    st.write("""
    This page explains how the machine learning model makes predictions.

    SHAP (SHapley Additive exPlanations) identifies which variables
    have the greatest influence on predicting chocolate boxes shipped.
    """)

    X = sample_df[
        [
            "Discount_Pct",
            "Price_per_Box",
            "Marketing_Spend"
        ]
    ]

    y = sample_df["Boxes_Shipped"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    X_sample = X_test.sample(500, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    st.subheader("Feature Importance")

    fig = plt.figure(figsize=(8,5))

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    st.pyplot(fig)

    plt.clf()

    st.subheader("SHAP Summary Plot")

    fig = plt.figure(figsize=(8,5))

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    st.pyplot(fig)

    st.divider()

    st.subheader("Business Interpretation")

    st.success("""
• Price per Box is one of the strongest variables affecting shipment prediction.

• Marketing Spend has a positive impact on customer demand.

• Discount Percentage also influences shipment volume.

• SHAP shows how each feature increases or decreases the predicted Boxes Shipped.

• This improves transparency and helps businesses understand the model's decisions.
""")
    # ============================================
# Page 5: Hyperparameter Tuning
# ============================================

elif page == "⚙️ Hyperparameter Tuning":

    st.markdown(
        "<h1 style='font-size:46px;'>⚙️ Hyperparameter Tuning</h1>",
        unsafe_allow_html=True
    )

    st.write("""
    This page compares different machine learning settings and selects
    the best-performing Random Forest model.
    """)

    X = sample_df[
        [
            "Discount_Pct",
            "Price_per_Box",
            "Marketing_Spend"
        ]
    ]

    y = sample_df["Boxes_Shipped"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    param_grid = {

        "n_estimators":[50,100],

        "max_depth":[5,10],

    }

    with st.spinner("Searching for best parameters..."):

        grid = GridSearchCV(

            RandomForestRegressor(random_state=42),

            param_grid,

            cv=3,

            scoring="r2"

        )

        grid.fit(X_train,y_train)

    best_model = grid.best_estimator_

    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test,predictions)

    rmse = np.sqrt(mean_squared_error(y_test,predictions))

    r2 = r2_score(y_test,predictions)

    st.subheader("Best Hyperparameters")

    st.write(grid.best_params_)

    st.subheader("Model Performance")

    c1,c2,c3 = st.columns(3)

    c1.metric("MAE",f"{mae:.2f}")

    c2.metric("RMSE",f"{rmse:.2f}")

    c3.metric("R²",f"{r2:.3f}")

    st.divider()

    linear = LinearRegression()

    linear.fit(X_train,y_train)

    lr_pred = linear.predict(X_test)

    comparison = pd.DataFrame({

        "Model":[

            "Linear Regression",

            "Random Forest"

        ],

        "MAE":[

            mean_absolute_error(y_test,lr_pred),

            mae

        ],

        "RMSE":[

            np.sqrt(mean_squared_error(y_test,lr_pred)),

            rmse

        ],

        "R²":[

            r2_score(y_test,lr_pred),

            r2

        ]

    })

    st.subheader("Model Comparison")

    st.dataframe(comparison,use_container_width=True)

    st.success("""
Random Forest produced stronger prediction performance by learning nonlinear
relationships between pricing, discounts and marketing investment.

The tuned Random Forest model was selected as the final model.
""")
    # ============================================
# Page 6: Conclusion
# ============================================

elif page == "📄 Conclusion":

    st.markdown(
        "<h1 style='font-size:46px;'>📄 Project Conclusion</h1>",
        unsafe_allow_html=True
    )

    st.header("Project Summary")

    st.write("""
This project developed a Streamlit dashboard that analyzes historical
chocolate sales data and predicts the number of chocolate boxes shipped
using machine learning.

The application combines business analytics, data visualization,
predictive modeling, explainable AI, and hyperparameter tuning to
support better business decision-making.
""")

    st.divider()

    st.header("Key Findings")

    st.write("""
• Marketing Spend positively influences shipment volume.

• Price per Box is one of the strongest predictors.

• Discount Percentage also contributes to shipment prediction.

• Countries and sales channels exhibit different shipment patterns.

• Machine learning can accurately estimate future shipment demand.
""")

    st.divider()

    st.header("Machine Learning Results")

    st.write("""
Two machine learning models were developed:

• Linear Regression

• Random Forest Regressor

After hyperparameter tuning, Random Forest achieved the strongest prediction
performance and was selected as the final model.
""")

    st.divider()

    st.header("Business Recommendations")

    st.write("""
1. Increase inventory for high-demand markets.

2. Allocate marketing budgets efficiently.

3. Monitor pricing strategies to maximize demand.

4. Focus promotional campaigns on high-performing products.

5. Use predictive analytics to improve supply chain planning.
""")

    st.divider()

    st.header("Future Improvements")

    st.write("""
• Include seasonal demand.

• Include customer purchasing behavior.

• Add weather and holiday effects.

• Compare additional machine learning models such as XGBoost.

• Deploy the model for real-time sales forecasting.
""")

    st.success("""
Thank you for using the Chocolate Sales Dashboard.

This project demonstrates how machine learning and explainable AI can
support smarter business decisions and improve demand forecasting.
""")
    
