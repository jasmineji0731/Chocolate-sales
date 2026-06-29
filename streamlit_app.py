import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import wandb
from sklearn.tree import DecisionTreeRegressor
import os

#if "WANDB_API_KEY" in os.environ:
    #wandb.login(key=os.environ["WANDB_API_KEY"])
    
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
        
    display_df = df.copy()
    display_df["Order_Date"] = display_df["Order_Date"].astype(str)
    
    st.dataframe(display_df.head(), use_container_width=True)
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

    st.title("📊 Data Visualization")

    st.write("""
This page explores key business patterns in the chocolate sales dataset.
The visualizations help identify important relationships between sales,
marketing, pricing, and shipment volume.
""")

    st.divider()

    # ----------------------------------
    # Sample data
    # ----------------------------------
    plot_df = df.sample(
        n=min(5000, len(df)),
        random_state=42
    )

    # ----------------------------------
    # Chart 1
    # ----------------------------------
    st.subheader("1. Total Boxes Shipped by Country")

    country = (
        plot_df.groupby("Country", as_index=False)["Boxes_Shipped"]
        .sum()
        .sort_values("Boxes_Shipped", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9,5))

    sns.barplot(
        data=country,
        x="Country",
        y="Boxes_Shipped",
        hue="Country",
        legend=False,
        palette="YlOrBr",
        ax=ax
    )

    plt.xticks(rotation=30)

    st.pyplot(fig)

    plt.close(fig)

    st.success(f"""
Highest shipment volume comes from **{country.iloc[0]['Country']}**,
making it an important market for inventory planning.
""")

    st.divider()

    # ----------------------------------
    # Chart 2
    # ----------------------------------
    st.subheader("2. Average Boxes Shipped by Channel")

    channel = (
        plot_df.groupby("Channel", as_index=False)["Boxes_Shipped"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        data=channel,
        x="Channel",
        y="Boxes_Shipped",
        hue="Channel",
        legend=False,
        palette="YlOrBr",
        ax=ax
    )

    st.pyplot(fig)

    plt.close(fig)

    st.info("""
Different sales channels produce different shipment volumes.
High-performing channels should receive additional marketing support.
""")

    st.divider()

    # ----------------------------------
    # Chart 3
    # ----------------------------------
    st.subheader("3. Top 10 Products")

    product = (
        plot_df.groupby("Product", as_index=False)["Boxes_Shipped"]
        .sum()
        .sort_values("Boxes_Shipped", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(
        data=product,
        x="Boxes_Shipped",
        y="Product",
        hue="Product",
        legend=False,
        palette="YlOrBr",
        ax=ax
    )

    st.pyplot(fig)

    plt.close(fig)

    st.info("""
The highest-selling products should receive priority inventory allocation.
""")

    st.divider()

    # ----------------------------------
    # Chart 4
    # ----------------------------------
    st.subheader("4. Marketing Spend vs Boxes Shipped")

    fig, ax = plt.subplots(figsize=(9,5))

    sns.scatterplot(
        data=plot_df,
        x="Marketing_Spend",
        y="Boxes_Shipped",
        alpha=0.5,
        ax=ax
    )

    st.pyplot(fig)

    plt.close(fig)

    st.info("""
Higher marketing spending is generally associated with higher shipment volume.
""")

    st.divider()

    # ----------------------------------
    # Chart 5
    # ----------------------------------
    st.subheader("5. Price per Box vs Boxes Shipped")

    fig, ax = plt.subplots(figsize=(9,5))

    sns.scatterplot(
        data=plot_df,
        x="Price_per_Box",
        y="Boxes_Shipped",
        alpha=0.5,
        ax=ax
    )

    st.pyplot(fig)

    plt.close(fig)

    st.info("""
Pricing strategy influences customer demand and shipment volume.
""")

    st.divider()

    # ----------------------------------
    # Chart 6
    # ----------------------------------
    st.subheader("6. Correlation Heatmap")

    corr = plot_df[
        [
            "Discount_Pct",
            "Price_per_Box",
            "Marketing_Spend",
            "Boxes_Shipped",
            "Amount"
        ]
    ].corr()

    fig, ax = plt.subplots(figsize=(7,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="YlOrBr",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)

    plt.close(fig)

    st.success("""
The heatmap identifies the strongest numerical relationships and helps
determine which variables are useful predictors for machine learning.
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
# Page 4: Explainable AI (SHAP)
# ============================================

elif page == "🧠 Explainable AI":

    st.markdown(
        "<h1 style='font-size:46px;'>🧠 Explainable AI (SHAP)</h1>",
        unsafe_allow_html=True
    )

    st.write("""
This page explains how the Random Forest model predicts
**Boxes Shipped** using SHAP (SHapley Additive exPlanations).
""")

    # -----------------------------
    # Prepare data
    # -----------------------------
    sample_df = df.sample(
        n=min(2000, len(df)),
        random_state=42
    )

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

    # -----------------------------
    # Train model once
    # -----------------------------
    @st.cache_resource
    def train_rf():

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)

        return model, explainer

    model, explainer = train_rf()

    # -----------------------------
    # Small sample for SHAP
    # -----------------------------
    sample = X_test.sample(
        n=min(100, len(X_test)),
        random_state=42
    )

    with st.spinner("Generating SHAP explanations..."):

        shap_values = explainer.shap_values(sample)

    st.divider()

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("📊 SHAP Feature Importance")

    plt.close("all")

    fig = plt.figure(figsize=(8,5))

    shap.summary_plot(
        shap_values,
        sample,
        plot_type="bar",
        show=False
    )

    st.pyplot(fig)

    plt.close(fig)

    st.info("""
The bar chart ranks variables according to their overall importance.

Features at the top contribute the most to predicting
Boxes Shipped.
""")

    st.divider()

    # -----------------------------
    # SHAP Summary
    # -----------------------------
    st.subheader("📈 SHAP Summary Plot")

    plt.close("all")

    fig = plt.figure(figsize=(8,5))

    shap.summary_plot(
        shap_values,
        sample,
        show=False
    )

    st.pyplot(fig)

    plt.close(fig)

    st.info("""
Each point represents one order.

• Red = higher feature value

• Blue = lower feature value

Features pushing predictions higher appear on the right,
while those lowering predictions appear on the left.
""")

    st.divider()

    st.subheader("Business Interpretation")

    st.success("""
Key insights from SHAP analysis:

• Price per Box has a strong influence on shipment prediction.

• Marketing Spend generally increases predicted shipment volume.

• Discount Percentage also contributes to prediction performance.

• Positive SHAP values increase predicted Boxes Shipped.

• Negative SHAP values decrease predicted Boxes Shipped.

These explanations improve model transparency and help managers understand how the Random Forest model makes business predictions.
""")
# ============================================
# Page 5: Hyperparameter Tuning
# ============================================

elif page == "⚙️ Hyperparameter Tuning":

    st.title("⚙️ Model Comparison & Hyperparameter Tuning")

    st.write("""
This page compares three machine learning models:

• Linear Regression

• Decision Tree Regressor

• Random Forest Regressor

Different hyperparameters are tested to identify the model with the
best predictive performance.
""")

    # ----------------------------
    # Prepare Data
    # ----------------------------

    sample_df = df.sample(
        n=min(5000, len(df)),
        random_state=42
    )

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

    results = []

    PROJECT_NAME = "Chocolate-Sales"

   # ======================================================
# Linear Regression Experiments
# ======================================================

for fit in [True, False]:

    run = wandb.init(
        project=PROJECT_NAME,
        mode="disabled",
        reinit=True,
        config={
            "Model": "Linear Regression",
            "fit_intercept": fit
        }
    )

    model = LinearRegression(
        fit_intercept=fit
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    wandb.log({
        "Model": "Linear Regression",
        "fit_intercept": fit,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    results.append({
        "Model": "Linear Regression",
        "Parameter": f"fit_intercept={fit}",
        "MAE": round(mae,2),
        "RMSE": round(rmse,2),
        "R²": round(r2,3)
    })

    run.finish()


# ======================================================
# Decision Tree Experiments
# ======================================================

for depth in [5,10]:

    run = wandb.init(
        project=PROJECT_NAME,
        mode="disabled",
        reinit=True,
        config={
            "Model":"Decision Tree",
            "max_depth":depth
        }
    )

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    mae=mean_absolute_error(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)

    wandb.log({
        "Model":"Decision Tree",
        "max_depth":depth,
        "MAE":mae,
        "RMSE":rmse,
        "R2":r2
    })

    results.append({
        "Model":"Decision Tree",
        "Parameter":f"Depth={depth}",
        "MAE":round(mae,2),
        "RMSE":round(rmse,2),
        "R²":round(r2,3)
    })

    run.finish()


# ======================================================
# Random Forest Experiments
# ======================================================

for trees in [50, 100, 150]:

    for depth in [5, 10]:

        run = wandb.init(
            project=PROJECT_NAME,
            mode="disabled",
            reinit=True,
            config={
                "Model": "Random Forest",
                "Trees": trees,
                "Max Depth": depth
            }
        )

        model = RandomForestRegressor(
            n_estimators=trees,
            max_depth=depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        wandb.log({
            "Model": "Random Forest",
            "Trees": trees,
            "Max Depth": depth,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        results.append({
            "Model": "Random Forest",
            "Parameter": f"{trees} Trees | Depth={depth}",
            "MAE": round(mae,2),
            "RMSE": round(rmse,2),
            "R²": round(r2,3)
        })

        run.finish()
        # ==========================================
    # Results Table
    # ==========================================
    
    result_df = pd.DataFrame(results)
    
    st.subheader("📊 Model Comparison Results")
    
    st.dataframe(result_df, use_container_width=True)
    
    st.divider()
    
    # ==========================================
    # Best Model
    # ==========================================
    
    best = result_df.loc[result_df["R²"].idxmax()]
    
    st.success(f"""
    ### 🏆 Best Performing Model
    
    **Model:** {best["Model"]}
    
    **Parameter:** {best["Parameter"]}
    
    **MAE:** {best["MAE"]}
    
    **RMSE:** {best["RMSE"]}
    
    **R² Score:** {best["R²"]}
    
    This model achieved the highest predictive accuracy among all experiments.
    """)
    
    st.divider()
    
    # ==========================================
    # Model Comparison Chart
    # ==========================================
    
    st.subheader("📈 R² Score Comparison")
    
    fig, ax = plt.subplots(figsize=(10,5))
    
    sns.barplot(
        data=result_df,
        x="Model",
        y="R²",
        hue="Parameter",
        palette="YlOrBr",
        ax=ax
    )
    
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Machine Learning Model")
    ax.set_title("Comparison of Machine Learning Models")
    
    plt.xticks(rotation=20)
    
    st.pyplot(fig)
    
    plt.close(fig)
    
    st.divider()
    
    # ==========================================
    # Interpretation
    # ==========================================
    
    st.subheader("Business Interpretation")
    
    st.write("""
    Ten machine learning experiments were conducted to compare the predictive
    performance of three algorithms: Linear Regression, Decision Tree Regressor,
    and Random Forest Regressor.
    
    Linear Regression served as the baseline model and assumes a linear
    relationship between the predictors and shipment volume.
    
    Decision Tree improved model flexibility by capturing nonlinear patterns,
    although deeper trees may increase the risk of overfitting.
    
    Random Forest consistently achieved the highest predictive performance by
    combining multiple decision trees, reducing model variance and improving
    generalization.
    
    Among all experiments, the model shown above achieved the highest R² score
    and was therefore selected as the final prediction model for this project.
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
    
