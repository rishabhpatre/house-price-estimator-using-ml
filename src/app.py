# =============================================
# 🏠 Beginner-Friendly California House Price Estimator
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# 1️⃣ Page Setup
# ----------------------------
st.set_page_config(page_title="🏠 House Price Estimator", layout="wide", page_icon="🏠")

st.title("🏠 California House Price Estimator")
st.markdown("""
Welcome!  
This simple web app predicts **median house prices** in different parts of California.

It also explains **how different factors like income, rooms, and age of houses** affect price —  
and shows how machine learning models work in plain language.
""")

# ----------------------------
# 2️⃣ Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df["MedHouseVal"] = df["MedHouseVal"] * 100000  # Convert to realistic $ value
    return df

df = load_data()
st.subheader("📘 About the Data")
st.markdown("""
Each row represents a **neighborhood** in California (based on 1990 census data).

| Column | Meaning |
|---------|----------|
| `MedInc` | Median income in the area (in $10,000s) |
| `HouseAge` | Average age of houses (years) |
| `AveRooms` | Average number of rooms per house |
| `AveBedrms` | Average number of bedrooms per house |
| `Population` | Total population in that neighborhood |
| `AveOccup` | Average number of people per household |
| `Latitude`, `Longitude` | Location coordinates |
| `MedHouseVal` | Median house price (our **target**) |
""")

# ----------------------------
# 3️⃣ Split + Preprocess + Train
# ----------------------------
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
numeric_cols = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols)
])

@st.cache_resource
def train_models():
    model1 = Pipeline([("prep", preprocessor), ("reg", LinearRegression())])
    model2 = Pipeline([("prep", preprocessor), ("reg", Ridge(alpha=1.0))])
    model3 = Pipeline([("prep", preprocessor), ("reg", RandomForestRegressor(n_estimators=150, random_state=42))])
    for m in [model1, model2, model3]:
        m.fit(X_train, y_train)
    return model1, model2, model3

model1, model2, model3 = train_models()

# ----------------------------
# 4️⃣ Predictions + Evaluation
# ----------------------------
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred_avg = (y_pred1 + y_pred2 + y_pred3) / 3
results = [
    evaluate(y_test, y_pred1, "Linear Regression"),
    evaluate(y_test, y_pred2, "Ridge Regression"),
    evaluate(y_test, y_pred3, "Random Forest"),
    evaluate(y_test, y_pred_avg, "Average Ensemble"),
]
res_df = pd.DataFrame(results)

# ----------------------------
# 5️⃣ Tabs Layout
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Price Estimator",
    "📊 Model Comparison",
    "📈 Data Insights",
    "🧾 Summary"
])

# ====================================
# 💰 TAB 1 – Price Estimator
# ====================================
with tab1:
    st.subheader("Enter House Details to Estimate Price")

    col1, col2, col3 = st.columns(3)
    with col1:
        MedInc = st.slider("Median Income (×10k USD)", 0.5, 15.0, 5.0, 0.1)
        HouseAge = st.slider("House Age (years)", 1, 50, 25)
        AveRooms = st.slider("Average Rooms", 2.0, 10.0, 5.0, 0.1)
    with col2:
        AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0, 0.1)
        Population = st.slider("Population in Area", 100, 10000, 1500, 100)
        AveOccup = st.slider("Average Occupancy", 1.0, 6.0, 3.0, 0.1)
    with col3:
        Latitude = st.slider("Latitude", 32.0, 42.0, 34.0, 0.1)
        Longitude = st.slider("Longitude", -124.0, -114.0, -118.0, 0.1)

    input_df = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }])

    if st.button("🔮 Estimate House Price"):
        preds = [model1.predict(input_df)[0], model2.predict(input_df)[0], model3.predict(input_df)[0]]
        avg_pred = np.mean(preds)
        st.success(f"🏡 **Estimated Median House Price:** ${avg_pred:,.0f}")
        st.caption(f"Linear: ${preds[0]:,.0f} | Ridge: ${preds[1]:,.0f} | Random Forest: ${preds[2]:,.0f}")
        st.markdown("""
        💡 **Note:** This prediction is based on patterns learned from 1990s California housing data.  
        Higher median income, newer homes, and more rooms generally increase predicted price.
        """)

# ====================================
# 📊 TAB 2 – Model Comparison
# ====================================
with tab2:
    st.subheader("How Each Model Performs")

    st.markdown("""
    Each model predicts prices slightly differently:  
    - **Linear Regression:** simple straight-line model.  
    - **Ridge Regression:** similar but more stable against noise.  
    - **Random Forest:** uses many decision trees for complex patterns.  
    The **ensemble** combines all three for balanced predictions.
    """)

    st.dataframe(res_df.style.format(subset=["MAE", "RMSE", "R2"], formatter="{:.3f}"))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.barplot(data=res_df, x="Model", y="R2", palette="viridis", ax=ax)
        ax.set_title("Model Accuracy (R² Score)")
        st.pyplot(fig)
        st.caption("R² closer to 1 means better prediction accuracy.")
    with col2:
        fig, ax = plt.subplots()
        sns.barplot(data=res_df, x="Model", y="RMSE", palette="magma", ax=ax)
        ax.set_title("Error Comparison (RMSE)")
        st.pyplot(fig)
        st.caption("Lower RMSE means smaller average prediction errors.")

# ====================================
# 📈 TAB 3 – Data Insights
# ====================================
with tab3:
    st.subheader("Understanding the Data")

    st.markdown("Here’s how prices and features relate in this dataset:")

    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(df["MedHouseVal"], bins=40, kde=True, color="#42A5F5", ax=ax)
    ax.set_title("Distribution of House Prices")
    st.pyplot(fig)
    st.caption("Most homes are priced below $300,000, with fewer very high-value homes.")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr().round(2), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Between Features")
    st.pyplot(fig)
    st.caption("""
    - Dark red = strong positive relation (price increases with that feature)  
    - Blue = negative relation (price decreases with that feature)  
    For example, higher **income (MedInc)** correlates strongly with higher prices.
    """)

    rf = model3.named_steps["reg"]
    imp = pd.Series(rf.feature_importances_, index=numeric_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=imp.values, y=imp.index, palette="crest", ax=ax)
    ax.set_title("Which Features Matter Most (Random Forest)")
    st.pyplot(fig)
    st.caption("""
    - **Median Income** is the strongest driver of price.  
    - **Latitude/Longitude** show that coastal (southern) areas are more expensive.  
    - Other features (rooms, house age) have smaller effects.
    """)

# ====================================
# 🧾 TAB 4 – Summary
# ====================================
with tab4:
    st.subheader("Key Takeaways")
    st.markdown("""
    ✅ **What this app does:**
    - Predicts California median house prices using real data.
    - Lets you play with features like income, rooms, and age.
    - Shows how machine learning models compare visually.

    ✅ **How to interpret results:**
    - Higher **Median Income**, **more rooms**, and **newer houses** → higher predicted prices.
    - Random Forest generally performs best (captures complex patterns).
    - Combining models (the *Ensemble*) gives the most balanced predictions.

    ✅ **Metrics explained:**
    - **MAE (Mean Absolute Error):** average error in dollars.
    - **RMSE (Root Mean Square Error):** penalizes big mistakes more.
    - **R²:** how well the model explains real-world prices (1.0 = perfect).

    💬 **In short:**  
    This app gives a realistic, easy-to-understand way to explore how different housing factors affect prices — and lets you estimate your own!
    """)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Scikit-learn, and real California housing data.")
