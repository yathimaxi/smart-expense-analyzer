import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest

st.set_page_config(page_title="Smart Expense Analyzer PRO", layout="wide")

st.title("💸 Smart Expense Analyzer PRO")
st.markdown("AI-powered expense insights, anomaly detection & prediction")

# Upload CSV
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # ----------------------------
    # DATA CLEANING
    # ----------------------------
    st.subheader("🧹 Data Cleaning")

    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.strip()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Category'] = df['Category'].astype(str).str.strip().str.title()
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    df = df.dropna(subset=['Date', 'Category', 'Amount'])
    df = df[df['Amount'] > 0]

    st.success("Data cleaned successfully!")

    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.dayofweek

    # ----------------------------
    # SIDEBAR FILTERS
    # ----------------------------
    st.sidebar.header("🎛️ Filters")

    categories = st.sidebar.multiselect(
        "Select Category",
        df['Category'].unique(),
        default=df['Category'].unique()
    )

    df = df[df['Category'].isin(categories)]

    # ----------------------------
    # EDA - INTERACTIVE
    # ----------------------------
    st.subheader("📊 Interactive Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(df, names='Category', values='Amount',
                      title='Category-wise Spending')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        monthly_data = df.groupby('Month')['Amount'].sum().reset_index()
        fig2 = px.line(monthly_data, x='Month', y='Amount',
                       markers=True, title='Monthly Spending Trend')
        st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # SMART INSIGHTS
    # ----------------------------
    st.subheader("🧠 AI Insights")

    total = df['Amount'].sum()
    avg = df['Amount'].mean()
    top_category = df.groupby('Category')['Amount'].sum().idxmax()

    st.metric("💰 Total Spending", f"₹{total}")
    st.metric("📊 Average Spending", f"₹{avg:.2f}")
    st.metric("🔥 Top Category", top_category)

    if avg > 500:
        st.warning("💸 Your average spending is quite high!")

    if top_category == "Food":
        st.info("🍔 You're spending heavily on food. Consider budgeting.")

    # Weekend vs weekday
    weekend = df[df['Weekday'] >= 5]['Amount'].sum()
    weekday = df[df['Weekday'] < 5]['Amount'].sum()

    if weekend > weekday:
        st.warning("⚠️ You spend more on weekends!")
    else:
        st.success("✅ You spend more on weekdays.")

    # ----------------------------
    # ANOMALY DETECTION
    # ----------------------------
    st.subheader("🚨 Anomaly Detection")

    iso = IsolationForest(contamination=0.1)
    df['Anomaly'] = iso.fit_predict(df[['Amount']])

    anomalies = df[df['Anomaly'] == -1]

    st.write("Unusual Transactions:")
    st.dataframe(anomalies)

    # ----------------------------
    # MACHINE LEARNING
    # ----------------------------
    st.subheader("🤖 Expense Prediction")

    monthly = df.groupby('Month')['Amount'].sum().reset_index()

    if len(monthly) > 1:
        X = monthly[['Month']]
        y = monthly['Amount']

        model = RandomForestRegressor()
        model.fit(X, y)

        next_month = np.array([[monthly['Month'].max() + 1]])
        prediction = model.predict(next_month)

        st.success(f"📈 Predicted Next Month Expense: ₹{prediction[0]:.2f}")
    else:
        st.warning("⚠️ Need at least 2 months of data for prediction.")

    # ----------------------------
    # DOWNLOAD CLEANED DATA
    # ----------------------------
    st.subheader("📥 Download Cleaned Data")

    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "cleaned_data.csv")