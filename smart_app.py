import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Inventory & Holiday Demand Planner",
    page_icon="📦",
    layout="wide"
)

# --- CACHED DATA LOADING & MODEL TRAINING ---
@st.cache_resource

def load_data_and_model():
    # Load dataset (update path if necessary)
    df = pd.read_csv("Walmart.csv")
    
    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # Store original date for plotting
    df_raw = df.copy()
    
    # One-hot encoding for stores
    df_encoded = pd.get_dummies(df, columns=['Store'], drop_first=True)
    
    X = df_encoded.drop(['Weekly_Sales', 'Date'], axis=1)
    y = df_encoded['Weekly_Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor (optimal performance from notebook)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return df_raw, df_encoded, model, X

try:
    df_raw, df_encoded, model, X = load_data_and_model()
except FileNotFoundError:
    st.error("Dataset 'Walmart.csv' not found! Please place it in the application directory.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔧 Planning Controls")

# Store Selection
stores = sorted(df_raw['Store'].unique())
selected_store = st.sidebar.selectbox("Select Store", stores)

# User-defined Threshold for Sales Alerts
max_sales = float(df_raw['Weekly_Sales'].max())
min_sales = float(df_raw['Weekly_Sales'].min())
mean_sales = float(df_raw['Weekly_Sales'].mean())

sales_threshold = st.sidebar.slider(
    "Sales Alert Threshold ($)",
    min_value=min_sales,
    max_value=max_sales,
    value=mean_sales * 1.25,
    step=10000.0,
    help="Weeks where predicted sales exceed this threshold will be flagged for extra inventory preparation."
)

# Demand Buffer Testing
demand_buffer = st.sidebar.slider(
    "Demand Buffer (%)",
    min_value=0,
    max_value=50,
    value=10,
    step=5,
    help="Percentage buffer added on top of predicted sales to test holding and safety stock requirements."
)

# --- MAIN DASHBOARD ---
st.title("📦 Smart Inventory & Holiday Demand Planning Dashboard")
st.markdown(
    "This application uses machine learning sales forecasts to help retail planners identify high-demand "
    "periods, evaluate holiday surges, and test inventory demand buffers."
)

# Important Limitation Notice per instructions
st.warning(
    "⚠️ **Important Planning Limitation:** This application predicts **weekly sales**, not final inventory quantities "
    "or stockouts. To calculate exact recommended order quantities, supplementary data such as current stock, lead times, "
    "and replenishment schedules would be required. Use these outputs as a sales-based inventory planning aid."
)

st.markdown("---")

# Filter data for selected store
store_mask_raw = df_raw['Store'] == selected_store
store_data_raw = df_raw[store_mask_raw].sort_values('Date').copy()

if selected_store != 1:
    store_mask_encoded = (
        df_encoded[f'Store_{selected_store}'] == True
        if f'Store_{selected_store}' in df_encoded.columns
        else pd.Series(False, index=df_encoded.index)
    )
else:
    store_cols = [
        col for col in df_encoded.columns
        if col.startswith('Store_')
    ]

    store_mask_encoded = ~df_encoded[store_cols].any(axis=1)# Predict using the model
X_store = df_encoded[df_encoded['Store_' + str(selected_store)] == True] if selected_store != 1 else df_encoded.loc[df_raw['Store'] == 1]
# Simpler approach: Predict for all rows belonging to the store
store_indices = df_raw['Store'] == selected_store
X_store_features = X[store_indices]
store_preds = model.predict(X_store_features)

store_data_raw['Predicted_Sales'] = store_preds
store_data_raw['Buffered_Sales'] = store_data_raw['Predicted_Sales'] * (1 + demand_buffer / 100.0)

# --- FEATURE 1 & 3: FORECASTED PEAKS & THRESHOLD ALERTS ---
st.subheader(f"📈 Sales Forecast & Peak Alerts for Store #{selected_store}")

# Flag weeks exceeding threshold
store_data_raw['Alert'] = store_data_raw['Predicted_Sales'] > sales_threshold

# Metrics Overview
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Predicted Sales", f"${store_data_raw['Predicted_Sales'].mean():,.2f}")
col2.metric("Peak Predicted Sales", f"${store_data_raw['Predicted_Sales'].max():,.2f}")
col3.metric("Peak Buffered Demand (+{0}%)".format(demand_buffer), f"${store_data_raw['Buffered_Sales'].max():,.2f}")
col4.metric("Weeks Exceeding Threshold", int(store_data_raw['Alert'].sum()))

# Plotting Sales vs Threshold
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(store_data_raw['Date'], store_data_raw['Weekly_Sales'], label='Actual Sales', color='lightblue', alpha=0.7)
ax.plot(store_data_raw['Date'], store_data_raw['Predicted_Sales'], label='Predicted Sales (ML)', color='blue', linewidth=2)
ax.axhline(y=sales_threshold, color='red', linestyle='--', label=f'Threshold (${sales_threshold:,.0f})')

# Highlight alert weeks
alert_weeks = store_data_raw[store_data_raw['Alert']]
ax.scatter(alert_weeks['Date'], alert_weeks['Predicted_Sales'], color='red', zorder=5, label='Demand Alert Peak')

ax.set_title(f"Weekly Sales Trajectory - Store #{selected_store}")
ax.set_xlabel("Date")
ax.set_ylabel("Weekly Sales ($)")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("---")

# --- FEATURE 2: COMPARE NORMAL WEEKS VS THANKSGIVING AND CHRISTMAS ---
st.subheader("🦃🎄 Normal Weeks vs. Holiday Periods (Thanksgiving & Christmas)")

# Walmart holiday identification helper
# Thanksgiving typically occurs in November (Week ~47-48), Christmas in December (Week ~51-52)
store_data_raw['Period_Type'] = 'Normal Week'
store_data_raw.loc[store_data_raw['Month'] == 11, 'Period_Type'] = 'Thanksgiving Period (Nov)'
store_data_raw.loc[store_data_raw['Month'] == 12, 'Period_Type'] = 'Christmas Period (Dec)'

comparison = store_data_raw.groupby('Period_Type')['Predicted_Sales'].mean().reset_index()

col_comp1, col_comp2 = st.columns([1, 1])

with col_comp1:
    st.markdown("### Average Sales by Period Type")
    st.dataframe(comparison.style.format({'Predicted_Sales': '${:,.2f}'}), use_container_width=True)

with col_comp2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(comparison['Period_Type'], comparison['Predicted_Sales'], color=['skyblue', 'orange', 'green'])
    ax2.set_ylabel("Average Predicted Sales ($)")
    ax2.set_title("Sales Comparison Across Periods")
    plt.xticks(rotation=15)
    st.pyplot(fig2)

st.markdown("---")

# --- FEATURE 4 & 5: STORE-LEVEL DEMAND PLANNING REPORT & BUFFER TESTING ---
st.subheader("📋 Store-Level Demand Planning Report")
st.markdown("Review flagged high-demand weeks incorporating your chosen **demand buffer** for inventory preparation.")

report_df = store_data_raw[['Date', 'Holiday_Flag', 'Weekly_Sales', 'Predicted_Sales', 'Buffered_Sales', 'Alert']].copy()
report_df['Status'] = report_df['Alert'].apply(lambda x: '🚨 HIGH DEMAND ALERT' - x if False else ('🚨 HIGH DEMAND ALERT' if x else 'Normal'))
report_df = report_df.rename(columns={
    'Date': 'Week Date',
    'Weekly_Sales': 'Historical Sales ($)',
    'Predicted_Sales': 'Forecasted Sales ($)',
    'Buffered_Sales': f'Buffered Target (+{demand_buffer}%) ($)'
})

# Display filter options
show_only_alerts = st.checkbox("Show only flagged threshold weeks", value=False)
if show_only_alerts:
    display_df = report_df[report_df['Alert'] == True]
else:
    display_df = report_df

st.dataframe(
    display_df[['Week Date', 'Holiday_Flag', 'Forecasted Sales ($)', f'Buffered Target (+{demand_buffer}%) ($)', 'Status']].style.format({
        'Forecasted Sales ($)': '${:,.2f}',
        f'Buffered Target (+{demand_buffer}%) ($)': '${:,.2f}'
    }),
    use_container_width=True
)

# Download report button
csv = report_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Complete Store Demand Report (CSV)",
    data=csv,
    file_name=f"demand_planning_report_store_{selected_store}.csv",
    mime='text/csv'
)