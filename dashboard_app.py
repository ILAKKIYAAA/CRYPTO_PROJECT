import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

CSV_PATH = Path("cryptocurrency_prices.csv")

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Cryptocurrency Top-10 Dashboard")

@st.cache_data
def load_data(path):
    if not path.exists():
        return None
    # Read CSV and parse timestamps robustly
    try:
        df = pd.read_csv(path, parse_dates=['Timestamp'], infer_datetime_format=True)
    except Exception:
        df = pd.read_csv(path)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', infer_datetime_format=True)
    # Drop rows with missing coin name or timestamp
    if 'Coin Name' in df.columns:
        df = df[df['Coin Name'].notna()]
    if 'Timestamp' in df.columns:
        df = df.dropna(subset=['Timestamp'])
    # Normalize coin names (take first line if multi-line and trim)
    if 'Coin Name' in df.columns:
        df['Coin Name'] = df['Coin Name'].astype(str).apply(lambda s: s.split('\n')[0].strip())
    return df

df = load_data(CSV_PATH)

if df is None or df.empty:
    st.warning(f"No data found at {CSV_PATH}. Run the scraper first to generate data.")
    st.stop()

# Use the latest timestamp snapshot if available
if 'Timestamp' in df.columns and not df['Timestamp'].isna().all():
    latest_ts = df['Timestamp'].max()
    snapshot = df[df['Timestamp'] == latest_ts].copy()
else:
    latest_ts = None
    snapshot = df.copy()

# Debug info for why snapshot may be small
# Debug info
st.write(f"Latest timestamp: {latest_ts}")
st.write(f"Rows in CSV: {len(df)}, rows in snapshot: {len(snapshot)}")

# --- Filtering controls (price and 24h change) ---
st.sidebar.header('Filters')
# Use parsed Price_val and Change_val from full df if available; recalc safely
def safe_parse_price_col(series):
    try:
        return series.astype(str).str.replace('[\\$,]', '', regex=True).str.replace(',', '').astype(float)
    except Exception:
        return pd.Series([None]*len(series))

if 'Price' in df.columns and '24h Change' in df.columns:
    df['Price_val_all'] = safe_parse_price_col(df['Price'])
    df['Change_val_all'] = df['24h Change'].astype(str).str.replace('%','').str.replace(',','').astype(str).apply(lambda x: pd.to_numeric(x, errors='coerce'))
else:
    df['Price_val_all'] = None
    df['Change_val_all'] = None

min_price = st.sidebar.number_input('Min price (USD)', value=0.0, step=1.0)
max_price = st.sidebar.number_input('Max price (USD)', value=1_000_000_000.0, step=1.0)
change_min = st.sidebar.slider('Min 24h change (%)', -100.0, 100.0, -100.0)
change_max = st.sidebar.slider('Max 24h change (%)', -100.0, 100.0, 100.0)

# Ensure numeric fields (move before filtering)
def parse_price(x):
    try:
        return float(str(x).replace('$','').replace(',','').strip())
    except Exception:
        return None

def parse_change(x):
    try:
        return float(str(x).replace('%','').replace(',','').strip())
    except Exception:
        return None

# Add numeric columns to snapshot for filtering/visuals
snapshot['Price_val'] = snapshot['Price'].apply(parse_price)
snapshot['Change_val'] = snapshot['24h Change'].apply(parse_change)

# Apply filters to the snapshot for display
filtered = snapshot.copy()
filtered = filtered.reset_index(drop=True)
if 'Price_val' in filtered.columns:
    filtered = filtered[(filtered['Price_val'].notna()) & (filtered['Price_val'] >= min_price) & (filtered['Price_val'] <= max_price)]
if 'Change_val' in filtered.columns:
    filtered = filtered[(filtered['Change_val'].notna()) & (filtered['Change_val'] >= change_min) & (filtered['Change_val'] <= change_max)]

# Sidebar controls
st.sidebar.header('Controls')
refresh = st.sidebar.button('Refresh data (reload CSV)')
if refresh:
    st.cache_data.clear()
    st.experimental_rerun()
# Populate history dropdown from the full dataset (historical list)
all_coins = sorted(df['Coin Name'].dropna().unique())
st.sidebar.markdown(f"**Available coins:** {len(all_coins)}")
st.sidebar.write(all_coins[:50])
if len(all_coins) == 0:
    st.sidebar.error('No coins found in CSV')
selected_coin = st.sidebar.selectbox('Select coin for history', options=all_coins, index=0 if len(all_coins)>0 else None)

# Main layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader(f"Top {len(filtered)} — Snapshot @ {latest_ts if 'latest_ts' in locals() else 'N/A'}")
    st.dataframe(filtered[['Rank','Coin Name','Price','24h Change','Market Cap','Timestamp']].sort_values('Rank'))
    # Price bar chart
    fig = px.bar(filtered.sort_values('Rank'), x='Coin Name', y='Price_val', color='Change_val', color_continuous_scale='RdYlGn', labels={'Price_val':'Price (USD)','Change_val':'24h %'}, height=400)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader('Market Cap (log scale)')
    try:
        snapshot['Market_val'] = snapshot['Market Cap'].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
        fig2 = px.bar(snapshot.sort_values('Rank'), x='Coin Name', y='Market_val', log_y=True, labels={'Market_val':'Market Cap (USD)'} , height=420)
        st.plotly_chart(fig2, width='stretch')
    except Exception:
        st.write('Market cap parse error — showing raw values')
        st.write(snapshot[['Coin Name','Market Cap']])

# Historical view for selected coin
st.markdown('---')
st.subheader(f'Price History — {selected_coin}')
coin_history = df[df['Coin Name'] == selected_coin].copy()
if coin_history.empty:
    st.write('No history available for this coin in the CSV.')
else:
    if 'Timestamp' in coin_history.columns:
        coin_history['Timestamp'] = pd.to_datetime(coin_history['Timestamp'])
        coin_history = coin_history.sort_values('Timestamp')
        coin_history['Price_val'] = coin_history['Price'].apply(parse_price)
        fig3 = px.line(coin_history, x='Timestamp', y='Price_val', markers=True, labels={'Price_val':'Price (USD)'})
        st.plotly_chart(fig3, width='stretch')
    else:
        st.write('Timestamps not available; cannot plot history.')

st.caption('Dashboard reads from `cryptocurrency_prices.csv`. Use the notebook scraper to append new snapshots.')
