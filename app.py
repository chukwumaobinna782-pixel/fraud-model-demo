# app.py
import streamlit as st, pandas as pd, simulator, model, time, plotly.express as px

st.set_page_config(page_title="Real-time Fraud Detector", layout="wide")
st.title("🔮 XGBoost Fraud Detection Demo")
st.markdown("Click **Generate** to create a synthetic transaction and see the model score.")

if 'log' not in st.session_state:
    st.session_state.log = []

def add_to_log(row, prob):
    st.session_state.log.append({
        'time': pd.Timestamp.now().strftime('%H:%M:%S'),
        'amount': f"${row.amount_usd:.2f}",
        'country': row.ip_country,
        'card_present': bool(row.card_present),
        'prob': f"{prob:.2%}",
        'flag': '🟢' if prob < 0.3 else '🟡' if prob < 0.7 else '🔴'
    })
    if len(st.session_state.log) > 50:
        st.session_state.log.pop(0)

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🎲 Generate new transaction"):
        with st.spinner("Scoring…"):
            row = simulator.gen_tx()
            prob = model.predict_fraud(row.to_frame().T)[0]
            add_to_log(row, prob)
            st.success(f"Score: **{prob:.2%}**")

    st.subheader("📜 Recent predictions")
    st.dataframe(pd.DataFrame(st.session_state.log), use_container_width=True)

with col2:
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        fig = px.histogram(df, x='prob', color='flag',
                           title="Score distribution last 50 tx",
                           color_discrete_map={'🟢':'green','🟡':'orange','🔴':'red'})
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
with st.expander("Upload your own CSV (must contain same columns)"):
    upl = st.file_uploader("", type=['csv'])
    if upl:
        df_up = pd.read_csv(upl)
        df_up['trans_ts'] = pd.to_datetime(df_up['trans_ts'])
        probs = model.predict_fraud(df_up)
        df_up['fraud_probability'] = probs
        st.write(df_up)
        csv = df_up.to_csv(index=False)
        st.download_button("Download scored file", csv, 'scored.csv')
