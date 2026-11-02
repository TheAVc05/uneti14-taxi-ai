from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import make_features

st.set_page_config(page_title="UNETI14 Taxi AI", page_icon="🚕", layout="centered")
st.title("🚕 NYC Taxi Trip Duration Predictor")
st.caption("UNETI Olympic AI 2026 · UNETI14 · Mô hình LightGBM")

MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

with st.form("trip_form"):
    c1, c2 = st.columns(2)
    with c1:
        vendor_id = st.selectbox("Nhà vận tải (vendor_id)", [1, 2])
        passenger_count = st.number_input("Số hành khách", min_value=1, max_value=8, value=1)
        pickup_month = st.selectbox("Tháng đón khách", list(range(1, 13)), index=5)
        pickup_dayofweek = st.selectbox("Thứ trong tuần (0–6)", list(range(7)), index=2)
        pickup_hour = st.slider("Giờ đón khách", 0, 23, 8)
        pickup_minute_bucket = st.selectbox("Mốc phút", [0, 15, 30, 45])
    with c2:
        pickup_lon_approx = st.number_input("Kinh độ điểm đón", value=-73.98, format="%.2f")
        pickup_lat_approx = st.number_input("Vĩ độ điểm đón", value=40.75, format="%.2f")
        dropoff_lon_approx = st.number_input("Kinh độ điểm đến", value=-73.96, format="%.2f")
        dropoff_lat_approx = st.number_input("Vĩ độ điểm đến", value=40.78, format="%.2f")
        straight_line_km_approx = st.number_input("Khoảng cách đường chim bay (km)", min_value=0.0, value=3.0, step=0.5)
        store_and_fwd_flag = st.selectbox("store_and_fwd_flag", ["N", "Y"])
    submitted = st.form_submit_button("Dự đoán thời gian")

if submitted:
    row = pd.DataFrame([{
        "vendor_id": vendor_id, "pickup_month": pickup_month,
        "pickup_dayofweek": pickup_dayofweek, "pickup_hour": pickup_hour,
        "pickup_minute_bucket": pickup_minute_bucket,
        "passenger_count": passenger_count,
        "pickup_lon_approx": pickup_lon_approx,
        "pickup_lat_approx": pickup_lat_approx,
        "dropoff_lon_approx": dropoff_lon_approx,
        "dropoff_lat_approx": dropoff_lat_approx,
        "straight_line_km_approx": straight_line_km_approx,
        "store_and_fwd_flag": store_and_fwd_flag
    }])
    model = load_model()
    pred = max(float(np.expm1(model.predict(make_features(row))[0])), 1.0)
    st.success(f"Thời gian dự đoán: **{pred:,.0f} giây** (khoảng **{pred/60:.1f} phút**)")
    st.caption("Kết quả do mô hình LightGBM đã huấn luyện tạo ra, không phải kết quả tra cứu có sẵn.")
