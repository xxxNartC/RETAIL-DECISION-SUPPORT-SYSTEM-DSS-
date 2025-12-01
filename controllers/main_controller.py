# controllers/main_controller.py
import streamlit as st
from dao.data_loader import load_raw_data
from controllers.segmentation_controller import segmentation_flow
from controllers.optimization_controller import optimization_flow
from controllers.forecasting_controller import forecasting_flow

def run_app():
    st.set_page_config(page_title="Dashboard DSS", layout="wide")
    st.title("Dashboard Hệ thống Hỗ trợ Quyết Định (DSS)")

    df_raw = load_raw_data()
    if df_raw is None:
        st.stop()

    choice = st.sidebar.radio(
        "Chọn Mô hình Phân tích",
        ("Phân khúc khách hàng", "Tối ưu lợi nhuận nhập hàng", "Dự báo Doanh thu nhóm sản phẩm")
    )
    if choice == "Phân khúc khách hàng":
        segmentation_flow(df_raw)
    elif choice == "Tối ưu lợi nhuận nhập hàng":
        optimization_flow(df_raw)
    else:
        forecasting_flow(df_raw)
