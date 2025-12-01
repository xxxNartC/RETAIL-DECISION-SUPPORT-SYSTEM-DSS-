import streamlit as st
from views.forecasting_view import render_setup_tab, render_results_tab, render_actions_tab

def forecasting_flow(df_raw):
    st.header("Mô hình: Dự báo Doanh thu nhóm sản phẩm (Time Series Forecasting)")
    tabs = st.tabs(["Thiết lập", "Kết quả", "Hành động"])

    # Tab 0: Thiết lập
    render_setup_tab(df_raw, tabs[0])

    # Tab 1: Kết quả
    render_results_tab(tabs[1])

    # Tab 2: Hành động
    render_actions_tab(tabs[2])
