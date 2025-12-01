import streamlit as st
from services.optimization_service import (
    preprocess_optimization_data,
    run_optimization
)
from views.optimization_view import (
    render_sidebar_optimization,
    render_preprocess_tab,
    render_optimization_results_tab,
    render_decision_tab
)

def optimization_flow(df_raw):
    # Tiêu đề chính
    st.header("Mô hình: Tối ưu lợi nhuận nhập hàng (Linear Programming)")

    # 1) Sidebar inputs
    keyword, budget, months = render_sidebar_optimization()

    # 2) Các tab
    tab1, tab2, tab3 = st.tabs([
        "Nhập dữ liệu & Tiền xử lý",
        "Kết quả tối ưu",
        "Quyết định tài chính"
    ])

    # --- Tab 1: Nhập & tiền xử lý ---
    with tab1:
        processed = preprocess_optimization_data(df_raw, keyword, months)
        run_pressed = render_preprocess_tab(processed, months)
        if run_pressed:
            # Lưu vào session để qua tab 2
            st.session_state.optim_processed_data = processed
            st.session_state.optim_current_budget = budget
            st.session_state.optim_current_months = months
            st.session_state.optim_run_triggered = True
            st.rerun()

    # --- Tab 2: Kết quả tối ưu ---
    with tab2:
        if st.session_state.get("optim_run_triggered", False) and \
           st.session_state.get("optim_processed_data") is not None:

            data = st.session_state.optim_processed_data
            budget_state = st.session_state.optim_current_budget

            try:
                sorted_df, top5, total_cost, total_profit = run_optimization(
                    data, budget_state
                )
            except ValueError as e:
                st.error(f"❌ {e}")
                st.session_state.optim_run_triggered = False
                st.stop()

            # Lưu kết quả vào session để tab 3 dùng
            st.session_state.optim_result_data = sorted_df
            st.session_state.optim_result_top5 = top5
            st.session_state.optim_total_cost = total_cost
            st.session_state.optim_total_profit = total_profit

            render_optimization_results_tab(
                sorted_df, top5, total_cost, total_profit
            )
            st.session_state.optim_run_triggered = False

        else:
            st.info("💡 Vui lòng nhập dữ liệu ở tab 'Nhập dữ liệu & Tiền xử lý' và nhấn '🚀 Tối ưu nhập hàng'.")

    # --- Tab 3: Quyết định tài chính ---
    with tab3:
        if st.session_state.get("optim_result_data") is not None:
            render_decision_tab(
                st.session_state.optim_result_data,
                st.session_state.optim_result_top5,
                st.session_state.optim_total_cost,
                st.session_state.optim_total_profit,
                st.session_state.optim_current_budget,
                st.session_state.optim_current_months
            )
        else:
            st.info("💡 Hãy tối ưu nhập hàng trước để hiển thị quyết định.")
