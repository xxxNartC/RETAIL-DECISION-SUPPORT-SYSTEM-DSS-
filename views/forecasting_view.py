import streamlit as st
import pandas as pd
from services.forecasting_service import ForecastModel

def render_setup_tab(df_raw, container):
    with container:
        st.header("Thiết lập mô hình")
        st.markdown("""
        💡 **Hướng dẫn sử dụng:**
        - **Tải file CSV**: Dữ liệu cần có 4 cột: Description, Quantity, UnitPrice, InvoiceDate.
        - **Từ khóa sản phẩm**: Hệ thống sẽ lọc những sản phẩm liên quan để dự báo.
        - **Số tháng phân tích**: Chọn dữ liệu gần đây để hệ thống học xu hướng.
        - **Dự báo trong bao lâu**: Số tháng tương lai cần hệ thống ước lượng doanh thu.
        - **Chi phí vốn**: Số tiền bạn chi để nhập 1 đơn vị sản phẩm.
        - **Ngưỡng MAPE**: Mức sai số tối đa bạn chấp nhận, để hệ thống tự chọn mô hình phù hợp.
        """)

        df_forecast = df_raw
        if df_forecast is not None:
            with st.sidebar:
                st.markdown("### Thiết lập mô hình")
                forecast_keyword = st.text_input(
                    "Từ khóa sản phẩm",
                    value=st.session_state.get("forecast_keyword", "CANDLE"),
                    key="forecast_keyword_input"
                )
                st.session_state["forecast_keyword"] = forecast_keyword

                forecast_history_months = st.selectbox(
                    "Số tháng phân tích", [12, 18, 24], index=0, key="forecast_history_months_select"
                )
                forecast_months = st.selectbox(
                    "Dự báo trong bao lâu (Tháng)", [3, 6, 12], index=1, key="forecast_months_select"
                )
                forecast_capital_cost = st.number_input(
                    "Chi phí vốn / đơn vị (£)", min_value=0.0, value=1.0, key="forecast_capital_cost_input"
                )
                st.caption("Ví dụ: nếu bạn mua 1 sp giá £1.00 → nhập 1.0")

                forecast_mape_threshold = st.number_input(
                    "Ngưỡng MAPE chấp nhận (%)", min_value=0.0, value=15.0, key="forecast_mape_threshold_input"
                )
                st.caption("MAPE càng thấp thì mô hình càng chính xác. <15% là đáng tin cậy")

                run_forecast = st.button("Chạy dự báo", key="run_forecast_button")

            # Danh sách sản phẩm chứa từ khóa
            filtered = df_forecast[
                df_forecast["Description"].str.contains(forecast_keyword, case=False, na=False)
            ]
            with st.expander("Danh sách sản phẩm chứa từ khóa", expanded=True):
                if not filtered.empty:
                    df_show = (
                        filtered[["Description","Quantity","UnitPrice"]]
                        .drop_duplicates().reset_index(drop=True)
                    )
                    st.dataframe(df_show, use_container_width=True)
                    st.caption(f"Tìm thấy {df_show.shape[0]} sản phẩm chứa từ khóa “{forecast_keyword}”.")
                else:
                    st.warning("Không tìm thấy sản phẩm nào khớp với từ khóa.")

            if run_forecast:
                model = ForecastModel(
                    df_forecast,
                    forecast_keyword,
                    forecast_history_months,
                    forecast_months,
                    forecast_capital_cost,
                    forecast_mape_threshold
                )
                if model.preprocess():
                    model.forecast("ARIMA")
                    if model.mape > forecast_mape_threshold:
                        st.warning(f"⚠️ MAPE {model.mape:.2f}% vượt ngưỡng. Chuyển sang SARIMA…")
                        model.forecast("SARIMA")
                        if model.mape > forecast_mape_threshold:
                            st.warning("⚠️ SARIMA vẫn chưa đạt yêu cầu. Chuyển sang Prophet…")
                            model.forecast("PROPHET")

                    st.session_state["forecast_model_instance"] = model
                    st.session_state["forecast_run_triggered"] = True
                    st.success(f"✅ Dự báo hoàn tất bằng mô hình {model.model_name}")
                    st.rerun()
                else:
                    st.session_state["forecast_model_instance"] = None
                    st.session_state["forecast_run_triggered"] = False
        else:
            st.info("📂 Vui lòng tải lên file CSV ở đầu sidebar để bắt đầu.")


def render_results_tab(container):
    with container:
        st.header("Kết quả dự báo")
        model = st.session_state.get("forecast_model_instance")
        triggered = st.session_state.get("forecast_run_triggered", False)

        if triggered and model:
            # Metrics
            st.metric("Tổng doanh thu dự báo", f"£{model.total_revenue:,.2f}")
            st.metric("Lợi nhuận gộp ước lượng", f"£{model.gross_profit:,.2f}")
            st.metric("MAPE", f"{model.mape:.2f}%")
            st.markdown(f"Mô hình đang sử dụng: **{model.model_name}**")

            # Chart + bảng chi tiết
            chart_data = model.get_chart_data()
            if not chart_data.empty:
                st.line_chart(chart_data)
                with st.expander("📅 Bảng doanh thu dự báo chi tiết theo tháng"):
                    df_tab = chart_data.reset_index().rename(
                        columns={"index":"Tháng","Forecast":"Doanh thu dự báo"}
                    )
                    st.dataframe(df_tab, use_container_width=True)

                # Phân tích biểu đồ
                st.markdown("### 📌 Phân tích biểu đồ doanh thu dự báo")
                first = chart_data.iloc[0,0]
                last  = chart_data.iloc[-1,0]
                delta = last - first
                pct   = (delta/first*100) if first!=0 else 0
                if delta > 0:
                    st.success(f"📈 Doanh thu có xu hướng tăng {pct:.1f}%.")
                elif delta < 0:
                    st.error(f"📉 Doanh thu giảm {abs(pct):.1f}%.")
                else:
                    st.info("🔵 Doanh thu ổn định.")

                # Sơ đồ luồng
                st.markdown("### 📊 Sơ đồ luồng mô hình dự báo")
                st.code(
                    "Người dùng nhập dữ liệu → ARIMA → (nếu MAPE > ngưỡng) → SARIMA → (nếu vẫn > ngưỡng) → Prophet",
                    language=None
                )

                # Phân tích chuyên sâu
                st.markdown(f"#### 🔍 Phân tích chuyên sâu: Mô hình {model.model_name}")
                if model.model_name == "ARIMA":
                    st.markdown("""
✔ Mô hình ARIMA được sử dụng vì dữ liệu có xu hướng ổn định, không có biến động theo mùa rõ rệt.  
➤ Bạn có thể dựa vào dự báo này để lập kế hoạch nhập hàng đều đặn theo tháng.
""")
                elif model.model_name == "SARIMA":
                    st.markdown("""
🔁 Mô hình SARIMA được sử dụng vì dữ liệu có yếu tố mùa vụ rõ ràng (ví dụ: doanh số tăng vào tháng lễ).  
➤ Bạn nên chú trọng nhập hàng và marketing vào các tháng cao điểm.
""")
                else:  # PROPHET
                    st.markdown("""
📈 Mô hình Prophet được áp dụng vì dữ liệu có trend + seasonality + biến động phức tạp.  
➤ Theo dõi sát để điều chỉnh nhập hàng linh hoạt.
""")

                # 🗓️ Gợi ý theo từng tháng từ biểu đồ
                st.subheader(f"🗓️ Gợi ý theo từng tháng từ biểu đồ {model.model_name}")
                monthly_forecast = model.forecast_series
                avg_forecast = monthly_forecast.mean()
                for month, value in monthly_forecast.items():
                    if value >= avg_forecast * 1.1:
                        status = "🔺 Cao điểm"
                        suggestion = "👉 Tăng nhập hàng, đẩy mạnh quảng bá."
                    elif value <= avg_forecast * 0.9:
                        status = "🔻 Thấp điểm"
                        suggestion = "👉 Giảm nhập hàng, khuyến mãi để kích cầu."
                    else:
                        status = "🟢 Ổn định"
                        suggestion = "👉 Giữ kế hoạch nhập hàng hiện tại."
                    st.markdown(
                        f"**{month.strftime('%B %Y')}**: Dự báo £{value:,.2f} → {status}  \n{suggestion}"
                    )
        else:
            st.info("Vui lòng chạy mô hình ở tab 'Thiết lập'")


def render_actions_tab(container):
    with container:
        st.header("Gợi ý hành động theo kết quả")
        model = st.session_state.get("forecast_model_instance")
        triggered = st.session_state.get("forecast_run_triggered", False)

        if triggered and model:
            past = model.get_last_month_sales()
            avg_rev = model.total_revenue / model.forecast_months if model.forecast_months>0 else 0
            avg_units = avg_rev / model.avg_unit_price if model.avg_unit_price>0 else 0
            gap = int(avg_units - past) if model.avg_unit_price>0 else 0

            st.markdown(f"**Tháng trước bán:** {past} SP")
            st.markdown(f"**Dự báo trung bình:** {int(avg_units)} SP/tháng")
            st.markdown(f"**Cần tăng thêm:** {gap if gap>0 else 0} SP")

            st.subheader("📌 Gợi ý hành động dành cho Bộ phận 📢 Marketing theo từng tháng")
            mean_hist = model.monthly.mean() if not model.monthly.empty else 0
            for month, value in model.forecast_series.items():
                deviation = value - mean_hist
                label = month.strftime('%B %Y')
                st.markdown(f"### 📅 {label}")
                st.markdown(f"- 📈 **Dự báo doanh thu:** £{value:,.2f}")
                if deviation > 0:
                    st.markdown(f"- 🔺 Cao hơn trung bình £{deviation:,.2f} → Tháng cao điểm.")
                    st.markdown("""
**🎯 Chiến lược Marketing gợi ý:**
- Tăng ngân sách quảng cáo.
- Tổ chức khuyến mãi hấp dẫn.
- Đẩy mạnh remarketing.
""")
                else:
                    st.markdown(f"- 🔻 Thấp hơn trung bình £{abs(deviation):,.2f} → Tháng trũng.")
                    st.markdown("""
**🎯 Chiến lược Marketing gợi ý:**
- Tập trung giữ khách hàng hiện tại.
- Tối ưu chi phí quảng cáo.
- Duy trì hiện diện thương hiệu.
""")
            st.subheader("💡 Gợi ý tự động từ mô hình")
            for s in model.get_suggestions():
                st.markdown(f"- {s}")
        else:
            st.info("Vui lòng chạy mô hình ở tab 'Thiết lập'")
