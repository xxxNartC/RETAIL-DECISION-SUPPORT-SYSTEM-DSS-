import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def render_sidebar_optimization():
    st.sidebar.subheader("Thông số Tối ưu nhập hàng")
    keyword = st.sidebar.text_input(
        "Từ khóa sản phẩm (VD: CANDLE)", value="CANDLE", key="optim_keyword_input"
    )
    budget = st.sidebar.number_input(
        "Ngân sách (£)", value=1000.0, min_value=0.0, key="optim_budget_input"
    )
    months = st.sidebar.number_input(
        "Dự báo nhu cầu cho bao nhiêu tháng tới? (tháng)",
        min_value=1, max_value=6, value=1, step=1, key="optim_months_input"
    )
    return keyword, budget, months

def render_preprocess_tab(processed: pd.DataFrame | None, months: int) -> bool:
    st.subheader("📥 Dữ liệu đầu vào & Tiền xử lý")
    st.info(
        "Dữ liệu được lấy từ file CSV và được lọc/tiền xử lý theo thông số sidebar."
    )
    if processed is not None and not processed.empty:
        st.dataframe(processed, use_container_width=True)
        st.success(f"Dữ liệu đã được tiền xử lý. Nhu cầu dự kiến cho {months} tháng tới.")
        run_pressed = st.button("🚀 Tối ưu nhập hàng", key="run_optimization_button")
    else:
        st.warning("Không có dữ liệu hợp lệ sau tiền xử lý.")
        run_pressed = False
    return run_pressed

def render_optimization_results_tab(
    df: pd.DataFrame,
    top5: pd.DataFrame,
    total_cost: float,
    total_profit: float
):
    st.subheader("📊 Kết quả tối ưu hóa lợi nhuận")
    if df.empty:
        st.warning("Không có sản phẩm nào được đề xuất nhập với số lượng > 0.")
        return

    st.dataframe(
        df[['Description','OrderQty','UnitPrice','TotalCost','ExpectedProfit']],
        use_container_width=True
    )
    st.markdown(f"💰 *Tổng chi phí đã dùng:* £{total_cost:,.2f}")
    st.markdown(f"📈 *Tổng lợi nhuận kỳ vọng:* £{total_profit:,.2f}")
    if total_cost > 0:
        st.markdown(f"📊 *Tỷ suất lợi nhuận:* {total_profit/total_cost*100:.2f}%")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(top5['Description'][::-1], top5['ExpectedProfit'][::-1])
    ax.set_xlabel("Lợi nhuận (£)")
    ax.set_title(f"🔝 Top {len(top5)} sản phẩm")
    st.pyplot(fig)

def render_decision_tab(
    df_result: pd.DataFrame,
    top5: pd.DataFrame,
    total_cost: float,
    total_profit: float,
    budget: float,
    months: int
):
    # 1) Forecast summary
    st.markdown(f"### Dự báo nhập hàng cho *{months} tháng tới*:")
    st.markdown(f"- *Tổng chi phí đã dùng:* £{total_cost:,.2f} / £{budget:,.2f}")
    st.markdown(f"- *Tổng lợi nhuận kỳ vọng:* £{total_profit:,.2f}")
    profit_margin = (total_profit / total_cost * 100) if total_cost > 0 else 0
    if total_cost > 0:
        st.markdown(f"- *Tỷ suất lợi nhuận:* {profit_margin:.2f}%")
    else:
        st.markdown("- *Tỷ suất lợi nhuận:* Không xác định (Tổng chi phí bằng 0)")

    # 2) Top-5 sản phẩm đề xuất
    st.markdown(f"### *Danh sách {len(top5)} sản phẩm nên ưu tiên nhập:*")
    for _, row in top5.iterrows():
        st.markdown(
            f"- {row['Description']}: Nhập {row['OrderQty']} đơn vị | "
            f"Lợi nhuận: £{row['ExpectedProfit']:.2f}"
        )

    # 3) Bảng Quyết định tài chính
    st.markdown(f"### Quyết định hỗ trợ – Bộ phận Tài chính (*{months} tháng tới*)")

    avg_monthly = total_cost / months if months > 0 else 0
    avg_weekly = avg_monthly / 4
    # nếu margin ≥95% thì tăng ngân sách 15%
    if profit_margin >= 95:
        budget_next = budget * 1.15
    else:
        budget_next = budget
    extra_profit = (budget_next - budget) * (total_profit / total_cost) if total_cost > 0 else 0
    expected_next = total_profit + extra_profit

    decision_data = {
        "Mục tiêu": [
            "1. Phân bổ ngân sách theo lợi nhuận",
            "2. Quản lý rủi ro dòng tiền",
            "3. Theo dõi hiệu suất tài chính",
            "4. Hoạch định thanh toán và dòng tiền",
            "5. Đề xuất tài chính kỳ sau"
        ],
        "Hành động tài chính cụ thể được đề xuất": [
            f"Phê duyệt nhập hàng đúng theo danh mục tối ưu "
            f"({len(df_result)} mặt hàng) với tổng chi là "
            f"£{total_cost:,.0f}/£{budget:,.0f}",
            
            f"Giữ lại ~£{budget - total_cost:,.0f} làm dự phòng tài chính linh hoạt "
            f"để tái đầu tư nếu sản phẩm nào vượt dự báo",
            
            f"Lập báo cáo lợi nhuận theo tuần trong {months} tháng, "
            f"so sánh với kỳ vọng £{total_profit:,.0f} để đánh giá hiệu suất từng sản phẩm",
            
            f"Thanh toán cho nhà cung cấp: trung bình £{avg_monthly:,.0f}/tháng "
            f"(~£{avg_weekly:,.0f}/tuần) để đảm bảo dòng tiền ổn định",
            
            f"Với tỷ suất lợi nhuận {profit_margin:.2f}%, đề xuất ngân sách kỳ tới: "
            f"£{budget_next:,.0f}, dự kiến lợi nhuận ~£{expected_next:,.0f}"
        ]
    }
    st.table(pd.DataFrame(decision_data))