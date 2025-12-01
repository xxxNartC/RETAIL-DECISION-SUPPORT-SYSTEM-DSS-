import streamlit as st
from services.segmentation_service import (
    load_and_preprocess_rfm_segmentation,
    compute_sse_segmentation,
    cluster_rfm,
    summarize_rfm
)
from views.segmentation_view import (
    render_elbow_chart,
    render_summary_table,
    render_proposals,
    render_details
)

def segmentation_flow(df_raw):
    st.header("📈 Mô hình: Phân khúc khách hàng (Customer Segmentation)")

    # --- Sidebar inputs ---
    k = st.sidebar.number_input(
        "Số nhóm (k)", min_value=2, max_value=6, value=3,
        key="k_segmentation_input",
        help="Chọn số phân khúc từ 2 đến 6. Số nhóm càng nhiều, phân tích càng chi tiết nhưng có thể phức tạp hơn."
    )
    threshold = st.sidebar.number_input(
        "Ngưỡng VIP (Monetary ≥)", min_value=0, max_value=1_000_000, value=500,
        key="threshold_segmentation_input",
        help="Tổng chi tiêu tối thiểu để một khách hàng được xem xét là VIP. Ví dụ: 500 (£)."
    )
    show_elbow = st.sidebar.checkbox(
        "Hiển thị Elbow Chart", False,
        key="show_elbow_segmentation_checkbox",
        help="Biểu đồ này giúp bạn xác định số nhóm tối ưu cho dữ liệu của mình."
    )

    # 1) Load & preprocess dữ liệu RFM
    rfm = load_and_preprocess_rfm_segmentation(df_raw)
    if rfm is None or rfm.empty:
        st.warning("Không đủ dữ liệu hợp lệ để phân tích phân khúc khách hàng.")
        st.stop()

    # 2) Nếu k > số khách hiện có, cảnh báo & điều chỉnh
    if k > len(rfm):
        st.warning(
            f"Số nhóm (k) đã chọn ({k}) lớn hơn số khách hàng hiện có ({len(rfm)}). "
            f"Tự điều chỉnh k xuống {len(rfm)}."
        )
        k = len(rfm)
        if k < 2:
            st.error("Không đủ khách hàng để phân cụm. Vui lòng tải lên dữ liệu có ít nhất 2 khách hàng.")
            st.stop()

    # 3) Tính SSE cho Elbow Chart (nếu được tick)
    sse = compute_sse_segmentation(rfm) if show_elbow else None

    # 4) Phân cụm với k đã chọn
    rfm_c = cluster_rfm(rfm, k)

    # 5) Tóm tắt & gán nhãn (logic nằm trong summarize_rfm)
    summary = summarize_rfm(rfm_c)

    # 6) Hiển thị các tab kết quả
    tab1, tab2, tab3 = st.tabs(["Tóm tắt", "Đề xuất", "Chi tiết khách"])

    with tab1:
        if show_elbow:
            render_elbow_chart(sse, k)
        render_summary_table(summary)

    with tab2:
        render_proposals(summary)

    with tab3:
        render_details(rfm_c, summary)
