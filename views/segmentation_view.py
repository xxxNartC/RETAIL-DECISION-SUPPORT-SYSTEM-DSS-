import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def render_elbow_chart(sse: list[float], k: int):
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.subheader("Biểu đồ Elbow Method")
        st.markdown("""
        Biểu đồ này giúp bạn chọn số nhóm khách hàng hợp lý nhất.
        - **Trục ngang (Số nhóm k)**: Số lượng nhóm bạn muốn chia khách hàng.
        - **Trục dọc (Chỉ số gắn kết - SSE)**: Cho biết các khách hàng trong cùng một nhóm giống nhau đến mức nào (số càng nhỏ, nhóm càng gắn kết).
        """)
    with col2:
        with st.popover("ℹ️", help="Bấm để xem hướng dẫn chi tiết về Biểu đồ Elbow"):
             st.markdown("""
                    - **Bạn có thể hình dung biểu đồ này giống như một cánh tay đang gập lại.**
                    - **"Điểm khuỷu tay" (Elbow point)**:
                        - Khi bạn bắt đầu chia nhóm (từ 1 nhóm lên 2, 3 nhóm...), chỉ số "gắn kết" (SSE) sẽ **giảm rất nhanh**, vì các khách hàng được gom vào các nhóm phù hợp hơn.
                        - Nhưng đến một lúc nào đó, việc tăng thêm số nhóm sẽ không làm cho các nhóm "gắn kết" hơn nhiều nữa (chỉ số SSE giảm chậm lại). **Điểm mà đường cong bắt đầu phẳng hơn chính là "điểm khuỷu tay"**.
                    - **Tại sao "điểm khuỷu tay" quan trọng?**: Đây là điểm số nhóm tối ưu. Nó giúp bạn tìm ra số nhóm vừa đủ để phân biệt các loại khách hàng rõ ràng, mà không làm quá phức tạp mọi thứ.
                    - **Cách xem biểu đồ**: Hãy tìm vị trí trên đường cong mà nó giống như một "khuỷu tay" – nơi độ dốc giảm đột ngột rồi sau đó gần như đi ngang. Đường **màu đỏ** trên biểu đồ đánh dấu số nhóm (k) bạn đang chọn. Nếu đường đỏ này nằm gần "điểm khuỷu tay", đó là một lựa chọn tốt!
                    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(sse)+1), sse, marker='o')
    ax.axvline(k, color='red', linestyle='--', label=f'Chọn k={k}')
    ax.set_xlabel("Số nhóm (k)")
    ax.set_ylabel("Chỉ số gắn kết (SSE)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def render_summary_table(summary: pd.DataFrame):
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.subheader("Bảng Tóm tắt Phân khúc")
    with col2:
        with st.popover("ℹ️", help="Bấm để xem hướng dẫn chi tiết về Bảng Tóm tắt Phân khúc"):
            st.markdown("""
            Bảng này tóm tắt các đặc điểm chính của từng nhóm khách hàng:
            - **Cluster**: ID của nhóm (cụm).
            - **Segment**: Tên ý nghĩa của nhóm (VIP, Churn, Potential…).
            - **Avg_Recency (days)**: Số ngày trung bình kể từ lần mua cuối cùng.  
            - **Avg_Frequency (orders)**: Số đơn trung bình.  
            - **Avg_Monetary (£)**: Chi tiêu trung bình.  
            - **Customers**: Số khách hàng trong nhóm.
            """)
    st.dataframe(
        summary[['Cluster','Segment','Avg_Recency','Avg_Frequency','Avg_Monetary','Customers']],
        use_container_width=True
    )


def render_proposals(summary: pd.DataFrame):
    # Tiêu đề + popover giải thích nhãn phân khúc
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Đề xuất Hành động")
    with col2:
        with st.popover("ℹ️", help="Xem ý nghĩa các phân khúc"):
            # Bảng mapping full các nhãn → ghi chú tiếng Việt
            segment_desc = {
                "VIP": "Khách hàng cao cấp, chi tiêu rất nhiều và thường xuyên nhất.",
                "Churn": "Khách hàng có nguy cơ rời bỏ, mua rất ít hoặc đã ngừng tương tác.",
                "Potential": "Khách hàng tiềm năng chung – chi tiêu và tần suất trung bình, cần theo dõi để phát triển.",
                "Active Potential": "Tiềm năng tích cực – tương tác đều đặn, dễ dàng upsell hoặc giữ chân lâu dài.",
                "Dormant Potential": "Tiềm năng ngủ đông – trước đây có tương tác, giờ giảm dần, cần chiến dịch tái kích hoạt.",
                "High-Value Potential": "Tiềm năng giá trị cao – đơn hàng lớn, có khả năng trở thành VIP nếu tăng tần suất mua.",
                "Engaged Potential": "Tiềm năng gắn kết – thường xuyên tương tác, mua đều đặn nhưng giá trị đơn hàng chưa cao nhất.",
                "Regular Potential": "Tiềm năng thông thường – mua ổn định, giá trị và tần suất ở mức trung bình.",
                "Needs Attention Potential": "Tiềm năng cần chú ý – tương tác ít, chi tiêu thấp; cần ưu đãi đặc biệt để kích thích."
            }
            present = summary['Segment'].unique().tolist()
            md = "|🔖 Nhãn|📝 Ý nghĩa|\n|---|---|\n"
            for seg in present:
                md += f"|**{seg}**|{segment_desc.get(seg,'')}|\n"
            st.markdown(md)

    # Chi tiết đề xuất cho từng cluster
    df = summary.sort_values('Avg_Monetary', ascending=False)
    for _, r in df.iterrows():
        st.markdown(f"**Cluster {r.Cluster} – {r.Segment}** ({int(r.Customers)} khách)")
        st.write(f"- Recency trung bình: {r.Avg_Recency} ngày")
        st.write(f"- Frequency trung bình: {r.Avg_Frequency} đơn")
        st.write(f"- Monetary trung bình: £{r.Avg_Monetary}")

        # Toàn bộ các case giữ nguyên như code gốc
        if r.Segment == 'VIP':
            st.write("→ **Chiến lược**: Gửi ưu đãi độc quyền, quà tri ân, mời tham gia chương trình khách hàng thân thiết cao cấp, cá nhân hóa trải nghiệm mua sắm.")
        elif r.Segment == 'Churn':
            st.write("→ **Chiến lược**: Gửi mã giảm giá mạnh để kích hoạt lại (re-engagement coupon), khảo sát ý kiến để tìm hiểu nguyên nhân rời bỏ, chương trình tri ân quay lại.")
        elif r.Segment == 'Potential':
            st.write("→ **Chiến lược**: Gợi ý combo sản phẩm, ưu đãi nhẹ để upsell, khuyến khích mua sắm thường xuyên hơn.")
        elif r.Segment == 'Active Potential':
            st.write("→ **Chiến lược**: Khuyến khích mua hàng lặp lại, giới thiệu sản phẩm mới, tham gia chương trình khách hàng thân thiết cơ bản để tăng cường gắn kết.")
        elif r.Segment == 'Dormant Potential':
            st.write("→ **Chiến lược**: Gửi ưu đãi đặc biệt để tái kích hoạt, email nhắc nhở về sản phẩm đã xem hoặc bỏ giỏ hàng, khảo sát nhẹ về trải nghiệm gần đây.")
        elif r.Segment == 'High-Value Potential':
            st.write("→ **Chiến lược**: Gợi ý các sản phẩm cao cấp hơn, mời tham gia chương trình khách hàng thân thiết cấp độ cao, khuyến khích mua sắm với giá trị lớn hơn.")
        elif r.Segment == 'Engaged Potential':
            st.write("→ **Chiến lược**: Tăng cường tương tác qua email/SMS, giới thiệu các sản phẩm liên quan, cung cấp ưu đãi dựa trên lịch sử mua hàng để thúc đẩy chuyển đổi.")
        elif r.Segment == 'Regular Potential':
            st.write("→ **Chiến lược**: Đề xuất các gói combo, chương trình tích điểm, ưu đãi định kỳ để duy trì tần suất mua hàng và tăng giá trị đơn hàng trung bình.")
        elif r.Segment == 'Needs Attention Potential':
            st.write("→ **Chiến lược**: Gửi voucher giảm giá hấp dẫn, thông báo về chương trình khuyến mãi đặc biệt, liên hệ cá nhân nếu có thể để hiểu rõ hơn nhu cầu.")
        elif r.Segment == 'General':
            st.write("→ **Chiến lược**: Tổng quan về khách hàng, xem xét mở rộng dữ liệu hoặc điều chỉnh tham số để phân khúc rõ hơn.")
        else:
            st.write("→ **Chiến lược**: Đề xuất chung cho nhóm khách hàng này để khám phá nhu cầu và tăng cường tương tác.")
        st.write("---")


def render_details(rfm: pd.DataFrame, summary: pd.DataFrame):
    st.subheader("Chi tiết Khách hàng theo Cụm")
    options = {
        c: f"Cluster {c} – {summary.loc[summary['Cluster']==c,'Segment'].iloc[0]}"
        for c in summary['Cluster']
    }
    choice = st.selectbox(
        "Chọn cụm",
        options.keys(),
        format_func=lambda x: options[x]
    )
    details = (
        rfm[rfm['Cluster']==choice]
        [['CustomerID','LastPurchase','Recency','Frequency','Monetary','AvgSpend']]
        .sort_values('Monetary', ascending=False)
        .reset_index(drop=True)
    )
    st.write(f"**Tổng số khách hàng trong cụm này:** {len(details)}")
    st.dataframe(details, use_container_width=True)
