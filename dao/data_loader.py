# dao/data_loader.py
import streamlit as st
import pandas as pd

def load_raw_data():
    st.sidebar.header("📥 Tải dữ liệu chung cho toàn bộ hệ thống")
    uploaded_file = st.sidebar.file_uploader(
        "Vui lòng tải lên file CSV chứa dữ liệu bán hàng của bạn",
        type=["csv"],
        help="File cần có các cột: CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice, Description."
    )
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            st.sidebar.success("✅ Đã tải dữ liệu thành công.")
            return df
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi khi đọc file: {e}. Vui lòng đảm bảo file là CSV hợp lệ và đúng định dạng.")
            return None
    else:
        st.sidebar.info("⬆️ Để bắt đầu, vui lòng tải lên một file CSV.")
        return None
