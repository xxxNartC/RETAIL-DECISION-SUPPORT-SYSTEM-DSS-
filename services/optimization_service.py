import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

@st.cache_data
def preprocess_optimization_data(df_raw: pd.DataFrame, keyword: str, months_forecast: int) -> pd.DataFrame | None:
    if df_raw is None or df_raw.empty:
        st.warning("Không có dữ liệu thô để xử lý tối ưu hóa. Vui lòng tải file lên.")
        return None

    required = ['Description', 'Quantity', 'UnitPrice', 'InvoiceNo']
    if not all(col in df_raw.columns for col in required):
        st.error(f"File CSV phải chứa cột: {', '.join(required)}.")
        return None

    df = df_raw.dropna(subset=required).copy()
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Description'].str.contains(keyword, case=False, na=False)]
    df = df[df['Quantity'] > 0]
    if df.empty:
        st.warning(f"Không tìm thấy sản phẩm nào chứa từ khóa '{keyword}' hoặc dữ liệu không hợp lệ sau lọc.")
        return None

    grouped = df.groupby('Description').agg({
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    }).reset_index()

    grouped['Quantity'] = np.ceil(grouped['Quantity'] / 12 * months_forecast).astype(int)
    grouped['ProfitPerUnit'] = grouped['UnitPrice'] * 0.40
    return grouped

def run_optimization(
    data: pd.DataFrame,
    budget: float
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    # chuẩn bị
    c = -data['ProfitPerUnit'].values
    A = [data['UnitPrice'].values]
    b = [budget]

    bounds = []
    for i in range(len(data)):
        max_demand = data.loc[i, 'Quantity']
        max_diversify = (budget * 0.4) // data.loc[i, 'UnitPrice'] if data.loc[i, 'UnitPrice'] > 0 else np.inf
        bounds.append((0, min(max_demand, max_diversify)))

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if not res.success:
        raise ValueError("Không tìm được phương án tối ưu.")

    data = data.copy()
    data['OrderQty'] = np.round(res.x).astype(int)
    data = data[data['OrderQty'] > 0].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

    data['TotalCost'] = data['OrderQty'] * data['UnitPrice']
    data['ExpectedProfit'] = data['OrderQty'] * data['ProfitPerUnit']

    total_cost = data['TotalCost'].sum()
    total_profit = data['ExpectedProfit'].sum()
    sorted_df = data.sort_values('ExpectedProfit', ascending=False).reset_index(drop=True)
    top5 = sorted_df.head(5)
    return sorted_df, top5, total_cost, total_profit

def build_decision_data(
    df_result: pd.DataFrame,
    top5: pd.DataFrame,
    total_cost: float,
    total_profit: float,
    budget: float,
    months: int
) -> pd.DataFrame:
    # tính toán
    profit_margin = (total_profit / total_cost * 100) if total_cost > 0 else 0
    if profit_margin >= 95:
        budget_next = budget * 1.15
    else:
        budget_next = budget
    add_profit = (budget_next - budget) * (total_profit / total_cost) if total_cost > 0 else 0
    expected_next = total_profit + add_profit
    avg_monthly = total_cost / months if months > 0 else 0
    avg_weekly = avg_monthly / 4

    decision = {
        "Mục tiêu": [
            "1. Phân bổ ngân sách theo lợi nhuận",
            "2. Quản lý rủi ro dòng tiền",
            "3. Theo dõi hiệu suất tài chính",
            "4. Hoạch định thanh toán và dòng tiền",
            "5. Đề xuất tài chính kỳ sau"
        ],
        "Hành động tài chính cụ thể được đề xuất": [
            f"Phê duyệt nhập hàng {len(df_result)} mặt hàng với tổng chi £{total_cost:,.0f}/£{budget:,.0f}",
            f"Giữ lại ~£{budget - total_cost:,.0f} làm dự phòng tài chính linh hoạt",
            f"Lập báo cáo lợi nhuận theo tuần trong {months} tháng, so sánh với kỳ vọng £{total_profit:,.0f}",
            f"Thanh toán cho nhà cung cấp: trung bình £{avg_monthly:,.0f}/tháng (~£{avg_weekly:,.0f}/tuần)",
            f"Với tỷ suất lợi nhuận {profit_margin:.2f}%, đề xuất ngân sách kỳ tới: £{budget_next:,.0f}, lợi nhuận ~£{expected_next:,.0f}"
        ]
    }
    return pd.DataFrame(decision)
