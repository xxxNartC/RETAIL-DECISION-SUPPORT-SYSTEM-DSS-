import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict

def load_and_preprocess_rfm_segmentation(df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Load raw DataFrame và trả về RFM DataFrame với cột:
      CustomerID, LastPurchase, Recency, Frequency, Monetary, AvgSpend
    Hoặc None nếu dữ liệu không hợp lệ.
    """
    if df_raw is None or df_raw.empty:
        return None

    required = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']
    if not all(col in df_raw.columns for col in required):
        return None

    df = df_raw.dropna(subset=['CustomerID']).copy()
    df['CustomerID'] = df['CustomerID'].astype(str)
    df = df[df['Quantity'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    if df['InvoiceDate'].empty:
        return None

    ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        LastPurchase=('InvoiceDate', 'max'),
        Recency=('InvoiceDate', lambda x: (ref_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    # Giữ những khách có Frequency>0 và Monetary>0
    rfm = rfm[(rfm['Frequency'] > 0) & (rfm['Monetary'] > 0)]
    rfm['AvgSpend'] = (rfm['Monetary'] / rfm['Frequency']).round(2)
    return rfm


def compute_sse_segmentation(rfm_df: pd.DataFrame, max_k: int = 6) -> List[float]:
    """
    Tính SSE cho các k=1..max_k (hoặc tới số khách).
    Trả về list SSE.
    """
    if rfm_df is None or rfm_df.empty:
        return [0.0] * max_k

    X = StandardScaler().fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    sse: List[float] = []
    limit = min(max_k, len(rfm_df))
    for k in range(1, limit + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto' if k > 1 else 1)
        sse.append(km.fit(X).inertia_)
    # nếu limit < max_k, bổ sung 0 cho đủ độ dài
    if limit < max_k:
        sse.extend([0.0] * (max_k - limit))
    return sse


def cluster_rfm(rfm_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Phân cụm RFM thành k cụm. Nếu chỉ 1 khách, gán cluster=0.
    """
    if rfm_df is None or rfm_df.empty:
        return rfm_df

    if len(rfm_df) < 2:
        rfm_df['Cluster'] = 0
        return rfm_df

    X = StandardScaler().fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    rfm_df['Cluster'] = km.fit_predict(X)
    return rfm_df


def summarize_rfm(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary RFM và assign segments exactly như định nghĩa:
      - VIP      = cluster có Avg_Monetary cao nhất
      - Churn    = cluster có Avg_Monetary thấp nhất
      - Remaining clusters được gán nhãn Potential đa dạng theo số lượng
      - Nếu chỉ 1 cluster: 'General'
    """
    summary = (
        rfm_df.groupby('Cluster')
              .agg(
                  Avg_Recency=('Recency', 'mean'),
                  Avg_Frequency=('Frequency', 'mean'),
                  Avg_Monetary=('Monetary', 'mean'),
                  Customers=('CustomerID', 'count')
              )
              .round(1)
              .reset_index()
    )

    temp = summary.sort_values('Avg_Monetary', ascending=False).reset_index(drop=True)
    segment_map: Dict[int, str] = {}

    if len(temp) >= 2:
        vip = temp.loc[0, 'Cluster']
        churn = temp.loc[len(temp)-1, 'Cluster']
        segment_map[vip] = 'VIP'
        segment_map[churn] = 'Churn'

        middle = temp[~temp['Cluster'].isin([vip, churn])].copy().reset_index(drop=True)
        # Sắp xếp middle để gán Potential chi tiết
        middle = middle.sort_values(
            by=['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary'],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        n = len(middle)
        if n == 1:
            segment_map[middle.loc[0,'Cluster']] = 'Potential'
        elif n == 2:
            segment_map[middle.loc[0,'Cluster']] = 'Active Potential'
            segment_map[middle.loc[1,'Cluster']] = 'Dormant Potential'
        elif n == 3:
            segment_map[middle.loc[0,'Cluster']] = 'High-Value Potential'
            segment_map[middle.loc[1,'Cluster']] = 'Engaged Potential'
            segment_map[middle.loc[2,'Cluster']] = 'Needs Attention Potential'
        elif n == 4:
            segment_map[middle.loc[0,'Cluster']] = 'High-Value Potential'
            segment_map[middle.loc[1,'Cluster']] = 'Engaged Potential'
            segment_map[middle.loc[2,'Cluster']] = 'Regular Potential'
            segment_map[middle.loc[3,'Cluster']] = 'Needs Attention Potential'
        else:
            for _, row in middle.iterrows():
                segment_map[row['Cluster']] = 'Potential'
    else:
        # Chỉ 1 cluster
        segment_map[temp.loc[0,'Cluster']] = 'General'

    summary['Segment'] = summary['Cluster'].map(segment_map)
    return summary
