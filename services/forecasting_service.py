import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

class ForecastModel:
    def __init__(
        self,
        df_raw: pd.DataFrame,
        keyword: str,
        history_months: int,
        forecast_months: int,
        capital_cost: float,
        mape_threshold: float
    ):
        self.df_raw = df_raw.copy()
        self.keyword = keyword
        self.history_months = history_months
        self.forecast_months = forecast_months
        self.capital_cost = capital_cost
        self.mape_threshold = mape_threshold

        self.model_name: str = ""
        self.mape: float = 0.0
        self.total_revenue: float = 0.0
        self.gross_profit: float = 0.0
        self.avg_unit_price: float = 0.0

        self.monthly: pd.Series = pd.Series(dtype=float)
        self.forecast_series: pd.Series = pd.Series(dtype=float)

    def preprocess(self) -> bool:
        required = ['Description', 'Quantity', 'UnitPrice', 'InvoiceDate']
        if not all(col in self.df_raw.columns for col in required):
            return False

        df = self.df_raw.dropna(subset=required).copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df = df.dropna(subset=['InvoiceDate'])
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        df = df[df['Description'].str.contains(self.keyword, case=False, na=False)]
        if df.empty:
            return False

        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        total_qty = df['Quantity'].sum()
        self.avg_unit_price = (df['Revenue'].sum() / total_qty) if total_qty > 0 else 0.0

        df.set_index('InvoiceDate', inplace=True)
        monthly_rev = df['Revenue'].resample('M').sum()
        if monthly_rev.empty:
            return False

        # Chỉ giữ lịch sử history_months
        if len(monthly_rev) >= self.history_months:
            self.monthly = monthly_rev[-self.history_months:]
        else:
            self.monthly = monthly_rev
        return True

    def forecast(self, model_type: str):
        """
        1) Dự báo trên train để tính MAPE
        2) Refit trên toàn bộ lịch sử để forecast thật
        3) Tính total_revenue & gross_profit
        """
        # --- 1) split train/test để tính MAPE ---
        if len(self.monthly) > self.forecast_months:
            train = self.monthly[:-self.forecast_months]
            test  = self.monthly[-self.forecast_months:]
        else:
            train = self.monthly
            test  = pd.Series(dtype=float)

        mt = model_type.upper()
        # Forecast trên train
        if mt == "ARIMA":
            self._arima_forecast(train)
        elif mt == "SARIMA":
            self._sarima_forecast(train)
        elif mt == "PROPHET":
            self._prophet_forecast_from(train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Lưu lại kết quả dự báo trên train để tính MAPE
        mape_preds = self.forecast_series.copy()
        if not test.empty:
            self.mape = mean_absolute_percentage_error(test.values, mape_preds.values[:len(test)]) * 100
        else:
            self.mape = 0.0

        # Gán model_name
        self.model_name = mt

        # --- 2) refit trên full history để forecast thật ---
        if mt == "ARIMA":
            self._arima_forecast(self.monthly)
        elif mt == "SARIMA":
            self._sarima_forecast(self.monthly)
        else:  # PROPHET
            self._prophet_forecast()

        # --- 3) Tính tổng và profit ---
        self.total_revenue = float(self.forecast_series.sum())
        if self.avg_unit_price > 0:
            total_units     = self.total_revenue / self.avg_unit_price
            self.gross_profit = self.total_revenue - total_units * self.capital_cost
        else:
            self.gross_profit = 0.0

    def _arima_forecast(self, series: pd.Series):
        res = ARIMA(series, order=(1,1,1)).fit()
        fc  = res.forecast(steps=self.forecast_months)
        last = series.index[-1]
        idx  = pd.date_range(last + pd.offsets.MonthEnd(), periods=self.forecast_months, freq='M')
        self.forecast_series = pd.Series(fc.values, index=idx)

    def _sarima_forecast(self, series: pd.Series):
        res = SARIMAX(
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        fc  = res.forecast(steps=self.forecast_months)
        last = series.index[-1]
        idx  = pd.date_range(last + pd.offsets.MonthEnd(), periods=self.forecast_months, freq='M')
        self.forecast_series = pd.Series(fc.values, index=idx)

    def _prophet_forecast(self):
        dfp = self.monthly.reset_index().rename(columns={'InvoiceDate':'ds','Revenue':'y'}) \
              if 'InvoiceDate' in self.monthly.index.names \
              else self.monthly.reset_index(name='y').rename(columns={'index':'ds'})
        m = Prophet(); m.fit(dfp)
        future = m.make_future_dataframe(periods=self.forecast_months, freq='M')
        pred   = m.predict(future).set_index('ds')['yhat']
        self.forecast_series = pred[-self.forecast_months:]

    def _prophet_forecast_from(self, series: pd.Series):
        dfp = series.reset_index().rename(columns={'InvoiceDate':'ds','Revenue':'y'}) \
              if 'InvoiceDate' in series.index.names \
              else series.reset_index(name='y').rename(columns={'index':'ds'})
        m = Prophet(); m.fit(dfp)
        future = m.make_future_dataframe(periods=self.forecast_months, freq='M')
        pred   = m.predict(future).set_index('ds')['yhat']
        self.forecast_series = pred[-self.forecast_months:]

    def get_chart_data(self) -> pd.DataFrame:
        return self.forecast_series.to_frame(name='Forecast')

    def get_last_month_sales(self) -> int:
        if not self.monthly.empty and self.avg_unit_price > 0:
            rev = self.monthly.iloc[-1]
            return int(round(rev / self.avg_unit_price))
        return 0

    def get_suggestions(self) -> list[str]:
        suggestions = []
        try:
            x = np.arange(len(self.monthly))
            y = self.monthly.values
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                if slope > 0:
                    suggestions.append("Xu hướng doanh thu tăng – cân nhắc tăng kế hoạch nhập hàng.")
                elif slope < 0:
                    suggestions.append("Xu hướng doanh thu giảm – cân nhắc giảm nhập hoặc đẩy mạnh marketing.")
                else:
                    suggestions.append("Doanh thu ổn định – duy trì mức nhập hiện tại.")
        except:
            pass

        suggestions.append(f"Sử dụng MAPE ({self.mape:.2f}%) để đánh giá độ chính xác mô hình.")
        suggestions.append("Chạy lại mô hình khi có thêm dữ liệu mới để cải thiện độ chính xác.")
        return suggestions
