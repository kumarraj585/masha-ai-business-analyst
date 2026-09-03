import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. LOAD FULL RETAIL DATA
# ============================================================

def load_full_data():

    file_path = "data/retail_raw/online_retail_II.xlsx"

    df1 = pd.read_excel(
        file_path,
        sheet_name="Year 2009-2010",
    )

    df2 = pd.read_excel(
        file_path,
        sheet_name="Year 2010-2011",
    )

    df = pd.concat(
        [df1, df2],
        ignore_index=True,
    )

    return df


# ============================================================
# 2. CREATE DAILY REVENUE
# ============================================================

def get_daily_revenue(df):

    df = df.copy()

    # Convert date
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"]
    )

    # Revenue = Quantity × Price
    df["Revenue"] = (
        df["Quantity"] * df["Price"]
    )

    # Keep valid sales
    df = df[
        (df["Quantity"] > 0)
        & (df["Price"] > 0)
    ]

    # Remove cancelled invoices
    df = df[
        ~df["Invoice"].astype(str).str.startswith(
            ("C", "c")
        )
    ]

    # Convert timestamp to calendar date
    df["Date"] = (
        df["InvoiceDate"]
        .dt.normalize()
    )

    # Aggregate revenue by date
    daily_revenue = (
        df.groupby("Date")["Revenue"]
        .sum()
        .sort_index()
    )

    # Create complete calendar date range
    full_date_range = pd.date_range(
        start=daily_revenue.index.min(),
        end=daily_revenue.index.max(),
        freq="D",
    )

    # Missing dates = zero sales
    daily_revenue = (
        daily_revenue
        .reindex(
            full_date_range,
            fill_value=0,
        )
    )

    daily_revenue.name = "Revenue"

    return daily_revenue


# ============================================================
# 3. CREATE FORECASTING FEATURES
# ============================================================

def create_features(daily_revenue):

    data = daily_revenue.reset_index()

    data.columns = [
        "Date",
        "Revenue",
    ]

    # Calendar features
    data["DayOfWeek"] = (
        data["Date"].dt.dayofweek
    )

    data["DayOfMonth"] = (
        data["Date"].dt.day
    )

    data["Month"] = (
        data["Date"].dt.month
    )

    data["Year"] = (
        data["Date"].dt.year
    )

    # Lag features
    data["Lag_1"] = (
        data["Revenue"].shift(1)
    )

    data["Lag_7"] = (
        data["Revenue"].shift(7)
    )

    data["Lag_14"] = (
        data["Revenue"].shift(14)
    )

    # Rolling features
    data["Rolling_7"] = (
        data["Revenue"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    data["Rolling_14"] = (
        data["Revenue"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    return data


# ============================================================
# 4. EVALUATE MODEL
# ============================================================

def evaluate_model(daily_revenue):

    data = create_features(
        daily_revenue
    )

    data = (
        data
        .dropna()
        .reset_index(drop=True)
    )

    features = [
        "DayOfWeek",
        "DayOfMonth",
        "Month",
        "Year",
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Rolling_7",
        "Rolling_14",
    ]

    # Chronological train/test split
    split_index = int(
        len(data) * 0.80
    )

    train = data.iloc[
        :split_index
    ]

    test = data.iloc[
        split_index:
    ]

    X_train = train[features]
    y_train = train["Revenue"]

    X_test = test[features]
    y_test = test["Revenue"]

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # Train
    model.fit(
        X_train,
        y_train,
    )

    # Predict
    y_pred = model.predict(
        X_test
    )

    # Metrics
    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred,
        )
    )

    r2 = r2_score(
        y_test,
        y_pred,
    )

    return (
        model,
        mae,
        rmse,
        r2,
    )


# ============================================================
# 5. TRAIN FINAL MODEL
# ============================================================

def train_final_model(
    daily_revenue
):

    data = create_features(
        daily_revenue
    )

    data = (
        data
        .dropna()
        .reset_index(drop=True)
    )

    features = [
        "DayOfWeek",
        "DayOfMonth",
        "Month",
        "Year",
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Rolling_7",
        "Rolling_14",
    ]

    X = data[features]
    y = data["Revenue"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# 6. FORECAST NEXT CALENDAR MONTH
# ============================================================

def forecast_next_month(
    daily_revenue
):

    model = train_final_model(
        daily_revenue
    )

    features = [
        "DayOfWeek",
        "DayOfMonth",
        "Month",
        "Year",
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Rolling_7",
        "Rolling_14",
    ]

    history = daily_revenue.copy()

    # Last historical month
    last_historical_month = (
        history.index[-1]
        .to_period("M")
    )

    # Next calendar month
    next_month_period = (
        last_historical_month + 1
    )

    # First day of next month
    first_future_date = (
        next_month_period
        .to_timestamp()
    )

    # Last day of next month
    last_future_date = (
        next_month_period
        .to_timestamp(
            how="end"
        )
        .normalize()
    )

    # Every date in next month
    future_dates = pd.date_range(
        start=first_future_date,
        end=last_future_date,
        freq="D",
    )

    predictions = []

    # Recursive forecasting
    for future_date in future_dates:

        revenue_history = (
            history.tolist()
        )

        lag_1 = revenue_history[-1]

        lag_7 = revenue_history[-7]

        lag_14 = revenue_history[-14]

        rolling_7 = np.mean(
            revenue_history[-7:]
        )

        rolling_14 = np.mean(
            revenue_history[-14:]
        )

        future_features = pd.DataFrame(
            {
                "DayOfWeek": [
                    future_date.dayofweek
                ],
                "DayOfMonth": [
                    future_date.day
                ],
                "Month": [
                    future_date.month
                ],
                "Year": [
                    future_date.year
                ],
                "Lag_1": [
                    lag_1
                ],
                "Lag_7": [
                    lag_7
                ],
                "Lag_14": [
                    lag_14
                ],
                "Rolling_7": [
                    rolling_7
                ],
                "Rolling_14": [
                    rolling_14
                ],
            }
        )

        prediction = model.predict(
            future_features[features]
        )[0]

        # Prevent negative revenue
        prediction = max(
            0,
            float(prediction)
        )

        predictions.append(
            {
                "Date": future_date,
                "Predicted_Revenue": prediction,
            }
        )

        # Add prediction to history
        history.loc[
            future_date
        ] = prediction

    forecast_df = pd.DataFrame(
        predictions
    )

    total_forecast = (
        forecast_df[
            "Predicted_Revenue"
        ].sum()
    )

    return (
        forecast_df,
        next_month_period,
        total_forecast,
    )


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    print("========================================")
    print("MASHA DAILY SALES FORECAST")
    print("========================================")

    # Load full dataset
    df = load_full_data()

    print("\nFull dataset shape:")
    print(df.shape)

    # Create daily revenue
    daily_revenue = get_daily_revenue(
        df
    )

    print("\n========================================")
    print("DAILY SALES DATA")
    print("========================================")

    print(
        "First date:",
        daily_revenue.index.min()
    )

    print(
        "Last date:",
        daily_revenue.index.max()
    )

    print(
        "Number of calendar days:",
        len(daily_revenue)
    )

    # Evaluate model
    (
        model,
        mae,
        rmse,
        r2,
    ) = evaluate_model(
        daily_revenue
    )

    print("\n========================================")
    print("RANDOM FOREST MODEL EVALUATION")
    print("========================================")

    print(
        f"MAE: £{mae:,.2f}"
    )

    print(
        f"RMSE: £{rmse:,.2f}"
    )

    print(
        f"R² Score: {r2:.2f}"
    )

    # Forecast next month
    (
        forecast_df,
        month,
        total_forecast,
    ) = forecast_next_month(
        daily_revenue
    )

    print("\n========================================")
    print("NEXT CALENDAR MONTH FORECAST")
    print("========================================")

    print(
        "Month:",
        month
    )

    print(
        f"Predicted revenue: "
        f"£{total_forecast:,.2f}"
    )

    print(
        "\nNumber of forecast days:",
        len(forecast_df)
    )

    print("\nDaily forecast:")

    print(
        forecast_df.to_string(
            index=False
        )
    )


# ============================================================
# RUN MAIN ONLY WHEN EXECUTED DIRECTLY
# ============================================================

if __name__ == "__main__":
    main()