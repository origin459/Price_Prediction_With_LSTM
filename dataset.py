import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
import joblib

def fetch_and_process_data(user_key, secret_key):
    binance_client = Client(user_key, secret_key)

    df = pd.DataFrame(binance_client.futures_historical_klines(
        symbol='BTCUSDT',
        interval='5m',
        start_str='2023-02-19',
        end_str='2024-02-19'
    ), columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])

    # Convert timestamp from ms to a datetime object for better readability
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

    # Function to calculate RSI
    def calculate_rsi(data, window):
        delta = data['close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        average_gain = up.rolling(window=window).mean()
        average_loss = abs(down.rolling(window=window).mean())
        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Calculate indicators
    df['rsi'] = calculate_rsi(df, 14)
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()

    # Shift feature variables to reflect the previous 5-minute cycle
    df[['open', 'high', 'low', 'volume', 'rsi', 'ma5', 'ma10', 'volume_ma5']] = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'ma5', 'ma10', 'volume_ma5']].shift(1)

    # Drop any rows with NaN values that were introduced by shifting
    df = df.dropna()

    # Save the processed data to a CSV file
    df.to_csv('BTC_5m.csv', index=False) 

    x = df[['open', 'high', 'low', 'volume', 'rsi', 'ma5', 'ma10', 'volume_ma5']]
    y = df['close']

    # Scale the feature data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.pkl')

    # Convert to DataFrame
    x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)

    # Concatenate features and target into a single DataFrame
    data = pd.concat([x_scaled_df, y], axis=1)

    # Convert DataFrame to NumPy array
    data_array = data.to_numpy()

    # Split features and target
    features = data_array[:, :-1]  # All columns except the last one
    target = data_array[:, -1]     # Last column

    return features, target