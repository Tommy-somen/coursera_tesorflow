  
import tensorflow as tf
print(tf.__version__)

# EXPECTED OUTPUT
# 2.0.0-beta1 (or later)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#時系列データの作成
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
#トレンドグラフの作成
def trend(time, slope=0):
    return slope * time
#周期的パターンの作成
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 7 * np.pi),
                    1 / np.exp(5 * season_time))
#周期性グラフの作成
def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)
#ノイズの作成
def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

#グラフ描画パラメータ
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.01
noise_level = 2

# Create the series：時系列データグラフの作成
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise：ノイズの追加
series += noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

# EXPECTED OUTPUT
# Chart as in the screencast. First should have 5 distinctive 'peaks'

#timeを1100:350でトレーニングセット：テストセットに分割する
split_time = 1100
#トレーニングのtime:value
time_train = time[:split_time]
x_train = series[:split_time]
#テスト用のtime:value
time_valid = time[split_time:]
x_valid = series[split_time:]
#トレーニングデータのグラフ描画
plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.show()

#テストデータのグラフ描画
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.show()

# EXPECTED OUTPUT
# Chart WITH 4 PEAKS between 50 and 65 and 3 troughs between -12 and 0
# Chart with 2 Peaks, first at slightly above 60, last at a little more than that, should also have a single trough at about 0

#last_timeを予測したいので、naive_forecastにはlast_timeの１つを除いたグラフデータをコピー
naive_forecast = series[split_time-1:-1] #次元数（データ数）を同一にするため、最初のデータを一つマイナスしている

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)

# Expected output: Chart similar to above, but with forecast overlay

#x_axisが1100-1250のレンジを把握してみる
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150) #x_axisのstart= , end= をインプットすることでグラフの一部を描画
plot_series(time_valid, naive_forecast, start=0, end=150)

# EXPECTED - Chart with X-Axis from 1100-1250 and Y Axes with series value and projections. Projections should be time stepped 1 unit 'after' series

#kerasにある平均2乗誤差ライブラリなどを用いて計算する
print(tf.keras.metrics.mean_squared_error(x_valid,naive_forecast))
print(tf.keras.metrics.mean_absolute_error(x_valid,naive_forecast))
# Expected Output
# 19.578304
# 2.6011968

#moving_avrは、window_size(何手先か)によって、移動する平均を計算して予測するアルゴリズム
#window_size=1なら、n番目のvalue ~ n+1番目のvalueの平均を取ることで予測するもの
def moving_average_forecast(series, window_size):
  forecasts = []
  for time in range(len(series)-window_size):
  #value情報であるseriesについて、time=0,win=1の時、series[0:1]の移動量平均を計算する
    forecasts.append(series[time:time+window_size].mean())
  return np.array(forecasts) #forecasts情報をnumpy形式で返す
  
  #window_sizeの関係上、window_size分前のseriesを指定して計算させる
moving_avg = moving_average_forecast(series[split_time-30:],30)

print(len(moving_avg))
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
    
# EXPECTED OUTPUT
# CHart with time series from 1100->1450+ on X
# Time series plotted
# Moving average plotted over it

#再びこの出力が前回のMSE,MAPEのベースラインとどう離れるかを比較するため、計算する
#time(1100(split_time)~1450)までのforecastsとvalueの平均値
print(tf.keras.metrics.mean_squared_error(x_valid,moving_avg))
print(tf.keras.metrics.mean_absolute_error(x_valid,moving_avg))
# EXPECTED OUTPUT
# 65.786224
# 4.3040023

plot_series(time,series,start=365,end=1450)
plot_series(time,series,start=0,end=1085)
plt.show()

#周期性をなくすために、n番目の周期-n-1番目の周期を行い、平坦化させる
diff_series = series[365:] - series[:-365]
diff_time = time[365:]

print(diff_series)
plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()

#周期性の差分グラフで、移動平均を行ってみる。
#diff_seriesは365日分失っていることに留意
diff_moving_avg = moving_average_forecast(diff_series[split_time-365-30:],30)

plt.figure(figsize=(10, 6))
#diff_seriesと、diff_movig_avrを比較してみる
plot_series(time_valid,diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()
            
# Expected output. Diff chart from 1100->1450 +
# Overlaid with moving average

plot_series(time,series,start=split_time - 365,end=1085)

#周期性の差分平均を加える
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()
# Expected output: Chart from 1100->1450+ on X. Same chart as earlier for time series, but projection overlaid looks close in value to it

rint(tf.keras.metrics.mean_squared_error(diff_moving_avg_plus_past,x_valid))
print(tf.keras.metrics.mean_absolute_error(diff_moving_avg_plus_past,x_valid))
# EXPECTED OUTPUT
# 8.498155
# 2.327179

#ノイズの除去を行う
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()
            
# EXPECTED OUTPUT:
# Similar chart to above, but the overlaid projections are much smoother

print(tf.keras.metrics.mean_squared_error(diff_moving_avg_plus_smooth_past ,x_valid))
print(tf.keras.metrics.mean_absolute_error(diff_moving_avg_plus_smooth_past ,x_valid))
# EXPECTED OUTPUT
# 12.527958
# 2.2034433
