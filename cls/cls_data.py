import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from yfinance import download
import os
import time
from pandas.tseries.holiday import USFederalHolidayCalendar as us_calendar

class Data():
    def __init__(self):
        pass

    def get_data(self, start_, end_, ticker):
        data = download('META', interval='1d', start=start_, end=end_, progress=False).reset_index()
        data.columns = data.columns.droplevel('Ticker')
        data.columns = data.columns.str.lower()
        data.dropna(inplace=True)
        del data['adj close']
        if data.shape[0] == 0:
            print("error get data!")
        return data

    def save_file(self, data, ticker, path):
        data.to_csv(path, index=True)
        print(f"File {path} đã được lưu.")


    def data_preprocessing(self, data):
        data = data.reset_index()
        data['date'] = pd.to_datetime(data['date'])
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        data.set_index('date', inplace=True)
        del data['Unnamed: 0']
        del data['index']
        data.index = pd.to_datetime(data.index)
        data = data.asfreq(freq='D', normalize=False).interpolate(method='linear', limit_direction=None)
        return data 
    
    # Technical Analysis Indicators (Chỉ số phân tích kỹ thuật) exogenous variables
    
    def holidays(self, data):
        start_date = data.index.min()
        end_date = data.index.max()
        holidays = us_calendar().holidays(start=start_date, end=end_date)

        data['holiday'] = data.index.isin(holidays).astype('int32')
        return data

    def time_and_Date_Indicators(self, data):
        data['year'] = data.index.year
        data['quarter'] = data.index.quarter
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['dayofweek'] = data.index.dayofweek
        data['is_weekend'] = data.index.dayofweek >= 5 # ngày cuối tuần
        data['month_end'] = data.index.is_month_end.astype('int32') # cuối tháng
        data = self.holidays(data)
        #  UNIX timestamp.--> biến thời gian liên tục 
        data['unix'] = data.index.astype('int64') // 10**9
        return data
    
    def Statistical_and_Transformation_Indicators(self, data):
        # Tạo feature daily_variation(biến động phần trăm hằng ngày so với close) and logs of volume 
        data = data.assign(daily_variation=lambda x: (x['high'] - x['low'])/ x['open'])
        data = data.assign(logs=lambda x: x['volume'].apply(np.log))
        # Vốn hóa thị trường
        data = data.assign(market_cap=lambda x: x['close'] * x['volume'])
        # Mức độ tăng giảm của giá đóng cữa theo ngày
        data = data.assign(close_change=lambda x: x['close'] - x['close'].shift())
        # tỷ lệ phần trăm thay đổi giữa giá mở cửa và giá đóng cửa trong 1 năm
        data = data.assign(year_change=lambda x: 100 * (x['close'] - x['open']) / x['open'])

        return data
    
    def technical_indicators(self, data):
        # EMA(trung bình động hàm mũ) với 14 phiên giao dịch --> theo dõi xu hướng ngắn hạng
        data = data.assign(ema_14=lambda x: x['close'].ewm(span=14, adjust=False).mean())

        # CMA(trung bình động lũy kế) theo chu kỳ 7 value -->  xu hướng dài hạn
        data = data.assign(cma_7=lambda x: x['close'].expanding(7).mean())

        # Độ lệch chuẩn 7 ngày
        data = data.assign(std_7=lambda x: x['close'].rolling('7D').std())
        
        # lợi nhuận hằng ngày được tính bàng phần trăm thay đổi giữa giá đóng cửa của ngày hiện tại và ngày trước đó.
        data = data.assign(daily_return=lambda x: (x['close'] /x['close'].shift()) - 1) # or x['close'].pct_change(periods=1)

        # Tỷ lệ chênh lệch giữa giá cao nhất và giá đóng cửa so với giá mở cửa
        data = data.assign(high_close=lambda x: (x['high'] - x['close'])/ x['open'])

        # Tỷ lệ chênh lệch giữa giá thấp nhất và giá mở cửa
        data = data.assign(low_open=lambda x: (x['low'] - x['open'])/ x['open'])

        # Tỷ lệ thay đổi tích lũy của giá đóng cửa so với tổng giá trị
        data = data.assign(cum_change=lambda x: (100 * x['close'].cumsum())/ x['close'].sum())

        return data
    
    def trend_indicators(self ,data,  period=14):
        # Tính toán các chỉ số ATR, DMI, ADX
        '''
            - ATR đo lường mức độ biến động (volatility) của giá trong một khoảng thời gian.
            - +DMI: Chỉ số chuyển động tích cực (Positive Directional Index), đo lường mức độ tăng giá so với biến động.
            - -DMI: Chỉ số chuyển động tiêu cực (Negative Directional Index), đo lường mức độ giảm giá so với biến động.
            - DMI: Tổng hợp chênh lệch giữa +DMI và -DMI, phản ánh sự mạnh mẽ của xu hướng giá.
            - ADX đo lường sức mạnh của xu hướng giá, bất kể xu hướng là tăng hay giảm.

        '''

        alpha = 1 / period

        # TR (true range)
        data['H-L'] = data['high'] - data['low']
        data['H-C'] = np.abs(data['high'] - data['close'].shift(1))
        data['L-C'] = np.abs(data['low'] - data['close'].shift(1))
        data['TR'] = data[['H-L', 'H-C', 'L-C']].max(axis=1)
        del data['H-L'], data['H-C'], data['L-C']

        # ATR (average true range)
        data['ATR'] = data['TR'].ewm(alpha=alpha, adjust=False).mean()

        # +DX & -DX
        data['H-pH'] = data['high'] - data['high'].shift(1)
        data['pL-L'] = data['low'].shift(1) - data['low']
        data['+DX'] = np.where((data['H-pH'] > data['pL-L']) & (data['H-pH'] > 0), data['H-pH'], 0.0)
        data['-DX'] = np.where((data['H-pH'] < data['pL-L']) & (data['pL-L'] > 0), data['pL-L'], 0.0)
        del data['H-pH'], data['pL-L']

        # DMI (directional movement index)
        data['S+DM'] = data['+DX'].ewm(alpha=alpha, adjust=False).mean()
        data['S-DM'] = data['-DX'].ewm(alpha=alpha, adjust=False).mean()
        data['+DMI'] = (data['S+DM']/ data['ATR']) * 100
        data['-DMI'] = (data['S-DM']/ data['ATR']) * 100
        data['DMI'] = 100 * np.abs(data['+DMI'] - data['-DMI'])/ np.abs(data['+DMI'] + data['-DMI'])
        del data['S+DM'], data['S-DM']

        # ADX (average directional index)
        data['DX'] = (np.abs(data['+DMI'] - data['-DMI']) /(data['+DMI'] + data['-DMI'])) * 100
        data['ADX'] = data['DX'].ewm(alpha=alpha, adjust=False).mean()
        del data['DX'], data['TR'], data['-DX'], data['+DX'], data['+DMI'], data['-DMI']
        
        data.columns = data.columns.str.lower()
        return data
    def hp_filter(self, data):
        '''
            Sử dụng bộ lọc Hodrick-Prescott để phân tách chu kỳ và xu hướng trong dữ liệu giá cổ phiếu.
            Mục tiêu: tăng cường phân tích chu kỳ ngắn hạn, loại bỏ những xu hướng dài hạn
        '''
        # for daily data
        lambda_ = 1600*(365/4)**4
        cycle = pd.Series(hpfilter(data['close'], lamb=lambda_)[0], name='cycle')
        
        data = data.join(cycle)
        
        return data
    
    def macd(self, data, short_period=12, long_period=26, signal_period=9):
        '''
            Tính toán chỉ báo MACD (đường trung bình động hội tụ phân kỳ). nhằm mục tiêu xác định xu hướng của giá cổ phiếu 
            trong tương lai.Ngoài ra nó còn giúp mô hình sarimax xác định được xu hướng và động lực giá.
        '''
        # 12-period EMA
        EMA_12 = data['close'].ewm(span=short_period, adjust=False).mean()
        # 26-period EMA
        EMA_26 = data['close'].ewm(span=long_period, adjust=False).mean()
        
        # MACD
        MACD = pd.Series(EMA_12 - EMA_26, name='macd')
        
        # optionally: signal line (9-period EMA of MACD)
        MACD_signal = pd.Series(MACD.ewm(span=signal_period, adjust=False).mean(), name='macd_signal')
        
        data = data.join(MACD)
        
        return data


    def bollinger_bands(self, data, n=20, m=2):
        '''
        Computes Bollinger bands. so sánh độ lệch giá so với đường trung bình B_MA, nhờ đó dự đoán tốt hơn các mức giá tiếp theo
        - B_MA: Đường trung bình di động, thể hiện xu hướng giá trung tâm.
        - BL: Dải dưới, cho thấy mức giá thấp tiềm năng.
        - BU: Dải trên, cho thấy mức giá cao tiềm năng.
        - So sánh giá hiện tại với dải trên (BU) và dải dưới (BL) để xác định các tín hiệu quá mua (overbought) hoặc quá bán (oversold). 
        - Khi dải Bollinger (BU - BL) thu hẹp: Biến động giá giảm, thị trường có thể chuẩn bị cho một biến động mạnh.

        '''
        # typical price
        TP = (data['high'] + data['low'] + data['close']) /3
        
        # Bollinger bands
        B_MA = pd.Series((TP.rolling(n, min_periods=n).mean()), name='b_ma')
        sigma = TP.rolling(window=n, min_periods=n).std()
        
        BL = pd.Series(B_MA - m * sigma, name='bl')
        BU = pd.Series(B_MA + m * sigma, name='bu')
        
        data = data.join(B_MA)
        data = data.join(BL)
        data = data.join(BU)
        
        return data    


    def sma(self, data):
        '''
        Computes SMA (simple moving averages) over a 7-day window & its confidence intervals.  Trung bình động đơn giản
        '''
        # 7-period EMA
        SMA_7 = pd.Series(data['close'].rolling('7D').mean(), name='sma_7')
        
        # confidence intervals
        interval = 2 * data['close'].rolling('7D').std()
        SMA_7_upper = pd.Series(SMA_7 + interval, name='sma_7_up')
        SMA_7_lower = pd.Series(SMA_7 - interval, name='sma_7_low')
        
        data = data.join(SMA_7)
        data = data.join(SMA_7_upper)
        data = data.join(SMA_7_lower)
        
        return data
    

    def rsi(self, data, period=14):
        '''
            - Chỉ số sức mạnh tương đối (RSI - Relative Strength Index)
            - Tính chênh lệch giữa giá đóng cửa ngày hôm nay và ngày hôm trước.
            - gain: Chỉ tính các ngày có mức tăng giá, còn ngày giảm giá được gán giá trị 0.
            - loss: Chỉ tính các ngày có mức giảm giá, còn ngày tăng giá được gán giá trị 0 (lấy giá trị âm của delta để thể hiện lỗ là số dương)
            - Tính trung bình lãi và lỗ (average gain & loss)
            - RS > 1: Trung bình lãi cao hơn, thị trường có xu hướng tăng.
            - RS < 1: Trung bình lỗ cao hơn, thị trường có xu hướng giảm.
            - Giá trị RSI càng cao thể hiện thị trường đang trong trạng thái quá mua (overbought), giá trị thấp thể hiện quá bán (oversold).

        '''
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = - delta.where(delta < 0, 0)
        
        # sum of average gains
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        # sum of average losses
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        RS = avg_gain / avg_loss
        RSI = pd.Series(100 - (100 / (1 + RS)), name='rsi')
        
        data = data.join(RSI)
    
        return data
    


    def stochastic_oscillator(self ,data, k_period=14, d_period=3):
        '''
            Tính Stochastic Oscillator, một chỉ báo dao động (oscillator) trong phân tích kỹ thuật. 
            Chỉ báo này đo lường vị trí tương đối của giá đóng cửa so với khoảng giá (high-low) 
            trong một khoảng thời gian nhất định, giúp xác định trạng thái quá mua (overbought) hoặc quá bán (oversold).

            - %K cho biết vị trí giá đóng cửa hiện tại (data['close']) so với khoảng giá (max_high - min_low).
            - %D là đường trung bình động đơn giản (Simple Moving Average - SMA) của %K trong d_period phiên gần nhất.


            Phát hiện tín hiệu mua/bán qua điểm giao cắt (crossovers):
                Khi %K cắt lên %D: Tín hiệu mua (bullish crossover).
                Khi %K cắt xuống %D: Tín hiệu bán (bearish crossover).
        '''
        # max value of previous 14 periods
        max_high = data['high'].rolling(k_period).max()
        # min value of previous 14 periods
        min_low = data['low'].rolling(k_period).min()
        
        # uses the min/max values to calculate the %K (as a percentage)
        K = pd.Series((data['close'] - min_low) * 100 / (max_high - min_low), name='k')
        # uses the %K to calculates a SMA over the past 3 values of %K
        D = pd.Series(K.rolling(d_period).mean(), name='d')
        
        data = data.join(K)
        data = data.join(D)
        
        return data
    


    def williams_r(self, data, period=14):
        '''
            Chỉ báo William’s %R (hay Williams Percent Range), một chỉ báo dao động trong phân tích kỹ thuật,Chỉ báo này 
            đo lường mức độ gần của giá đóng cửa so với khoảng giá cao nhất và thấp nhất trong một khoảng thời gian xác định.
            Chỉ báo tính toán mức độ gần của giá đóng cửa (data['close']) với giá cao nhất (max_high) trong khoảng giá 
            (max_high - min_low).

                WR > -20%: Thị trường có thể quá mua (overbought).
                WR < -80%: Thị trường có thể quá bán (oversold)
                Khi WR rơi vào vùng quá mua/quá bán, đây có thể là tín hiệu thị trường chuẩn bị đảo chiều.
                Khi WR vượt xuống dưới -20%: Có thể là tín hiệu bán.
                Khi WR vượt lên trên -80%: Có thể là tín hiệu mua.
        '''
        # max value of previous 14 periods
        max_high = data['high'].rolling(period).max()
        # min value of previous 14 periods
        min_low = data['low'].rolling(period).min()
        
        wr = -100 * ((max_high - data['close']) / (max_high - min_low))
        WR = pd.Series(wr, name='wr')
                                                
        data = data.join(WR)                                        
        
        return data



    def cci(self, data, period=14):
        '''
            Tính toán CCI (Commodity Channel Index), một chỉ báo dao động thường được sử dụng trong phân tích kỹ thuật 
            để đánh giá xu hướng giá cả và xác định các trạng thái quá mua (overbought) hoặc quá bán (oversold).

            TP - SMA: Khoảng cách giữa giá tiêu biểu và trung bình động.
            0.015: Hằng số điều chỉnh để chuẩn hóa CCI trong khoảng từ -100 đến +100 trong điều kiện thị trường bình thường.

            CCI > +100: Giá đang trong trạng thái quá mua (overbought), có thể chuẩn bị đảo chiều giảm.

            CCI < -100: Giá đang trong trạng thái quá bán (oversold), có thể chuẩn bị đảo chiều tăng.

        '''
        # typical price
        TP = (data['high'] + data['low'] + data['close']) /3
        SMA = TP.rolling(period).mean()
        MAD = TP.rolling(period).std()
        
        cci = (TP - TP.rolling(period).mean()) /(0.015 * MAD)
        CCI = pd.Series(cci, name='cci')
        
        data = data.join(CCI)
        
        return data



    def ppo(self, data, short_period=12, long_period=26, signal_period=9):
        '''
            Chỉ báo PPO (Percentage Price Oscillator), một công cụ phân tích kỹ thuật dựa trên đường trung bình động hàm mũ 
            (EMA) để đánh giá động lực giá tương đối.

            PPO là phiên bản tỷ lệ phần trăm của chỉ báo MACD, được sử dụng để so sánh sự khác biệt giữa hai đường trung 
            bình động hàm mũ với một tín hiệu chung.

            PPO > 0: Động lực tăng giá, EMA ngắn hạn lớn hơn EMA dài hạn.
            PPO < 0: Động lực giảm giá, EMA ngắn hạn nhỏ hơn EMA dài hạn.

            Khi PPO cắt lên trên đường tín hiệu (PPO_signal): Tín hiệu mua.
            Khi PPO cắt xuống dưới đường tín hiệu (PPO_signal): Tín hiệu bán.

        '''
        # 12-period EMA
        EMA_12 = data['close'].ewm(span=short_period, adjust=False).mean()
        # 26-period EMA
        EMA_26 = data['close'].ewm(span=long_period, adjust=False).mean()
        
        # PPO
        PPO = pd.Series(100 *(EMA_12 - EMA_26) /EMA_26, name='ppo')
        
        # optionally: signal line (9-period EMA of PPO)
        PPO_signal = pd.Series(PPO.ewm(span=signal_period, adjust=False).mean(), name='ppo_signal')
        
        data = data.join(PPO)
        #df = df.join(PPO_signal)
        
        return data

    def Technical_Analysis_Indicators(self, data):
        data = self.time_and_Date_Indicators(data)

        data = self.Statistical_and_Transformation_Indicators(data)

        data = self.technical_indicators(data)

        data = self.trend_indicators(data)
        data = self.hp_filter(data)
        data = self.macd(data)
        data = self.bollinger_bands(data)
        data = self.sma(data)
        data = self.rsi(data)
        data = self.stochastic_oscillator(data)
        data = self.williams_r(data)
        data = self.cci(data)
        data = self.ppo(data)
        return data

    

if __name__ == '__main__':
    print("-----------------RUN-----------------")

    _data = Data()
    start_date="2020-01-01"
    end_date ="2024-08-01"
    ticker = 'TSLA'
    path = "../data/{}_basic.csv".format(ticker)
    if not os.path.exists(path):
        data  = _data.get_data(start_date, end_date, ticker)
        _data.save_file(data, ticker, path)
        

    # Expension with Technical Analysis Indicators (Chỉ số phân tích kỹ thuật) exogenous variables
    path_tech = '../data/{}_Technical.csv'.format(ticker)
    if not os.path.exists(path_tech):
        data = pd.read_csv(path)
        data = _data.data_preprocessing(data)
        data = _data.Technical_Analysis_Indicators(data)
        _data.save_file(data, ticker, path_tech)


    
    