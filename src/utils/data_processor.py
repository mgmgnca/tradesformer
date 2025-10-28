import os
import logging
import pandas as pd
import numpy as np
from finta import TA
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# Configure logging
logger = logging.getLogger(__name__)

def patch_missing_data(df, dt_col_name='time', cf=None):
    min_bars = cf.data_processing_parameters("min_bars_per_week")

        # ["time","open", "high", "low", "close"]
    required_cols = cf.data_processing_parameters("required_cols")    
    
    # df မှာ 6 columns ရှိရင် vol ပါထည့်မယ် 
    if df.shape[1] == 6:
        df.columns = required_cols + ['vol']  
    elif df.shape[1] == 5:
        df.columns = required_cols
    else:
        raise ValueError(f"Invalid number of columns: {df.shape[1]} =>{required_cols}")
    
    logger.warning(f"shape of  column: {df.shape[1]}")
    # 1. Column validation
    if missing := set(required_cols) - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    # 2. Auto-detect datetime column
    dt_candidates = {'time', 'timestamp', 'date', 'datetime'}
    if dt_col_name not in df.columns:
        found = list(dt_candidates & set(df.columns))
        if not found:
            raise KeyError(f"No datetime column found. Tried: {dt_candidates}")
        dt_col_name = found[0]
        logger.info(f"Using datetime column: {dt_col_name}")

    # 3. Convert to datetime index
    df[dt_col_name] = pd.to_datetime(df[dt_col_name], utc=True)
    df = df.set_index(dt_col_name).sort_index()

    # Week by Week Group (Friday-end week)
    groups = df.groupby(pd.Grouper(freq='W-FRI'))

    patched_weeks = []  # patched weekly df storage

    for w, week_df in groups:
        if week_df.empty:
            continue

        if len(week_df) != min_bars:
            logger.warning(f"Week {w} has {len(week_df)}/{min_bars} bars")

        # Create 5-minute frequency index
        new_index = pd.date_range(
            start=week_df.index.min(),
            end=week_df.index.max(),
            freq='5min',
            tz='UTC'
        )

        # Reindex + forward fill
        week_df = week_df.reindex(new_index)
        fill_limit = 12 # ဥပမာ: 1 နာရီ (12 bars) ထက်ပိုတဲ့ ကွက်လပ်ကို မဖြည့်ပါ
        fill_cols = ['open', 'high', 'low', 'close', 'vol'] if 'vol' in df.columns else ['open', 'high', 'low', 'close']
        # FFill: ရှေ့က data ဖြင့် ဖြည့်ပါ
        week_df[fill_cols] = week_df[fill_cols].ffill(limit=fill_limit)
        patched_weeks.append(week_df)

    # Merge back all weeks
    if patched_weeks:
        all_df = pd.concat(patched_weeks)
    else:
        all_df = df.copy()

    return all_df.reset_index().rename(columns={'index': dt_col_name})

def add_time_feature(df, symbol):
    """Add temporal features with proper index handling"""
    
    if 'time' not in df.columns:
        raise KeyError("'time' column missing after patch_missing_data")
        
    df = df.set_index('time')
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Cyclical time features
    df['weekday'] = df.index.dayofweek  # 0=Monday
    df['day'] = df.index.day
    df['week'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24).round(6)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24).round(6)
    df['minute_block'] = df.index.minute // 5  # 0-11
    df['minute_sin'] = np.sin(2 * np.pi * df['minute_block']/12).round(6)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute_block']/12).round(6)
    
    # Market sessions (GMT)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
    
    df['symbol'] = symbol
    return df.reset_index()

def tech_indicators(df, cf=None):  # 288 = 24hrs in 5-min bars
    """Calculate technical indicators with proper NaN handling"""
    period = cf.data_processing_parameters("indicator_period")
    # 1. Preserve raw prices before normalization
    raw_cols = ['mean_std_open','mean_std_high','mean_std_low','mean_std_close']
    df[raw_cols] = df[['open','high','low','close']].copy()
    # Calculate indicators
    df['macd'] = TA.MACD(df).SIGNAL.ffill().round(6)
    bb = TA.BBANDS(df)
    df['boll_ub'] = bb['BB_UPPER'].ffill()
    df['boll_lb'] = bb['BB_LOWER'].ffill()
    
    df['rsi_30'] = TA.RSI(df, period=period).ffill()
    df['dx_30'] = TA.ADX(df, period=period).ffill()
    df['close_30_sma'] = TA.SMA(df, period=period).ffill()
    df['close_60_sma'] = TA.SMA(df, period=period*2).ffill()
    df['atr'] = TA.ATR(df, period=period).ffill()
     # Add returns and volatility ratio
    df['returns_5'] = df['close'].pct_change(5,fill_method=None).round(6)
    df['returns_24'] = df['close'].pct_change(24,fill_method=None).round(6)
    df['volatility_ratio'] = (df['high'] - df['low']) / df['close'].round(6)
        
    # Normalize
    scaler = StandardScaler()
    scale_cols = cf.data_processing_parameters("scale_cols")  

    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # 1. Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 2. Apply clipping only to numeric features
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e5, upper=1e5)
    # 3. Round decimal values
    df[numeric_cols] = df[numeric_cols].round(6).clip(-1e5, 1e5)
    return df

class TimeSeriesScaler:
    """
    Manages the MinMax Scaling process for time series features.
    It fits the scaler on the first chunk of data (expected to be the training start)
    and uses that fitted scaler to transform all subsequent data chunks (including eval).
    """
    def __init__(self):
        # MinMaxScaler ကို အသုံးပြုပြီး 0 နဲ့ 1 ကြားကို ပြောင်းပါ
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.price_cols = ['mean_std_open', 'mean_std_high', 'mean_std_low', 'mean_std_close']
        
    def fit_and_transform(self, df):
        """Fit the scaler on the data and transform it."""
        logger.info("Fitting Scaler on current week data (TRAIN set base)")
        # .copy() လုပ်ပြီးမှ transform လုပ်ပါ
        df_copy = df.copy() 
        df_copy[self.price_cols] = self.scaler.fit_transform(df_copy[self.price_cols])
        self.is_fitted = True
        return df_copy
    
    def transform(self, df):
        """Transform data using the previously fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted on the training data first!")
            
        logger.info("Transforming current week data using fitted scaler.")
        # .copy() လုပ်ပြီးမှ transform လုပ်ပါ
        df_copy = df.copy() 
        df_copy[self.price_cols] = self.scaler.transform(df_copy[self.price_cols])
        return df_copy
    
def split_time_series_v2(df, symbol='EURUSD', cf=None, scaler_manager=None):
    """
    Split data with weekly alignment, adds a lookback context (overlap) from the
    previous week for continuous sequence processing (e.g., Transformer), 
    and performs MinMax scaling.
    
    Args:
        df (pd.DataFrame): Input Time Series Data.
        freq (str): Frequency string for pandas Grouper (e.g., 'W-FRI' for weekly split ending Friday).
        symbol (str): Trading symbol.
        cf (object): Configuration manager.
        scaler_manager (object): TimeSeriesScaler instance.
        sequence_length (int): The lookback window size needed for the Transformer.
    """
    if scaler_manager is None:
        raise ValueError("scaler_manager (TimeSeriesScaler instance) must be provided.")

    split_cfg = cf.data_processing_parameters("train_eval_split")
    base_path = split_cfg["base_path"].format(symbol=symbol)

    sequence_length = cf.data_processing_parameters("sequence_length")
        
    # Align with Forex week (Monday-Friday/Sunday)
    # df['time'] သည် ဤနေရာတွင် datetime object ဖြစ်ရမည်။
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time')
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame must have a 'time' column or a datetime index.")
    
    # W-FRI သည် သောကြာနေ့တွင် အဆုံးသတ်သော အပတ်ကို ကိုယ်စားပြုသည်။
    groups = df.groupby(pd.Grouper(freq='W-FRI'))
    
    # Indicators columns
    indicator_cols = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'atr']
    
    
    prev_week_df = None # ယခင် Week ရဲ့ DataFrame အပြည့်အစုံကို သိမ်းဆည်းရန်
    
    # Loop စတင်ခြင်း
    for week_start, week_df in groups:
        if week_df.empty:
            continue
        
        # 1. Context (Overlap Data) ကို ဆုံးဖြတ်ခြင်း
        # [NEW ACTION] ယခင် Week ရဲ့ နောက်ဆုံး sequence_length စာရှိတဲ့ data ကို ဖြတ်ယူပြီး ကပ်ပါ
        context_df = pd.DataFrame() 
        if prev_week_df is not None:
            # နောက်ဆုံး sequence_length စာ rows ကို ဖြတ်ယူပါ
            # NOTE: Index Slicing မှန်စေရန် .iloc ကို အသုံးပြုပါ
            context_df = prev_week_df.iloc[-sequence_length:].copy()
        
        # [NEW ACTION] လက်ရှိ week_df နဲ့ Context ကို ပေါင်းစပ်ခြင်း
        # Concat လုပ်ရာတွင် index ကို ဆက်ထိန်းထားရပါမည် (ignore_index=False)
        # Context သည် week_df ၏ ရှေ့တွင် ရှိရမည်
        current_chunk = pd.concat([context_df, week_df])
        
        # 2. Check raw indicators to determine Eval set (Data Leakage မဖြစ်စေရန်)
        # Check လုပ်ရာတွင် context မပါသော week_df ကိုသာ အသုံးပြုသင့်သည်၊ သို့မဟုတ်
        # context မပါသော ပထမဆုံး row ကိုသာ အသုံးပြုသင့်သည်။
        first_row = week_df[indicator_cols].iloc[0] # week_df (context မပါ) ကိုသာ စစ်ဆေး
        has_nan = first_row.isna().any()
        has_zero = (first_row == 0).any()
        is_eval = has_nan or has_zero # Indicator များ မပြည့်စုံသေးသော အပတ်ကို Eval အဖြစ် သတ်မှတ်
        
        # # Data အရေအတွက် စစ်ဆေးခြင်း (1440 bars per week)
        # if len(week_df) < 1440: # Context ပါသော current_chunk ကို စစ်ဆေးရန် မလို
        #     logger.warning(f"Skipping {week_start}: {len(week_df)}/{1440} bars (original week)")
        #     continue
        
        # 3. Normalize and validate (Fit-Transform Logic)
        if not scaler_manager.is_fitted and not is_eval:
            # Scaler ကို ပထမဆုံးသော၊ Indicators ပြည့်စုံသော (is_eval=False) Training Set တွင် Fit လုပ်ပါ
            # [ACTION] Fit လုပ်ပြီး Transform လုပ်မည့် data မှာ Context ပါဝင်ရန် မလို၊ Original Data ကိုသာ Fit လုပ်ရမည်
            # [ACTION] Fit လုပ်ပြီး Transform လုပ်မည့် data မှာ Context မပါဝင်ရန် 
            week_df_transformed = scaler_manager.fit_and_transform(current_chunk) # Context ပါသော chunk ကို Transform
            dir_type = 'train'
        elif scaler_manager.is_fitted:
            # Scaler Fit ပြီးပါက၊ Train နှင့် Eval နှစ်ခုလုံးကို Transform လုပ်ပါ
            # [ACTION] Context ပါသော chunk ကို Transform
            week_df_transformed = scaler_manager.transform(current_chunk)
            dir_type = 'eval' if is_eval else 'train'
        else:
            # Fit မလုပ်ရသေးဘဲ is_eval ဖြစ်နေရင် ကျော်သွားပါ
            logger.warning(f"Skipping {week_start}: Indicators not ready for fitting and not fitted yet.")
            # [ACTION] နောက်တစ်ကြိမ်အတွက် prev_week_df ကိုလည်း update လုပ်ရန် လိုအပ်သည် (မသိမ်းမီ)
            prev_week_df = week_df.copy() 
            continue

        # 4. Save to appropriate directory
        path = os.path.join(base_path, split_cfg[f"{dir_type}_dir"])
        os.makedirs(path, exist_ok=True)
        
        iso_year, iso_week, _ = week_start.isocalendar()
        fname = f"{symbol}_{iso_year}_{iso_week:02d}.csv"
        
        # [ACTION] Context ပါဝင်ပြီး၊ Normalize ပြီးသော DataFrame ကိုသာ သိမ်းပါ
        week_df_transformed.reset_index().to_csv(f"{path}/{fname}", index=False)
        logger.critical(f"Saved {dir_type} file: {fname} (Total rows: {len(week_df_transformed)})")

        # 5. လက်ရှိ week_df ကို နောက်တစ်ကြိမ်အတွက် Context အဖြစ် မှတ်သားခြင်း
        # [ACTION] prev_week_df သည် Context မပါဝင်သေးသော Original Week Data ဖြစ်ရမည်။
        prev_week_df = week_df.copy()