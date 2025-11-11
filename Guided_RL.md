### Market regime

- Trend
    UPTREND | DOWNTREND

    trend_strength_i = (price - ema_i) / ema_i
    trend_direction_i = sign(trend_strength_i)
    trend_confidence_i = abs(trend_strength_i) (optional)
    ဒါဆိုရင် EMAs 3 ခု → အနည်းဆုံး features = 3 × (strength + direction) = 6 features (confidence ထည့်ရင် 9)။

    Useful cross-scale features (highly recommended)
    ema50_minus_ema100 = (ema50 - ema100) / ema100 (relative gap)
    ema100_minus_ema200 = (ema100 - ema200) / ema200
    ema50_vs_ema200 = (ema50 - ema200) / ema200



- Pullback
    UPTREND_PULLBACK | UPTREND_BREAKOUT |  DOWNTREND_PULLBACK | DOWNTREND_BREAKOUT
- Volatility
    atr_norm | LOW_VOL | MED_VOL | HIGH_VOL
        0.5768      0           0         1


- XX Liquidity / Spread / Market Microstructure XX
- Momentum / Short-term Bias
    momentum_score = (rsi - 50) / 50

    if rsi < 45:
        mom_bearish, mom_neutral, mom_bullish = 1, 0, 0
    elif rsi > 55:
        mom_bearish, mom_neutral, mom_bullish = 0, 0, 1
    else:
        mom_bearish, mom_neutral, mom_bullish = 0, 1, 0

- XX Correlation / Cross-pair Regime XX



- News / Economic Calendar Flags
    Pre-news → 0 … 1 (news candle)
    Post-news → 1 … 0 (30 min after)


- Seasonality / Time-of-day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
