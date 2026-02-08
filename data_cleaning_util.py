import os
import pandas as pd
import yfinance as yf
from config import MOTLEY_FOOL_DATA_PATH, STOCK_DATA_KAGGLE_PATH

def earnings_call_prepare(DATA_PATH=MOTLEY_FOOL_DATA_PATH, get_tickers_only=False):
    """
    Inputs:
        DATA_PATH - File path containing the transcripts from Motley fool

    Returns:
        If get_tickers_only=True:
            tickers - list of tickers in the transcript dataset
        Else:
            earnings_call_data - Dataframe containing an adjusted_date field which is the trading date
            stock_prices - downloaded price panel from yfinance for those tickers over the dataset range
    """
    # Motley-fool data with transcript for different tickers (company code) and dates
    # FIX: use the passed DATA_PATH (previously ignored)
    earnings_call_data = pd.read_pickle(DATA_PATH)

    # adjusting the date format and aligning with market closing time date
    earnings_call_data['date'] = earnings_call_data['date'].str.replace(r'\s*ET$', '', regex=True)
    earnings_call_data['date'] = (
        earnings_call_data['date']
        .str.replace('a.m.', 'AM', regex=False)
        .str.replace('p.m.', 'PM', regex=False)
        .str.strip()
    )
    earnings_call_data['date'] = pd.to_datetime(earnings_call_data['date'], format='mixed', errors='coerce')

    earnings_call_data['adjusted_date'] = earnings_call_data['date'].apply(
        lambda x: x + pd.offsets.BDay(1) if x.hour >= 16 else x
    )
    earnings_call_data['adjusted_date'] = earnings_call_data['adjusted_date'].dt.normalize()
    # FIX: redundant pd.to_datetime removed (already datetime64[ns] after normalize)

    # defining start and end date to get the time range for the dataset we have
    start_date = earnings_call_data['adjusted_date'].min().strftime('%Y-%m-%d')
    end_date = earnings_call_data['adjusted_date'].max().strftime('%Y-%m-%d')
    tickers = earnings_call_data['ticker'].unique().tolist()

    if get_tickers_only:
        return tickers

    #stock_prices = yf.download(tickers, start=start_date, end=end_date, threads=False, progress=False)
    return earnings_call_data


def get_sp500_data(
    DATA_PTAH=STOCK_DATA_KAGGLE_PATH,
    start_date="2018-01-01",
    end_date="2022-12-12",
    tickers=None,  # FIX: allow passing tickers to avoid re-calling earnings_call_prepare()
):
    """
    Inputs:
        DATA_PATH - File path containing the stock data for S&P 500 companies
        start_date - Start date for the data extraction
        end_date - End date for the data extraction

    Returns:
        df_with_market - Dataframe containing stock prices and calculated abnormal returns
    """
    folder = STOCK_DATA_KAGGLE_PATH
    dates = pd.date_range(start_date, end_date, freq="D")

    # FIX: avoid calling earnings_call_prepare() (expensive) if tickers already provided
    if tickers is None:
        tickers = earnings_call_prepare(get_tickers_only=True)

    # FIX: compute required_tickers ONCE (was being recomputed inside the loop every iteration)
    sp500_tickers = []
    for file in os.listdir(folder):
        ticker = os.path.splitext(file)[0]
        sp500_tickers.append(ticker)

    required_tickers = [t for t in sp500_tickers if t in set(tickers)]

    series_by_ticker = {}  # dictionary to hold ticker prices for each ticker

    for t in required_tickers:
        path = os.path.join(folder, f"{t}.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        adj_price = df[["Date", "Close"]]
        s = adj_price.set_index('Date')["Close"].rename(t)
        series_by_ticker[t] = s

    prices_df = pd.DataFrame(series_by_ticker)

    prices_df.index = pd.to_datetime(prices_df.index).normalize()
    start = pd.to_datetime(start_date) - pd.tseries.offsets.BDay(20)
    end = pd.to_datetime(end_date)

    # slice by datetime range
    new_df = prices_df.loc[start:end]

    new_cols = {}

    for ticker in new_df.columns:
        s = new_df[ticker]
        new_cols[f'{ticker}_bcallday_r1'] = s.pct_change(1).shift(1)
        new_cols[f'{ticker}_bcallday_r5'] = s.pct_change(5).shift(1)
        new_cols[f'{ticker}_bcallday_r20'] = s.pct_change(20).shift(1)
        new_cols[f'{ticker}_r1d'] = s.shift(-1) / s - 1
        new_cols[f'{ticker}_r5d'] = s.shift(-5) / s - 1

    new_df = pd.concat([new_df, pd.DataFrame(new_cols, index=new_df.index)], axis=1)

    spy_series = yf.download(
        "SPY",
        start=(start - pd.tseries.offsets.BDay(20)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False
    )["Close"]

    # normalize index, sort, compute returns
    spy_series.index = pd.to_datetime(spy_series.index).normalize()
    spy_series = spy_series.sort_index()

    market = pd.DataFrame(index=spy_series.index)
    market["Close"] = spy_series
    market["m_bcallday_r1"] = market["Close"].pct_change(1).shift(1)
    market["m_bcallday_r5"] = market["Close"].pct_change(5).shift(1)
    market["m_bcallday_r20"] = market["Close"].pct_change(20).shift(1)
    market["m_r1d"] = market["Close"].shift(-1) / market["Close"] - 1
    market["m_r5d"] = market["Close"].shift(-5) / market["Close"] - 1

    df_with_market = pd.merge(new_df, market, left_index=True, right_index=True, how="left")

    abn_cols = {}

    for t in required_tickers:
        abn_cols[f'{t}_abn_r1d'] = df_with_market[f'{t}_r1d'] - df_with_market['m_r1d']
        abn_cols[f'{t}_abn_r5d'] = df_with_market[f'{t}_r5d'] - df_with_market['m_r5d']
        abn_cols[f'{t}_abn_bcallday_r1'] = df_with_market[f'{t}_bcallday_r1'] - df_with_market['m_bcallday_r1']
        abn_cols[f'{t}_abn_bcallday_r5'] = df_with_market[f'{t}_bcallday_r5'] - df_with_market['m_bcallday_r5']
        abn_cols[f'{t}_abn_bcallday_r20'] = df_with_market[f'{t}_bcallday_r20'] - df_with_market['m_bcallday_r20']

    # concat the 5 abnormal-return blocks at once
    df_with_market = pd.concat([df_with_market, pd.DataFrame(abn_cols, index=df_with_market.index)], axis=1)

    # compute vol20d AFTER (needs abn_r1d columns to exist)
    vol_cols = {}
    for t in required_tickers:
        vol_cols[f'{t}_abn_vol20d'] = df_with_market[f'{t}_abn_r1d'].rolling(window=20).std()

    df_with_market = pd.concat([df_with_market, pd.DataFrame(vol_cols, index=df_with_market.index)], axis=1)

    return df_with_market


def format_earnings_data(df_with_market, earnings_call_data):
    """
    Inputs:
        df_with_market - Dataframe containing stock prices and calculated abnormal returns
        earnings_call_data - Dataframe containing an adjusted_date field which is the trading date of the earnings call

    Returns:
        final_df - Formatted dataframe ready for analysis with relevant abnormal return columns merged
    """
    # convert wide df_with_market columns like "TICK_abn_r1d" -> long table and merge
    df_with_market.index = pd.to_datetime(df_with_market.index).normalize()
    earnings_call_data['adjusted_date'] = pd.to_datetime(earnings_call_data['adjusted_date']).dt.normalize()

    # 1d abnormal returns long
    abn1 = df_with_market.filter(regex=r'_abn_r1d$').copy()
    abn1.columns = [c.rsplit('_abn_r1d', 1)[0] for c in abn1.columns]   # rename cols -> ticker
    abn1_long = abn1.stack().rename('abret_1d').reset_index()
    abn1_long.columns = ['adjusted_date', 'ticker', 'abret_1d']

    # 5d abnormal returns long
    abn5 = df_with_market.filter(regex=r'_abn_r5d$').copy()
    abn5.columns = [c.rsplit('_abn_r5d', 1)[0] for c in abn5.columns]
    abn5_long = abn5.stack().rename('abret_5d').reset_index()
    abn5_long.columns = ['adjusted_date', 'ticker', 'abret_5d']

    # 20-day abnormal volatility long
    abnvol20 = df_with_market.filter(regex=r'_abn_vol20d$').copy()
    abnvol20.columns = [c.rsplit('_abn_vol20d', 1)[0] for c in abnvol20.columns]
    abnvol20_long = abnvol20.stack().rename('abvol_20d').reset_index()
    abnvol20_long.columns = ['adjusted_date', 'ticker', 'abvol_20d']

    # abnormal before call day 1-day returns long
    abncallday = df_with_market.filter(regex=r'_abn_bcallday_r1$').copy()
    abncallday.columns = [c.rsplit('_abn_bcallday_r1', 1)[0] for c in abncallday.columns]
    abncallday_long = abncallday.stack().rename('abcallday_r1').reset_index()
    abncallday_long.columns = ['adjusted_date', 'ticker', 'abcallday_r1']

    # abnormal before call day 5-day returns long
    abncallday5 = df_with_market.filter(regex=r'_abn_bcallday_r5$').copy()
    abncallday5.columns = [c.rsplit('_abn_bcallday_r5', 1)[0] for c in abncallday5.columns]
    abncallday5_long = abncallday5.stack().rename('abcallday_r5').reset_index()
    abncallday5_long.columns = ['adjusted_date', 'ticker', 'abcallday_r5']

    # abnormal before call day 20-day returns long
    abncallday20 = df_with_market.filter(regex=r'_abn_bcallday_r20$').copy()
    abncallday20.columns = [c.rsplit('_abn_bcallday_r20', 1)[0] for c in abncallday20.columns]
    abncallday20_long = abncallday20.stack().rename('abcallday_r20').reset_index()
    abncallday20_long.columns = ['adjusted_date', 'ticker', 'abcallday_r20']

    # merge all into earnings_call_data
    earnings_call_data = (
        earnings_call_data
        .merge(abn1_long, on=['adjusted_date', 'ticker'], how='left')
        .merge(abn5_long, on=['adjusted_date', 'ticker'], how='left')
        .merge(abnvol20_long, on=['adjusted_date', 'ticker'], how='left')
        .merge(abncallday_long, on=['adjusted_date', 'ticker'], how='left')
        .merge(abncallday5_long, on=['adjusted_date', 'ticker'], how='left')
        .merge(abncallday20_long, on=['adjusted_date', 'ticker'], how='left')
    )

    clean_df = earnings_call_data.copy().dropna()
    clean_df['r1d_direction'] = (clean_df['abret_1d'] > 0).astype(int)
    clean_df['r5d_direction'] = (clean_df['abret_5d'] > 0).astype(int)
    final_df = clean_df[
        ["adjusted_date", "ticker", "transcript", "abvol_20d",
         "abcallday_r1", "abcallday_r5", "abcallday_r20",
         "abret_1d", "abret_5d", "r1d_direction", "r5d_direction"]
    ]

    return final_df


def prepare_earnings_data():
    """
    Returns:
        formated_data - Formatted dataframe ready for analysis with relevant abnormal return columns merged
    """
    # FIX: call earnings_call_prepare ONCE and reuse tickers
    earnings_call_data = earnings_call_prepare(get_tickers_only=False)
    tickers = earnings_call_data['ticker'].unique().tolist()

    # FIX: pass tickers into get_sp500_data so it doesn't call earnings_call_prepare again
    stock_data = get_sp500_data(tickers=tickers)

    formated_data = format_earnings_data(stock_data, earnings_call_data)
    return formated_data
