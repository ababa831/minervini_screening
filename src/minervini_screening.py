import os
from pathlib import Path
import io
import time
import pickle
import gc
import warnings
from datetime import datetime, timedelta
import dateutil.parser

import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
import requests
import numpy as np
from tqdm import tqdm
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import slack
from slack.errors import SlackApiError
import nest_asyncio
import click


warnings.filterwarnings('ignore')


def get_tickers(country):
    """
    Download symbol list
    """
    tickers = None
    df_stockcode = None
    if country == 'us':
        ex_nas = pd.read_csv('ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt', sep='|')
        nas = pd.read_csv('ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt', sep='|')

        filter_ignore = '[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]'
        filter_contains = '[A-Z]{,4}'
        tickers = []
        for e in [ex_nas, nas]:
            e.rename(columns={'ACT Symbol': 'Symbol'}, inplace=True)
            c1 = ~e['Symbol'].str.contains(filter_ignore)
            c2 = e['Symbol'].str.match(filter_contains)
            pd_symbols = e[c1&c2]['Symbol']
            tickers += pd_symbols.tolist()[:-1]
    elif country == 'ja':
        url = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'
        res = requests.get(url)
        res.raise_for_status()
        with io.BytesIO(res.content) as fh:
            df_stockcode = pd.io.excel.read_excel(fh)
        tickers = df_stockcode['コード']
    else:
        raise NotImplementedError('Other contry are not yet supported') 

    return tickers, df_stockcode


def get_ibd_rs(tickers, range='max', country='us'):
    ibd_rs_dict = {}
    charts = {}
    for t in tqdm(tickers):
        time.sleep(1) 
        df_chart = None
        if country == 'ja':
            jp = yf.Ticker(f'{t}.T')
            df_chart = jp.history(period=range, progress=False)
        elif country == 'us':
            df_chart = yf.download(t, period=range, progress=False)
        else:
            NotImplementedError('Other countries are not yet supported.')

        if 'Close' not in df_chart.columns:
            continue

        c_pasts = {
            'c63': 63,
            'c126': 126,
            'c189': 189,
            'c252': 252
            }
        for k, v in c_pasts.items():
            df_chart[k] = df_chart.shift(v)['Close']

        # 2 x c / c63 + c / c126 + c / c189 + c / c252
        df_chart['ibdRS'] = \
            df_chart['Close'] * (2/df_chart['c63'] + 1/df_chart['c126'] + 1/df_chart['c189'] + 1/df_chart['c252'])
        ibd_rs_dict[t] = df_chart['ibdRS']
        charts[t] = df_chart

    return ibd_rs_dict, charts


def get_rs_rank(ibd_rs_dict):
    """
    Get relative strength ranking DataFrame
    """
    # 欠損＆インデックス重複対策
    idbrs_dict_f = {k: v.dropna() for k, v in ibd_rs_dict.items() if v.shape[0] != 0 and not np.isnan(v).all()}

    idbrs_dict_f = {}
    for k, v in ibd_rs_dict.items():
        if v.shape[0] == 0 or np.isnan(v).all():
            continue
        if v.index[-1] == v.index[-2]:
            idbrs_dict_f[k] = v.iloc[:-1].dropna()
        else:
            idbrs_dict_f[k] = v.dropna()

    max_len = 0
    arg_max = None
    for k, v in idbrs_dict_f.items():
        if v.shape[0] > max_len:
            arg_max = k

    df_all_rs = pd.DataFrame(idbrs_dict_f, index=idbrs_dict_f[arg_max].index)

    for idx, row in tqdm(df_all_rs.iterrows()):
        demo_rank = row.dropna().sort_values(ascending=False)
        demo_rank_w_na = row.sort_values(ascending=False)
        len_diff = demo_rank_w_na.shape[0] - demo_rank.shape[0]

        thr = demo_rank.shape[0] * 0.01
        rs_rank = 100 - np.ceil(np.arange(demo_rank.shape[0]) / thr)
        rs_rank = rs_rank.astype(np.int8)

        nan_array = np.array([np.nan]*len_diff)
        rank_w_nan =  np.append(rs_rank, nan_array)

        use_rank = pd.Series(rank_w_nan, index=demo_rank_w_na.index)

        df_all_rs.loc[idx].update(use_rank)

    return df_all_rs


def get_recent_strong_tickers(df_rs_rank, rs_thres=90):
    """
    Get recent RS growth symbols
    """
    latest = df_rs_rank.iloc[-1]
    latest = latest[~latest.isnull()]
    latest_rs_elite = latest[latest >= rs_thres]

    tickers_rs_elite = latest_rs_elite.index

    tickers_rs_growth_last60 = []
    for col in tqdm(tickers_rs_elite):
        df_tmp = df_rs_rank[[col]].rolling(5).mean()
        c30 = df_tmp[col].iloc[-30]
        c60 = df_tmp[col].iloc[-60]
        c = df_tmp[col].iloc[-1]
        if np.isnan(c60):
            continue
        if c30 <= c and c60 <= c30:
            tickers_rs_growth_last60.append(col)
    return tickers_rs_growth_last60


def get_growth_stocks(tickers, range='max', charts=None, country='us'):
    growth_stocks = {}
    rank_of_diff_highest = {}
    for t in tqdm(tickers):
        df_chart = None
        if charts:
            df_chart = charts[t]
        elif country == 'ja':  # データがない場合
            jp = yf.Ticker(f'{t}.T')
            df_chart = jp.history(period=range)
            time.sleep(1)
        elif country == 'us':  # データがない場合
            df_chart = yf.download(t, period=range)  
            time.sleep(1)
        else:
            raise NotImplementedError('Other countries are not yet supported.')

        # MA calculation
        df_chart['MA200'] = df_chart['Close'].rolling(200).mean()
        df_chart['MA150'] = df_chart['Close'].rolling(150).mean()
        df_chart['MA100'] = df_chart['Close'].rolling(100).mean()
        df_chart['MA50'] = df_chart['Close'].rolling(50).mean()

        # filtering: MA comparison
        df_latest = df_chart.iloc[-1]
        c1 = df_latest['MA200'] < df_latest['MA150']
        c2 = df_latest['MA150'] < df_latest['MA100']
        c3 = df_latest['MA100'] < df_latest['MA50']
        c4 = df_latest['MA150'] < df_latest['Close']
        if not (c1 and c2 and c3 and c4):
            continue

        # filtering: MA200 trend (for over a month)
        df_last25 = df_chart.iloc[-25:]
        c5 = df_last25['MA200'].iloc[0] < df_last25['MA200'].iloc[-1]
        if not c5:
            continue

        # filtering: 52weeks (260 days) low & high
        df_last52weeks = df_chart.iloc[-260:]
        high = df_last52weeks['Close'].max()
        low = df_last52weeks['Close'].min()
        latest = df_latest['Close']
        c6 = latest > low * 1.25
        c7 = latest > high * 0.75
        if not (c6 and c7):
            continue

        # filtering: exclude penny (< $10)
        c8 = latest < 10
        if c8:
            continue

        # Get the highest price -> Ranking: diff of latest and highest
        all_latest = df_chart['Close'].max()
        diff = (latest - all_latest) / all_latest

        # Volatility
        df_chart['volatility'] = df_chart['Close'].rolling(25).std()
        vol_pct_mean = df_chart.iloc[-5:]['volatility'].pct_change().mean()

        # Append the filtered ticker
        growth_stocks[t] = df_chart
        rank_of_diff_highest[t] = (diff, vol_pct_mean)

    return growth_stocks, rank_of_diff_highest


def filter_excellent_tickers(rank_diff, df_stockcode, country='us', ja_stockcode=None):
    """
    Filter excellent tickers from growth stocks
    """
    excellent_tickers = list(rank_diff.keys())
    excellent_diffs = [rank_diff[k][0] for k in excellent_tickers]
    excellent_vols = [rank_diff[k][1] for k in excellent_tickers]
    excellent_dict = None
    if country == 'us':
        excellent_dict = {
            'ticker': excellent_tickers,
            'price_diff_from_highest': excellent_diffs,
            'volatility_change_in5days': excellent_vols
        }
    elif country == 'ja' and not isinstance(ja_stockcode, type(None)):
        df_ex_tickerinfo = df_stockcode[df_stockcode['コード'].isin(excellent_tickers)]
        excellent_dict = {
            'ticker': excellent_tickers,
            'price_diff_from_highest': excellent_diffs,
            'volatility_change_in5days': excellent_vols,
            'name': df_ex_tickerinfo['銘柄名'].to_list(),
            'market': df_ex_tickerinfo['市場・商品区分'].to_list(),
            '33sectors': df_ex_tickerinfo['33業種区分'].to_list(),
            '17sectors': df_ex_tickerinfo['17業種区分'].to_list(),
            'stock_size': df_ex_tickerinfo['規模区分'].to_list()
        }
    df_excellent = pd.DataFrame(excellent_dict)
    df_excellent.sort_values(by='volatility_change_in5days', inplace=True)
    df_excellent = df_excellent[df_excellent['price_diff_from_highest'] >= -0.25]
    
    return df_excellent


def add_next_earnings_date(df_excellent):
    """
    Add next earnings date to excellent tickers df
    """
    DAYS_AHEAD = 365

    # setting the dates
    start_date = datetime.now().date()
    end_date = (datetime.now().date() + timedelta(days=DAYS_AHEAD))

    # downloading the earnings calendar
    yec = YahooEarningsCalendar()
    next_earnings_date = []
    for t in tqdm(df_excellent['ticker']):
        earnings_list = yec.get_earnings_of(t)
        earnings_df = pd.DataFrame(earnings_list)

        # extracting the date from the string and filtering for the period of interest
        if 'startdatetime' in earnings_df.columns:
            earnings_df['report_date'] = earnings_df['startdatetime'].apply(lambda x: dateutil.parser.isoparse(x).date())
            earnings_df = earnings_df.loc[earnings_df['report_date'].between(start_date, end_date)] \
                                    .sort_values('report_date')
        
        report_date = None
        if earnings_df.shape[0] == 0:
            report_date = 'TBD'
        else:
            report_date = earnings_df.iloc[0]['report_date'].strftime('%Y-%m-%d')
        next_earnings_date.append(report_date)

        time.sleep(1)

    df_excellent['earnings_date'] = next_earnings_date

    return df_excellent


def postfile(converted_data, title, country, channel='#stock_batch'):
    client = slack.WebClient(token=os.environ['SLACK_TOKEN'])
    try:
        text = f'minervini_screening_{title}'
        response = client.chat_postMessage(
            channel=channel,
            text=text
        )
        response = client.files_upload(
            content=converted_data,
            channels=channel,
            filename=text+'.csv',
            title=text
        )
        assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")
        print(e.response)
    time.sleep(0.5)


def save(obj, path):
    """
    Save output as a pickle file.
    """
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


def load(path):
    """
    Load a output from .pkl
    """
    with open(path, mode='rb') as f:
        obj = pickle.load(f)
    return obj


@click.command()
@click.option('--country', '-c', default='us')
def main(country):
    tickers, ja_stockcode = get_tickers(country)
    # ja_stockcodeがNoneでない場合はcountryがjaでなければならない
    if not isinstance(ja_stockcode, type(None)):
        if country != 'ja':
            raise ValueError('日本株選定しているはずなのにja_stockcodeがNoneの場合はおかしい')
    
    ibd_rs_dict, charts = get_ibd_rs(tickers, country=country)
    df_rs_rank = get_rs_rank(ibd_rs_dict)
    tickers_rs_growth = get_recent_strong_tickers(df_rs_rank, rs_thres=90)
    growth_stocks, rank_diff = \
        get_growth_stocks(tickers_rs_growth, charts=charts, country=country)
    df_excellent = \
        filter_excellent_tickers(rank_diff, ja_stockcode, country=country, ja_stockcode=ja_stockcode)
    df_excellent = add_next_earnings_date(df_excellent)
    
    dstdir = Path(__file__).parent
    title = f'{datetime.now().strftime("%Y%m%d")}_{country}'
    converted_data = df_excellent.to_csv(index=False).encode('utf-8')

    postfile(converted_data, title, country, channel='#stock_batch')


if __name__ == "__main__":
    main()
