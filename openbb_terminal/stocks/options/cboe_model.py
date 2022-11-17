"""CBOE Model"""
__docformat__ = "numpy"

import logging
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame as pdf
import requests

from openbb_terminal.decorators import log_start_end
from openbb_terminal.rich_config import console
from openbb_terminal.stocks.options.op_helpers import get_dte_from_expiration as get_dte

logger = logging.getLogger(__name__)
# pylint: disable=unsupported-assignment-operation

stock = 'stock'
index = 'index'

ticker_exceptions = ['NDX', 'RUT']

@log_start_end(log=logger)
def get_expirations(symbol: str) -> List[str]:
    """ Get option chains for a given symbol """

    symbol_cboe, quotes_url, _, _ = _get_ticker_urls_cboe(ticker=symbol)

    _, _, expirations_list = _get_ticker_info_cboe(ticker_cboe = symbol_cboe)

    return expirations_list

@log_start_end(log=logger)
def get_option_chain(symbol: str, expiry: str = None) -> pd.DataFrame:
    """ Get option chains for a given symbol """

    symbol_cboe, quotes_url, _, _ = _get_ticker_urls_cboe(ticker=symbol)

    symbol_info_json, symbol_details, expirations_list = _get_ticker_info_cboe(ticker_cboe = symbol_cboe)
    options_quotes = _get_options_quotes(symbol_info_json, quotes_url)
    options_chains = _get_options_chains_from_options_quotes(options_quotes)

    if expiry:
        options_chains = options_chains[options_chains['Expiration'] == expiry]

    return options_chains

@log_start_end(log=logger)
def _get_options_quotes(symbol_info_json, quotes_url):
    ### Gets quotes and greeks data and returns a dataframe: options_quotes ###

    if symbol_info_json.success == False:
        print('No Data Found')
    else:
        r = requests.get(quotes_url)

    r_json = r.json()
    data = pd.DataFrame(r_json['data'])

    symbol_columns = ['symbol', 'security_type', 'exchange_id', 'current_price', 'price_change', 'price_change_percent', 'bid','bid_size', 'ask', 'ask_size', 'open', 'high', 'low', 'close', 'prev_day_close', 'volume', 'iv30', 'iv30_change', 'iv30_change_percent', 'seqno', 'last_trade_time', 'tick']
        
    # symbol_info = pd.DataFrame(data = data[0:1], columns = symbol_columns)

    options = pd.Series(data.options, index = data.index)
    options_columns = list(options[0])
    options_data = list(options[:])
    options_df = pd.DataFrame(options_data, columns = options_columns)

    pd.DataFrame(options_df).rename(columns = {
        'option': 'Option Symbol',
        'bid': 'Bid',
        'bid_size': 'Bid Size',
        'ask': 'Ask',
        'ask_size': 'Ask Size',
        'iv': 'IV',
        'open_interest': 'OI',
        'volume': 'Vol',
        'delta': 'Delta',
        'gamma': 'Gamma',
        'theta': 'Theta',
        'rho': 'Rho',
        'vega': 'Vega',
        'theo': 'Theoretical Price',
        'change': 'Change',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'tick': 'Tick',
        'last_trade_price': 'Last Price',
        'last_trade_time': 'Last Timestamp',
        'percent_change': '% Change',
        'prev_day_close': 'Prev Close',
    }, inplace=True)

    options_df_order = ['Option Symbol', 'Theoretical Price', 'Last Price', 'Tick', 'Prev Close', '% Change', 'Open', 'High', 'Low','Bid Size', 'Bid', 'Ask', 'Ask Size', 'Vol', 'OI', 'IV', 'Theta', 'Delta', 'Gamma', 'Vega', 'Rho', 'Last Timestamp']

    options_df = pd.DataFrame(options_df, columns = options_df_order).set_index(keys = 'Option Symbol')

    options_quotes = options_df

    return options_quotes

def _get_options_chains_from_options_quotes(options_quotes):
    ### Breaks down options_quotes into a multi-index dataframe and returns: options_chains ###
    options_chains = pd.Series(options_quotes.index).str.extractall(r'^(?P<Ticker>\D*)(?P<Expiration>\d*)(?P<Type>\D*)(?P<Strike>\d*)')
    options_chains = pdf(options_chains, columns = ['Expiration', 'Strike', 'Type', 'Prev Close', 'Last', 'Bid Size', 'Bid', 'Ask', 'Ask Size', 'Tick', 'OI', 'Vol', 'IV', 'Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'Theoretical', 'Last Timestamp'])
    options_chains['Last Timestamp'] = options_quotes['Last Timestamp'].values
    options_chains['Prev Close'] = options_quotes['Prev Close'].values
    options_chains.Last = options_quotes['Last Price'].values
    options_chains['Bid Size'] = options_quotes['Bid Size'].values
    options_chains.Bid = options_quotes['Bid'].values
    options_chains.Ask = options_quotes['Ask'].values
    options_chains['Ask Size'] = options_quotes['Ask Size'].values
    options_chains.Underlying = ticker
    
    options_quotes.Tick = pd.Series(options_quotes.Tick).str.replace(r'\_', repl = ' ', regex = True)
    options_chains.Tick = options_quotes['Tick'].values
    options_chains.IV = options_quotes['IV'].values
    options_chains.Delta = options_quotes['Delta'].values
    options_chains.Gamma = options_quotes['Gamma'].values
    options_chains.Rho = options_quotes['Rho'].values
    options_chains.Theta = options_quotes['Theta'].values
    options_chains.Vega = options_quotes['Vega'].values
    options_chains.Vol = options_quotes['Vol'].values
    options_chains.OI = options_quotes['OI'].values
    options_chains.Theoretical = options_quotes['Theoretical Price'].values
    options_chains = pdf(options_chains).set_index(keys = ['Expiration', 'Strike', 'Type'])
    options_chains.index.get_level_values(1).drop_duplicates(keep = 'first')
    options_chains.sort_index(axis = 0, level = ['Expiration', 'Strike', 'Type'], sort_remaining = False, ignore_index = False, inplace = True)
    options_chains = options_chains.convert_dtypes()

    return options_chains

def _get_directory_cboe():
    ### Gets the symbol directory from CBOE and returns a multi-index dataframe, grouping names with multiple ticker symbols: directory_cboe ###

    book = pd.read_json(path_or_buf='https://cdn.cboe.com/api/global/delayed_quotes/symbol_book/symbol-book.json')
    directory = list(book.data)
    directory_columns = list(directory[0])
    directory_cboe = pd.DataFrame(directory, columns = directory_columns)

    directory_cboe = pd.DataFrame(directory_cboe).rename(columns={
        'name': 'Symbol',
        'company_name': 'Name',
    },)

    directory_cboe = pd.DataFrame(directory_cboe).reset_index()
    return directory_cboe

def _get_countries_cboe():
    ### Gets a dataframe of Countries and ISO 2 & 3 letter country codes: countries_cboe ###
    countries_json = pd.read_json(path_or_buf='https://cdn.cboe.com/resources/general/countries.json')
    countries_columns = list(countries_json)
    countries_cboe = pd.DataFrame(countries_json, columns = countries_columns)
    countries_cboe = pd.DataFrame(countries_cboe).rename(columns = {
        'name': 'Country',
        'alpha_2_code': 'Two Letter Code',
        'alpha_3_code': 'Three Letter Code',
    })

    countries_cboe = pd.DataFrame(countries_cboe).set_index(keys = 'Country')
    return countries_cboe


def _get_options_roots_cboe():
    ### Gets a Series of options roots: options_roots ###
    roots = requests.get('https://cdn.cboe.com/api/global/delayed_quotes/symbol_book/option-roots.json')
    roots_json = pd.DataFrame(roots.json())
    options_roots = pd.Series(roots_json.data)
    return options_roots

def _get_all_indices_cboe():
    ### Gets all indices listed on the CBOE and returns the dataframe: cboe_us_indices
    indices = pd.read_json(path_or_buf='https://cdn.cboe.com/api/global/us_indices/definitions/all_indices.json')
    indices = pdf(indices).rename(columns = {
        'calc_end_time': 'Close Time',
        'calc_start_time': 'Open Time',
        'currency': 'Currency',
        'description': 'Description',
        'display': 'Display',
        'featured': 'Featured',
        'featured_order': 'Featured Order',
        'index_symbol': 'Ticker',
        'mkt_data_delay': 'Data Delay',
        'name': 'Name',
        'tick_days': 'Tick Days',
        'tick_frequency': 'Frequency',
        'tick_period': 'Period',
        'time_zone': 'Time Zone',
    },)

    indices_order = ['Ticker', 'Description', 'Currency', 'Tick Days', 'Frequency', 'Period', 'Time Zone']
    indices.set_index(keys = ['Name'], inplace = True)
    cboe_us_indices = pdf(indices, columns = indices_order)
    indexes = list(cboe_us_indices.Ticker[:])

    return indexes

def _get_all_futures_cboe():
    ### Gets all futures listed on the CBOE and returns the dataframe: cboe_futures
    futures = pd.read_json(path_or_buf='https://cdn.cboe.com/api/global/delayed_quotes/symbol_book/futures-roots.json')
    futures_list = list(futures.data)
    futures_columns = ['future_root', 'underlying', 'family', 'sort_order', 'name']
    futures_directory = pdf(futures_list, columns = futures_columns)

    futures_directory.rename(columns = {
        'future_root': 'Future Root',
        'underlying': 'Underlying',
        'family': 'Family',
        'sort_order': 'Sort Order',
        'name': 'Name',
    },inplace = True)

    futures_columns_order = ['Name', 'Family', 'Future Root', 'Underlying']
    cboe_futures = pdf(futures_directory, columns = futures_columns_order)
    cboe_futures = cboe_futures.set_index(keys = ['Family'], inplace = True)

    return cboe_futures

def _get_ticker_info_cboe(ticker_cboe: str):
    ### Gets basic info for the ticker, and if results, dataframes are returned as: ticker_info, expirations_list ###
    symbol_info_url = 'https://www.cboe.com/education/tools/trade-optimizer/symbol-info/?symbol='f"{ticker_cboe}"

    symbol_info = requests.get(symbol_info_url)
    symbol_info_json = pd.Series(symbol_info.json())
        
    if symbol_info_json.success == True:
        symbol_details = pd.Series(symbol_info_json['details'])
        symbol_details = pdf(symbol_details).transpose()
        symbol_details.reset_index()

    if symbol_info_json.success == False:
        pdf()

    if symbol_info_json.success == True:
        expirations_list = pd.Series(symbol_info_json['expirations'])

    return symbol_info_json, symbol_details, expirations_list

def _get_expirations_cboe(ticker_cboe: str):
    ### Gets the expiration dates for the ticker, and if results, a dataframe is returned as: expirations_list ###
    _, _, expirations_list = _get_ticker_info_cboe(ticker_cboe)

    return expirations_list

def _get_ticker_urls_cboe(ticker: str):
    ### Checks ticker to determine if ticker is an index or an exception that requires modifying the request's URLs ### 

    indexes = _get_all_indices_cboe()

    if ticker in ticker_exceptions:
        quotes_url = 'https://cdn.cboe.com/api/global/delayed_quotes/options/_'f"{ticker}"'.json'
        quotes_iv_url = 'https://cdn.cboe.com/api/global/delayed_quotes/historical_data/_'f"{ticker}"'.json'
        intraday_url = 'https://cdn.cboe.com/api/global/delayed_quotes/charts/intraday/_'f"{ticker}"'.json'
        ticker_cboe = '^'+ticker

    else:
        if ticker not in indexes:
            quotes_url= 'https://cdn.cboe.com/api/global/delayed_quotes/options/'f"{ticker}"'.json'
            quotes_iv_url = 'https://cdn.cboe.com/api/global/delayed_quotes/historical_data/'f"{ticker}"'.json'
            intraday_url = 'https://cdn.cboe.com/api/global/delayed_quotes/charts/intraday/'f"{ticker}"'.json'
            ticker_cboe = ticker

        if ticker in indexes:
            quotes_url = 'https://cdn.cboe.com/api/global/delayed_quotes/options/_'f"{ticker}"'.json'
            quotes_iv_url = 'https://cdn.cboe.com/api/global/delayed_quotes/historical_data/_'f"{ticker}"'.json'
            intraday_url = 'https://cdn.cboe.com/api/global/delayed_quotes/charts/intraday/_'f"{ticker}"'.json'
            ticker_cboe = '^'+ticker

    # print(ticker, ticker_cboe)
    return ticker_cboe, quotes_url, quotes_iv_url, intraday_url

def _get_ticker_details_cboe(symbol_details):
    ### If no options data is found for ticker, then a message is printed; else, cleans columns depending on if the security type is a stock or an index and returns the dataframe: 'ticker_details'. ###
    type:str = symbol_details.security_type

    if stock[0] in type[0]:
        stock_details = symbol_details
        ticker_details = pdf(stock_details).rename(columns = {
        'symbol': 'Symbol',
        'current_price': 'Current Price',
        'bid': 'Bid',
        'ask': 'Ask',
        'bid_size': 'Bid Size',
        'ask_size': 'Ask Size',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'iv30': 'IV30',
        'prev_day_close': 'Previous Close',
        'price_change': 'Price Change',
        'price_change_percent': 'Price Change %',
        'iv30_change': 'IV30 Change',
        'iv30_percent_change': 'IV30 Change %',
        'last_trade_time': 'Last Trade Time',
        'exchange_id': 'Exchange ID',
        'tick': 'Tick',
        'security_type': 'Type',
        })
        details_columns = ['Symbol','Type', 'Tick', 'Bid', 'Bid Size', 'Ask Size', 'Ask', 'Current Price', 'Open', 'High', 'Low', 'Close', 'Volume', 'Previous Close', 'Price Change', 'Price Change %', 'IV30', 'IV30 Change', 'IV30 Change %', 'Last Trade Time']
        ticker_details = pdf(ticker_details, columns = details_columns)
        ticker_details = pdf(ticker_details).set_index(keys = 'Symbol')
        ticker_details = pdf(ticker_details).dropna(axis = 1)
        
    elif index[0] in type[0]:
        index_details = symbol_details
        ticker_details = pdf(index_details).rename(columns = {
            'symbol': 'Symbol',
            'security_type': 'Type',
            'current_price': 'Current',
            'price_change': 'Change',
            'price_change_percent': 'Change %',
            'tick': 'Tick',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'prev_day_close': 'Previous Close',
            'iv30': 'IV30',
            'iv30_change': 'IV30 Change',
            'iv30_change_percent': 'IV30 Change %',
            'last_trade_time': 'Last Trade Time',
        })
        index_columns = ['Symbol', 'Type', 'Tick', 'Current', 'Open', 'High', 'Low', 'Close', 'Previous Close', 'Change', 'Change %', 'IV30', 'IV30 Change', 'IV30 Change %', 'Last Trade Time']
        ticker_details = pdf(ticker_details, columns = index_columns)
        ticker_details = pdf(ticker_details).set_index(keys = 'Symbol')
        ticker_details = pdf(ticker_details).dropna(axis = 1)
    else:
        print('No options data found for ticker.')
        ticker_details = pdf()

    return ticker_details

def _get_ticker_iv_cboe(quotes_iv_url):
    ### Gets 30d, 60d, 90d, and 1Y implied and historical volatility for the ticker and returns dataframe: ticker_iv. ###
    h_iv = requests.get(quotes_iv_url)
    h_iv_json = pd.DataFrame(h_iv.json())

    h_columns = ['annual_high', 'annual_low', 'hv30_annual_high', 'hv30_annual_low', 'hv60_annual_high', 'hv60_annual_low', 'hv90_annual_high', 'hv90_annual_low', 'iv30_annual_high', 'iv30_annual_low','iv60_annual_high', 'iv60_annual_low', 'iv90_annual_high', 'iv90_annual_low', 'symbol']
    h_data  = h_iv_json[1:]
    h_data = pd.DataFrame(h_iv_json).transpose()
    h_data = h_data[1:2]
    quotes_iv_df = pd.DataFrame(data = h_data, columns = h_columns).reset_index()

    pd.DataFrame(quotes_iv_df).rename(columns = {
        'annual_high':'1Y High',
        'annual_low':'1Y Low',
        'hv30_annual_high': 'HV30 1Y High',
        'hv30_annual_low': 'HV30 1Y Low',
        'hv60_annual_high': 'HV60 1Y High',
        'hv60_annual_low':'HV60 1Y Low',
        'hv90_annual_high': 'HV90 1Y High',
        'hv90_annual_low': 'HV90 1Y Low',
        'iv30_annual_high': 'IV30 1Y High',
        'iv30_annual_low': 'IV30 1Y Low',
        'iv60_annual_high': 'IV60 1Y High',
        'iv60_annual_low': 'IV60 1Y Low',
        'iv90_annual_high': 'IV90 1Y High',
        'iv90_annual_low': 'IV90 1Y Low',
        'symbol': 'Symbol',
    },inplace = True)

    quotes_iv_df.set_index(keys = 'Symbol', inplace=True)

    iv_order = ['IV30 1Y High', 'HV30 1Y High', 'IV30 1Y Low', 'HV30 1Y Low', 'IV60 1Y High', 'HV60 1Y High', 'IV60 1Y Low', 'HV60 1Y low', 'IV90 1Y High', 'HV90 1Y High', 'IV90 1Y Low', 'HV 90 1Y Low']

    ticker_iv = pdf(quotes_iv_df, columns = iv_order)

    return ticker_iv

def _get_ticker_intraday(symbol_info_json, intraday_url):
    # Gets one-minute candles for the underlying stock and returns the dataframe: underlying_intraday ###

    if symbol_info_json.success == False:
        print('No Data Found')

    else:
        intraday = requests.get(intraday_url)
        intraday_json = pd.DataFrame(intraday.json())
        intraday_data = list(intraday_json.data)
        intraday_data_columns = ['datetime', 'sequence_number']

        options_intraday = pd.DataFrame.from_dict(intraday_data)
        options_intraday = pd.DataFrame(options_intraday).set_index(keys = ['datetime', 'sequence_number'])
        intraday_price = list(options_intraday.price)
        price_columns = ['open', 'high', 'low', 'close']
        intraday_price = pd.DataFrame(intraday_price, columns = price_columns)
        volume = list(options_intraday.volume)
        volume_columns = ['stock_volume', 'calls_volume', 'puts_volume', 'total_options_volume']
        intraday_volume = pdf(volume, columns = volume_columns)
        options_columns = intraday_data_columns + price_columns + volume_columns
        timestamp = pdf(intraday_data, columns = intraday_data_columns)

        intraday_price.to_numpy()
        intraday_volume.to_numpy()
        timestamp.to_numpy()
        array = np.append(intraday_volume[:], intraday_price[:], axis = 1)

        array = np.append(array, timestamp, axis = 1)
        array_columns = ['stock_volume', 'calls_volume', 'puts_volume', 'total_options_volume', 'open', 'high', 'low', 'close', 'datetime', 'sequence_number']
        intraday_options = pdf(array, columns = array_columns)
        pdf(intraday_options).rename(columns={
            'stock_volume': 'Stock Volume',
            'calls_volume': 'Calls Volume',
            'puts_volume': 'Puts Volume',
            'total_options_volume': 'Options Volume',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'datetime': 'Timestamp',
            'sequence_number': 'Sequence Number',
        },inplace = True)

        new_intraday_options_order = ['Timestamp', 'Sequence Number', 'Open', 'High', 'Low', 'Close', 'Stock Volume', 'Calls Volume', 'Puts Volume', 'Options Volume']


        intraday_options = pd.DataFrame(data = intraday_options, columns = new_intraday_options_order)
        intraday_options.set_index(keys = ['Timestamp', 'Sequence Number'], inplace = True)
        intraday_options.sort_index(ascending = False, inplace = True)
        underlying_intraday = intraday_options

        return underlying_intraday