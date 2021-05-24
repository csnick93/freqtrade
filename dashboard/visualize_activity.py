from binance.client import Client
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import streamlit as st

DB_PATH = os.environ.get('DB_PATH')
STAKE_CURRENCY = 'USDT'
FEE_PERCENTAGE = 0.001


def query_data(table_name, sqlite_file, columns):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    c.execute(f'SELECT * FROM {table_name}')
    results = c.fetchall()
    result_df = pd.DataFrame(results, columns=columns)
    return result_df


def get_current_crypto_value(closed_buy_orders, closed_sell_orders):
    return closed_buy_orders.cost.sum() - closed_sell_orders.cost.sum()


def get_profit_history(closed_buy_orders, closed_sell_orders):
    # compute value of portfolio over time
    #   - portfolio value changes only for sales, i.e. finished deal
    #   - changes by the amount made
    deal_pairs = compute_trade_pair_results(closed_buy_orders,
                                            closed_sell_orders)
    deal_pairs.sort_values('order_date_sell', ascending=True, inplace=True)
    profit_value_history = np.cumsum(deal_pairs['net_profit'].values)
    profit_timepoints = deal_pairs['order_date_sell'].values

    return profit_value_history, profit_timepoints


def compute_trade_pair_results(closed_buy_orders, closed_sell_orders):
    deal_pairs = closed_sell_orders.merge(closed_buy_orders,
                                          on='trade_id',
                                          how='inner',
                                          suffixes=('_sell', '_buy'))[[
                                              'ft_pair_sell', 'cost_buy',
                                              'cost_sell', 'order_date_sell'
                                          ]]
    deal_pairs['perc_profit'] = (
        deal_pairs['cost_sell'] -
        deal_pairs['cost_buy']) / deal_pairs['cost_buy'] * 100
    deal_pairs['net_profit'] = deal_pairs['cost_sell'] - deal_pairs['cost_buy']
    del deal_pairs['cost_buy']
    del deal_pairs['cost_sell']

    deal_pairs.sort_values('order_date_sell', ascending=False, inplace=True)

    return deal_pairs[[
        'ft_pair_sell', 'perc_profit', 'net_profit', 'order_date_sell'
    ]]


def get_binance_client():
    api_key = os.environ.get('BINANCE_API_KEY', None)
    secret_key = os.environ.get('BINANCE_SECRET_KEY', None)
    if api_key is None or secret_key is None:
        raise RuntimeError('No binance credentials available')
    client = Client(api_key, secret_key)
    return client


def get_current_price(client, ticker: str) -> float:
    price = float(client.get_symbol_ticker(symbol=ticker)['price'])
    return price


def get_current_crypto_portfolio(closed_buy_orders, closed_sell_orders):
    client = get_binance_client()
    deal_pairs = closed_sell_orders.merge(closed_buy_orders,
                                          on='trade_id',
                                          how='inner',
                                          suffixes=('_sell', '_buy'))
    crypto_positions = closed_buy_orders[~closed_buy_orders.id.
                                         isin(deal_pairs['id_buy'])]
    crypto_positions = crypto_positions[['symbol', 'order_date', 'price', 'amount']]
    crypto_positions['current_price'] = crypto_positions.symbol.apply(
        lambda x: get_current_price(client, x.replace('/', '')))
    crypto_positions['current_perc'] = (
        crypto_positions['current_price'] -
        crypto_positions['price']) / crypto_positions['price']
    crypto_positions = crypto_positions[[
        'symbol', 'current_perc', 'current_price', 'price', 'order_date', 'amount'
    ]]
    crypto_positions.sort_values('order_date', ascending=True, inplace=True)
    return crypto_positions


columns = [
    'id', 'trade_id', 'ft_order_side', 'ft_pair', 'ft_is_open', 'order_id',
    'status', 'symbol', 'order_type', 'side', 'price', 'amount', 'filled',
    'remaining', 'cost', 'order_date', 'order_filled_date', 'order_update_date'
]
orders = query_data('orders', DB_PATH, columns)

open_orders = orders[orders.amount != orders.filled]
closed_orders = orders[orders.amount == orders.filled]

open_buy_orders = open_orders[open_orders.ft_order_side == 'buy']
open_sell_orders = open_orders[open_orders.ft_order_side == 'sell']

closed_buy_orders = closed_orders[closed_orders.ft_order_side == 'buy']
closed_sell_orders = closed_orders[closed_orders.ft_order_side == 'sell']

profit_development, profit_timepoints = get_profit_history(
    closed_buy_orders, closed_sell_orders)

############################################################

traces = []

trace = go.Scatter(x=open_buy_orders.order_date,
                   y=open_buy_orders.cost,
                   mode='markers',
                   marker={
                       'color': 'green',
                       'size': 10,
                       'symbol': 'circle-open',
                   },
                   name='Open buy orders')
traces.append(trace)

trace = go.Scatter(x=open_sell_orders.order_date,
                   y=open_sell_orders.cost,
                   mode='markers',
                   marker={
                       'color': 'blue',
                       'size': 10,
                       'symbol': 'circle-open',
                   },
                   name='Open sell orders')
traces.append(trace)

trace = go.Scatter(x=closed_buy_orders.order_date,
                   y=closed_buy_orders.cost,
                   mode='markers',
                   marker={
                       'color': 'green',
                       'size': 10,
                       'symbol': 'circle',
                   },
                   name='Closed buy orders')
traces.append(trace)

trace = go.Scatter(x=closed_sell_orders.order_date,
                   y=closed_sell_orders.cost,
                   mode='markers',
                   marker={
                       'color': 'blue',
                       'size': 10,
                       'symbol': 'circle',
                   },
                   name='Closed sell orders')
traces.append(trace)

trace = go.Scatter(x=profit_timepoints,
                   y=profit_development,
                   name='Profit History')
traces.append(trace)

g = go.Figure(data=traces)

g.update_layout(title='Profit Development',
                xaxis_title='Time',
                yaxis_title=STAKE_CURRENCY)

######################################################
st.title('Freqtrade Bot Activity')

st.plotly_chart(g, config={'displayModeBar': False})

cash_spent = closed_buy_orders.cost.sum()
cash_earned = closed_sell_orders.cost.sum()
crypto_portfolio = get_current_crypto_portfolio(closed_buy_orders, closed_sell_orders)

fees = orders.cost.sum() * FEE_PERCENTAGE
net_spending = closed_buy_orders.cost.sum() - closed_sell_orders.cost.sum()

trade_pair_profits = compute_trade_pair_results(closed_buy_orders,
                                                closed_sell_orders)
deal_percentages = trade_pair_profits.perc_profit
perc_profit = np.round(np.mean(deal_percentages), 2)
net_profit = trade_pair_profits.net_profit.sum()

st.markdown("## Key Indicators")
st.json({
    "Average Deal Percentage Profit": f'{perc_profit} %',
    "Net Profit": f'$ {round(net_profit, 2)}',
    "Net Spending": f'$ {round(net_spending, 2)}',
    "Number of Trades": len(deal_percentages),
    "Fees": f'$ {np.round(fees, 2)}',
    "Crypto Value": f'$ {(crypto_portfolio.current_price * crypto_portfolio.amount).sum()}'
})

st.markdown('## Deal Outcomes')
fig = go.Figure(
    data=[go.Histogram(x=deal_percentages, autobinx=False, nbinsx=20)])
fig.update_layout(title='Deal Percentage Distributions')
st.plotly_chart(fig, config={'displayModeBar': False})

wins, losses = sum(deal_percentages > 0.2), sum(deal_percentages <= 0.2)
fig = go.Figure([go.Bar(x=['losses', 'wins'], y=[losses, wins])])
fig.update_layout(title='Deal Outcomes')
st.plotly_chart(fig, config={'displayModeBar': False})

st.markdown('## Crypto Portfolio')
st.dataframe(crypto_portfolio)

st.markdown('## Trade Pair Profits')
st.dataframe(trade_pair_profits)

st.markdown("## Open Orders")
open_orders.sort_values('order_date', ascending=False, inplace=True)
st.dataframe(open_orders[[
    'ft_pair', 'ft_order_side', 'price', 'amount', 'cost', 'order_date'
]])

st.markdown("## Closed Orders")
closed_orders.sort_values('order_date', ascending=False, inplace=True)
st.dataframe(closed_orders[[
    'ft_pair', 'ft_order_side', 'price', 'amount', 'cost', 'order_date'
]])
