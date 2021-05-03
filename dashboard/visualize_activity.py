from dateutil.parser import parse
import datetime
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import pytz
import sqlite3
import streamlit as st

DB_PATH = os.environ.get('DB_PATH')
MAX_TICKERS = 10
STAKE_CURRENCY = 'BTC'
STAKE_AMOUNT = 0.01
START_DATE = datetime.datetime(2021, 5, 1)


def query_data(table_name, sqlite_file, columns):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    c.execute(f'SELECT * FROM {table_name}')
    results = c.fetchall()
    result_df = pd.DataFrame(results, columns=columns)
    return result_df


def get_current_cash(closed_buy_orders, closed_sell_orders):
    initial_cash = MAX_TICKERS * STAKE_AMOUNT
    purchase_sum = closed_buy_orders.cost.sum()
    sale_sum = closed_sell_orders.cost.sum()
    return initial_cash - purchase_sum + sale_sum


def get_signed_costs(orders):
    side_factors = (orders.side == 'sell').values.astype(int)
    side_factors[np.where(side_factors == 0)] = -1
    signed_costs = orders.cost.values * side_factors
    return signed_costs


def get_cash_history(closed_orders):
    signed_costs = get_signed_costs(closed_orders)
    cash_development = np.cumsum(np.insert(signed_costs, 0, MAX_TICKERS*STAKE_AMOUNT))

    execution_dates = closed_orders.order_filled_date.values
    execution_dates = [parse(x).astimezone(pytz.timezone('CET')) for x in execution_dates]
    execution_dates.insert(0, START_DATE)

    return execution_dates, cash_development


def get_portfolio_history_value(closed_orders):
    # a purchase increases ticker portfolio value by the costs
    # sale decreases ticker portfolio value by the costs
    closed_orders['portfolio_delta'] = closed_orders['cost']
    sell_idxs = closed_orders[closed_orders.side == 'sell'].index
    for idx in sell_idxs:
        trade_id = closed_orders['trade_id'].loc[idx]
        previous_orders = closed_orders[closed_orders.index < idx]
        original_cost = previous_orders[previous_orders.trade_id == trade_id].cost.iloc[0]
        closed_orders['portfolio_delta'].loc[idx] = -original_cost
    signed_portfolio_value_deltas = np.cumsum(
        np.insert(closed_orders['portfolio_delta'].values, 0, 0))
    return signed_portfolio_value_deltas


columns = ['id', 'trade_id', 'ft_order_side', 'ft_pair', 'ft_is_open', 'order_id', 'status',
           'symbol', 'order_type', 'side', 'price', 'amount', 'filled', 'remaining', 'cost',
           'order_date', 'order_filled_date', 'order_update_date']
orders = query_data('orders', DB_PATH, columns)


open_orders = orders[orders.amount != orders.filled]
closed_orders = orders[orders.amount == orders.filled]

open_buy_orders = open_orders[open_orders.ft_order_side == 'buy']
open_sell_orders = open_orders[open_orders.ft_order_side == 'sell']

closed_buy_orders = closed_orders[closed_orders.ft_order_side == 'buy']
closed_sell_orders = closed_orders[closed_orders.ft_order_side == 'sell']


cash_execution_dates, cash_development = get_cash_history(closed_orders)
portfolio_value_deltas = get_portfolio_history_value(closed_orders)

assert(len(portfolio_value_deltas) == len(cash_development))

portfolio_development = cash_development + portfolio_value_deltas

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

trace = go.Scatter(x=open_sell_orders.order_date,
                   y=open_sell_orders.cost,
                   mode='markers',
                   marker={
                       'color': 'blue',
                       'size': 10,
                        'symbol': 'circle',
                   },
                   name='Closed sell orders')
traces.append(trace)

trace = go.Scatter(x=cash_execution_dates,
                   y=cash_development,
                   name='Cash Development')
traces.append(trace)

trace = go.Scatter(x=cash_execution_dates,
                   y=portfolio_value_deltas,
                   name='Portfolio Deltas')
traces.append(trace)


trace = go.Scatter(x=cash_execution_dates,
                   y=portfolio_development,
                   name='Portfolio Development')
traces.append(trace)


g = go.Figure(data=traces)

g.update_layout(title='Portfolio Development',
                xaxis_title='Time',
                yaxis_title='BTC')

######################################################
st.title('Freqtrade Bot Activity')

st.plotly_chart(g)

start_cash = MAX_TICKERS * STAKE_AMOUNT
portfolio_value = portfolio_development[-1]
fees = orders.cost.sum() * 0.001
profit_perc = (portfolio_value - start_cash - fees) / start_cash

st.markdown("## Key Indicators")
st.json({"Percentage Profit": round(profit_perc * 100, 2),
         "Portfolio Value": portfolio_value,
         "Available Cash": cash_development[-1],
         "Fees": fees})

st.markdown("## Open Orders")
st.dataframe(open_orders)

st.markdown("## Closed Orders")
st.dataframe(closed_orders)
