from dateutil.parser import parse
from telegram.ext import MessageHandler, Filters
from telegram.ext import Updater
from typing import Tuple
import datetime
import os
import pandas as pd
import pytz
import sqlite3
import subprocess
import telegram
import time

DB_PATH = os.environ['DB_PATH']
CHAT_ID = os.environ['CHAT_ID']
BOT_TOKEN = os.environ['BOT_TOKEN']


# Database
def query_data(db_path):
    columns = [
        'id', 'trade_id', 'ft_order_side', 'ft_pair', 'ft_is_open', 'order_id',
        'status', 'symbol', 'order_type', 'side', 'price', 'amount', 'filled',
        'remaining', 'cost', 'order_date', 'order_filled_date',
        'order_update_date'
    ]
    table_name = 'orders'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f'SELECT * FROM {table_name}')
    results = c.fetchall()
    result_df = pd.DataFrame(results, columns=columns)
    return result_df


def get_closed_buy_sell_orders(
        orders: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    closed_orders = orders[orders.amount == orders.filled]
    closed_buy_orders = closed_orders[closed_orders.ft_order_side == 'buy']
    closed_sell_orders = closed_orders[closed_orders.ft_order_side == 'sell']
    return closed_buy_orders, closed_sell_orders


def compute_trade_pair_results(
        closed_buy_orders: pd.DataFrame,
        closed_sell_orders: pd.DataFrame) -> pd.DataFrame:
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


def key_metrics(db_path) -> str:
    orders = query_data(db_path)
    closed_buy_orders, closed_sell_orders = get_closed_buy_sell_orders(orders)
    trade_pair_profits = compute_trade_pair_results(closed_buy_orders,
                                                    closed_sell_orders)
    fees = round((closed_sell_orders.cost.sum() + closed_buy_orders.cost.sum()) * 0.001, 2)
    msg = (
        f'''Avg Trade Percentage: {round(trade_pair_profits.perc_profit.mean(), 2)}%\n'''
        f'''Net Profit: ${round(trade_pair_profits.net_profit.sum() - fees, 2)}''')

    return msg


def check_for_alarms(db_path: str) -> str:
    orders = query_data(db_path)
    closed_buy_orders, closed_sell_orders = get_closed_buy_sell_orders(orders)
    trade_pair_profits = compute_trade_pair_results(closed_buy_orders,
                                                    closed_sell_orders)
    trade_pair_profits['order_date_sell'] = trade_pair_profits[
        'order_date_sell'].apply(
            lambda x: parse(x).astimezone(pytz.timezone('UTC')))

    now = datetime.datetime.now(pytz.timezone('UTC'))

    alarm = ''

    # alarm 1: sale with loss < -20 % in last hour
    alarm_I_df = trade_pair_profits[(trade_pair_profits.order_date_sell >=
                                     now - datetime.timedelta(hours=1))
                                    & (trade_pair_profits.perc_profit < -20)]
    if alarm_I_df.shape[0] > 0:
        alarm += 'Encountered trades with extreme negative loss in last hour:\n'
        for r, row in alarm_I_df.iterrows():
            alarm += f'{row.ft_pair_sell}: {round(row.perc_profit, 2)}%\n'

    # alarm 2: average deal percentages of last hour < -5 %
    alarm_II_df = trade_pair_profits[(trade_pair_profits.order_date_sell >=
                                      now - datetime.timedelta(hours=1))]
    avg_deal_percentage = alarm_II_df.perc_profit.mean()
    if avg_deal_percentage < -5:
        alarm += f'The average deal percentage over the last hour was {round(avg_deal_percentage, 2)}%\n'

    # alarm 3: average deal percentage in last 24 hours < 0
    alarm_III_df = trade_pair_profits[(trade_pair_profits.order_date_sell >=
                                       now - datetime.timedelta(hours=24))]
    avg_deal_percentage = alarm_III_df.perc_profit.mean()
    if avg_deal_percentage < 0:
        alarm += f'The average deal percentage during the last day was {round(avg_deal_percentage, 2)}%'

    return alarm


def shutdown_bot():
    os.chdir(os.path.expanduser('~/freqtrade_I/ft_userdata'))
    cmd = "docker-compose down"
    response = os.system(cmd)
    status = 'Success' if response == 0 else 'Failed'
    return status


def restart_bot():
    os.chdir(os.path.expanduser('~/freqtrade_I/ft_userdata'))
    cmd = "docker-compose up -d"
    response = os.system(cmd)
    status = 'Success' if response == 0 else 'Failed'
    return status


# Telegram
def send(msg, chat_id, token):
    """
    Send a mensage to a telegram user specified on chatId
    chat_id must be a number!
    """
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=msg)


def echo(update, context):
    command = update.message.text
    if command == 'update':
        update_metrics = key_metrics(DB_PATH)
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=update_metrics)
    elif command == 'shutdown':
        response = shutdown_bot()
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=response)
    elif command == 'restart':
        st_response = shutdown_bot()
        restart_response = restart_bot()
        msg = f'Shutdown: {st_response}\nRestart: {restart_response}'
        context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    elif command == 'chat_id':
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text='Chat ID: {update.effective_chat.id}')


if __name__ == '__main__':
    updater = Updater(token=BOT_TOKEN, use_context=True)

    echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)
    updater.dispatcher.add_handler(echo_handler)

    updater.start_polling()

    while (True):
        alarms = check_for_alarms(DB_PATH)
        if len(alarms) > 0:
            send(alarms, CHAT_ID, BOT_TOKEN)
        time.sleep(60 * 60)
