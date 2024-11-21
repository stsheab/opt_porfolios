
import functions

import datetime

import numpy as np
import pandas as pd
import streamlit as st


# daily data of adjusted close prices
# notice that stocks in default_stock_list must be in csv
# or change default_stock_list

file_name = r'stock_10_adj_close_2010_2021.csv'

default_stock_list = ['NFLX', 'MSFT', 'AMZN', 'GE', 'F']


def run():

    @ st.cache
    def load_local_data(file_name) -> pd.DataFrame:
        df = functions.read_csv(file_name)
        return df

    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])
        if uploaded_file is not None:
            data = load_local_data(uploaded_file)
        else:
            data = load_local_data(file_name)

        portfolios_num = st.slider(
            "Number of Monte-Carlo simulations", min_value=1000, max_value=10000, value=5000)

        stock_list = data.columns.to_list()

        stocks = st.multiselect('Stocks', key=1, options=stock_list,
                                default=default_stock_list)

    daily_data = data[stocks]

    stocks_num = len(stocks)

    max_sharpe_ratio, res, fig = functions.main(
        daily_data, stocks_num, portfolios_num, stocks)

    st.pyplot(fig)
    st.write(f'max Sharpe = {max_sharpe_ratio:.3f}')
    st.dataframe(res)


if __name__ == '__main__':
    run()
