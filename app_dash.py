
import plotly.express as px
from dash import Dash, html, dcc
import functions

app = Dash(__name__)


file_name = r'stock_10_adj_close_2010_2021.csv'
default_stock_list = ['NFLX', 'MSFT', 'AMZN', 'GE', 'F']
stocks = default_stock_list
portfolios_num = 5000

data = functions.read_csv(file_name)

daily_data = data[stocks]
stocks_num = len(stocks)

max_sharpe_ratio, res, fig = functions.main(
    daily_data, stocks_num, portfolios_num, stocks)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run(debug=True)


fig
