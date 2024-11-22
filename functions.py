
import datetime
from functools import partial

import matplotlib
import matplotlib.pyplot as plt

# import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import scipy.optimize as opt
import sklearn.covariance as skcov


def read_csv(file_name) -> pd.DataFrame:
    data = pd.read_csv(file_name)
    data.Date = pd.to_datetime(data.Date)
    data.set_index('Date', inplace=True)
    return data


def calc_returns(daily_data: pd.DataFrame, *, log_returns=True, period='Y') -> pd.DataFrame:
    if log_returns:
        returns_daily = np.log((daily_data/daily_data.shift(1))).dropna()
    else:
        returns_daily = daily_data.pct_change().dropna()

    return returns_daily.resample(period).sum()


def calc_covariance(returns: pd.DataFrame, *, shrinkage=False) -> pd.DataFrame:
    if shrinkage:
        cov_yearly_oas, alpha = skcov.oas(returns)
        return pd.DataFrame(cov_yearly_oas)
    else:
        return returns.cov()


def calc_volatility(x, covariance_matrix: pd.DataFrame) -> float:
    return np.sqrt(x.T@covariance_matrix.values@x).item()


def calc_portfolios_monte_carlo(portfolios_num: int, stocks_num: int,
                                covariance_matrix: pd.DataFrame, returns_mean: np.array) -> pd.DataFrame:
    portfolio_returns = []
    portfolio_volatility = []
    sharpe_ratio = []
    stock_weights = []
    for every_portfolio in range(portfolios_num):
        weights_raw = np.random.random(stocks_num).reshape(stocks_num, 1)
        weights = weights_raw / weights_raw.sum()
        returns = (returns_mean.T@weights).item()
        volatility = calc_volatility(weights, covariance_matrix)
        sharpe = returns / volatility
        portfolio_returns.append(returns)
        portfolio_volatility.append(volatility)
        sharpe_ratio.append(sharpe)
        stock_weights.append(weights)
        portfolio = {'returns': portfolio_returns,
                     'volatility': portfolio_volatility,
                     'sharpe_ratio': sharpe_ratio, }
    return pd.DataFrame(portfolio)


def calc_efficient_frontier(return_: float, stocks_num: int, covariance_matrix: pd.DataFrame, returns_mean: np.array) -> np.array:
    weights_initial = np.ones(stocks_num) / stocks_num
    return_constr = {'type': 'eq',
                     'fun': lambda x: (returns_mean.T@x).item() - return_
                     }
    weights_constr = {'type': 'eq',
                      'fun': lambda x: np.sum(x) - 1
                      }
    bnds = opt.Bounds(np.zeros_like(weights_initial),
                      np.ones_like(weights_initial) * np.inf)

    res = opt.minimize(calc_volatility, weights_initial, args=(covariance_matrix), method='SLSQP',
                       constraints=[return_constr, weights_constr], bounds=bnds)
    return res.x


def plot_results(portf_vol: pd.Series, portf_ret: pd.Series,
                 vol_opt: list, ret_range: list, v_shrp_max: float, r_shrp_max: float) -> plt:

    fig = plt.figure(num=1, figsize=(6, 4), dpi=72,
                     facecolor='w')

    plt.grid()
    plt.scatter(portf_vol, portf_ret,
                s=30, c='navy', edgecolors='k', linewidths=0.5, alpha=0.7, edgecolor='k')
    plt.plot(vol_opt, ret_range, 'r--')
    plt.scatter(v_shrp_max, r_shrp_max, s=100, c='g')
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    # plt.title(f'Efficient Frontier')
    # plt.legend()
    return fig
    # plt.show()


def plot_results_plotly(portf_vol: pd.Series, portf_ret: pd.Series,
                        vol_opt: list, ret_range: list, v_shrp_max: float, r_shrp_max: float) -> plt:

    m = 20

    data = go.Scatter(x=portf_vol, y=portf_ret, mode='markers',
                      marker=dict(color='rgba(10, 10, 100, 0.75)', size=5, line=dict(width=0.5, color='black')))
    fig = go.Figure(data, layout=go.Layout(width=700, height=500,
                    margin=go.layout.Margin(t=m, b=m, r=m, l=m)))

    # plt.plot(vol_opt, ret_range, 'r--')
    # plt.scatter(v_shrp_max, r_shrp_max, s=100, c='g')

    return fig


def main(daily_data: pd.DataFrame, stocks_num: int, portfolios_num: int, stocks: list) -> tuple:
    returns = calc_returns(daily_data, log_returns=True, period='Y')

    covariance = calc_covariance(returns, shrinkage=False)

    returns_mean = returns.mean().values.reshape(stocks_num, 1)

    portfolios_df = calc_portfolios_monte_carlo(
        portfolios_num, stocks_num, covariance, returns_mean)

    returns_range = np.arange(portfolios_df.returns.min(),
                              portfolios_df.returns.max(), .01)

    weights_optimal = np.array(list(map(partial(
        calc_efficient_frontier, stocks_num=stocks_num, covariance_matrix=covariance,
        returns_mean=returns_mean), returns_range)))

    volatilities_optimal = [calc_volatility(
        i, covariance) for i in weights_optimal]

    shrp_argmax = (returns_range/volatilities_optimal).argmax()
    ret_shrp_max = returns_range[shrp_argmax]
    vol_shrp_max = volatilities_optimal[shrp_argmax]
    max_sharpe_ratio = ret_shrp_max/vol_shrp_max
    weights_max_sharpe = weights_optimal[shrp_argmax]

    res = pd.DataFrame(
        {'weights': weights_max_sharpe.round(3)}, index=stocks).T

    fig = plot_results(portfolios_df.volatility, portfolios_df.returns,
                              volatilities_optimal, returns_range, vol_shrp_max, ret_shrp_max)

    return max_sharpe_ratio, res, fig
