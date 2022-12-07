import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime

import config
from preprocessor.yahoodownloader import YahooDownloader
from preprocessor.preprocessors import FeatureEngineer, data_split, series_decomposition
# from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from env.env_portfolio import StockPortfolioEnv
from models import DRLAgent
from plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, convert_daily_return_to_pyfolio_ts

import itertools


def train_stock_trading(dataset):
    """
    train an agent
    """
    if dataset == 'DOW_30_TICKER':
        Ticker_list = config.DOW_30_TICKER
    if dataset == 'HSI_50_TICKER':
        Ticker_list = config.HSI_50_TICKER
    if dataset == 'SSE_50_TICKER':
        Ticker_list = config.SSE_50_TICKER

    print("==============Start Fetching Data===========")
    df = YahooDownloader(
        portfolio_name=dataset,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=Ticker_list,
    ).fetch_data()

    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False,
    )

    df = fe.preprocess_data(df)


    # add covariance matrix as states
    # print(df.head())
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
    # cov_list = []
    return_list = []
    price_list = []
    dec_price = []

    # look back is one year
    lookback=251
    for i in range(lookback,len(df.index.unique())):
      data_lookback = df.loc[i-lookback:i,:]
      price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
      price_list.append(price_lookback.values)
      # dec_price.append(series_decomposition(price_lookback.values, config.MAX_LEVEL))
      return_lookback = price_lookback.pct_change().dropna()
      return_list.append(return_lookback)
      # covs = return_lookback.cov().values 
      # cov_list.append(covs)

    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'price_list':price_list})
    #after merged, df[day,:][cov_list] has 30 array with shape[30,30], ang df length from 3145-->2893
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    # Training & Trading data split
    train = data_split(df, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(df, config.START_TRADE_DATE, config.END_DATE)

    # test = train.loc[1,:]
    # close_price = test["price_list"].values[0]
    # state= np.append(
    #         np.array(covs),
    #         [test[tech].values.tolist() for tech in config.TECHNICAL_INDICATORS_LIST],
    #         axis=0,
    #     )
    # print(close_price.shape)
    # exit()

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        "lookback": lookback+1, #lookback=251, but the data length is 252
        "agent_num": config.AGENT_NUM
        }

    e_train_gym = StockPortfolioEnv(df=train, **env_kwargs)
    e_trade_gym = StockPortfolioEnv(df=trade, turbulence_threshold=250, **env_kwargs)
    # env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=e_train_gym)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    model_a2c = agent.get_model("a2c")
    trained_a2c = agent.train_model(
        model=model_a2c, tb_log_name="a2c", total_timesteps=50000, eval_env = e_trade_gym
    )

    print("==============Start Trading===========")
    

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_a2c, environment = e_trade_gym
    )
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + dataset + str(config.AGENT_NUM) + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + dataset + str(config.AGENT_NUM) + ".csv")

    print("==============Get Backtest Results===========")
 
    from pyfolio import timeseries
    DRL_strat = convert_daily_return_to_pyfolio_ts(df_account_value)
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func( returns=DRL_strat, 
                              factor_returns=DRL_strat, 
                                positions=None, transactions=None, turnover_denom="AGB")
    print("==============DRL Strategy Stats===========")
    print(perf_stats_all)

    #baseline stats
    # print("==============Get Baseline Stats===========")
    # baseline_df = get_baseline(
    #         dataset = dataset,
    #         ticker="^DJI", 
    #         start = df_account_value.loc[0,'date'],
    #         end = df_account_value.loc[len(df_account_value)-1,'date'])

    # stats = backtest_stats(baseline_df, value_col_name = 'close')
