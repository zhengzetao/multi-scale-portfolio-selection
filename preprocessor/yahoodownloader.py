"""Contains methods and classes to collect data from
Yahoo Finance API
"""
import os
import pandas as pd
import yfinance as yf


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, portfolio_name: str, start_date: str, end_date: str, ticker_list: list):

        self.portfolio_name = portfolio_name
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        portfolio_save_path = "./datasets" + '/' + self.portfolio_name
        if os.path.exists(portfolio_save_path):
            standard_file = pd.read_csv(portfolio_save_path + '/' + self.ticker_list[0] + '.csv',index_col=False)
            standard_date = pd.to_datetime(standard_file['Date'], errors='coerce')
            for tic in self.ticker_list:
                temp_df = pd.read_csv(portfolio_save_path + '/' + tic + '.csv',index_col=False)
                temp_df["tic"] = tic
                condition1 = temp_df['Date']>=self.start_date
                condition2 = temp_df['Date']<=self.end_date
                temp_df = temp_df[condition1 & condition2]
                temp_df['Date'] = pd.to_datetime(temp_df['Date'], errors='coerce')
                temp_df = pd.merge(standard_date, temp_df, how='left', on="Date")
                temp_df = temp_df.fillna(method='ffill').fillna(method="bfill")
                temp_df.set_index('Date',inplace=True)
                data_df = data_df.append(temp_df)
        else:
            for tic in self.ticker_list:
                temp_df = yf.download(tic, start=self.start_date, end=self.end_date)
                temp_df["tic"] = tic
                data_df = data_df.append(temp_df)

        # exit()
        # reset the index, we want to use numbers as index instead of dates
        # print(data_df.head(5))
        data_df = data_df.reset_index()
        # print(temp_df.head(5))
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop("adjcp", 1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)


        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
