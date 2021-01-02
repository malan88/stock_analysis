import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse as ap

class Stock:
    def __init__(self, ticker, period="5y"):
        ticker = ticker.strip()
        self.ticker = ticker
        self.yfitem = yf.Ticker(ticker)
        hist = self.yfitem.history(period=period)
        self.hist = self.yfitem.history(period=period)
        dropcols = list(hist.columns)
        dropcols.remove('Close')
        hist = hist.drop(dropcols, axis=1)
        hist = hist.dropna()
        self.lr = LinearRegression()
        self.X = np.array(hist.index).reshape((-1,1))
        self.y = hist["Close"]
        self.lr.fit(self.X, self.y)
        self.coef = self.lr.coef_[0]

    def __str__(self):
        return f'"{self.ticker}": "{self.coef}"'

    def __repr__(self):
        return f'"{self.ticker}": "{self.coef}"'


def buildportfolio(tickers, period):
    portfolio = []
    for ticker in tickers:
        portfolio.append(Stock(ticker, period=period))
    return portfolio


def main(filein, period):
    with open(filein, 'rt') as fin:
        tickers = fin.readlines()
    portfolio = buildportfolio(tickers, period)
    portfolio.sort(key=lambda stock: stock.coef)
    with open('stocks.csv', 'wt') as fout:
        for stock in portfolio:
            fout.write(f"{stock.ticker},{stock.coef}\r\n")


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Get coefficients for stock tickers")
    parser.add_argument('fin', metavar='fin', nargs='?', type=str,
                        help="The text file list of ticker symbols")
    parser.add_argument('period', metavar='period', nargs='?', type=str,
                        help="Period")
    args = parser.parse_args()
    print(args.fin)
    main(args.fin)
