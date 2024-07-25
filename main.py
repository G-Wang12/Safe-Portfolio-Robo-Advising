from IPython.display import display, Math, Latex

import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import random

def get_average__monthly_volume(stock_volume):
    # # get the ticker
    # ticker = yf.Ticker(ticker_symbol)
    # # get the stock data
    # stock_data = ticker.history(start='2023-01-01', end='2023-10-31')

    # # Resample data to monthly frequency and calculate average volume
    # stock_volume = stock_data['Volume']
    # Resample data to monthly frequency and calculate the number of trading days per month
    trading_days_per_month = stock_volume.resample('M').count()

    # Filter out months with less than 18 trading days
    valid_months = trading_days_per_month[trading_days_per_month >= 18]

    # Filter the stock data for the valid months
    stock_volume = stock_volume[stock_volume.index.month.isin(valid_months.index.month)]

    # return the result
    return stock_volume.sum()/len(valid_months)

def validtickers(df):
    ticker_lst = []
    ##start and end dates to check if the ticker is avaliable in those times
    start_date = '2023-01-01'
    end_date = '2023-10-01'
    ##in case there's no title and the column title is a ticker (so in the ticker_example.csv, the title was AAPL, which is a ticker)
    try:
        ticker = df.columns[0]
        stock = yf.Ticker(ticker)
        ##get historical data from it
        history = stock.history(start=start_date, end=end_date)
        ##if it is a valid ticker, this would run and if it hits the requirements, it appends the ticker to the tickerlist
        if ((stock.fast_info['currency'] == "CAD" or stock.fast_info['currency'] == "USD") and (get_average__monthly_volume(history['Volume']) > 150000)): # added my function here
          ticker_lst.append(df.columns[0])
          x = 0
        else:
          ##otherwise, it's not a ticker that hits the requirements,
          print('not a valid stock')
          x = 0
    except:
      ##if the code outputs an error, the ticker isn't valid, so outputs not a ticker
        print('not a ticker')
        x = 0
    while x < len(df):
      ##for the rest of the column, exactly the same process as above, try getting an output from the ticker, and if it doesn't work, go next
        try:
            ticker = df.iloc[x,0]
            stock = yf.Ticker(ticker)
            history = stock.history(start=start_date, end=end_date)
            if ((stock.fast_info['currency'] == 'CAD' or stock.fast_info['currency'] == 'USD') and (get_average__monthly_volume(history['Volume']) > 150000)):
             ticker_lst.append(df.iloc[x,0])
             x += 1
            else:
              print('not a valid stock')
              x += 1
        except:
            print('not a ticker')
            x += 1

    return ticker_lst
    ##return random.sample(ticker_lst, random.randint(10,22))

tickerlist = validtickers(pd.read_csv('Tickers_Example.csv'))
print(tickerlist)

stock_currency = pd.DataFrame({'Ticker':[],
                         'Currency':[]})
for i in tickerlist:
  tempdf = pd.DataFrame(columns=['Ticker','Currency'])
  temp = yf.Ticker(i)
  tempdf['Ticker'] = [i]
  ##Code for industry if avaliable
  tempdf['Currency'] = [temp.fast_info['currency']]
  stock_currency = pd.concat([stock_currency, tempdf],ignore_index=True)

stock_currency

pctchangelist = pd.DataFrame()
stockprice = pd.DataFrame()

# Define the date range
start_date = '2023-01-01'
end_date = '2023-10-01'

# Loop through each ticker
for ticker_symbol in tickerlist:
    # Retrieve the historical data for the ticker
    ticker = yf.Ticker(ticker_symbol)

    #Make a list with the stock values
    stockprice[ticker_symbol] = ticker.history(start=start_date, end=end_date).Close

    # Extract and calculate the percentage change of closing prices
    pct_change = ticker.history(start=start_date, end=end_date).Close.pct_change()

    # Add the percentage change data to the DataFrame
    pctchangelist[ticker_symbol] = pct_change

# Drop the first row (index 0) since it contains NaN values due to percentage change calculation
pctchangelist = pctchangelist.dropna()
pctchangelist2 = pctchangelist

# Localize timezone to only display date
pctchangelist.index = pctchangelist.index.tz_localize(None)
stockprice.index = stockprice.index.tz_localize(None)

# Display the resulting DataFrame
print(stockprice)

# optional that we can add to reduce the number of tickers by removing the most volatile stocks
# might cause problems if there are only like 45 stocks and the remaining ones are from similar industries -> hence the lower volatility
# given that it's being run for only like two weeks, doubt it matters too much
if len(tickerlist) > 44:
  # create a list of averages, append the average of the absolute value to get the variation from zero
  # since its percent change, higher average percent change means high volatility
    avelist = []
    for i in tickerlist:
        avelist.append(pctchangelist[i].abs().mean())

    avelist = pd.Series(avelist)
    # get the median, cut the number of stocks in half
    avemid = stdlist.median()
    i = 0
    while i < len(stdlist):
      # remove them from the percent change list if the stock average is above the overall average
        if avelist[i] > avemid:
            print(avelist[i])
            pctchangelist = pctchangelist.drop(columns = [tickerlist[i]])
            i += 1
        else:
            i += 1

print(pctchangelist)

tempchangelist = pctchangelist
# get all negative correlations
# to be a safe portfolio, we want stocks that cancel each other out, so any big change in one stock is slightly negated by the other stock
# thus, we want the lowest correlation values, at least below zero
# this guarantees at least two stocks, even if all correlation values are positive
corrvalues = tempchangelist .corr()
# get the minimum of the correlation values
minvalues = corrvalues.min()
# correlation dataframe is sorted in a way such that the smallest value occurs twice, once in both stocks
stocks = np.where(minvalues == minvalues.min())[0]
# get the index and find the stock associated with the index - there would be two
firststock = tickerlist[stocks[0]]
secondstock = tickerlist[stocks[1]]
# create a portfolio pct change and remove the two stocks from the percent change dataframe
# then we can compare them percent change dataframe again to get new correlation values
stockprice['Portfolio'] = stockprice[firststock] + stockprice[secondstock]
tempchangelist['Portfolio'] = stockprice['Portfolio'].ffill().pct_change().dropna()
tempchangelist = tempchangelist.drop(columns=[firststock, secondstock])
# create a list with the stocks and a list with the correlation values
# since correlation can't be above 1, set the starting index to 2
stocklist = [[firststock, secondstock]]

i = 2
while i < 22:
  # loop at max 22 times for 22 stocks
  # create a temporary list for the correlation values and get the minimum for the last column (portfolio column)
    temp = tempchangelist.corr()
    mintemp = temp.iloc[:, -1].min()
    # if the minimum is bigger than 0, then there is a positive correlation, no matter how slight, so we woul rather not include it
    if mintemp > 0:
      # then get the minimum value in the list, which should be lower than if the correlation is positive
        minvalues = temp.min()
        # if it is still positive, then there is only positive correlations between the stocks left, so break and we fill the rest of the
        # stocks in the following block of code - this guarantees the minimum isn't between the portfolio and one of the stocks
        if minvalues.min() > 0:
            break
        else:
          # if the minimum isn't above zero, we find which stocks are associated with the low correlation, and add them to the list
          # same process as above
            stocks = np.where(minvalues == minvalues.min())[0]
            firststock = tempchangelist.columns[stocks[0]]
            secondstock = tempchangelist.columns[stocks[1]]
            stockprice['Portfolio'] = stockprice[firststock] + stockprice[secondstock]
            tempchangelist['Portfolio'] = stockprice['Portfolio'].ffill().pct_change().dropna()
            tempchangelist = tempchangelist.drop(columns=[firststock, secondstock])
            stocklist = stocklist + [[firststock, secondstock]]
            # add two because adding two new stocks
            i += 2
    else:
      # else, there is a negative correlation between the portfolio and one of the stocks, in which case
      # find the index associated with the number and add it to the portfolio and the stock/correlation list, and remove it from
      # the pctchange list
        minvalues = mintemp
        stock = temp.iloc[:, -1].idxmin()
        stockprice['Portfolio'] = stockprice['Portfolio'] + stockprice[stock]
        tempchangelist['Portfolio'] = stockprice['Portfolio'].ffill().pct_change().dropna()
        tempchangelist = tempchangelist.drop(columns=[stock])
        stocklist[len(stocklist)-1].append(stock)
        i += 1

print(stocklist)

# flatten the stocklist, as to find the number of stocks theyre are
def flatten(A):
    lst = []
    for i in A:
        if isinstance(i,list): lst.extend(flatten(i))
        else: lst.append(i)
    return lst

flatstocklist = flatten(stocklist)
print(flatstocklist)

# make new correlation dataframe with the new percent change list
corrvalues2 = pctchangelist2.corr()
# remove all items in the stocklist from the rows and all items not in the stocklist from the columns
# this guarantees 22 different stocks, which is safer for our portfolio, and will be weighted according to the correlation
corrvalues2 = corrvalues2.drop(columns = [x for x in flatstocklist])
corrvalues2 = corrvalues2.drop(columns = ['Portfolio'])
corrvalues2 = corrvalues2.drop([x for x in tickerlist if x not in flatstocklist])

# get the remining stocks by finding the highest (22 - however many current stocks) number of correlation values
maxcorr = pd.DataFrame(corrvalues2.stack().sort_values(ascending = False).iloc[:24])
maxcorr = maxcorr.tail(22-len(stocklist))

# we love recursion
# go through the entire list and find where the index equals the stock associated with the correlation values
# outputs in a way such that if there's correlation, list becomes nested - eg.
# [['SHOP.TO', 'UNH'], [['C', 'BLK', 'C'], 'LLY', ['BAC', 'TD.TO', 'BAC']], ['ABBV', 'AMZN'], [['ACN', 'TXN', 'ACN'], 'MRK'],
# [['AXP', 'BK', 'AXP'], ['CL', 'KO', 'CL']], [['BMY', 'PFE'], ['QCOM', 'BLK']]]

# go through the list of correlations
i = 0
while i < len(maxcorr):
  # go through the list of stocks to find the appropriate one
    j = 0
    while j < len(stocklist):
      # since each is a nested list, go through each of the nested lists
        k = 0
        while k < len(stocklist[j]):
          # if the first is already a stock, we want it shared amongst at most three, so we check for that
            if isinstance(stocklist[j][k], list):
                if maxcorr.index[i][0] == stocklist[j][k][0]:
                    if len(stocklist[j][k]) < 3:
                      # if it's less than three, append the stock, else break
                        stocklist[j][k].append(maxcorr.index[i][0])
                        break
                    else:
                        break
                else:
                  # if it's not the first item, keep checking the nested list
                    k += 1
            elif (maxcorr.index[i][0] == stocklist[j][k]):
              # if it's not a list, check it's index, then if it matches, create a list and break
                stocklist[j][k] = [maxcorr.index[i][0], maxcorr.index[i][1]]
                break
                # otherwise, check the next one
            else: k += 1
        j += 1
    i += 1


print(stocklist)

# averageBeta returns the average beta of list of stocks, or the beta of a single stock
def averageBeta (element):
    # element ((listof Str) or Str): Either a list of stocks to be recursively called on, or a stock,
    #                                represented by its ticker as a string
    if type(element) is list:
        count = 0
        for i in range (len(element)):
            count += averageBeta(element[i])
        return count/(len(element))
    return yf.Ticker(element).info['beta']

# splitPercentages divides the % of the portfolio evenly amongst all stocks in tickers, and adds a bonus
#                  percent if the ticker is
def splitPercentages (tickers, total, added):
    # element ((listof Str) or Str): Either a list of stocks to be recursively called on, or a stock,
    #                                represented by its ticker as a string
    # total (0 <= Num <= 1): The percentage to be divided amongst the stocks
    allTickers = []
    allWeights = []
    for i in range (len(tickers)):
        if type(tickers[i]) is list:
            if i > (len(tickers)/2 - 1):
                added = 0
            nested = splitPercentages(tickers[i], total/len(tickers), added/len(tickers[i]))
            allTickers = allTickers + nested[0]
            allWeights = allWeights + nested[1]
            continue
        if i > (len(tickers)/2 - 1):
            added = 0
        allTickers.append(tickers[i])
        allWeights.append(total/len(tickers) + added*2)
    return [allTickers, allWeights]

# assignWeights returns a list of length 2. The first element in the list is a list of stocks, and the second
#               element is a list, where the element at index i is the weight of the stock at index i in the
#               first list
def assignWeights (tickers):
    # tickers (listof (Str or (listof Str))):

    temp = tickers
    # sorting the groups by their beta, or their average beta
    temp.sort(key=averageBeta)

    # we remove a portion from the total percentage to be set aside for biased redistribution later
    totalPercentage = 1-(len(temp)*0.005)
    returns = splitPercentages(temp, totalPercentage, 0.005)

    # this occurs when len(temp) is odd. However, it is a happy accident, as now we can add more weight to the
    # stock with the lowest beta
    if sum(returns[1]) < 1:
        returns[1][0] += 1-sum(returns[1])

    return returns

stock_weight = assignWeights(stocklist)
print(stock_weight)

def price(ticker):
    start_date = '2023-10-23'
    end_date = '2023-10-24'
    stock = yf.Ticker(ticker)
    stock_hist = stock.history(start=start_date, end=end_date)
    return stock_hist.Close.iloc[0]
def get_num_shares(price, value):
    return value / price

def final_data_clean(stocks, weighing):
    total_value = 750000
    final = pd.DataFrame({"Ticker": stocks}, index=range(1, len(stocks) + 1))
    initial_prices = []
    currencies = []
    values = []
    shares = []
    for i in range(len(stocks)):
        curr_price = price(stocks[i])
        initial_prices.append(curr_price)
        currencies.append(stock_currency[stock_currency['Ticker'] == stocks[i]].iloc[0, 1])
        curr_value = total_value * weighing[i]
        values.append(curr_value)
        shares.append(get_num_shares(curr_price, curr_value))
    final['Price'] = initial_prices
    final['Currency'] = currencies
    final['Shares'] = shares
    final['Value'] = values
    final['Weight'] = weighing
    return final
Portfolio_Final = final_data_clean(flatten(stocklist), [0.25, 0.25, 0.25, 0.25])
print(Portfolio_Final)
Stocks_Final = Portfolio_Final[['Ticker', 'Shares']]
Stocks_Final.to_csv('Stocks_Group_10.csv')

# define function to get the price of a ticker
def price(ticker):
    # set start and end date
    start_date = '2023-10-23'
    end_date = '2023-10-24'
    stock = yf.Ticker(ticker)
    # get the ticker history
    stock_hist = stock.history(start=start_date, end=end_date)
    # return the first cell of the data
    return stock_hist.Close.iloc[0]
# create a function to get the currency exchange rate from that currency to CAD
def usd_exchange():
    exchange_ticker = yf.Ticker('CADUSD=x')
    # get data history
    start_date = '2023-10-23'
    end_date = '2023-10-24'
    exchange_hist = exchange_ticker.history(start=start_date, end=end_date)
    # make a dataframe from the closing exchange rate
    exchange_rates = pd.DataFrame(1/exchange_hist.Close)
    exchange_rates.columns = ['cad']
    # return the first cell of the exchange rate
    return exchange_rates.iloc[0,0]
# get the current usd exchange rate
usd_exchange_rate = usd_exchange()
# function to get number of shares of a stock given the price and value
def get_num_shares(price, value, currency):
    # if currency is usd we consider the usd exchange rate
    if currency == 'USD':
        return value / (price * usd_exchange_rate)
    else:
        return value / price

# function to get the cleaned version of the output
def final_data_clean(stocks, weighing):
    # set initial total value
    total_value = 750000
    # create a final dataframe with a ticker column
    final = pd.DataFrame({"Ticker": stocks}, index=range(1, len(stocks) + 1))
    # set an initial empty list to hold every initial price, currency, value, and shares for each stock
    initial_prices = []
    currencies = []
    values = []
    shares = []
    # loop through each stock to get the initial price, currency, value, and shares of every stock
    for i in range(len(stocks)):
        curr_price = price(stocks[i])
        initial_prices.append(curr_price)
        curr_currency = stocklist[stocklist['Ticker'] == stocks[i]].iloc[0, 1]
        currencies.append(curr_currency)
        curr_value = total_value * weighing[i]
        values.append(curr_value)
        shares.append(get_num_shares(curr_price, curr_value, curr_currency))
    # create a column for each set of data
    final['Price'] = initial_prices
    final['Currency'] = currencies
    final['Shares'] = shares
    final['Value'] = values
    final['Weight'] = weighing
    # return the final dataframe
    return final
Portfolio_Final = final_data_clean(['AAPL', 'ABBV', 'ABT', 'MO'], [0.25, 0.25, 0.25, 0.25])
print(Portfolio_Final)
Stocks_Final = Portfolio_Final[['Ticker', 'Shares']]
Stocks_Final.to_csv('Stocks_Group_10.csv')

