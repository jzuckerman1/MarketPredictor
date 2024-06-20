# import packages --------------------------------------------------------------
import pandas as pd
import yfinance as yf

class Ticker:
    '''
    A class to represent a Stock Ticker.
    
    Attributes:
      saveAs : str
        The filename to save the stock data under.
        
      name : str
        The ticker name as it appears on the Yahoo Finance API.
    '''
    
    def __init__(self, saveAs, name):
        self.saveAs = saveAs
        self.name = name
    
    def __str__(self):
        return self.saveAs
    

# Gathering data --------------------------------------------------------------()
class IndexData:
    '''
    A class to gather and save stock index data.
    
    Attributes:
      tickers : list of Ticker
        List of stock tickers to save data from.
        
      saveCols : list of str
        A list of columns that we want to keep from the Yahoo Finance API.
    '''
    tickers : [Ticker] = [Ticker("snp", "^GSPC"), Ticker("vix", "^VIX"), 
        Ticker("dow", "^DJI"), Ticker("nasdaq", "^IXIC")]
      
    saveCols : [str] = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
      
    @staticmethod
    def getYahooData(tick : Ticker):
        '''
        Gathers the past ten years of historical data from a ticker and saves it as a CSV.
        
        Parameters:
          tick : Ticker
            The desired ticker to collect data.
        '''
        ticker = yf.Ticker(tick.name)
        historical_data = ticker.history(period='10y')
        data = pd.DataFrame(historical_data).reset_index()
        data['Date'] = [str(date)[0:10] for date in data['Date']] # Formats date to drop hours and minutes
        data = data[IndexData.saveCols] #Takes only the requested columns
        data.columns = [f'{tick.saveAs}_{col}' if not(col == 'Date') else col for col in data.columns] #Rename columns
        data.to_csv("../data/stocks/" + tick.saveAs + ".csv")
      
def main():
    for ticker in IndexData.tickers:
      IndexData.getYahooData(ticker)
        

if __name__ == '__main__':
    main()
    
