# import packages --------------------------------------------------------------
import pandas as pd
import yfinance as yf

class Ticker:
    def __init__(self, saveAs, name):
        self.saveAs = saveAs
        self.name = name
    
    def __str__(self):
        return self.saveAs
    

# Gathering data --------------------------------------------------------------()
class IndexData:
    tickers : [Ticker] = [Ticker("snp", "^GSPC"), Ticker("vix", "^VIX"), 
        Ticker("dow", "^DJI"), Ticker("nasdaq", "^IXIC")]
      
    saveCols : [str] = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
      
    @staticmethod
    def getYahooData(tick : Ticker):
        ticker = yf.Ticker(tick.name)
        historical_data = ticker.history(period='10y')
        data = pd.DataFrame(historical_data).reset_index()
        data['Date'] = [str(date)[0:10] for date in data['Date']]
        data = data[IndexData.saveCols]
        data.columns = [f'{tick.saveAs}_{col}' if not(col == 'Date') else col for col in data.columns]
        data.to_csv("../data/stocks/" + tick.saveAs + ".csv")
      
    @staticmethod
    def main():
        for ticker in IndexData.tickers:
          IndexData.getYahooData(ticker)
        

if __name__ == '__main__':
    IndexData.main()
    
