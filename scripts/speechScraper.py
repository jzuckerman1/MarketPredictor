# import packages --------------------------------------------------------------
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime as dt

from selenium import webdriver
from selenium.webdriver import Chrome
# If running this for your own, use  Service(ChromeDriverManager().install())
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

# FOMC Reader ----------------------------------------------------------------
class FOMCData:
    @staticmethod
    def get_speech_links(soup : BeautifulSoup) -> [str]:
        '''
        Helper Function: 
          From a given screen of BeautifulSoup HTML, returns a list of links 
          to each individual speech site
        
        Parameters:
          soup : BeautifulSoup
            All html info from the page
        '''
        
        speeches_tag = ".itemTitle"
        all_speeches = soup.select(speeches_tag)
        speech_links = ["https://www.federalreserve.gov" + speech.a['href'] for speech in all_speeches]
        return speech_links
      
    @staticmethod
    def scrape_main_page() -> [str]:
        '''
        Runs across the federal reserve website and collects speeches to be scraped
        '''
        
        url = "https://www.federalreserve.gov/newsevents/speeches.htm"
        # cService = webdriver.ChromeService(executable_path=ChromeDriverManager().install())
        driver = webdriver.Chrome() #service = cService
        
        speech_links = []
        
        driver.get(url)
        
        # CSS Selector to get the 'Next' button
        next_button_tag = ".pagination-next .ng-binding"
        
        # 100 ensures an end to the loop
        for _ in range(100):
            time.sleep(0.001) #Minimal loading w/ next button
            soup = BeautifulSoup(driver.page_source, "html.parser")
            new_links = FOMCData.get_speech_links(soup)
            speech_links.extend(new_links) #appends each new link
            
            next_button_tag = ".pagination-next .ng-binding"
            next_button = driver.find_element(By.CSS_SELECTOR, next_button_tag)
            
            if next_button.get_attribute('disabled'):
                break
            else:
                next_button.click()
        
        driver.quit()
        return speech_links

# Scraper data ---------------------------------------------------------------
class Scraper:
    @staticmethod
    def scrape_page(url):
      
      #pause for respectful period of time
      time.sleep(0.2)
      
      # read page
      page = requests.get(url).content
      soup = BeautifulSoup(page, "html.parser")
      
      # scrape heading and top section
      heading = soup.select(".heading")
      metaData = [elmnt.text for elmnt in heading][0]
      info = metaData.split("\n")[1:5]
      
      # adjusts the date formatting
      dt_object = dt.strptime(info[0], '%B %d, %Y')
      date = dt.strftime(dt_object, '%Y-%m-%d')
      
      # possbile titles for the speaker
      possible_titles = ["Chairman", "Vice Chairman", "Governor", "Chair", "Vice Chair"]
      prefix = info[2].split(' ')
      title = ""
      for diffTitle in possible_titles:
          if (prefix[0] == diffTitle.split(' ')[0]):
              title = diffTitle
              break
      name = ' '.join(prefix[len(title.split(' ')):])
      
      # grab actual speech text
      speech = soup.select(".col-md-8:nth-child(3)")
      speech_text = [section.text for section in speech][0]
      speech_breakup = speech_text.split("\n")
      
      # Remove empty
      speech_breakup = list(filter(None, speech_breakup))
      
      # Remove citations
      speech_breakup = [element for element in speech_breakup if not(element[0].isnumeric())]
      
      full_speech = ' '.join(speech_breakup)
      
      speech_length = full_speech.count(' ')
      
      framed_data = pd.DataFrame({
        "Title": title,
        "Name": name,
        "Date": date,
        "Length": speech_length,
        "Text": speech_breakup
      })
      
      return framed_data
  
  
def main():
    all_speeches = FOMCData.scrape_main_page()
    df = Scraper.scrape_page(all_speeches.pop())
    df = df.groupby(["Title", "Name", "Date", "Length"])['Text'].apply(list).reset_index()
    for speech in all_speeches:
        try:
          new_speech = Scraper.scrape_page(speech)
          addOn = new_speech.groupby(["Title", "Name", "Date", "Length"])['Text'].apply(list).reset_index()
          df = pd.concat([df, addOn]).reset_index().iloc[:,1:]
        except:
          print(f'{speech} failed due to inalignment')
    df.to_csv("../data/speechData.csv")
    
if __name__ == '__main__':
    main()
      

