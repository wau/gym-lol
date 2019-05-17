import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
#/usr/local/bin/chromedriver

# r = requests.get('https://rankedboost.com/league-of-legends/build/lee-sin/')
# print(r.text)

    
def fetch_current_champions():
    """Fetches a list of all current league of legends champions and returns them as a list
    """
    champions_list = []
    html = open_with_selenium('https://na.leagueoflegends.com/en/game-info/champions/', '//*[@id="champion-grid-content"]/div/ul')
    soup = BeautifulSoup(html) 
    for ultag in soup.find_all('ul', {'class': 'champion-grid grid-list gs-container gs-no-gutter default-7-col content-center'}):
        for litag in ultag.find_all('li'):
            champions_list.append(litag.text)
    return champions_list

def fetch_champion_statistics(champion_name):
    """Fetches match statistics for a given champion
            Args:
                champion_name: name of a champion (Can include spaces & uppercase letters)

            Returns: 
                A String: The percent for (WinRate, PickRate, BanRate)
    """
    champion_statistics = []
    html = open_with_selenium('https://rankedboost.com/league-of-legends/build/{0}/'.format(champion_name.replace(' ', '-').lower()), '//*[@id="overview"]/p/span')
    soup = BeautifulSoup(html) 
    for span in soup.find_all('span', {"class" : "top-10-number-text"}):
        for stats in span.find_all('span'):
            champion_statistics.append(stats.text)
    return champion_statistics

def open_with_selenium(uri, xpath=None, timeout=5):
    driver = webdriver.Chrome('/usr/local/bin/chromedriver')
    driver.get(uri)
    if xpath != None:
        try:
            element_present = EC.presence_of_element_located((By.XPATH, xpath))
            WebDriverWait(driver, timeout).until(element_present)
        except TimeoutException:
            print("Timed out waiting for League of Legends game-info page to load")   
    html = driver.page_source
    driver.close() 
    return html


if __name__ == "__main__":
    print(fetch_current_champions())
    print(fetch_champion_statistics('Kindred'))
        

