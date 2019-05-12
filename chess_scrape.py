from splinter import Browser
from bs4 import BeautifulSoup as bs
import pandas as pd
import time

def init_browser():
    executable_path = {"executable_path": "chromedriver.exe"}
    return Browser("chrome", **executable_path, headless=False)

def scrape():
    browser = init_browser()

    hrefs=[]
    titles=[]
    images=[]

    url = "https://new.uschess.org/home/"
    browser.visit(url)

    html = browser.html
    soup = bs(html, "html.parser")


    contents = soup.find_all("li", class_="soliloquy-item")

    for content in contents:

        try:
            if (content.a):
                a = content.find("a")
                href = a["href"]
                title = a["title"]
                img = a.find("img")
                image = img["src"]
                
                hrefs.append(href)
                titles.append(title)
                images.append(image)
                
        except AttributeError as e:
            print(e)

    browser.quit()


    return [hrefs,titles,images]
