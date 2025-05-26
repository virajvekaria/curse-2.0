import requests
from bs4 import BeautifulSoup
import json

class WebScraper:
    def __init__(self, url):
        self.url = url

    def fetch_headlines(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = [headline.get_text() for headline in soup.find_all('h1')]
            return headlines
        except requests.RequestException as e:
            print(f"Error fetching the webpage: {e}")
            return []

def main():
    scraper = WebScraper('https://news.example.com')
    headlines = scraper.fetch_headlines()
    if headlines:
        with open('headlines.json', 'w') as f:
            json.dump(headlines, f)
        print(f"Headlines saved to headlines.json")
    else:
        print("No headlines found or an error occurred.")

if __name__ == "__main__":
    main()