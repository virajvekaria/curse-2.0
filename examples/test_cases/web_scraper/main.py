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
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {self.url}: {e}")
            return None

class HTMLParser:
    def extract_headlines(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        headlines = []
        for headline in soup.find_all('h1'):
            headlines.append(headline.get_text())
        return headlines

class DataExtractor:
    def save_to_json(self, data, filename='headlines.json'):
        try:
            with open(filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

def main():
    url = 'https://httpbin.org/html'  # Test URL that returns HTML
    scraper = WebScraper(url)
    html_content = scraper.fetch_headlines()

    if html_content:
        parser = HTMLParser()
        headlines = parser.extract_headlines(html_content)

        extractor = DataExtractor()
        extractor.save_to_json(headlines)

if __name__ == "__main__":
    main()