import requests
from bs4 import BeautifulSoup
import json

def parse_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        headlines = []
        for headline in soup.find_all('h2', class_='headline'):
            headlines.append(headline.get_text(strip=True))
        
        return headlines
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    url = "https://example-news-website.com"
    headlines = parse_html(url)
    
    if headlines:
        print("Headlines fetched successfully:")
        for headline in headlines:
            print(headline)
        
        with open('headlines.json', 'w') as json_file:
            json.dump(headlines, json_file, indent=4)
            print("Headlines saved to headlines.json")