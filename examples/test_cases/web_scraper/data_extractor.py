import requests
from bs4 import BeautifulSoup
import json

def parse_html(html_content):
    """
    Parse HTML content and return a BeautifulSoup object.
    
    :param html_content: str, HTML content to be parsed
    :return: BeautifulSoup object
    """
    return BeautifulSoup(html_content, 'html.parser')

def extract_headlines(url):
    """
    Extract news headlines from a given URL.
    
    :param url: str, URL of the website to scrape
    :return: list, list of news headlines
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = parse_html(response.text)
        
        # Example CSS selector to find headline elements
        headlines = [h.get_text(strip=True) for h in soup.select('h1, h2, h3')]
        
        return headlines
    
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    url = "https://news.example.com"
    headlines = extract_headlines(url)
    if headlines:
        print("Extracted Headlines:")
        for headline in headlines:
            print(headline)
        
        # Save to JSON
        with open('headlines.json', 'w') as f:
            json.dump(headlines, f, indent=4)
        print("Headlines saved to headlines.json")
    else:
        print("No headlines extracted.")