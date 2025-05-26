import requests
from bs4 import BeautifulSoup
import json

def save_to_json(data, filename='news_headlines.json'):
    """
    Save data to a JSON file.

    Args:
        data (list of dict): List of dictionaries containing news headlines.
        filename (str): Name of the file to save the data to. Defaults to 'news_headlines.json'.

    Returns:
        None
    """
    try:
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

if __name__ == "__main__":
    url = 'https://news.example.com'  # Replace with the actual news website URL
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        exit(1)

    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = []
    for headline_tag in soup.find_all('h2'):  # Adjust the tag and class according to the website's HTML structure
        headlines.append({
            'title': headline_tag.get_text(strip=True),
            'url': headline_tag.find('a')['href'] if headline_tag.find('a') else None
        })

    save_to_json(headlines)