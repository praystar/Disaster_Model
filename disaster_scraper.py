import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta

class DisasterNewsScraper:
    def __init__(self, news_api_key):
        self.news_api = NewsApiClient(api_key=news_api_key)
        
    def fetch_disaster_news(self, days_back=7):
        """
        Fetch disaster-related news articles using NewsAPI
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Keywords related to disasters
        disaster_keywords = (
            'natural disaster OR earthquake OR flood OR hurricane OR '
            'tsunami OR wildfire OR tornado OR landslide'
        )
        
        articles = self.news_api.get_everything(
            q=disaster_keywords,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy'
        )
        
        return self._process_articles(articles)
    
    def _process_articles(self, articles):
        """
        Process and clean the fetched articles
        """
        processed_articles = []
        
        for article in articles['articles']:
            # Extract main text using BeautifulSoup if URL is available
            full_text = ''
            if article['url']:
                try:
                    response = requests.get(article['url'])
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Get text from paragraphs
                    paragraphs = soup.find_all('p')
                    full_text = ' '.join([p.get_text() for p in paragraphs])
                except:
                    full_text = article['description'] or ''
            
            processed_articles.append({
                'title': article['title'],
                'description': article['description'],
                'full_text': full_text,
                'url': article['url'],
                'published_at': article['publishedAt']
            })
        
        return pd.DataFrame(processed_articles) 