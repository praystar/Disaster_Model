import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import os

class DisasterNewsScraper:
    def __init__(self, news_api_key):
        self.news_api = NewsApiClient(api_key=news_api_key)
        
    def fetch_disaster_news(self, days_back=7):
        """
        Fetch disaster-related news articles using NewsAPI
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Comprehensive list of disaster-related keywords
        disaster_keywords = (
            'natural disaster OR earthquake OR flood OR hurricane OR '
            'tsunami OR wildfire OR tornado OR landslide OR '
            'cyclone OR typhoon OR volcanic eruption OR drought OR '
            'avalanche OR mudslide OR storm surge OR heatwave OR '
            'cold wave OR blizzard OR hailstorm OR thunderstorm OR '
            'severe weather OR emergency response OR disaster relief OR '
            'humanitarian crisis OR evacuation OR emergency shelter'
        )
        
        # Exclude non-disaster related terms
        exclude_terms = (
            'NOT sports NOT entertainment NOT politics NOT business NOT '
            'NOT technology NOT celebrity NOT fashion NOT music NOT movie'
        )
        
        try:
            print(f"\nSearching for articles from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"Using query: {disaster_keywords} {exclude_terms}")
            
            articles = self.news_api.get_everything(
                q=f"{disaster_keywords} {exclude_terms}",
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            if articles['status'] != 'ok':
                print(f"\nAPI Error: {articles.get('message', 'Unknown error')}")
                return pd.DataFrame()
                
            print(f"\nAPI Response: {articles['status']}")
            print(f"Total articles found: {articles['totalResults']}")
            
            if articles['totalResults'] == 0:
                print("\nNo articles found. Trying a simpler query...")
                # Try a simpler query without exclusions
                articles = self.news_api.get_everything(
                    q=disaster_keywords,
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy'
                )
                print(f"Articles found with simpler query: {articles['totalResults']}")
            
            return self._process_articles(articles)
            
        except Exception as e:
            print(f"\nError fetching articles: {str(e)}")
            return pd.DataFrame()
    
    def _process_articles(self, articles):
        """
        Process and clean the fetched articles
        """
        processed_articles = []
        
        if not articles.get('articles'):
            print("\nNo articles to process")
            return pd.DataFrame()
            
        print(f"\nProcessing {len(articles['articles'])} articles...")
        
        for article in articles['articles']:
            # Skip articles without content
            if not article['title'] and not article['description']:
                continue
                
            # Extract main text using BeautifulSoup if URL is available
            full_text = ''
            if article['url']:
                try:
                    response = requests.get(article['url'], timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                        element.decompose()
                    
                    # Get text from paragraphs and headings
                    paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
                    full_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    
                    # Clean the text
                    full_text = re.sub(r'\s+', ' ', full_text)  # Remove extra whitespace
                    full_text = full_text.strip()
                    
                except Exception as e:
                    print(f"Error processing article {article['url']}: {str(e)}")
                    full_text = article['description'] or ''
            
            # Skip articles that are too short
            if len(full_text) < 100 and len(article['description'] or '') < 50:
                continue
            
            processed_articles.append({
                'title': article['title'],
                'description': article['description'],
                'full_text': full_text,
                'url': article['url'],
                'published_at': article['publishedAt']
            })
        
        print(f"Successfully processed {len(processed_articles)} articles")
        return pd.DataFrame(processed_articles)

def test_scraper():
    """
    Test function to run when the script is executed directly
    """
    try:
        print("Testing Disaster News Scraper")
        print("=" * 80)
        
        # Load environment variables
        load_dotenv()
        NEWS_API_KEY = os.getenv('NEWS_API_KEY')
        
        if not NEWS_API_KEY:
            print("Error: NEWS_API_KEY not found in environment variables")
            return
        
        # Initialize scraper
        scraper = DisasterNewsScraper(NEWS_API_KEY)
        
        # Fetch news
        print("\nFetching disaster news...")
        articles_df = scraper.fetch_disaster_news(days_back=7)
        
        if articles_df.empty:
            print("\nNo articles were fetched. Please check:")
            print("1. Your NewsAPI key is valid")
            print("2. You have sufficient API credits")
            print("3. The query parameters are correct")
            return
        
        # Print results
        print(f"\nFetched {len(articles_df)} articles")
        print("\nSample of fetched articles:")
        print(articles_df[['title', 'published_at']].head())
        
        # Save to CSV for inspection
        articles_df.to_csv('raw_disaster_articles.csv', index=False)
        print("\nSaved raw articles to 'raw_disaster_articles.csv'")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")

test_scraper() 