from disaster_scraper import DisasterNewsScraper
from text_processor import TextProcessor
from disaster_classifier import DisasterClassifier
import pandas as pd
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    
    # Initialize components
    scraper = DisasterNewsScraper(NEWS_API_KEY)
    processor = TextProcessor()
    classifier = DisasterClassifier()
    
    # Fetch disaster news
    print("Fetching disaster news...")
    articles_df = scraper.fetch_disaster_news(days_back=30)
    
    # Process articles
    print("Processing articles...")
    processed_data = []
    
    for _, article in articles_df.iterrows():
        # Combine title and full text
        full_text = f"{article['title']} {article['full_text']}"
        
        # Preprocess text
        processed_text = processor.preprocess_text(full_text)
        
        # Extract disaster type and severity
        disaster_type, severity = processor.extract_disaster_info(full_text)
        
        # Extract locations
        locations = processor.extract_location(full_text)
        
        processed_data.append({
            'text': processed_text,
            'disaster_type': disaster_type,
            'severity': severity,
            'locations': locations,
            'url': article['url'],
            'published_at': article['published_at']
        })
    
    # Create DataFrame with processed data
    results_df = pd.DataFrame(processed_data)
    
    # Save results
    results_df.to_csv('disaster_analysis_results.csv', index=False)
    print("Analysis complete! Results saved to disaster_analysis_results.csv")

if __name__ == "__main__":
    main() 