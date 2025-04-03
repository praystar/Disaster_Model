from disaster_scraper import DisasterNewsScraper
from text_processor import TextProcessor
from disaster_classifier import DisasterClassifier
from disaster_deduplicator import DisasterDeduplicator
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
    deduplicator = DisasterDeduplicator()
    
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
        
        # Extract locations (this still returns a list, but will be processed into a single location later)
        locations = processor.extract_location(full_text)
        
        processed_data.append({
            'text': processed_text,
            'disaster_type': disaster_type,
            'severity': severity,
            'locations': locations,  # Still plural here as it's the initial extraction
            'url': article['url'],
            'published_at': article['published_at']
        })
    
    # Create DataFrame with processed data
    results_df = pd.DataFrame(processed_data)
    
    # Deduplicate and combine similar disaster reports
    print("Deduplicating and combining similar disaster reports...")
    unique_disasters_df = deduplicator.combine_duplicate_disasters(results_df)
    
    # Save both detailed and combined results
    results_df.to_csv('all_disaster_reports.csv', index=False)
    unique_disasters_df.to_csv('unique_disasters_combined.csv', index=False)
    
    print("\nAnalysis complete!")
    print(f"Found {len(unique_disasters_df)} unique disaster events from {len(results_df)} news articles")
    print("Results saved to:")
    print("- all_disaster_reports.csv (all articles)")
    print("- unique_disasters_combined.csv (combined unique disasters)")
    
    # Print summary of unique disasters
    print("\nSummary of Unique Disasters:")
    for _, disaster in unique_disasters_df.iterrows():
        print(f"\nDisaster Type: {disaster['disaster_type']}")
        print(f"Severity: {disaster['severity']}")
        print(f"Location: {disaster['location']}")  # Changed from locations to location
        print(f"Number of related articles: {disaster['article_count']}")
        print(f"Latest update: {disaster['published_at']}")
        print("-" * 50)

if __name__ == "__main__":
    main() 