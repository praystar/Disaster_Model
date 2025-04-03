import time
import schedule
from datetime import datetime
import os
from disaster_scraper import DisasterNewsScraper
from text_processor import TextProcessor
from disaster_classifier import DisasterClassifier
from disaster_deduplicator import DisasterDeduplicator
from relief_supply_manager import ReliefSupplyManager
from predict_disaster_allocations import predict_disaster_allocations
from dotenv import load_dotenv
import pandas as pd

def run_pipeline():
    """
    Run the complete disaster relief pipeline
    """
    try:
        print(f"\n{'='*80}")
        print(f"Starting Pipeline Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Load environment variables
        load_dotenv()
        NEWS_API_KEY = os.getenv('NEWS_API_KEY')
        
        # Initialize components
        scraper = DisasterNewsScraper(NEWS_API_KEY)
        processor = TextProcessor()
        classifier = DisasterClassifier()
        deduplicator = DisasterDeduplicator()
        supply_manager = ReliefSupplyManager()
        
        # Step 1: Fetch and Process Disaster News
        print("1. Disaster News Processing Phase")
        print("-" * 40)
        articles_df = scraper.fetch_disaster_news(days_back=30)
        processed_data = []
        
        for _, article in articles_df.iterrows():
            full_text = f"{article['title']} {article['full_text']}"
            processed_text = processor.preprocess_text(full_text)
            disaster_type, severity = processor.extract_disaster_info(full_text)
            locations = processor.extract_location(full_text)
            
            processed_data.append({
                'text': processed_text,
                'disaster_type': disaster_type,
                'severity': severity,
                'locations': locations,
                'url': article['url'],
                'published_at': article['published_at']
            })
        
        results_df = pd.DataFrame(processed_data)
        print("News processing completed successfully\n")
        
        # Step 2: Deduplicate and Analyze Disasters
        print("2. Disaster Analysis Phase")
        print("-" * 40)
        unique_disasters_df = deduplicator.combine_duplicate_disasters(results_df)
        results_df.to_csv('all_disaster_reports.csv', index=False)
        unique_disasters_df.to_csv('unique_disasters_combined.csv', index=False)
        print("Disaster analysis completed successfully\n")
        
        # Step 3: Predict and Allocate Relief
        print("3. Relief Allocation Phase")
        print("-" * 40)
        # Convert DataFrame to list of disaster dictionaries
        disaster_list = []
        for _, disaster in unique_disasters_df.iterrows():
            disaster_data = {
                'disaster_type': disaster.get('disaster_type', 'unknown'),
                'severity': disaster.get('severity', 'medium'),
                'population_density': 1000,  # Default value
                'urban_rural': 'mixed',  # Default value
                'infrastructure_damage': 'moderate',  # Default value
                'accessibility': 'moderate',  # Default value
                'time_since_disaster': 1,  # Default value
                'location': disaster.get('location', 'Unknown Location')
            }
            disaster_list.append(disaster_data)
        
        # Allocate supplies using ReliefSupplyManager
        allocations = supply_manager.allocate_supplies(disaster_list)
        supply_manager.print_allocation_report(allocations)
        print("Relief allocation completed successfully\n")
        
        print(f"{'='*80}")
        print(f"Pipeline Run Completed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\nError in pipeline run: {str(e)}")
        print("Continuing to next iteration...\n")

def main():
    """
    Main function to run the pipeline on a schedule
    """
    # Run the pipeline immediately on startup
    run_pipeline()
    
    # Schedule the pipeline to run every hour
    schedule.every(1).hours.do(run_pipeline)
    
    print("\nPipeline scheduler started. Running every hour.")
    print("Press Ctrl+C to stop the pipeline.\n")
    
    # Keep the script running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\nPipeline stopped by user.")
            break
        except Exception as e:
            print(f"\nError in scheduler: {str(e)}")
            print("Continuing to next iteration...\n")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    main() 