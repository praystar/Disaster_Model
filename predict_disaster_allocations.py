import pandas as pd
from relief_supply_manager import ReliefSupplyManager
import json

def predict_disaster_allocations(unique_disasters_df):
    print("Predicting Relief Allocations for Unique Disasters")
    print("=" * 80)
    
    try:
        print(f"Processing {len(unique_disasters_df)} unique disasters")
        
        # Print available columns for debugging
        print("\nAvailable columns in the dataset:")
        print(unique_disasters_df.columns.tolist())
        
        # Initialize ReliefSupplyManager
        manager = ReliefSupplyManager(buffer_percentage=20)
        
        # Convert disaster data to the required format
        disaster_list = []
        for _, disaster in unique_disasters_df.iterrows():
            # Extract location from the text or use a default
            location = disaster.get('location', 'Unknown Location')
            
            # Create disaster data with default values for missing fields
            disaster_data = {
                'disaster_type': disaster.get('disaster_type', 'unknown'),
                'severity': disaster.get('severity', 'medium'),
                'population_density': 1000,  # Default value
                'urban_rural': 'mixed',  # Default value
                'infrastructure_damage': 'moderate',  # Default value
                'accessibility': 'moderate',  # Default value
                'time_since_disaster': 1,  # Default value
                'location': location  # Add location to the data
            }
            disaster_list.append(disaster_data)
        
        # Get allocations
        allocations = manager.allocate_supplies(disaster_list)
        
        # Create detailed report
        report = {
            'buffer_percentage': 20,
            'total_disasters': len(disaster_list),
            'allocations': {}
        }
        
        # Add allocation details to report
        for disaster_type, disaster_allocations in allocations.items():
            # Find the corresponding disaster data
            disaster_data = next((d for d in disaster_list if d['disaster_type'] == disaster_type), None)
            if disaster_data:
                report['allocations'][disaster_type] = {
                    'location': disaster_data['location'],
                    'severity': disaster_data['severity'],
                    'supplies': disaster_allocations
                }
        
        # Save report to JSON file
        with open('disaster_allocations_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Print summary
        print("\nAllocation Summary:")
        print("-" * 40)
        for disaster_type, details in report['allocations'].items():
            print(f"\nDisaster: {disaster_type.upper()}")
            print(f"Location: {details['location']}")
            print(f"Severity: {details['severity']}")
            print("Supply Allocations:")
            for supply_type, quantity in details['supplies'].items():
                print(f"  {supply_type}: {quantity:.2f} units")
        
        print("\nDetailed report saved to 'disaster_allocations_report.json'")
        return allocations
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nDebugging Information:")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # For standalone execution, read from file
    try:
        unique_disasters_df = pd.read_csv('unique_disasters_combined.csv')
        predict_disaster_allocations(unique_disasters_df)
    except FileNotFoundError:
        print("Error: Could not find 'unique_disasters_combined.csv'")
        print("Please make sure you have run the disaster analysis pipeline first") 