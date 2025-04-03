from relief_allocator import ReliefAllocator
import pandas as pd

def test_predictions():
    # Load the training data
    try:
        training_data = pd.read_csv('relief_allocation_training_data.csv')
    except FileNotFoundError:
        print("Training data not found. Please run generate_training_data.py first.")
        return

    # Initialize and train the model
    print("Training the model...")
    allocator = ReliefAllocator()
    allocator.train(training_data)

    # Example disaster scenarios to test
    test_scenarios = [
        {
            'name': 'Urban Earthquake',
            'data': {
                'disaster_type': 'earthquake',
                'severity': 'high',
                'population_density': 5000,
                'urban_rural': 'urban',
                'infrastructure_damage': 'severe',
                'accessibility': 'difficult',
                'time_since_disaster': 1
            }
        },
        {
            'name': 'Rural Flood',
            'data': {
                'disaster_type': 'flood',
                'severity': 'medium',
                'population_density': 200,
                'urban_rural': 'rural',
                'infrastructure_damage': 'moderate',
                'accessibility': 'moderate',
                'time_since_disaster': 3
            }
        },
        {
            'name': 'Coastal Hurricane',
            'data': {
                'disaster_type': 'hurricane',
                'severity': 'high',
                'population_density': 3000,
                'urban_rural': 'mixed',
                'infrastructure_damage': 'severe',
                'accessibility': 'difficult',
                'time_since_disaster': 2
            }
        }
    ]

    # Make predictions for each scenario
    print("\nPredictions for Different Disaster Scenarios:")
    print("=" * 80)
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 40)
        
        # Get predictions
        allocations = allocator.predict_needs(scenario['data'])
        
        # Print detailed information
        print("Disaster Details:")
        for key, value in scenario['data'].items():
            print(f"  {key}: {value}")
        
        print("\nRecommended Resource Allocations:")
        for supply_type, percentage in allocations.items():
            print(f"  {supply_type}: {percentage:.1f}%")
        
        print("=" * 80)

if __name__ == "__main__":
    test_predictions() 