import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic training data for relief allocation model
    """
    # Define possible values for categorical features
    disaster_types = ['earthquake', 'flood', 'hurricane', 'wildfire', 'tornado', 'landslide', 'tsunami']
    severity_levels = ['low', 'medium', 'high']
    urban_rural = ['urban', 'rural', 'mixed']
    infrastructure_damage = ['none', 'light', 'moderate', 'severe']
    accessibility = ['easy', 'moderate', 'difficult']
    
    # Base allocation percentages for different disaster types
    base_allocations = {
        'earthquake': {'food': 30, 'water': 25, 'medicine': 25, 'shelter': 20},
        'flood': {'food': 25, 'water': 35, 'medicine': 20, 'shelter': 20},
        'hurricane': {'food': 30, 'water': 30, 'medicine': 20, 'shelter': 20},
        'wildfire': {'food': 25, 'water': 25, 'medicine': 20, 'shelter': 30},
        'tornado': {'food': 30, 'water': 25, 'medicine': 25, 'shelter': 20},
        'landslide': {'food': 25, 'water': 25, 'medicine': 30, 'shelter': 20},
        'tsunami': {'food': 25, 'water': 30, 'medicine': 25, 'shelter': 20}
    }
    
    data = []
    
    for _ in range(num_samples):
        # Generate random features
        disaster_type = random.choice(disaster_types)
        severity = random.choice(severity_levels)
        population_density = random.randint(100, 10000)  # people per square km
        urban_rural_type = random.choice(urban_rural)
        damage_level = random.choice(infrastructure_damage)
        access_level = random.choice(accessibility)
        time_since_disaster = random.randint(1, 30)  # days
        
        # Get base allocation for this disaster type
        base_allocation = base_allocations[disaster_type]
        
        # Adjust allocations based on severity
        severity_multiplier = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.2
        }[severity]
        
        # Adjust allocations based on infrastructure damage
        damage_multiplier = {
            'none': 0.9,
            'light': 1.0,
            'moderate': 1.1,
            'severe': 1.2
        }[damage_level]
        
        # Adjust allocations based on accessibility
        access_multiplier = {
            'easy': 0.9,
            'moderate': 1.0,
            'difficult': 1.1
        }[access_level]
        
        # Adjust allocations based on urban/rural setting
        urban_rural_multiplier = {
            'urban': 1.1,
            'rural': 0.9,
            'mixed': 1.0
        }[urban_rural_type]
        
        # Adjust allocations based on population density
        pop_density_multiplier = min(1.0 + (population_density / 10000), 1.5)
        
        # Adjust allocations based on time since disaster
        time_multiplier = max(1.0 - (time_since_disaster / 30), 0.7)
        
        # Calculate final allocations with all adjustments
        total_multiplier = (
            severity_multiplier * 
            damage_multiplier * 
            access_multiplier * 
            urban_rural_multiplier * 
            pop_density_multiplier * 
            time_multiplier
        )
        
        # Generate final allocations with some random variation
        allocations = {}
        for supply_type, base_percent in base_allocation.items():
            # Add some random variation (±5%)
            variation = random.uniform(-5, 5)
            adjusted_percent = base_percent * total_multiplier + variation
            allocations[f'{supply_type}_allocation_percent'] = max(0, min(100, adjusted_percent))
        
        # Normalize to ensure total is 100%
        total = sum(allocations.values())
        for supply_type in allocations:
            allocations[supply_type] = (allocations[supply_type] / total) * 100
        
        # Create data point
        data_point = {
            'disaster_type': disaster_type,
            'severity': severity,
            'population_density': population_density,
            'urban_rural': urban_rural_type,
            'infrastructure_damage': damage_level,
            'accessibility': access_level,
            'time_since_disaster': time_since_disaster,
            **allocations
        }
        
        data.append(data_point)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('relief_allocation_training_data.csv', index=False)
    print(f"Generated {num_samples} synthetic data points and saved to relief_allocation_training_data.csv")
    
    # Print sample of the generated data
    print("\nSample of generated data:")
    print(df.head())
    
    # Print summary statistics
    print("\nSummary statistics for allocation percentages:")
    for supply_type in ['food', 'water', 'medicine', 'shelter']:
        col_name = f'{supply_type}_allocation_percent'
        print(f"\n{supply_type.capitalize()} Allocation:")
        print(f"  Mean: {df[col_name].mean():.2f}%")
        print(f"  Min: {df[col_name].min():.2f}%")
        print(f"  Max: {df[col_name].max():.2f}%")
    
    return df

if __name__ == "__main__":
    generate_synthetic_data()