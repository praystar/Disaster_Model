import pandas as pd
import numpy as np
from relief_allocator import ReliefAllocator
from datetime import datetime

class ReliefSupplyManager:
    def __init__(self, buffer_percentage=20):
        """
        Initialize the ReliefSupplyManager
        :param buffer_percentage: Percentage of total supplies to keep as buffer (default: 20%)
        """
        self.buffer_percentage = buffer_percentage
        self.allocator = ReliefAllocator()
        self.current_supplies = {
            'food': 1000,  # units
            'water': 1000,  # liters
            'medicine': 100,  # units
            'shelter': 500  # units
        }
        
    def set_current_supplies(self, supplies):
        """
        Set the current available supplies
        :param supplies: Dictionary of current supplies
        """
        self.current_supplies = supplies
        
    def calculate_buffer_stock(self):
        """
        Calculate the buffer stock for each supply type
        """
        buffer_stock = {}
        for supply_type, quantity in self.current_supplies.items():
            buffer_stock[supply_type] = quantity * (self.buffer_percentage / 100)
        return buffer_stock
    
    def calculate_available_supplies(self):
        """
        Calculate available supplies after buffer stock
        """
        buffer_stock = self.calculate_buffer_stock()
        available_supplies = {}
        for supply_type, quantity in self.current_supplies.items():
            available_supplies[supply_type] = quantity - buffer_stock[supply_type]
        return available_supplies
    
    def calculate_disaster_priority(self, disaster_data):
        """
        Calculate priority score for a disaster
        Higher score means higher priority
        """
        # Base priority factors with default values for unknown
        severity_scores = {'low': 1, 'medium': 2, 'high': 3, 'unknown': 1}
        damage_scores = {'none': 1, 'light': 2, 'moderate': 3, 'severe': 4, 'unknown': 2}
        access_scores = {'easy': 1, 'moderate': 2, 'difficult': 3, 'unknown': 2}
        
        # Get values with defaults for missing or unknown values
        severity = disaster_data.get('severity', 'unknown')
        damage = disaster_data.get('infrastructure_damage', 'unknown')
        access = disaster_data.get('accessibility', 'unknown')
        pop_density = disaster_data.get('population_density', 1000)
        time_since = disaster_data.get('time_since_disaster', 1)
        
        # Calculate priority score
        priority_score = (
            severity_scores.get(severity, 1) * 3 +  # Severity is most important
            damage_scores.get(damage, 2) * 2 +
            access_scores.get(access, 2) +
            (pop_density / 1000) +  # Population density factor
            (1 / max(time_since, 1))  # More recent disasters get higher priority
        )
        
        return priority_score
    
    def allocate_supplies(self, disaster_list):
        """
        Allocate supplies to multiple disasters based on priority
        :param disaster_list: List of disaster data dictionaries
        :return: Dictionary of allocations for each disaster
        """
        # Calculate available supplies after buffer
        available_supplies = self.calculate_available_supplies()
        
        # Calculate priority scores for all disasters
        disaster_priorities = []
        for disaster in disaster_list:
            priority_score = self.calculate_disaster_priority(disaster)
            disaster_priorities.append({
                'disaster': disaster,
                'priority_score': priority_score
            })
        
        # Sort disasters by priority
        disaster_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Take top 5 disasters
        top_disasters = disaster_priorities[:5]
        
        # Calculate total priority score
        total_priority = sum(d['priority_score'] for d in top_disasters)
        
        # Base percentage allocations based on disaster type
        disaster_type_allocations = {
            'earthquake': {'food': 25, 'water': 25, 'medicine': 30, 'shelter': 20},
            'flood': {'food': 20, 'water': 40, 'medicine': 20, 'shelter': 20},
            'hurricane': {'food': 30, 'water': 30, 'medicine': 20, 'shelter': 20},
            'wildfire': {'food': 25, 'water': 35, 'medicine': 20, 'shelter': 20},
            'tornado': {'food': 30, 'water': 25, 'medicine': 25, 'shelter': 20},
            'default': {'food': 30, 'water': 30, 'medicine': 20, 'shelter': 20}
        }
        
        # Severity multipliers for different supply types
        severity_multipliers = {
            'low': {'food': 0.8, 'water': 0.8, 'medicine': 0.7, 'shelter': 0.7},
            'medium': {'food': 1.0, 'water': 1.0, 'medicine': 1.0, 'shelter': 1.0},
            'high': {'food': 1.2, 'water': 1.2, 'medicine': 1.3, 'shelter': 1.3},
            'unknown': {'food': 1.0, 'water': 1.0, 'medicine': 1.0, 'shelter': 1.0}
        }
        
        # Damage multipliers for different supply types
        damage_multipliers = {
            'none': {'food': 0.7, 'water': 0.7, 'medicine': 0.6, 'shelter': 0.6},
            'light': {'food': 0.8, 'water': 0.8, 'medicine': 0.8, 'shelter': 0.8},
            'moderate': {'food': 1.0, 'water': 1.0, 'medicine': 1.0, 'shelter': 1.0},
            'severe': {'food': 1.3, 'water': 1.3, 'medicine': 1.4, 'shelter': 1.4},
            'unknown': {'food': 1.0, 'water': 1.0, 'medicine': 1.0, 'shelter': 1.0}
        }
        
        # Allocate supplies based on priority
        allocations = {}
        for disaster_data in top_disasters:
            disaster = disaster_data['disaster']
            priority_ratio = disaster_data['priority_score'] / total_priority
            
            # Get disaster-specific base allocations
            disaster_type = disaster.get('disaster_type', 'default').lower()
            base_allocations = disaster_type_allocations.get(disaster_type, disaster_type_allocations['default'])
            
            # Get severity and damage multipliers
            severity = disaster.get('severity', 'unknown')
            damage = disaster.get('infrastructure_damage', 'unknown')
            severity_mult = severity_multipliers.get(severity, severity_multipliers['unknown'])
            damage_mult = damage_multipliers.get(damage, damage_multipliers['unknown'])
            
            # Calculate adjusted percentages based on severity and damage
            adjusted_allocations = {}
            for supply_type, base_percentage in base_allocations.items():
                # Apply both severity and damage multipliers
                adjusted_percentage = base_percentage * severity_mult[supply_type] * damage_mult[supply_type]
                adjusted_allocations[supply_type] = adjusted_percentage
            
            # Normalize percentages to ensure they sum to 100
            total_percentage = sum(adjusted_allocations.values())
            if total_percentage > 0:
                for supply_type in adjusted_allocations:
                    adjusted_allocations[supply_type] = (adjusted_allocations[supply_type] / total_percentage) * 100
            
            # Calculate actual quantities based on available supplies and priority
            disaster_allocations = {}
            for supply_type, percentage in adjusted_allocations.items():
                available = available_supplies[supply_type]
                # Apply priority ratio to the total available supplies
                disaster_allocations[supply_type] = (available * priority_ratio) * (percentage / 100)
            
            allocations[disaster['disaster_type']] = disaster_allocations
        
        return allocations
    
    def print_allocation_report(self, allocations):
        """
        Print a detailed report of the allocations
        """
        print("\nRelief Supply Allocation Report")
        print("=" * 80)
        
        # Print buffer stock
        buffer_stock = self.calculate_buffer_stock()
        print("\nBuffer Stock:")
        for supply_type, quantity in buffer_stock.items():
            print(f"  {supply_type}: {quantity:.2f} units")
        
        # Print available supplies
        available_supplies = self.calculate_available_supplies()
        print("\nAvailable Supplies (after buffer):")
        for supply_type, quantity in available_supplies.items():
            print(f"  {supply_type}: {quantity:.2f} units")
        
        # Print allocations
        print("\nAllocations by Disaster:")
        for disaster_type, disaster_allocations in allocations.items():
            print(f"\n{disaster_type.upper()}:")
            for supply_type, quantity in disaster_allocations.items():
                print(f"  {supply_type}: {quantity:.2f} units")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    # Example usage
    manager = ReliefSupplyManager(buffer_percentage=20)
    
    # Example disaster list
    disaster_list = [
        {
            'disaster_type': 'earthquake',
            'severity': 'high',
            'population_density': 5000,
            'urban_rural': 'urban',
            'infrastructure_damage': 'severe',
            'accessibility': 'difficult',
            'time_since_disaster': 1
        },
        {
            'disaster_type': 'flood',
            'severity': 'medium',
            'population_density': 3000,
            'urban_rural': 'mixed',
            'infrastructure_damage': 'moderate',
            'accessibility': 'moderate',
            'time_since_disaster': 2
        },
        {
            'disaster_type': 'hurricane',
            'severity': 'high',
            'population_density': 4000,
            'urban_rural': 'urban',
            'infrastructure_damage': 'severe',
            'accessibility': 'difficult',
            'time_since_disaster': 1
        },
        {
            'disaster_type': 'wildfire',
            'severity': 'high',
            'population_density': 2000,
            'urban_rural': 'rural',
            'infrastructure_damage': 'severe',
            'accessibility': 'difficult',
            'time_since_disaster': 1
        },
        {
            'disaster_type': 'tornado',
            'severity': 'medium',
            'population_density': 2500,
            'urban_rural': 'mixed',
            'infrastructure_damage': 'moderate',
            'accessibility': 'moderate',
            'time_since_disaster': 2
        }
    ]
    
    # Allocate supplies
    allocations = manager.allocate_supplies(disaster_list)
    
    # Print report
    manager.print_allocation_report(allocations) 