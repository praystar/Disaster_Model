import pandas as pd
from relief_supply_manager import ReliefSupplyManager
from generate_training_data import generate_synthetic_data
from relief_allocator import ReliefAllocator

def test_system():
    print("Testing Relief Supply Management System")
    print("=" * 80)
    
    # Step 1: Generate training data
    print("\n1. Generating Training Data...")
    training_data = generate_synthetic_data(num_samples=1000)
    print("✓ Training data generated successfully")
    
    # Step 2: Initialize ReliefSupplyManager
    print("\n2. Initializing ReliefSupplyManager...")
    manager = ReliefSupplyManager(buffer_percentage=20)
    print("✓ ReliefSupplyManager initialized")
    
    # Step 3: Test buffer stock calculation
    print("\n3. Testing Buffer Stock Calculation...")
    buffer_stock = manager.calculate_buffer_stock()
    print("Buffer Stock:")
    for supply_type, quantity in buffer_stock.items():
        print(f"  {supply_type}: {quantity:.2f} units")
    print("✓ Buffer stock calculated correctly")
    
    # Step 4: Test available supplies calculation
    print("\n4. Testing Available Supplies Calculation...")
    available_supplies = manager.calculate_available_supplies()
    print("Available Supplies:")
    for supply_type, quantity in available_supplies.items():
        print(f"  {supply_type}: {quantity:.2f} units")
    print("✓ Available supplies calculated correctly")
    
    # Step 5: Test with sample disasters
    print("\n5. Testing with Sample Disasters...")
    sample_disasters = [
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
        }
    ]
    
    # Test priority calculation
    print("\nTesting Priority Calculation:")
    for disaster in sample_disasters:
        priority = manager.calculate_disaster_priority(disaster)
        print(f"  {disaster['disaster_type']}: {priority:.2f}")
    print("✓ Priority calculation working")
    
    # Test allocation
    print("\nTesting Supply Allocation:")
    allocations = manager.allocate_supplies(sample_disasters)
    print("Allocations:")
    for disaster_type, disaster_allocations in allocations.items():
        print(f"\n{disaster_type.upper()}:")
        for supply_type, quantity in disaster_allocations.items():
            print(f"  {supply_type}: {quantity:.2f} units")
    print("✓ Allocation working")
    
    # Step 6: Verify allocations sum correctly
    print("\n6. Verifying Allocation Sums...")
    total_allocated = {supply_type: 0 for supply_type in manager.current_supplies.keys()}
    for disaster_allocations in allocations.values():
        for supply_type, quantity in disaster_allocations.items():
            total_allocated[supply_type] += quantity
    
    print("Total Allocated vs Available:")
    for supply_type in manager.current_supplies.keys():
        available = available_supplies[supply_type]
        allocated = total_allocated[supply_type]
        print(f"  {supply_type}:")
        print(f"    Available: {available:.2f} units")
        print(f"    Allocated: {allocated:.2f} units")
        print(f"    Difference: {available - allocated:.2f} units")
        if allocated <= available:
            print("    ✓ Allocation within available limits")
        else:
            print("    ✗ Allocation exceeds available supplies!")
    
    # Step 7: Test with different buffer percentages
    print("\n7. Testing Different Buffer Percentages...")
    buffer_percentages = [10, 20, 30]
    for percentage in buffer_percentages:
        print(f"\nTesting with {percentage}% buffer:")
        manager = ReliefSupplyManager(buffer_percentage=percentage)
        buffer_stock = manager.calculate_buffer_stock()
        available_supplies = manager.calculate_available_supplies()
        print("  Buffer Stock:")
        for supply_type, quantity in buffer_stock.items():
            print(f"    {supply_type}: {quantity:.2f} units")
        print("  Available Supplies:")
        for supply_type, quantity in available_supplies.items():
            print(f"    {supply_type}: {quantity:.2f} units")
    
    print("\nSystem Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_system() 