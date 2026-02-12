#!/usr/bin/env python3
"""
Test script to demonstrate the validation functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from agents.hypotheis_ag import HypothesisAgent

def test_validation():
    # Test with a problematic requirement
    test_requirements = [
        '''
        Do an incompressible lid-driven cavity flow.
        The cavity is a square with dimensions 1×1, thin z (0.1), 2D.
        Grid: 35 × 35 × 1; front/back 'empty'.
        'movingWall' at y=1 moves in +x with U=1 m/s. Others 'fixedWalls' no-slip.
        Run from time=0 to time=0.001 with Δt = 0.01; write every 100 steps.
        nu = 1/400 = 0.0025 m^2/s.
        Visualize Visualization
        ''',
        '''
        Normal requirement with good parameters.
        Run from time=0 to time=10 with Δt = 0.005; write every 100 steps.
        '''
    ]
    
    print("🔍 Testing validation functionality...")
    print("="*60)
    
    # Create validator (don't need LLM for this)
    validator = HypothesisAgent()
    
    # Validate requirements
    results = validator.validate_user_requirements(test_requirements)
    
    for i, result in enumerate(results, 1):
        print(f"\nRequirement {i}:")
        print(f"Original: {result['original'].strip()[:100]}...")
        
        if result['issues']:
            print(f"❌ Issues found:")
            for issue in result['issues']:
                print(f"   - {issue}")
            print(f"✅ Corrected: {result['corrected'].strip()[:100]}...")
        else:
            print(f"✅ No issues found")
            
        print(f"Parsed: t0={result['parsed']['t0']}, t1={result['parsed']['t1']}, dt={result['parsed']['dt']}")

if __name__ == "__main__":
    test_validation()