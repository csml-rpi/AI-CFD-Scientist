#!/usr/bin/env python3
"""
Test the improved validation with the exact requirements from main.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.hypotheis_ag import HypothesisAgent

def test_improved_validation():
    """Test the improved validation with the requirements from main.py"""
    
    # Requirements from main.py - including the problematic one
    user_requirements = [
        '''
    Do an incompressible lid-driven cavity flow.
    The cavity is a square with dimensions 1 (x) × 1 (y) and very thin in z (0.1), making it effectively 2D.
    Use a grid of 35 × 35 in x and y, and 1 cell in z. The front and back faces are 'empty'.
    The top wall ('movingWall') at y=1 moves in +x with U=1 m/s.
    All other walls ('fixedWalls') are no-slip (U=0).
    Run from time=0 to time=10 with time step Δt = 0.00025; write results every 100 steps.
    Set kinematic viscosity nu = 1/100 = 0.01 m^2/s (constant).
    Visualize Visualization
    ''',
        '''
    Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
    Grid: 41 × 41 × 1; front/back 'empty'.
    Top lid (y=1) U=1 m/s in +x; other walls no-slip.
    Run from time=10 to time=0 with Δt = 0.0015; write every 100 steps.
    nu = 1/2000 = 0.0005 m^2/s.
    Visualize Visualization
    ''',
        '''
    Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
    Grid: 47 × 47 × 1; front/back 'empty'.
    'movingWall' at y=1: U=1 m/s in +x. Others no-slip.
    Run from time=0 to time=100 with Δt = 0.001; write every 100 steps.
    nu = 1/5000 = 0.0002 m^2/s.
    Visualize Visualization
    '''
    ]
    
    print("="*80)
    print("Testing Improved HypothesisAgent Validation")
    print("="*80)
    
    agent = HypothesisAgent()
    results = agent.validate_user_requirements(user_requirements)
    
    for i, result in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"Requirement {i}")
        print(f"{'='*60}")
        
        parsed = result['parsed']
        print(f"Parsed: t0={parsed['t0']}, t1={parsed['t1']}, dt={parsed['dt']}")
        if parsed['total_time']:
            print(f"Total time: {parsed['total_time']}")
        if parsed['num_timesteps']:
            print(f"Number of timesteps: {parsed['num_timesteps']:.1f}")
        
        if result['issues']:
            print(f"\n🔍 Issues found:")
            for issue in result['issues']:
                print(f"  ⚠️  {issue}")
            
            print(f"\n📝 Original:")
            print(result['original'])
            print(f"\n✅ Corrected:")
            print(result['corrected'])
            print(f"\n🔍 Changes made:")
            original_lines = result['original'].strip().split('\n')
            corrected_lines = result['corrected'].strip().split('\n')
            for j, (orig, corr) in enumerate(zip(original_lines, corrected_lines)):
                if orig != corr:
                    print(f"  Line {j+1}: '{orig.strip()}' → '{corr.strip()}'")
        else:
            print(f"\n✅ No issues found - requirement is valid as-is")
    
    print(f"\n{'='*80}")
    print(f"Validation Complete")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_improved_validation()