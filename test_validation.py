#!/usr/bin/env python3
"""
Test script to demonstrate the HypothesisAgent validation functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.hypotheis_ag import HypothesisAgent

def main():
    """Test the validation functionality with the user requirements from main.py"""
    
    # User requirements from main.py (exactly as provided)
    user_requirements = []
    
    # ------------ 2D Square Cavity --------------
    # Re = 100 (35×35)
    user_requirements.append('''
    Do an incompressible lid-driven cavity flow.
    The cavity is a square with dimensions 1 (x) × 1 (y) and very thin in z (0.1), making it effectively 2D.
    Use a grid of 35 × 35 in x and y, and 1 cell in z. The front and back faces are 'empty'.
    The top wall ('movingWall') at y=1 moves in +x with U=1 m/s.
    All other walls ('fixedWalls') are no-slip (U=0).
    Run from time=0 to time=10 with time step Δt = 0.00025; write results every 100 steps.
    Set kinematic viscosity nu = 1/100 = 0.01 m^2/s (constant).
    Visualize Visualization
    ''')
    
    # Re = 400 (35×35)
    user_requirements.append('''
    Do an incompressible lid-driven cavity flow.
    Square cavity 1 (x) × 1 (y), thin z (0.1), effectively 2D.
    Grid: 35 × 35 × 1; front/back 'empty'.
    'movingWall' at y=1 moves in +x with U=1 m/s. Others 'fixedWalls' no-slip.
    Run from time=0 to time=25 with Δt = 0.0015; write every 100 steps.
    nu = 1/400 = 0.0025 m^2/s.
    Visualize Visualization
    ''')
    
    # Re = 1000 (35×35)
    user_requirements.append('''
    Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
    Grid: 35 × 35 × 1; front/back 'empty'.
    'movingWall' (y=1): U=(1,0,0) m/s. Other walls no-slip.
    Run from time=0 to time=50 with Δt=0.0015; write every 100 steps.
    nu = 1/1000 = 0.001 m^2/s.
    Visualize Visualization
    ''')
    
    # Re = 2000 (41×41)
    user_requirements.append('''
    Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
    Grid: 41 × 41 × 1; front/back 'empty'.
    Top lid (y=1) U=1 m/s in +x; other walls no-slip.
    Run from time=0 to time=70 with Δt = 0.0015; write every 100 steps.
    nu = 1/2000 = 0.0005 m^2/s.
    Visualize Visualization
    ''')
    
    # Re = 5000 (47×47)
    user_requirements.append('''
    Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
    Grid: 47 × 47 × 1; front/back 'empty'.
    'movingWall' at y=1: U=1 m/s in +x. Others no-slip.
    Run from time=0 to time=100 with Δt = 0.001; write every 100 steps.
    nu = 1/5000 = 0.0002 m^2/s.
    Visualize Visualization
    ''')
    
    # Re = 10000 (61×61)
    user_requirements.append('''
      Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
      Grid: 61 × 61 × 1; front/back 'empty'.
      Top lid y=1 moves U=1 m/s in +x; other walls no-slip.
      Run from time=0 to time=200 with Δt = 0.0008; write every 100 steps.
      nu = 1/10000 = 0.0001 m^2/s.
      Visualize Visualization
    ''')
    
    print("="*80)
    print("CFD User Requirements Validation Test")
    print("="*80)
    
    # Initialize HypothesisAgent (no model needed for validation)
    agent = HypothesisAgent()
    
    # Validate requirements
    print(f"\nValidating {len(user_requirements)} user requirements...\n")
    
    validation_results = agent.validate_user_requirements(user_requirements)
    
    # Display results
    corrected_requirements = []
    
    for i, result in enumerate(validation_results, 1):
        print(f"{'='*60}")
        print(f"Requirement {i} (Re = {[100, 400, 1000, 2000, 5000, 10000][i-1]})")
        print(f"{'='*60}")
        
        parsed = result['parsed']
        print(f"Parsed values:")
        print(f"  t0 = {parsed['t0']}")
        print(f"  t1 = {parsed['t1']}")  
        print(f"  dt = {parsed['dt']}")
        print(f"  total_time = {parsed['total_time']}")
        print(f"  write_every_steps = {parsed['write_every_steps']}")
        
        if result['issues']:
            print(f"\n🔍 Issues found:")
            for issue in result['issues']:
                print(f"  ⚠️  {issue}")
            print(f"\n📝 Corrected requirement:")
            print(f"```")
            print(result['corrected'])
            print(f"```")
            corrected_requirements.append(result['corrected'])
        else:
            print(f"\n✅ No issues found - requirement is valid")
            corrected_requirements.append(result['original'])
        
        print()
    
    print("="*80)
    print("Summary")
    print("="*80)
    
    total_issues = sum(len(r['issues']) for r in validation_results)
    print(f"Total requirements: {len(user_requirements)}")
    print(f"Requirements with issues: {sum(1 for r in validation_results if r['issues'])}")
    print(f"Total issues found: {total_issues}")
    
    if total_issues > 0:
        print(f"\n💡 All corrected requirements can be used by replacing the original list.")
        print(f"   The corrected versions fix timing inconsistencies and maintain all other parameters.")
    else:
        print(f"\n✅ All requirements are valid and ready to use!")
    
    return corrected_requirements

if __name__ == "__main__":
    corrected_reqs = main()