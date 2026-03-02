#!/usr/bin/env python3
"""
Quick test script to test OpenAI model functionality.
"""


import os
import sys
sys.path.append('.')
from utils.base_llm import create_client, get_response_from_llm
from pathlib import Path
import argparse
import json
import subprocess
from datetime import datetime
import uuid

# Set default model for all LLM/agent calls (Bedrock by default).
# For Bedrock, always use the Claude Sonnet 4 application profile.
DEFAULT_BEDROCK_MODEL = os.environ.get(
    "CFD_SCIENTIST_MODEL",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
)

# Allow override via command-line argument
def parse_args():
    parser = argparse.ArgumentParser(description="CFD Scientist Main Runner")
    parser.add_argument('--model', type=str, default=DEFAULT_BEDROCK_MODEL, 
                       help='LLM model to use (Bedrock ARN, model ID, or OpenAI/Anthropic)')
    parser.add_argument('--rerun-batch', type=str, default=None,
                       help='Rerun failed experiments from a specific batch (provide batch name)')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip automatic analysis after simulations (default: False)')
    parser.add_argument('--auto-rerun', action='store_true',
                       help='Automatically rerun failed experiments without asking (default: False)')
    parser.add_argument('--analyze-after-rerun', action='store_true',
                       help='After reruns complete, analyze the batch again (default: False)')
    parser.add_argument('--auto-rerun-threshold', type=float, default=5.0,
                       help='Threshold below which cases are automatically rerun (default: 5.0)')
    parser.add_argument('--max-rerun-iterations', type=int, default=3,
                       help='Maximum rerun attempts per case (default: 3)')
    parser.add_argument('--auto-execute-recommendations', action='store_true',
                       help='Automatically execute study recommendations as new experiments (default: False)')
    return parser.parse_args()

args = parse_args()
os.environ["CFD_SCIENTIST_MODEL"] = args.model
print(f"[CFD Scientist] Using LLM model: {args.model}")

def run_foam(user_requirement: str, show_output: bool = False, run_index: int = None, experiment_dir: Path = None, original_run_name: str = None):
    """
    Run Foam-Agent with a user requirement.
    
    Args:
        user_requirement (str): The user requirement for the simulation
        show_output (bool): Whether to show output in terminal
        run_index (int): Optional run index for unique naming
        experiment_dir (Path): Optional specific experiment directory to use
        original_run_name (str): If this is a rerun, the name of the original run (e.g., "run_001")
    
    Returns:
        dict: Result dictionary with success status and paths
    """
    # Ensure all paths are absolute and exist
    src_dir = Path(__file__).parent.resolve()
    project_root = src_dir.parent.resolve()
    foam_bench = project_root / "Foam-Agent" / "foambench_main.py"
    
    # Verify Foam-Agent exists
    if not foam_bench.exists():
        print(f"❌ Error: Foam-Agent not found at {foam_bench}")
        return {
            'success': False,
            'return_code': -1,
            'error': f'Foam-Agent not found at {foam_bench}'
        }

    # Create unique run directory in the specified experiment folder
    from datetime import datetime
    import time
    
    # Use provided experiment directory or create default one under data/output
    if experiment_dir is None:
        base_experiments_dir = project_root / "data" / "experiments"
        base_experiments_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        experiment_dir = base_experiments_dir / f"sim_{timestamp}_{short_id}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique run directory name within the experiment
    if original_run_name is not None:
        # This is a rerun - create a clearly named rerun directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = experiment_dir / f"{original_run_name}_rerun_{timestamp}"
    elif run_index is not None:
        run_dir = experiment_dir / f"run_{run_index:03d}"
    else:
        # Use microseconds for uniqueness when no index provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = experiment_dir / f"run_{timestamp}"
    
    # Ensure directory doesn't already exist (add counter if needed)
    counter = 1
    original_run_dir = run_dir
    while run_dir.exists():
        if original_run_name is not None:
            # For reruns, add a counter to the rerun name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = experiment_dir / f"{original_run_name}_rerun_{timestamp}_{counter:02d}"
        elif run_index is not None:
            run_dir = experiment_dir / f"run_{run_index:03d}_{counter:02d}"
        else:
            run_dir = original_run_dir.parent / f"{original_run_dir.name}_{counter:02d}"
        counter += 1
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    if original_run_name is not None:
        print(f"📁 Created RERUN directory: {run_dir}")
        print(f"🔁 This is a rerun of: {original_run_name}")
    else:
        print(f"📁 Created run directory: {run_dir}")
    
    prompt_file = run_dir / "user_requirement.txt"
    # Foam-Agent expects an output directory; place it under each run folder as 'output'
    output_dir = run_dir / "output"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
  
    try:
        prompt_file.write_text(user_requirement)
        print(f"📄 User requirement written to: {prompt_file}")
        
    except Exception as e:
        print(f"❌ Error writing user requirement file: {e}")
        return {
            'run_dir': str(run_dir),
            'output_dir': str(output_dir),
            'success': False,
            'return_code': -1,
            'error': f'Failed to write user requirement file: {e}'
        }

    # Build command
    cmd = [
        "python", str(foam_bench),
        "--openfoam_path", os.environ.get("WM_PROJECT_DIR", "/opt/openfoam10"),
        "--output", str(output_dir),
        "--prompt_path", str(prompt_file),
    ]

    print(f"🚀 Running Foam-Agent: {' '.join(cmd)}")
    print(f"📁 Run folder: {run_dir}")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    
    print(f"🔧 Working directory: {project_root}")
    print(f"🔧 Foam-Agent path: {foam_bench}")
    print(f"🔧 Output directory: {output_dir}")
    
    # Run Foam-Agent
    try:
        if show_output:
            # Run with real-time output display
            result = subprocess.run(
                cmd, 
                cwd=str(project_root), 
                env=env,
                capture_output=False,
                text=True
            )
        else:
            # Run with captured output
            result = subprocess.run(
                cmd, 
                cwd=str(project_root), 
                env=env,
                capture_output=True,
                text=True
            )
            
    except Exception as e:
        print(f"❌ Error running Foam-Agent: {e}")
        return {
            'run_dir': str(run_dir),
            'output_dir': str(output_dir),
            'success': False,
            'return_code': -1,
            'error': f'Failed to run Foam-Agent: {e}'
        }

    # Check result
    if result.returncode != 0:
        print(f"❌ Foam-Agent failed (exit code {result.returncode})")
        if hasattr(result, 'stderr') and result.stderr:
            print(f"Error: {result.stderr}")
    else:
        print(f"✅ Foam-Agent succeeded! Outputs in: {output_dir}")
    
    return {
        'run_dir': str(run_dir),
        'output_dir': str(output_dir),
        'success': result.returncode == 0,
        'return_code': result.returncode
    }


def rerun_failed_experiments(batch_name: str, analyze_after: bool = False):
    """
    Rerun experiments that were flagged as needing improvement by the analysis agent.
    
    Args:
        batch_name: Name of the batch to rerun experiments from
    """
    project_root = Path(__file__).parent.parent.resolve()
    batch_dir = project_root / "data" / "experiments" / batch_name
    
    if not batch_dir.exists():
        print(f"❌ Batch directory not found: {batch_dir}")
        return
    
    suggestions_file = batch_dir / "rerun_suggestions.json"
    if not suggestions_file.exists():
        print(f"❌ No rerun suggestions found at {suggestions_file}")
        print("Run analysis first: python agents/analysis_ag.py")
        return
    
    # Load rerun suggestions
    try:
        rerun_suggestions = json.loads(suggestions_file.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"❌ Failed to load rerun suggestions: {e}")
        return
    
    if not rerun_suggestions:
        print("✅ No experiments need reruns!")
        return
    
    print(f"\n{'='*60}")
    print(f"🔁 Rerunning {len(rerun_suggestions)} experiments with updated requirements")
    print(f"{'='*60}\n")
    
    rerun_results = []
    
    for idx, suggestion in enumerate(rerun_suggestions, 1):
        exp_name = suggestion['experiment']
        run_name = suggestion['run']
        original_req = suggestion['original_requirement']
        updated_req = suggestion['updated_requirement']
        accuracy = suggestion['accuracy']
        
        print(f"\n[{idx}/{len(rerun_suggestions)}] Rerunning: {exp_name}/{run_name}")
        print(f"   Original accuracy: {accuracy:.1f}/10")
        print(f"   Original requirement: {original_req[:100]}...")
        print(f"   Updated requirement: {updated_req[:100]}...")
        
        # Use the experiment directory for this specific experiment
        exp_dir = batch_dir / exp_name
        
        # Extract the original run name from the full run path
        original_run_name = run_name  # This should be like "run_001"
        
        try:
            result = run_foam(
                user_requirement=updated_req,
                show_output=True,
                run_index=None,
                experiment_dir=exp_dir,
                original_run_name=original_run_name
            )
            
            rerun_results.append({
                **suggestion,
                "rerun_success": result.get("success", False),
                "rerun_dir": str(result.get("run_dir", "")),
                "rerun_error": result.get("error", None)
            })
            
            if result.get("success"):
                print(f"   ✅ Rerun successful: {result.get('run_dir')}")
            else:
                print(f"   ❌ Rerun failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Exception during rerun: {e}")
            rerun_results.append({
                **suggestion,
                "rerun_success": False,
                "rerun_error": str(e)
            })
    
    # Save rerun results
    try:
        results_file = batch_dir / "rerun_results.json"
        results_file.write_text(json.dumps(rerun_results, indent=2), encoding='utf-8')
        print(f"\n📋 Saved rerun results to {results_file}")
    except Exception as e:
        print(f"Failed to save rerun results: {e}")
    
    # Print summary
    successful = sum(1 for r in rerun_results if r.get("rerun_success", False))
    print(f"\n{'='*60}")
    print(f"📊 Rerun Summary: {successful}/{len(rerun_results)} successful")
    print(f"{'='*60}")

    # Optionally analyze the batch again after reruns complete
    if analyze_after:
        try:
            print("\n" + "="*60)
            print("🔬 Analyzing batch again after reruns...")
            print("="*60 + "\n")
            from agents.analysis_ag import analyze_batch
            analysis_text, rerun_suggestions = analyze_batch(
                batch_dir=batch_dir,
                model=os.environ.get("CFD_SCIENTIST_MODEL", DEFAULT_BEDROCK_MODEL),
                temperature=0.0
            )
            if analysis_text:
                print("\n✅ Post-rerun analysis completed.")
                print(f"📄 Analysis saved to: {batch_dir / 'analysis_summary.txt'}")
            if rerun_suggestions:
                print(f"\n⚠️  Found {len(rerun_suggestions)} experiments that still need reruns")
                print(f"   You can run: python src/main.py --rerun-batch {batch_name}")
            else:
                print("\n✅ All rerun experiments now meet quality standards!")
        except Exception as e:
            print(f"\n⚠️  Post-rerun analysis failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    # If re-run is in command, rerun experiments
    if args.rerun_batch:
        rerun_failed_experiments(args.rerun_batch)
        return
  
    print("=== CFD SCIENTIST - LID-DRIVEN CAVITY STUDY ===")
    
    # Create unique experiment directory for this batch of runs under data/experiments
    from datetime import datetime 
    import uuid
    base_experiments_dir = Path(__file__).parent.parent / "data" / "experiments"
    base_experiments_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]

    # Create an outer batch folder to group all simulations from this invocation
    batch_name = f"batch_{timestamp}_{short_id}"
    batch_dir = base_experiments_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Inside the batch folder, create the experiment folder that will contain runs
    experiment_name = f"sim_{timestamp}_{short_id}"
    current_experiment_dir = batch_dir / experiment_name
    current_experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Created batch directory: {batch_dir}")
    print(f"📁 Created experiment directory: {current_experiment_dir}")
    print(f"🔬 Experiment: {experiment_name}")
    
    # User requirements as a list - easy to add multiple requirements
    # skip ideation for now, use lid-driven cavity flow pre-defined user requirements 
    
    simulation_description = "Lid-driven cavity flow simulations at various Reynolds numbers following Ghia et al. (1982) benchmark study"
    simulation_instructions = """
    Analysis Instructions:
    1. Extract centerline velocity profiles (u-velocity along vertical centerline at x=0.5, v-velocity along horizontal centerline at y=0.5)
    2. Compare flow structures: primary vortex location and strength, corner eddies formation
    3. Analyze Reynolds number effects on flow transition and vortex structure
    4. For 2D cases: compare with benchmark data from literature (Ghia et al.)
    5. For 3D cases: analyze differences from 2D counterparts, check for 3D effects
    6. Document convergence behavior and grid resolution adequacy
    7. Identify any numerical artifacts or unphysical behavior
    """
    
    # Structure: (case_id, simulation_description, simulation_instructions, user_requirement)
    simulation_configs = []
    
    # ------------ 2D Square Cavity (Unit Aspect Ratio) --------------
    # Case 2D-SQ-100: Re = 100 (35×35)
    simulation_configs.append({
        'case_id': '2D-SQ-100',
        'profile_id': 'UNIT-2D',
        'reynolds_number': 100,
        'geometry': '2D Square Cavity (1×1×0.1)',
        'user_requirement': '''
    Do an incompressible lid-driven cavity flow.
    The cavity is a square with dimensions 1 (x) × 1 (y) and very thin in z (0.1), making it effectively 2D.
    Use a grid of 35 × 35 in x and y, and 1 cell in z. The front and back faces are 'empty'.
    The top wall ('movingWall') at y=1 moves in +x with U=1 m/s.
    All other walls ('fixedWalls') are no-slip (U=0).
    Run from time=0 to time=10 with time step Δt = 0.00025; write results every 100 steps.
    Set kinematic viscosity nu = 1/100 = 0.01 m^2/s (constant).
    Visualization:
    - Generate a 2D snapshot at the final time (t=10) on the mid-plane z=0.05.
    - Plot velocity magnitude |U| as a filled contour (colormap) over the cavity.
    - Overlay velocity vectors (quiver) sparsely so they are readable.
    - Keep any title/annotations minimal and positioned so they do not overlap the domain or the colorbar.
    '''
    })
    
    # Case 2D-SQ-400: Re = 400 (35×35)
    # simulation_configs.append({
    #     'case_id': '2D-SQ-400',
    #     'profile_id': 'UNIT-2D',
    #     'reynolds_number': 400,
    #     'geometry': '2D Square Cavity (1×1×0.1)',
    #     'user_requirement': '''
    # Do an incompressible lid-driven cavity flow.
    # Square cavity 1 (x) × 1 (y), thin z (0.1), effectively 2D.
    # Grid: 35 × 35 × 1; front/back 'empty'.
    # 'movingWall' at y=1 moves in +x with U=1 m/s. Others 'fixedWalls' no-slip.
    # Run from time=0 to time=25 with Δt = 0.0015; write every 100 steps.
    # nu = 1/400 = 0.0025 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    
    # Case 2D-SQ-1000: Re = 1000 (35×35)
    # simulation_configs.append({
    #     'case_id': '2D-SQ-1000',
    #     'profile_id': 'UNIT-2D',
    #     'reynolds_number': 1000,
    #     'geometry': '2D Square Cavity (1×1×0.1)',
    #     'user_requirement': '''
    # Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
    # Grid: 35 × 35 × 1; front/back 'empty'.
    # 'movingWall' (y=1): U=(1,0,0) m/s. Other walls no-slip.
    # Run from time=0 to time=50 with Δt=0.0015; write every 100 steps.
    # nu = 1/1000 = 0.001 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    
    # Case 2D-SQ-2000: Re = 2000 (41×41)
    # simulation_configs.append({
    #     'case_id': '2D-SQ-2000',
    #     'profile_id': 'UNIT-2D',
    #     'reynolds_number': 2000,
    #     'geometry': '2D Square Cavity (1×1×0.1)',
    #     'user_requirement': '''
    # Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
    # Grid: 41 × 41 × 1; front/back 'empty'.
    # Top lid (y=1) U=1 m/s in +x; other walls no-slip.
    # Run from time=0 to time=70 with Δt = 0.0015; write every 100 steps.
    # nu = 1/2000 = 0.0005 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    
    # Case 2D-SQ-5000: Re = 5000 (47×47)
    # simulation_configs.append({
    #     'case_id': '2D-SQ-5000',
    #     'profile_id': 'UNIT-2D',
    #     'reynolds_number': 5000,
    #     'geometry': '2D Square Cavity (1×1×0.1)',
    #     'user_requirement': '''
    # Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
    # Grid: 47 × 47 × 1; front/back 'empty'.
    # 'movingWall' at y=1: U=1 m/s in +x. Others no-slip.
    # Run from time=0 to time=100 with Δt = 0.001; write every 100 steps.
    # nu = 1/5000 = 0.0002 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    
    # Case 2D-SQ-10000: Re = 10000 (61×61)
    # simulation_configs.append({
    #     'case_id': '2D-SQ-10000',
    #     'profile_id': 'UNIT-2D',
    #     'reynolds_number': 10000,
    #     'geometry': '2D Square Cavity (1×1×0.1)',
    #     'user_requirement': '''
    #   Incompressible lid-driven cavity, square 1×1, thin z (0.1), 2D.
    #   Grid: 61 × 61 × 1; front/back 'empty'.
    #   Top lid y=1 moves U=1 m/s in +x; other walls no-slip.
    #   Run from time=0 to time=200 with Δt = 0.0005; write every 200 steps.
    #   nu = 1/10000 = 0.0001 m^2/s.
    #   Visualize velocity magnitude contours and streamlines
    #   '''
    # })
    
    # ------------ 3D Cubic Cavity --------------
    # # Case 3D-CB-100: Re = 100 (35×35×35)
    # simulation_configs.append({
    #     'case_id': '3D-CB-100',
    #     'profile_id': 'UNIT-3D-CUBE',
    #     'reynolds_number': 100,
    #     'geometry': '3D Cubic Cavity (1×1×1)',
    #     'user_requirement': '''
    # Do an incompressible lid-driven cavity flow in 3D.
    # Cube 1 (x) × 1 (y) × 1 (z). Grid: 35 × 35 × 35.
    # Top face at z=1 ('movingWall') moves in +x with U=1 m/s. All other faces no-slip.
    # Run from time=0 to time=10 with Δt = 0.00025; write results every 100 steps.
    # nu = 1/100 = 0.01 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 3D-CB-400: Re = 400 (35×35×35)
    # simulation_configs.append({
    #     'case_id': '3D-CB-400',
    #     'profile_id': 'UNIT-3D-CUBE',
    #     'reynolds_number': 400,
    #     'geometry': '3D Cubic Cavity (1×1×1)',
    #     'user_requirement': '''
    # Incompressible 3D lid-driven cavity, cube 1×1×1. Grid: 35 × 35 × 35.
    # Lid z=1 moves +x at 1 m/s; other walls no-slip.
    # Run from time=0 to time=25 with Δt = 0.0015; write results every 100 steps.
    # nu = 1/400 = 0.0025 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 3D-CB-1000: Re = 1000 (35×35×35)
    # simulation_configs.append({
    #     'case_id': '3D-CB-1000',
    #     'profile_id': 'UNIT-3D-CUBE',
    #     'reynolds_number': 1000,
    #     'geometry': '3D Cubic Cavity (1×1×1)',
    #     'user_requirement': '''
    # 3D cube lid-driven cavity 1×1×1; grid 35×35×35.
    # 'movingWall' at z=1: U=(1,0,0) m/s; other walls no-slip.
    # Run from time=0 to time=50 with Δt = 0.0015; write results every 100 steps.
    # nu = 1/1000 = 0.001 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 3D-CB-2000: Re = 2000 (41×41×41)
    # simulation_configs.append({
    #     'case_id': '3D-CB-2000',
    #     'profile_id': 'UNIT-3D-CUBE',
    #     'reynolds_number': 2000,
    #     'geometry': '3D Cubic Cavity (1×1×1)',
    #     'user_requirement': '''
    # 3D incompressible cube cavity 1×1×1; grid 41×41×41.
    # Top lid z=1 moves +x at 1 m/s; others no-slip.
    # Run from time=0 to time=70 with Δt = 0.0015; write results every 100 steps.
    # nu = 1/2000 = 0.0005 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 3D-CB-5000: Re = 5000 (47×47×47)
    # simulation_configs.append({
    #     'case_id': '3D-CB-5000',
    #     'profile_id': 'UNIT-3D-CUBE',
    #     'reynolds_number': 5000,
    #     'geometry': '3D Cubic Cavity (1×1×1)',
    #     'user_requirement': '''
    # 3D incompressible lid-driven cube 1×1×1; grid 47×47×47.
    # Lid z=1 moves in +x with U=1 m/s; others no-slip.
    # Run from time=0 to time=100 with Δt = 0.001; write results every 100 steps.
    # nu = 1/5000 = 0.0002 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    
    
    # ------------ 2D Rectangle Cavity --------------
    # # Case 2D-RC-100: Re = 100 (35×70×1)
    # simulation_configs.append({
    #     'case_id': '2D-RC-100',
    #     'profile_id': 'RECT-2D',
    #     'reynolds_number': 100,
    #     'geometry': '2D Rectangle Cavity (1×2×0.1)',
    #     'user_requirement': '''
    # Incompressible lid-driven cavity, rectangle 1×2, thin z (0.1), 2D.
    # Grid: 35 × 70 × 1; front/back 'empty'.
    # Lid y=2 U=1 m/s in +x; other walls no-slip.
    # Run from time=0 to time=15 with Δt = 0.00025; write results every 100 steps.
    # nu = 1/100 = 0.01 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 2D-RC-400: Re = 400 (35×70×1)
    # simulation_configs.append({
    #     'case_id': '2D-RC-400',
    #     'profile_id': 'RECT-2D',
    #     'reynolds_number': 400,
    #     'geometry': '2D Rectangle Cavity (1×2×0.1)',
    #     'user_requirement': '''
    # Incompressible lid-driven cavity, rectangle 1×2, thin z (0.1), 2D.
    # Grid: 35 × 70 × 1; front/back 'empty'.
    # Lid y=2 U=1 m/s in +x; other walls no-slip.
    # Run from time=0 to time=25 with Δt = 0.0015; write results every 100 steps.
    # nu = 1/400 = 0.0025 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 2D-RC-1000: Re = 1000 (35×70×1)
    # simulation_configs.append({
    #     'case_id': '2D-RC-1000',
    #     'profile_id': 'RECT-2D',
    #     'reynolds_number': 1000,
    #     'geometry': '2D Rectangle Cavity (1×2×0.1)',
    #     'user_requirement': '''
    # 2D incompressible rectangle 1×2 (thin z=0.1).
    # Grid: 35 × 70 × 1; 'empty' front/back.
    # Top lid y=2 moves +x at 1 m/s; others no-slip.
    # Run from time=0 to time=50 with Δt = 0.0015; write results every 100 steps.
    # nu = 1/1000 = 0.001 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 2D-RC-2000: Re = 2000 (41×82×1)
    # simulation_configs.append({
    #     'case_id': '2D-RC-2000',
    #     'profile_id': 'RECT-2D',
    #     'reynolds_number': 2000,
    #     'geometry': '2D Rectangle Cavity (1×2×0.1)',
    #     'user_requirement': '''
    # 2D lid-driven cavity, rectangle 1×2, thin z=0.1.
    # Grid: 41 × 82 × 1; 'empty' front/back.
    # Lid y=2: U=1 m/s in +x; others no-slip.
    # Run from time=0 to time=70 with Δt = 0.0015; write results every 100 steps.
    # nu = 1/2000 = 0.0005 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 2D-RC-5000: Re = 5000 (47×94×1)
    # simulation_configs.append({
    #     'case_id': '2D-RC-5000',
    #     'profile_id': 'RECT-2D',
    #     'reynolds_number': 5000,
    #     'geometry': '2D Rectangle Cavity (1×2×0.1)',
    #     'user_requirement': '''
    # 2D incompressible lid-driven rectangle 1×2, thin z=0.1.
    # Grid: 47 × 94 × 1; 'empty' front/back.
    # Top wall y=2 moves +x at 1 m/s; others no-slip.
    # Run from time=0 to time=150 with Δt = 0.001; write results every 100 steps.
    # nu = 1/5000 = 0.0002 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 2D-RC-10000: Re = 10000 (61×122×1)
    # simulation_configs.append({
    #     'case_id': '2D-RC-10000',
    #     'profile_id': 'RECT-2D',
    #     'reynolds_number': 10000,
    #     'geometry': '2D Rectangle Cavity (1×2×0.1)',
    #     'user_requirement': '''
    # 2D lid-driven cavity, rectangle 1×2, thin z=0.1.
    # Grid: 61 × 122 × 1; front/back 'empty'.
    # Lid y=2: U=1 m/s in +x; others no-slip.
    # Run from time=0 to time=200 with Δt = 0.0008; write results every 100 steps.
    # nu = 1/10000 = 0.0001 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })    
    # 
    # # ------------ 3D Prism Cavity --------------
    # # Case 3D-PR-100: Re = 100 (35×35×70)
    # simulation_configs.append({
    #     'case_id': '3D-PR-100',
    #     'profile_id': 'PRISM-3D',
    #     'reynolds_number': 100,
    #     'geometry': '3D Prism Cavity (1×1×2)',
    #     'user_requirement': '''
    # Incompressible 3D lid-driven cavity.
    # Prism 1 (x) × 1 (y) × 2 (z). Grid: 35 × 35 × 70.
    # Top face at z=2 ('movingWall') moves in +x with U=1 m/s; other faces no-slip.
    # Run from time=0 to time=15 with Δt = 0.00025; write results every 100 steps.
    # nu = 1/100 = 0.01 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 3D-PR-400: Re = 400 (35×35×70)
    # simulation_configs.append({
    #     'case_id': '3D-PR-400',
    #     'profile_id': 'PRISM-3D',
    #     'reynolds_number': 400,
    #     'geometry': '3D Prism Cavity (1×1×2)',
    #     'user_requirement': '''
    # 3D incompressible lid-driven prism 1×1×2; grid 35×35×70.
    # Lid z=2 U=1 m/s in +x; others no-slip.
    # Run from time=0 to time=25 with Δt = 0.001; write results every 100 steps.
    # nu = 1/400 = 0.0025 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 3D-PR-1000: Re = 1000 (35×35×70)
    # simulation_configs.append({
    #     'case_id': '3D-PR-1000',
    #     'profile_id': 'PRISM-3D',
    #     'reynolds_number': 1000,
    #     'geometry': '3D Prism Cavity (1×1×2)',
    #     'user_requirement': '''
    # 3D prism cavity 1×1×2; grid 35×35×70.
    # Top wall z=2 moves +x at 1 m/s; others no-slip.
    # Run from time=0 to time=50 with Δt = 0.0015; write results every 100 steps.
    # nu = 1/1000 = 0.001 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 3D-PR-2000: Re = 2000 (41×41×82)
    # simulation_configs.append({
    #     'case_id': '3D-PR-2000',
    #     'profile_id': 'PRISM-3D',
    #     'reynolds_number': 2000,
    #     'geometry': '3D Prism Cavity (1×1×2)',
    #     'user_requirement': '''
    # 3D incompressible prism 1×1×2; grid 41×41×82.
    # Lid z=2 U=1 m/s in +x; others no-slip.
    # Run from time=0 to time=70 with Δt = 0.0015; write results every 100 steps.
    # nu = 1/2000 = 0.0005 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    # 
    # # Case 3D-PR-5000: Re = 5000 (47×47×94)
    # simulation_configs.append({
    #     'case_id': '3D-PR-5000',
    #     'profile_id': 'PRISM-3D',
    #     'reynolds_number': 5000,
    #     'geometry': '3D Prism Cavity (1×1×2)',
    #     'user_requirement': '''
    # 3D lid-driven prism 1×1×2; grid 47×47×94.
    # Top face z=2 moves +x at 1 m/s; others no-slip.
    # Run from time=0 to time=100 with Δt = 0.001; write results every 100 steps.
    # nu = 1/5000 = 0.0002 m^2/s.
    # Visualize velocity magnitude contours and streamlines
    # '''
    # })
    
    print("📝 FINISHED CASE CONFIGS AND Starting simulations for lid-driven cavity flow cases...")
    
    # Extract user requirements from configs for backward compatibility
    user_requirements = [config['user_requirement'] for config in simulation_configs]
    
    # Validate and correct user requirements before running simulations
    print("\n🔍 Validating user requirements...")
    from agents.hypotheis_ag import HypothesisAgent
    validator = HypothesisAgent()
    validation_results = validator.validate_user_requirements(user_requirements)
    
    corrected_requirements = []
    for i, result in enumerate(validation_results, 1):
        if result['issues']:
            print(f"\n⚠️  Requirement {i} has issues:")
            for issue in result['issues']:
                print(f"   - {issue}")
            print(f"   Using corrected version.")
            corrected_requirements.append(result['corrected'].strip())
        else:
            print(f"✅ Requirement {i} looks good")
            corrected_requirements.append(result['original'].strip())
    
    # Use corrected requirements for simulation
    user_requirements = corrected_requirements
    print(f"\n✅ All user requirements validated and corrected as needed")
    
    print("\n" + "="*50 + "\n")
    
    # Run Foam-Agent for each user requirement
    results = []
    
    # default - uses bedrock
    for i, user_requirement in enumerate(user_requirements, 1):
        # Check if this is one of the structured simulation configs
        sim_config = None
        if i <= len(simulation_configs):
            sim_config = simulation_configs[i-1]
            case_id = sim_config['case_id']
            print(f" Running Foam-Agent for case {case_id} ({i}/{len(user_requirements)})...")
            print(f" Case ID: {case_id}")
            print(f" Profile: {sim_config['profile_id']}")
            print(f" Reynolds Number: {sim_config['reynolds_number']}")
            print(f" Geometry: {sim_config['geometry']}")
        else:
            print(f" Running Foam-Agent for requirement {i}/{len(user_requirements)}...")
        
        print(f"\n💼 User Requirement:")
        print(user_requirement.strip())
        print("\n" + "-"*30 + "\n")
        
        # Pass experiment directory to ensure all runs go to the same experiment folder
        result = run_foam(
            user_requirement.strip(), 
            show_output=True, 
            run_index=i, 
            experiment_dir=current_experiment_dir
        )
        
        result_entry = {
            'requirement_index': i,
            'result': result
        }
        
        # Add simulation config info if available
        if sim_config:
            result_entry['case_id'] = case_id
            result_entry['simulation_config'] = sim_config
        
        results.append(result_entry)
        
        if result['success']:
            success_msg = f"\n✅ Requirement {i} completed successfully!"
            if sim_config:
                success_msg = f"\n✅ Case {case_id} completed successfully!"
            print(success_msg)
            print(f"📁 Run directory: {result['run_dir']}")
            print(f"📁 Output directory: {result['output_dir']}")
        else:
            error_msg = f"\n❌ Requirement {i} failed!"
            if sim_config:
                error_msg = f"\n❌ Case {case_id} failed!"
            print(error_msg)
            print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"Return code: {result['return_code']}")
        
        print("\n" + "="*50 + "\n")
    
    
    successful_runs = sum(1 for r in results if r['result']['success'])
    failed_runs = len(results) - successful_runs
    
    print(f"📊 EXPERIMENT SUMMARY:")
    print(f"   Experiment: {experiment_name}")
    print(f"   Experiment directory: {current_experiment_dir}")
    print(f"   Total requirements: {len(results)}")
    print(f"   Successful runs: {successful_runs}")
    print(f"   Failed runs: {failed_runs}")
    print(f"   All results saved in: {current_experiment_dir}")
    
    # Save experiment results to JSON for analysis and future reference
    try:
        results_file = batch_dir / "experiment_results.json"
        experiment_summary = {
            'batch_name': batch_name,
            'experiment_name': experiment_name,
            'experiment_dir': str(current_experiment_dir),
            'timestamp': timestamp,
            'simulation_description': simulation_description,
            'simulation_instructions': simulation_instructions,
            'total_cases': len(results),
            'successful_cases': successful_runs,
            'failed_cases': failed_runs,
            'results': results
        }
        results_file.write_text(json.dumps(experiment_summary, indent=2), encoding='utf-8')
        print(f"📋 Saved experiment summary to: {results_file}")
    except Exception as e:
        print(f"⚠️  Failed to save experiment results: {e}")
    
    # Automatically analyze the batch after simulations complete (unless skipped)
    ############################
    if not args.skip_analysis:
        print("\n" + "="*60)
        print("🔬 Starting automatic analysis of batch...")
        print("="*60 + "\n")
        
        analysis_text = None 
        rerun_suggestions = []
        
        try:
            from agents.analysis_ag import analyze_batch
            # analyze the latest batch, which corresponds to the Foam-Agent runs just completed
            # Pass simulation configurations for enhanced analysis
            analysis_text, rerun_suggestions = analyze_batch(
                batch_dir=batch_dir,
                model=args.model,
                temperature=0.0,
                simulation_description=simulation_description,
                simulation_instructions=simulation_instructions,
                simulation_configs=simulation_configs,
                auto_rerun_threshold=args.auto_rerun_threshold,
                enable_auto_rerun=args.auto_rerun,
                max_rerun_iterations=args.max_rerun_iterations,
                auto_execute_recommendations=args.auto_execute_recommendations
            )
            
            if analysis_text:
                print("\n✅ Analysis completed successfully!")
                print(f"📄 Analysis saved to: {batch_dir / 'analysis_summary.txt'}")
            else:
                print("\n⚠️  Analysis completed with warnings")
                
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            rerun_suggestions = []
        
        # If there are rerun suggestions after auto-rerun has been handled
        if rerun_suggestions:
            print(f"\n{'='*60}")
            print(f"⚠️  Found {len(rerun_suggestions)} experiments that still need reruns")
            print(f"{'='*60}")
            
            for idx, suggestion in enumerate(rerun_suggestions, 1):
                print(f"\n{idx}. {suggestion['experiment']}/{suggestion['run']}")
                print(f"   Accuracy: {suggestion['accuracy']:.1f}/10")
                print(f"   Issue: {suggestion['explanation'][:100]}...")
            
            print(f"\n{'='*60}")
            
            # If auto-rerun is enabled, these are cases that couldn't be auto-fixed
            if args.auto_rerun:
                print(f"\n🔁 Auto-rerun was enabled but these cases still need attention:")
                print(f"   • Cases < {args.auto_rerun_threshold} were automatically rerun {args.max_rerun_iterations} times")
                print(f"   • The remaining {len(rerun_suggestions)} cases either:")
                print(f"     - Scored {args.auto_rerun_threshold}-7.0 (moderate quality, manual review recommended)")
                print(f"     - Failed to improve after {args.max_rerun_iterations} auto-rerun attempts")
                print(f"\n⚠️  NOTE: Cases scoring > 7.0 should NOT appear here - this may be a bug!")
                print("\n💡 To rerun these remaining cases manually:")
                print(f"   python src/main.py --rerun-batch {batch_name}")
            else:
                # Manual mode - ask user
                response = input("\nDo you want to rerun these experiments now? (yes/no): ").strip().lower()
                
                if response in ['yes', 'y']:
                    print("\n🔁 Starting reruns with updated requirements...")
                    rerun_failed_experiments(batch_name)
                else:
                    print(f"\n💡 To rerun later, use:")
                    print(f"   python src/main.py --rerun-batch {batch_name}")
        else:
            print("\n" + "="*60)
            print("✅ All experiments meet quality standards!")
            print("   No reruns needed.")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("⏭️  Automatic analysis skipped (use --skip-analysis flag)")
        print(f"💡 To analyze later, use:")
        print(f"   python agents/analysis_ag.py --batch {batch_name}")
        print("="*60)
        analysis_text = None 
        rerun_suggestions = []  
    
    return {
        'batch_name': batch_name,
        'experiment_name': experiment_name,
        'experiment_dir': str(current_experiment_dir),
        'results': results,
        'summary': {
            'total': len(results),
            'successful': successful_runs,
            'failed': failed_runs
        },
        'analysis_completed': analysis_text is not None,
        'rerun_suggestions_count': len(rerun_suggestions) if rerun_suggestions else 0
    } 


if __name__ == "__main__":
    main()
