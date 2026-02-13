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

# Set default model for all LLM/agent calls (Bedrock by default)
DEFAULT_BEDROCK_MODEL = os.environ.get("CFD_SCIENTIST_MODEL","arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2")

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
  
    print("=== CFD SCIENTIST - SMALL POOL FIRE CASE STUDY ===")
    
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
    
    # User requirements are generated by ideation (once) into multiple parametric experiments.
    # We keep the rest of the pipeline the same: validate requirements -> run Foam-Agent -> analyze.

    from agents.idea_ag import IdeationAgent
    from agents.hypotheis_ag import HypothesisAgent

    print("🧠 Running ideation to generate parametric experiments...")
    idea_agent = IdeationAgent(model=os.environ.get("CFD_SCIENTIST_MODEL"))
    ideas = idea_agent.generate_candidates(num_calls=1)
    if not ideas:
        raise RuntimeError("IdeationAgent returned no ideas; cannot generate user requirements")

    idea_json = ideas[0]

    # Keep these for downstream analysis prompt context (even if you skip analysis later).
    simulation_description = str(idea_json.get("description", ""))
    simulation_instructions = str(idea_json.get("post", {}).get("objective", ""))

    # Build explicit candidate experiments (deterministic cartesian product), then select up to 10
    # that best support or reject the hypothesis.

    import re as _re

    def _parse_box_2d_dims(box_str: str):
        """Parse an OpenFOAM box like '(x0 y0 z0)(x1 y1 z1)' and return (width, height).

        Returns (None, None) if parsing fails.
        """
        if not isinstance(box_str, str):
            return None, None
        m = _re.search(
            r"\(\s*([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s*\)\s*\(\s*([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s*\)",
            box_str,
        )
        if not m:
            return None, None
        x0, y0, _z0, x1, y1, _z1 = [float(m.group(i)) for i in range(1, 7)]
        return abs(x1 - x0), abs(y1 - y0)

    case_list = idea_json.get("cases", []) if isinstance(idea_json, dict) else []
    if not case_list:
        raise RuntimeError("Ideation JSON missing 'cases'; cannot build parametric study")

    candidate_sims = []
    candidate_idx = 0

    for case in case_list:
        case_name = str(case.get("name", f"case_{candidate_idx+1}"))
        dims = case.get("dimensions", [1.0, 1.0, 0.1])
        topology = str(case.get("topology", "2d")).lower()

        fuel_speeds = case.get("fuel_speed_list") or case.get("fuel speed")
        if not isinstance(fuel_speeds, list):
            fuel_speeds = [fuel_speeds]
        fuel_speeds = [fs for fs in fuel_speeds if fs is not None]

        box_sizes = case.get("box_size_list") or case.get("box size")
        if not isinstance(box_sizes, list):
            box_sizes = [box_sizes]
        box_sizes = [bs for bs in box_sizes if bs is not None]

        if not fuel_speeds:
            fuel_speeds = [0.1]  # #ASSUMPTION
        if not box_sizes:
            box_sizes = ["#ASSUMPTION(no_box_size_provided)"]

        for fs in fuel_speeds:
            for bs in box_sizes:
                candidate_idx += 1
                box_str = str(bs)
                box_w, box_h = _parse_box_2d_dims(box_str)
                candidate_sims.append(
                    {
                        "candidate_id": f"{case_name}_{candidate_idx:03d}",
                        "case_name": case_name,
                        "fuel_velocity": fs,
                        "box": box_str,
                        "box_width": box_w,
                        "box_height": box_h,
                        "dims": dims,
                        "topology": topology,
                    }
                )

    print(f"🧪 Ideation generated {len(candidate_sims)} candidate experiment(s)")

    # Choose ONE "hero" case for a beautiful single-case study.
    hypothesis_text = "Study the effect of fuel velocity and inlet box sizes in 2D small pool fire."

    def _nearly_equal(a, b, tol=1e-12):
        try:
            return abs(float(a) - float(b)) <= tol
        except Exception:
            return False

    cand_by_id = {c["candidate_id"]: c for c in candidate_sims}

    hero_candidate = None
    hero_reason = None

    try:
        case0 = (case_list[0] if isinstance(case_list, list) and case_list else {})
        fuels = case0.get("fuel_speed_list") or []
        boxes = case0.get("box_size_list") or []

        # Choose the middle settings: 3rd of 5 fuel speeds and 2nd of 3 box sizes.
        fuel_mid = fuels[len(fuels) // 2] if isinstance(fuels, list) and fuels else None
        box_mid = boxes[len(boxes) // 2] if isinstance(boxes, list) and boxes else None

        if fuel_mid is not None and box_mid is not None:
            for c in candidate_sims:
                if _nearly_equal(c.get("fuel_velocity"), fuel_mid) and str(c.get("box")) == str(box_mid):
                    hero_candidate = c
                    hero_reason = "middle fuel speed + middle inlet box from ideation lists"
                    break
    except Exception:
        hero_candidate = None

    if hero_candidate is None:
        # Fallback: choose the median candidate in the deterministic enumeration.
        if candidate_sims:
            hero_candidate = candidate_sims[len(candidate_sims) // 2]
            hero_reason = "fallback median candidate"

    if hero_candidate is None:
        raise RuntimeError("No candidates available to run")

    selected_candidates = [hero_candidate]

    print("✅ Selected 1 hero experiment for execution:", hero_candidate["candidate_id"])
    if hero_reason:
        print("Selection rationale:", hero_reason)

    # Save selection for reproducibility/debugging.
    try:
        (batch_dir / "selector_selection.json").write_text(
            json.dumps(
                {
                    "mode": "hero_single_case",
                    "hypothesis": hypothesis_text,
                    "selected_ids": [hero_candidate["candidate_id"]],
                    "reason": hero_reason,
                    "hero": {
                        "candidate_id": hero_candidate.get("candidate_id"),
                        "fuel_velocity": hero_candidate.get("fuel_velocity"),
                        "box": hero_candidate.get("box"),
                        "box_width": hero_candidate.get("box_width"),
                        "box_height": hero_candidate.get("box_height"),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"⚠️  Could not write selector_selection.json: {e}")

    # Convert selected candidates -> Foam-Agent user requirements.
    hypothesis_agent = HypothesisAgent(model=os.environ.get("CFD_SCIENTIST_MODEL"))

    simulation_configs = []
    for i, cand in enumerate(selected_candidates, 1):
        simulation = {
            "case_name": cand["case_name"],
            "simulation_id": f"sim_{i:03d}",
            "parameter_value": cand["fuel_velocity"],
            "inlet_box": cand["box"],
            "inlet_box_width": cand.get("box_width"),
            "inlet_box_height": cand.get("box_height"),
            "description": f"Small pool fire param sweep. fuel_velocity={cand['fuel_velocity']}, inlet_box={cand['box']}",
            "visualization": str(idea_json.get("post", {}).get("visualization", "")),
            "case_data": {
                "dimensions": cand["dims"],
                "topology": cand["topology"],
            },
        }

        user_req = hypothesis_agent.generate_user_requirement(idea_json, simulation)
        simulation_configs.append(
            {
                "case_id": cand["candidate_id"],
                "profile_id": "IDEATION",
                "reynolds_number": None,
                "geometry": f"{cand['topology'].upper()} {cand['dims']}",
                "user_requirement": user_req,
            }
        )

    print(f"🧾 Generated {len(simulation_configs)} Foam-Agent user requirement(s) after selection")
    
    # (Legacy hardcoded lid-driven cavity simulation_configs removed; now generated via ideation.)

    print("📝 FINISHED IDEATION CONFIGS AND Starting simulations for generated cases...")
    
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

        # Deterministic post-processing to produce requirement-compliant artifacts.
        try:
            from src.postprocess import postprocess_case
            out_dir = Path(result.get('output_dir', ''))
            req_path = Path(result.get('run_dir', '')) / 'user_requirement.txt'
            req_text = ''
            try:
                if req_path.exists():
                    req_text = req_path.read_text(encoding='utf-8')
            except Exception:
                req_text = user_requirement

            pp = postprocess_case(out_dir, user_requirement=req_text)
            result['postprocess'] = pp
            if pp.get('success'):
                print(f"🧾 Postprocess artifacts written under: {out_dir}")
            else:
                print(f"⚠️  Postprocess skipped/failed: {pp.get('error')}")
        except Exception as e:
            print(f"⚠️  Postprocess exception: {e}")
        
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
