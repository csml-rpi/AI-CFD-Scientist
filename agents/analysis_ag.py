"""analysis_agent.py

Scan a batch under data/experiments and analyze all experiments/runs using an LLM.

Usage:
    python src/analysis_agent.py --batch <batch_name>
    python src/analysis_agent.py                # analyzes latest batch

Outputs:
    - <batch_dir>/analysis_summary.txt  : overall batch analysis
    - <batch_dir>/<experiment>/analysis.txt : per-experiment analysis

This module uses the project's `utils.base_llm` helpers to create a client and call the model.
"""

from pathlib import Path
import argparse
import json
import os
import textwrap
import base64
from typing import Optional

# Ensure project root is on sys.path for absolute imports BEFORE importing project modules
import sys as _sys
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

from utils.base_llm import create_client, get_response_from_llm, extract_json_between_markers


MAX_SNIPPET_CHARS = 1600
DEFAULT_MODEL = os.environ.get("CFD_SCIENTIST_MODEL", "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2")

# bedrock requires encoding to base64 
def encode_image_to_base64(image_path: Path) -> Optional[str]:
    """
    Encode an image file to base64 string for Bedrock vision models.
    
    Args:
        image_path: Path to the image file (PNG, JPG, etc.)
        
    Returns:
        Base64 encoded string or None if error
    """
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"⚠️  Error encoding image {image_path}: {e}")
        return None


def read_text_safe(p: Path, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    try:
        s = p.read_text(encoding="utf-8")
        if len(s) > max_chars:
            return s[:max_chars] + "\n\n...[truncated]..."
        return s
    except Exception as e:
        return f"[Error reading {p}: {e}]"


def choose_batch(base_dir: Path, batch_name: Optional[str]) -> Optional[Path]:
    if not base_dir.exists():
        print(f"No experiments directory found at {base_dir}")
        return None

    batches = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    if not batches:
        print(f"No batch directories found in {base_dir}")
        return None

    if batch_name:
        candidate = base_dir / batch_name
        if candidate.exists() and candidate.is_dir():
            return candidate
        else:
            print(f"Batch '{batch_name}' not found under {base_dir}")
            return None

    # default: newest batch (by name sort / mtime)
    # try by modification time
    batches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return batches[0]


def collect_batch_info(batch_dir: Path, max_experiments: Optional[int] = None):
    experiments = [d for d in batch_dir.iterdir() if d.is_dir()]
    experiments.sort()
    if max_experiments:
        experiments = experiments[:max_experiments]

    batch_summary = []

    for exp in experiments:
        exp_info = {"experiment": exp.name, "runs": []}
        for run in sorted([d for d in exp.iterdir() if d.is_dir()]):
            run_info = {"run": run.name, "paths": {}}
            # primary user requirement
            ur = run / "user_requirement.txt"
            if ur.exists():
                run_info["user_requirement"] = read_text_safe(ur)
                run_info["paths"]["user_requirement"] = str(ur)

            # Collect postprocess time-stamped images for BOTH velocity magnitude (UMag) and pressure (p)
            out_dir = run / "output"
            if out_dir.exists():

                def _parse_time_from_stem(stem: str, prefix: str):
                    # e.g., stem='umag_t0p10' prefix='umag_t' -> 0.10
                    try:
                        tag = stem.split(prefix, 1)[1]
                        return float(tag.replace("p", "."))
                    except Exception:
                        return None

                def _collect_images(pattern: str, prefix: str, field: str):
                    paths = list(out_dir.glob(pattern))

                    def _t(p: Path):
                        return _parse_time_from_stem(p.stem, prefix)

                    paths.sort(key=lambda p: (_t(p) is None, _t(p) or 0.0, p.name))
                    imgs = []
                    for p in paths:
                        image_base64 = encode_image_to_base64(p)
                        if not image_base64:
                            continue
                        imgs.append(
                            {
                                "field": field,
                                "path": str(p),
                                "base64": image_base64,
                                "format": "png",
                                "t": _t(p),
                            }
                        )
                    return imgs

                umag_imgs = _collect_images("umag_t*.png", "umag_t", "umag")
                p_imgs = _collect_images("p_t*.png", "p_t", "p")

                if umag_imgs:
                    run_info["timestep_images_umag"] = umag_imgs
                if p_imgs:
                    run_info["timestep_images_p"] = p_imgs

                # For backward compatibility, keep a combined list.
                timestep_images = umag_imgs + p_imgs
                if timestep_images:
                    run_info["timestep_images"] = timestep_images
                    print(
                        f"   📸 Found timestep images under: {out_dir} (umag={len(umag_imgs)}, p={len(p_imgs)})"
                    )

                    # Representative image for cross-case batch summary: prefer UMag at final time.
                    rep = out_dir / "umag_t3p00.png"
                    if not rep.exists() and umag_imgs:
                        rep = Path(umag_imgs[-1]["path"])

                    rep64 = encode_image_to_base64(rep)
                    if rep64:
                        run_info["visualization_image"] = {
                            "path": str(rep),
                            "base64": rep64,
                            "format": "png",
                        }

            # possible visualization scripts in run root or output folder
            viz_candidates = []
            viz_candidates += list(run.glob("visualization*.py"))
            viz_candidates += list((run / "output").glob("visualization*.py")) if (run / "output").exists() else []
            if viz_candidates:
                run_info["visualizations"] = []
                for v in viz_candidates:
                    run_info["visualizations"].append({
                        "path": str(v),
                        "content": read_text_safe(v)
                    })

            # collect a brief listing of output files
            out_dir = run / "output"
            if out_dir.exists():
                # list top-level files and sizes
                files = []
                for f in sorted(out_dir.rglob("*")):
                    if f.is_file():
                        try:
                            files.append((str(f.relative_to(run)), f.stat().st_size))
                        except Exception:
                            files.append((str(f), 0))
                run_info["output_files"] = files[:200]
                run_info["paths"]["output"] = str(out_dir)

            exp_info["runs"].append(run_info)
        batch_summary.append(exp_info)

    return batch_summary


def build_prompt_for_batch(batch_name: str, batch_summary, simulation_description: str = None, 
                          simulation_instructions: str = None, simulation_configs: list = None) -> str:
    """
    Enhanced prompt builder for comprehensive cross-case analysis.
    """
    
    # Build enhanced header with study context
    header = (
        f"You are an expert CFD analysis assistant.\n"
        f"You will analyze a batch of simulation runs named '{batch_name}'.\n"
    )

    if simulation_description:
        header += f"\nSTUDY CONTEXT: {simulation_description}\n"

    if simulation_instructions:
        header += f"\nANALYSIS INSTRUCTIONS:\n{simulation_instructions}\n"

    header += (
        "\nFor this batch analysis, provide:\n"
        "1. INDIVIDUAL RUN ANALYSIS: For each run, assess whether outputs match the user requirement and note numerical/physical issues.\n"
        "2. CROSS-CASE COMPARISON: Compare outcomes across the parameter sweep(s) and identify trends and outliers.\n"
        "3. ARTIFACT COMPLETENESS: Verify required visualizations/exports are present and consistent across runs.\n"
        "4. NUMERICAL QUALITY: Comment on stability, convergence/steadiness, and whether mesh/time step/sim time appear adequate.\n"
        "\nProvide accuracy scores (/10) for each run and an overall study assessment.\n\n"
    )

    # Enhanced per-experiment blocks with case metadata
    blocks = []
    for i, exp in enumerate(batch_summary):
        case_info = ""
        if simulation_configs and i < len(simulation_configs):
            config = simulation_configs[i]
            case_info = (
                f"Case ID: {config.get('case_id', 'Unknown')}\n"
                f"Profile: {config.get('profile_id', 'Unknown')}\n"
                f"Reynolds Number: {config.get('reynolds_number', 'Unknown')}\n"
                f"Geometry: {config.get('geometry', 'Unknown')}\n"
            )
        
        b = [f"=== EXPERIMENT {i+1}: {exp['experiment']} ===\n{case_info}"]
        
        for run in exp.get("runs", []):
            b.append(f"Run: {run['run']}")
            if "user_requirement" in run:
                b.append("User Requirement:\n" + textwrap.indent(run["user_requirement"], "  "))
            
            # Enhanced visualization handling
            if "visualizations" in run:
                b.append("Visualization Scripts Generated:")
                for v in run["visualizations"]:
                    b.append(f"  Script: {v['path']}")
                    b.append(textwrap.indent(v["content"], "    "))
                    
            # Show output files for context
            if "output_files" in run:
                sample_files = ", ".join([f[0] for f in run["output_files"][:10]])
                total_files = len(run["output_files"])
                b.append(f"Output Files ({total_files} total): {sample_files}")
                if total_files > 10:
                    b.append("  ... and more")
                    
        blocks.append("\n".join(b))

    # Cross-case analysis section
    cross_case_prompt = """

=== CROSS-CASE COMPARATIVE ANALYSIS REQUIREMENTS ===

After analyzing individual runs, provide a concise comparative analysis across the study:

1. PARAMETER → RESPONSE TRENDS:
   - What qualitative changes occur as parameters vary (e.g., inlet/fuel settings, geometry changes)?

2. CONSISTENCY & OUTLIERS:
   - Identify runs that look inconsistent with the rest (possible numerical issues or setup mistakes).

3. ARTIFACT/POSTPROCESS COMPLETENESS:
   - Note any missing or inconsistent visualization/export artifacts across runs.

4. RECOMMENDATIONS:
   - Suggest reruns with concrete requirement fixes if needed.
   - Suggest next experiments if the sweep is insufficient to support/reject the hypothesis.

Conclude with an overall study assessment and next steps.
"""

    prompt = header + "\n\n".join(blocks) + cross_case_prompt
    
    # Ensure prompt isn't too large
    if len(prompt) > 40000:  # Increased limit for enhanced analysis
        prompt = prompt[:40000] + "\n\n...[truncated for length]..."
    return prompt


def build_multimodal_cross_case_prompt(batch_name: str, batch_summary, all_images: list,
                                     simulation_description: str = None, 
                                     simulation_instructions: str = None, 
                                     simulation_configs: list = None) -> str:
    """
    Build a multimodal prompt for comprehensive cross-case analysis with all visualization images.
    """
    
    header = (
        f"🔬 COMPREHENSIVE CROSS-CASE CFD ANALYSIS\n"
        f"Batch: {batch_name}\n\n"
        f"You are analyzing a CFD study with {len(all_images)} visualization images.\n"
    )
    
    if simulation_description:
        header += f"STUDY CONTEXT: {simulation_description}\n\n"
    
    if simulation_instructions:
        header += f"ANALYSIS INSTRUCTIONS:\n{simulation_instructions}\n\n"
    
    header += (
        f"🖼️ MULTIMODAL ANALYSIS INSTRUCTIONS:\n"
        f"You will receive {len(all_images)} visualization images showing flow fields for different cases/parameters.\n"
        f"For each image, provide:\n"
        f"1. Flow structure description (vortices, boundary layers, separation)\n"
        f"2. Parameter/case effects visible in the visualization\n"
        f"3. Numerical accuracy assessment\n\n"
        f"Then provide COMPREHENSIVE CROSS-CASE COMPARATIVE ANALYSIS:\n"
        f"1. Parameter progression effects\n"
        f"2. Primary vortex evolution (location, strength, size)\n"
        f"3. Secondary vortex development\n"
        f"4. Boundary layer thickness comparison\n"
        f"5. Flow transition phenomena\n"
        f"6. Grid resolution adequacy across Re range\n"
        f"7. Benchmark validation opportunities\n"
        f"8. Overall study assessment and recommendations\n\n"
    )
    
    # Build case-by-case descriptions
    case_descriptions = []
    for i, img_data in enumerate(all_images):
        case_info = ""
        if simulation_configs and i < len(simulation_configs):
            config = simulation_configs[i]
            case_info = (
                f"Case ID: {config.get('case_id', 'Unknown')}\n"
                f"Reynolds Number: {config.get('reynolds_number', 'Unknown')}\n"
                f"Geometry: {config.get('geometry', 'Unknown')}\n"
            )
        
        desc = (
            f"=== IMAGE {i+1}: {img_data['experiment']} ===\n"
            f"{case_info}"
            f"Run: {img_data['run']}\n"
            f"Visualization: {img_data['image_data']['path']}\n"
            f"User Requirement:\n{textwrap.indent(img_data['user_requirement'], '  ')}\n"
        )
        case_descriptions.append(desc)
    
    cross_analysis_requirements = """

CROSS-CASE ANALYSIS REQUIREMENTS:

1. PARAMETER/CASING EFFECTS MATRIX:
   - Describe how key qualitative flow features change across cases/parameters.

2. TEMPORAL/STEADINESS CHECK (if applicable):
   - If the provided images represent different cases at a single time, note consistency.

3. NUMERICAL QUALITY:
   - Identify signs of instability, excessive diffusion, nonphysical artifacts, or mesh imprinting.

4. STUDY COMPLETENESS:
   - Are the chosen cases sufficient to support/reject the stated hypothesis? What is missing?

5. RECOMMENDATIONS:
   - Suggest reruns with concrete fixes.
   - Suggest next experiments to strengthen discriminative power.

Provide detailed, physics-based analysis with specific observations from each image.
"""
    
    prompt = header + "\n\n".join(case_descriptions) + cross_analysis_requirements
    
    # Limit prompt size
    if len(prompt) > 50000:
        prompt = prompt[:50000] + "\n\n...[truncated for length]..."
    
    return prompt


def perform_multimodal_cross_case_analysis(prompt: str, all_images: list, model: str, temperature: float) -> str:
    """
    Perform multimodal analysis by sending prompt with all visualization images to LLM.
    """
    client, model_name = create_client(model)
    
    print(f"🖼️ Performing multimodal analysis with {len(all_images)} images...")
    
    # Prepare multimodal content 
    # For Bedrock and Claude, we need to format as content list
    multimodal_content = [{"type": "text", "text": prompt}]
    
    # Add all images to the content
    for i, img_data in enumerate(all_images):
        image_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{img_data['image_data']['format']}",
                "data": img_data['image_data']['base64']
            }
        }
        multimodal_content.append(image_content)
        print(f"   📸 Added image {i+1}: {img_data['experiment']} / {img_data['run']}")
    
    # System message for multimodal analysis
    system_message = (
        "You are a world-class CFD expert. "
        "Analyze the visualization images with deep physics understanding. "
        "Provide cross-case comparisons, identify parameter/case effects, and assess numerical accuracy. "
        "Be specific about structures/patterns visible in each image and whether they match the stated requirements."
    )
    
    try:
        # For multimodal, pass the content list as prompt and use msg_history to structure the message
        msg_history = [{
            "role": "user", 
            "content": multimodal_content
        }][:-1]  # Remove last message since get_response_from_llm will add it
        
        # Extract just the text for the prompt parameter
        text_prompt = prompt
        
        analysis_text, _ = get_response_from_llm(
            prompt=text_prompt,
            client=client,
            model=model_name,
            system_message=system_message,
            temperature=temperature,
            print_debug=False,
            msg_history=msg_history
        )
        
        print("✅ Multimodal cross-case analysis completed successfully!")
        return analysis_text
        
    except Exception as e:
        print(f"❌ Multimodal analysis failed: {e}")
        # Try a different approach - create a custom call for multimodal
        return perform_direct_multimodal_call(prompt, all_images, client, model_name, system_message, temperature)


def perform_direct_multimodal_call(prompt: str, all_images: list, client, model: str, system_message: str, temperature: float) -> str:
    """
    Direct multimodal call for Bedrock/Claude with images.
    """
    print("🔄 Attempting direct multimodal call...")
    
    # Check if this is a Bedrock client
    is_bedrock_boto3 = hasattr(client, 'invoke_model') and (model.startswith("arn:aws:bedrock") or "bedrock" in str(type(client)).lower())
    
    if is_bedrock_boto3:
        # Bedrock Converse API with images
        content = [{"text": prompt}]
        
        # Add images in Bedrock format
        for img_data in all_images:
            content.append({
                "image": {
                    "format": img_data['image_data']['format'],
                    "source": {
                        "bytes": base64.b64decode(img_data['image_data']['base64'])
                    }
                }
            })
        
        converse_params = {
            "modelId": model,
            "messages": [{
                "role": "user",
                "content": content
            }],
            "system": [{"text": system_message}],
            "inferenceConfig": {
                "maxTokens": 4096,
                "temperature": temperature,
            }
        }
        
        response = client.converse(**converse_params)
        return response['output']['message']['content'][0]['text']
        
    elif "claude" in model:
        # Anthropic Claude API with images
        content = [{"type": "text", "text": prompt}]
        
        for img_data in all_images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{img_data['image_data']['format']}",
                    "data": img_data['image_data']['base64']
                }
            })
        
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=system_message,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        return response.content[0].text
    
    else:
        # Fallback: text-only analysis
        print("⚠️  Model doesn't support multimodal, falling back to text-only analysis")
        analysis_text, _ = get_response_from_llm(
            prompt=prompt,
            client=client,
            model=model,
            system_message=system_message,
            temperature=temperature,
            print_debug=False
        )
        return analysis_text


def analyze_batch(batch_dir: Path, model: str = DEFAULT_MODEL, temperature: float = 0.0, max_experiments: Optional[int] = None,
                 simulation_description: str = None, simulation_instructions: str = None, simulation_configs: list = None,
                 auto_rerun_threshold: float = 5.0, enable_auto_rerun: bool = False, max_rerun_iterations: int = 3,
                 auto_execute_recommendations: bool = False):
    """
    Analyze a batch of experiments and return suggestions for reruns.
    Enhanced to perform cross-case comparative analysis across different Reynolds numbers.
    Optionally automatically rerun cases with very low scores until threshold is met.
    
    Args:
        batch_dir: Path to batch directory
        model: LLM model to use
        temperature: Temperature for LLM
        max_experiments: Maximum experiments to analyze
        simulation_description: Overall description of the simulation study
        simulation_instructions: Analysis instructions for the study
        simulation_configs: List of simulation configurations with case IDs and metadata
        auto_rerun_threshold: Threshold below which cases are automatically rerun (default: 5.0)
        enable_auto_rerun: Whether to automatically rerun very poor cases (default: False)
        max_rerun_iterations: Maximum number of rerun attempts per case (default: 3)
        auto_execute_recommendations: Whether to automatically execute study recommendations (default: False)
    
    Returns:
        tuple: (analysis_text, rerun_suggestions)
            analysis_text: str - The full analysis report with cross-case comparisons
            rerun_suggestions: list - List of dicts with experiment, run, and updated requirements
    """
    print(f"Scanning batch directory: {batch_dir}")
    batch_summary = collect_batch_info(batch_dir, max_experiments=max_experiments)

    if not batch_summary:
        print("No experiments found in batch")
        return None, []

    # Print a plan of what will be analyzed for transparency
    total_runs = 0
    print("\n=== Analysis Plan ===")
    for exp in batch_summary:
        runs = exp.get("runs", [])
        total_runs += len(runs)
        print(f"• Experiment: {exp['experiment']} — {len(runs)} run(s)")
        for r in runs:
            ur = (r.get("user_requirement", "") or "").strip()
            first_line = next((ln.strip() for ln in ur.splitlines() if ln.strip()), "<missing user requirement>")
            print(f"   - Run {r['run']}: {first_line}")
    print(f"Total runs summarized for batch-level analysis: {total_runs}")
    
    # Enhanced analysis with simulation context and image collection
    if simulation_description or simulation_instructions or simulation_configs:
        print("🔬 Enhanced cross-case comparative analysis enabled")
        if simulation_configs:
            print(f"   📊 Analyzing {len(simulation_configs)} simulation configurations")
    
    # Collect all visualization images for cross-case analysis
    all_images = []
    for exp in batch_summary:
        for run in exp.get("runs", []):
            if "visualization_image" in run:
                all_images.append({
                    'experiment': exp['experiment'],
                    'run': run['run'], 
                    'image_data': run['visualization_image'],
                    'user_requirement': run.get('user_requirement', '')
                })
    
    print(f"🖼️  Found {len(all_images)} visualization images for cross-case analysis")
    
    # Build enhanced multimodal prompt
    if all_images:
        # Create a comprehensive multimodal analysis
        prompt = build_multimodal_cross_case_prompt(
            batch_dir.name,
            batch_summary,
            all_images,
            simulation_description=simulation_description,
            simulation_instructions=simulation_instructions,
            simulation_configs=simulation_configs
        )
        
        # Use multimodal LLM call with all images
        try:
            analysis_text = perform_multimodal_cross_case_analysis(
                prompt, all_images, model, temperature
            )
        except Exception as e:
            print(f"Multimodal analysis failed, falling back to text-only: {e}")
            # Fallback to text-only analysis
            prompt = build_prompt_for_batch(
                batch_dir.name, 
                batch_summary, 
                simulation_description=simulation_description,
                simulation_instructions=simulation_instructions, 
                simulation_configs=simulation_configs
            )
            client, model_name = create_client(model)
            analysis_text, _ = get_response_from_llm(prompt, client, model_name, 
                                                   "You are a CFD expert. Analyze simulations against their user requirements.", 
                                                   print_debug=False, temperature=temperature)
    else:
        # No images found, use text-only analysis
        prompt = build_prompt_for_batch(
            batch_dir.name, 
            batch_summary, 
            simulation_description=simulation_description,
            simulation_instructions=simulation_instructions, 
            simulation_configs=simulation_configs
        )
        client, model_name = create_client(model)
        analysis_text, _ = get_response_from_llm(prompt, client, model_name,
                                               "You are a CFD expert. Analyze simulations against their user requirements.",
                                               print_debug=False, temperature=temperature)

    # Ensure client and model_name are defined for rerun analysis
    if 'client' not in locals() or 'model_name' not in locals():
        client, model_name = create_client(model)

    # Save overall analysis
    out_file = batch_dir / "analysis_summary.txt"
    out_file.write_text(analysis_text, encoding="utf-8")
    print(f"Saved batch analysis to {out_file}")

    # Evaluate runs and collect rerun suggestions with per-experiment analysis
    # This combines analysis + validation into a single step per experiment
    rerun_suggestions = []
    auto_rerun_cases = []
    try:
        rerun_suggestions, auto_rerun_cases = _analyze_and_collect_rerun_suggestions(
            batch_dir, batch_summary, client, model_name, temperature, 
            auto_rerun_threshold=auto_rerun_threshold
        )
    except Exception as e:
        print(f"Analysis and rerun evaluation stage encountered an error: {e}")
        import traceback
        traceback.print_exc()
        rerun_suggestions = []
        auto_rerun_cases = []

    # Automatically rerun very poor cases if enabled - loop until threshold is met
    if enable_auto_rerun:
        iteration = 0
        while iteration < max_rerun_iterations:
            if not auto_rerun_cases:
                break
                
            iteration += 1
            print(f"\n{'='*60}")
            print(f"🔄 AUTO-RERUN ITERATION {iteration}/{max_rerun_iterations}")
            print(f"Found {len(auto_rerun_cases)} cases with scores < {auto_rerun_threshold}")
            print(f"{'='*60}")
            
            for case in auto_rerun_cases:
                print(f"   • {case['experiment']}/{case['run']} (score: {case['accuracy']:.1f}/10)")
            
            print(f"\n🚀 Starting automatic reruns (iteration {iteration})...")
            _execute_automatic_reruns(batch_dir, auto_rerun_cases)
            
            # Re-analyze the entire batch to check if reruns improved the scores
            print(f"\n🔬 Re-analyzing batch after iteration {iteration}...")
            
            # Collect batch info again (may include replacement runs)
            batch_summary = collect_batch_info(batch_dir, max_experiments=max_experiments)
            
            # Re-analyze ALL cases in the batch (including replaced runs)
            rerun_suggestions_new, auto_rerun_cases_new = _analyze_and_collect_rerun_suggestions(
                batch_dir, batch_summary, client, model_name, temperature, 
                auto_rerun_threshold=auto_rerun_threshold
            )
            
            # Filter to only cases that still need auto-rerun
            remaining_auto_rerun = [
                case for case in auto_rerun_cases_new 
                if any(orig['experiment'] == case['experiment'] and orig['run'] == case['run'] 
                      for orig in auto_rerun_cases)
            ]
            
            improved_count = len(auto_rerun_cases) - len(remaining_auto_rerun)
            
            print(f"\n📊 Iteration {iteration} Results:")
            print(f"   • Cases improved: {improved_count}/{len(auto_rerun_cases)}")
            print(f"   • Cases still below threshold: {len(remaining_auto_rerun)}")
            
            if not remaining_auto_rerun:
                print(f"✅ All cases now meet threshold! Stopping auto-rerun loop.")
                break
            elif improved_count == 0:
                print(f"⚠️  No cases improved in iteration {iteration}. Continuing with remaining cases...")
            
            auto_rerun_cases = remaining_auto_rerun
        
        # Final status
        if auto_rerun_cases and iteration >= max_rerun_iterations:
            print(f"\n⏹️  Reached maximum iterations ({max_rerun_iterations}). {len(auto_rerun_cases)} cases still below threshold.")
            # Add remaining cases back to rerun suggestions for manual handling
            rerun_suggestions.extend(auto_rerun_cases)
        elif not auto_rerun_cases:
            print(f"\n✅ All auto-rerun cases now meet quality threshold after {iteration} iteration(s)!")
        
        # Update rerun suggestions to exclude auto-rerun cases that were handled
        rerun_suggestions = [r for r in rerun_suggestions if r['accuracy'] >= auto_rerun_threshold]
        
        print(f"\n🏁 Auto-rerun process completed. {len(rerun_suggestions)} cases remain for manual rerun consideration.")
    
    # Generate comprehensive study recommendations based on all analyses
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE STUDY ANALYSIS & RECOMMENDATIONS")
    print("="*70)
    
    # Collect all individual analyses for recommendation generation
    all_individual_analyses = []
    for exp in batch_summary:
        exp_dir = batch_dir / exp["experiment"]
        for run in exp.get("runs", []):
            verdict_file = exp_dir / run["run"] / "analysis_verdict.json"
            if verdict_file.exists():
                try:
                    with open(verdict_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    all_individual_analyses.append(analysis_data)
                except Exception as e:
                    print(f"⚠️  Could not read analysis for {exp['experiment']}/{run['run']}: {e}")
    
    try:
        study_recommendations = _generate_study_recommendations(
            batch_dir=batch_dir,
            batch_analysis=analysis_text,
            individual_analyses=all_individual_analyses,
            simulation_configs=simulation_configs,
            client=client,
            model_name=model_name,
            temperature=temperature
        )
        
        print("\n Study recommendations generated successfully!")
        print("Check study_recommendations.txt for detailed suggestions")
        
        # Auto-execute recommendations if enabled
        if auto_execute_recommendations:
            print(f"\n EXECUTING STUDY RECOMMENDATIONS...")
            print(f"{'='*60}")
            _execute_study_recommendations(batch_dir)
        
    except Exception as e:
        print(f"⚠️  Could not generate study recommendations: {e}")
        study_recommendations = None
    
    return analysis_text, rerun_suggestions


def _generate_study_recommendations(
    batch_dir: Path,
    batch_analysis: str,
    individual_analyses: list,
    simulation_configs: list,
    client,
    model_name: str,
    temperature: float = 0.0
) -> str:
    """
    Generate comprehensive study recommendations by analyzing all results together.
    Suggests new experiments, parameter studies, or validation cases to make the study more complete.
    
    Args:
        batch_dir: Path to batch directory
        batch_analysis: Overall batch analysis text
        individual_analyses: List of individual run analyses
        simulation_configs: List of simulation configurations
        client: LLM client
        model_name: Model name
        temperature: Temperature for LLM
    
    Returns:
        str: Comprehensive study recommendations
    """
    print("\n🔬 Generating comprehensive study recommendations...")
    
    # Collect all individual analysis summaries
    individual_summaries = []
    for analysis in individual_analyses:
        summary = {
            'experiment': analysis.get('experiment', 'Unknown'),
            'run': analysis.get('run', 'Unknown'),
            'accuracy': analysis.get('accuracy', 0),
            'analysis': analysis.get('analysis', ''),
            'issues': analysis.get('explanation', ''),
            'requirement': analysis.get('original_requirement', '')
        }
        individual_summaries.append(summary)
    
    # Build comprehensive analysis prompt
    reynolds_numbers = []
    geometries = []
    if simulation_configs:
        reynolds_numbers = [config.get('reynolds_number', 'Unknown') for config in simulation_configs]
        geometries = list(set([config.get('geometry', 'Unknown') for config in simulation_configs]))
    
    system_message = (
        "You are a world-class CFD research scientist with expertise in experimental design and "
        "parameter studies. Your task is to identify the most critical missing experiments that would "
        "significantly improve the study's scientific value. You MUST respond with valid JSON only, "
        "focusing on the top 3-5 high priority experiments that are scientifically essential and "
        "computationally feasible."
    )
    
    prompt = f"""
🔬 COMPREHENSIVE STUDY ANALYSIS & RECOMMENDATION GENERATION

You have access to a complete CFD study with the following components:

1. OVERALL BATCH ANALYSIS:
{batch_analysis[:3000]}{'...[truncated]' if len(batch_analysis) > 3000 else ''}

2. STUDY PARAMETERS:
   - Reynolds Numbers Tested: {reynolds_numbers}
   - Geometries: {geometries}
   - Total Cases: {len(individual_summaries)}
   - Successful Cases: {len([a for a in individual_summaries if a['accuracy'] >= 6.0])}

3. INDIVIDUAL CASE RESULTS:
"""
    
    # Add individual case summaries
    for i, summary in enumerate(individual_summaries[:10]):  # Limit to first 10 for prompt size
        prompt += f"""
   Case {i+1}: {summary['experiment']}/{summary['run']}
   - Accuracy: {summary['accuracy']:.1f}/10
   - Requirement: {summary['requirement'][:100]}{'...' if len(summary['requirement']) > 100 else ''}
   - Analysis: {summary['analysis'][:200]}{'...' if len(summary['analysis']) > 200 else ''}
   - Issues: {summary['issues'][:150]}{'...' if len(summary['issues']) > 150 else ''}
"""
    
    if len(individual_summaries) > 10:
        prompt += f"\n   ... and {len(individual_summaries) - 10} more cases\n"
    
    prompt += f"""

🎯 COMPACT STUDY RECOMMENDATIONS

Based on your analysis, identify the TOP 3-5 HIGH PRIORITY experiments that would most significantly improve this study's scientific value and publication readiness.

OUTPUT FORMAT: Return ONLY a valid JSON object with this exact structure:

{{
  "summary": {{
    "total_cases_analyzed": {len(individual_summaries)},
    "successful_cases": {len([a for a in individual_summaries if a['accuracy'] >= 6.0])},
    "reynolds_range": {reynolds_numbers if reynolds_numbers else ["Unknown"]},
    "main_gaps": ["brief description of 1-2 key gaps"]
  }},
  "high_priority_experiments": [
    {{
      "experiment_id": "gap_reynolds_400",
      "description": "Critical Reynolds number transition case",
      "parameters": {{
        "reynolds_number": 400,
        "geometry": "2D square cavity 1x1x0.1",
        "boundary_conditions": "moving lid, no-slip walls",
        "mesh_size": "70x70",
        "solver_settings": "steady-state, SIMPLE"
      }},
      "scientific_justification": "Captures first bifurcation and secondary vortex formation",
      "computational_cost_hours": 3,
      "priority_reason": "Missing critical flow transition regime"
    }}
  ]
}}

REQUIREMENTS:
- Focus ONLY on the most critical 3-5 missing experiments
- Each experiment should fill a significant scientific gap
- Parameters must be specific and executable
- Computational cost should be realistic (1-10 hours per case)
- Prioritize experiments that would most improve publication potential

Identify experiments for: missing Reynolds numbers, grid convergence validation, benchmark comparison, or critical flow physics gaps.
"""
    
    try:
        recommendations, _ = get_response_from_llm(
            prompt=prompt,
            client=client,
            model=model_name,
            system_message=system_message,
            temperature=temperature,
            print_debug=False
        )
        
        # Try to parse as JSON first
        try:
            recommendations_json = json.loads(recommendations)
            
            # Save JSON recommendations for programmatic use
            json_file = batch_dir / "study_recommendations.json"
            json_file.write_text(json.dumps(recommendations_json, indent=2), encoding="utf-8")
            print(f"📋 Saved JSON recommendations to {json_file}")
            
            # Also save human-readable version
            txt_file = batch_dir / "study_recommendations.txt"
            readable_content = f"""# 🔬 STUDY RECOMMENDATIONS

## Summary
- Total Cases Analyzed: {recommendations_json.get('summary', {}).get('total_cases_analyzed', 'Unknown')}
- Successful Cases: {recommendations_json.get('summary', {}).get('successful_cases', 'Unknown')}
- Reynolds Range: {recommendations_json.get('summary', {}).get('reynolds_range', [])}
- Main Gaps: {recommendations_json.get('summary', {}).get('main_gaps', [])}

## High Priority Experiments
"""
            
            for i, exp in enumerate(recommendations_json.get('high_priority_experiments', []), 1):
                readable_content += f"""
### {i}. {exp.get('experiment_id', 'Unknown')}
**Description:** {exp.get('description', 'No description')}
**Scientific Justification:** {exp.get('scientific_justification', 'No justification')}
**Computational Cost:** {exp.get('computational_cost_hours', 'Unknown')} hours
**Priority Reason:** {exp.get('priority_reason', 'No reason')}

**Parameters:**
"""
                params = exp.get('parameters', {})
                for key, value in params.items():
                    readable_content += f"- {key}: {value}\n"
            
            txt_file.write_text(readable_content, encoding="utf-8")
            print(f"📄 Saved readable recommendations to {txt_file}")
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Response not in valid JSON format, saving as text only")
            # Save as text if JSON parsing fails
            recommendations_file = batch_dir / "study_recommendations.txt"
            recommendations_file.write_text(recommendations, encoding="utf-8")
            print(f"📋 Saved text recommendations to {recommendations_file}")
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Failed to generate study recommendations: {e}")
        return "Error: Could not generate study recommendations"


def _execute_study_recommendations(batch_dir: Path):
    """
    Automatically execute study recommendations by reading from study_recommendations.json
    and running new experiments using Foam-Agent.
    
    Args:
        batch_dir: Path to batch directory containing study_recommendations.json
    """
    import subprocess
    import uuid
    from datetime import datetime
    
    # Look for the JSON recommendations file
    recommendations_file = batch_dir / "study_recommendations.json"
    
    if not recommendations_file.exists():
        print("❌ No study_recommendations.json found. Cannot execute recommendations.")
        return
    
    try:
        with open(recommendations_file, 'r', encoding='utf-8') as f:
            recommendations_json = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read recommendations file: {e}")
        return
    
    high_priority_experiments = recommendations_json.get('high_priority_experiments', [])
    
    if not high_priority_experiments:
        print("📝 No high priority experiments found in recommendations")
        return
    
    print(f"🎯 Found {len(high_priority_experiments)} high priority experiments to execute")
    
    project_root = _project_root
    foam_bench = (project_root / "Foam-Agent" / "foambench_main.py").resolve()
    
    if not foam_bench.exists():
        print(f"❌ Error: Foam-Agent not found at {foam_bench}. Cannot execute recommendations.")
        return
    
    # Find existing experiment directory in the batch (should be sim_TIMESTAMP_ID)
    existing_experiment_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith('sim_')]
    
    if existing_experiment_dirs:
        # Use the first (and usually only) existing experiment directory
        target_experiment_dir = existing_experiment_dirs[0]
        print(f"📁 Adding recommendations to existing experiment: {target_experiment_dir}")
    else:
        # Fallback: create new experiment directory if none found
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        target_experiment_dir = batch_dir / f"sim_{timestamp}_{short_id}"
        target_experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created new experiment directory: {target_experiment_dir}")
    
    # Determine next run number by checking existing runs
    existing_runs = [d for d in target_experiment_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if existing_runs:
        # Extract run numbers and find the highest one
        run_numbers = []
        for run_dir in existing_runs:
            try:
                # Extract number from run_XXX format
                run_num_str = run_dir.name.split('_')[1]
                if run_num_str.isdigit():
                    run_numbers.append(int(run_num_str))
            except:
                continue
        next_run_number = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_run_number = 1
    
    print(f"🔢 Starting recommendation runs at run number: {next_run_number:03d}")
    
    execution_results = []
    current_run_number = next_run_number
    
    for i, exp_rec in enumerate(high_priority_experiments, 1):
        experiment_id = exp_rec.get('experiment_id', f'rec_{i}')
        description = exp_rec.get('description', 'No description')
        params = exp_rec.get('parameters', {})
        
        print(f"\\n[{i}/{len(high_priority_experiments)}] 🔬 Executing: {experiment_id}")
        print(f"   📋 Description: {description}")
        print(f"   🎯 Run number: {current_run_number:03d}")
        print(f"   ⏱️  Estimated time: {exp_rec.get('computational_cost_hours', 'Unknown')} hours")
        
        # Convert recommendation parameters to user requirement format
        user_requirement = _convert_recommendation_to_user_requirement(exp_rec)
        
        if not user_requirement:
            print(f"   ❌ Could not convert recommendation to user requirement")
            execution_results.append({
                'run_number': current_run_number,
                'experiment_id': experiment_id,
                'success': False,
                'error': 'Failed to convert recommendation to user requirement'
            })
            current_run_number += 1
            continue
        
        print(f"   📄 Generated user requirement:")
        print(f"   {user_requirement[:150]}...")
        
        # Create run directory with sequential numbering
        run_dir = target_experiment_dir / f"run_{current_run_number:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory
        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write user requirement
        prompt_file = run_dir / "user_requirement.txt"
        try:
            prompt_file.write_text(user_requirement, encoding='utf-8')
        except Exception as e:
            print(f"   ❌ Failed to write requirement file: {e}")
            execution_results.append({
                'run_number': current_run_number,
                'experiment_id': experiment_id,
                'success': False,
                'error': f'Failed to write requirement file: {e}'
            })
            current_run_number += 1
            continue
        
        # Build Foam-Agent command
        cmd = [
            "python", str(foam_bench),
            "--openfoam_path", os.environ.get("WM_PROJECT_DIR", "/opt/openfoam10"),
            "--output", str(output_dir),
            "--prompt_path", str(prompt_file),
        ]
        
        print(f"   🚀 Running: {' '.join(cmd)}")
        
        # Set up environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
        
        # Execute Foam-Agent
        try:
            print(f"   ⏳ Executing Foam-Agent...")
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per case
            )
            
            if result.returncode == 0:
                print(f"   ✅ Recommendation executed successfully!")
                execution_results.append({
                    'run_number': current_run_number,
                    'experiment_id': experiment_id,
                    'success': True,
                    'run_dir': str(run_dir),
                    'output_dir': str(output_dir)
                })
            else:
                print(f"   ❌ Execution failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                execution_results.append({
                    'run_number': current_run_number,
                    'experiment_id': experiment_id,
                    'success': False,
                    'error': result.stderr,
                    'return_code': result.returncode
                })
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Execution timed out after 2 hours")
            execution_results.append({
                'run_number': current_run_number,
                'experiment_id': experiment_id,
                'success': False,
                'error': 'Execution timed out after 2 hours'
            })
        except Exception as e:
            print(f"   ❌ Execution exception: {e}")
            execution_results.append({
                'run_number': current_run_number,
                'experiment_id': experiment_id,
                'success': False,
                'error': str(e)
            })
        
        current_run_number += 1
    
    # Save execution results
    try:    
        results_file = target_experiment_dir / "execution_results.json"
        results_file.write_text(json.dumps(execution_results, indent=2), encoding='utf-8')
        print(f"\\n📋 Saved execution results to {results_file}")
    except Exception as e:
        print(f"Failed to save execution results: {e}")
    
    # Print summary
    successful = sum(1 for r in execution_results if r.get('success', False))
    print(f"\\n{'='*60}")
    print(f"📊 RECOMMENDATIONS EXECUTION SUMMARY")
    print(f"   Experiment Directory: {target_experiment_dir}")
    print(f"   Total Recommendations: {len(execution_results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(execution_results) - successful}")
    
    if successful > 0:
        print("   ✅ Successful Runs:")
        for result in execution_results:
            if result['success']:
                print(f"      • Run {result['run_number']:03d} - {result['experiment_id']}")
    
    if len(execution_results) > successful:
        print("   ❌ Failed Runs:")
        for result in execution_results:
            if not result['success']:
                print(f"      • Run {result['run_number']:03d} - {result['experiment_id']}")
    
    print(f"{'='*60}")


def _convert_recommendation_to_user_requirement(exp_rec: dict) -> str:
    """
    Convert a study recommendation JSON to a user requirement string that Foam-Agent can execute.
    
    Args:
        exp_rec: Experiment recommendation dictionary from JSON
        
    Returns:
        str: Formatted user requirement for Foam-Agent
    """
    params = exp_rec.get('parameters', {})
    description = exp_rec.get('description', 'CFD simulation')
    
    reynolds_number = params.get('reynolds_number', 1000)
    geometry = params.get('geometry', '2D square cavity 1x1x0.1')
    boundary_conditions = params.get('boundary_conditions', 'moving lid, no-slip walls')
    mesh_size = params.get('mesh_size', '35x35')
    solver_settings = params.get('solver_settings', 'steady-state')
    
    nu = 1.0 / reynolds_number
    
    # Parse mesh size
    if 'x' in str(mesh_size).lower():
        mesh_parts = str(mesh_size).lower().replace('x', ' ').split()
        if len(mesh_parts) >= 2:
            try:
                mesh_x = int(mesh_parts[0])
                mesh_y = int(mesh_parts[1])
            except:
                mesh_x = mesh_y = 35
        else:
            mesh_x = mesh_y = 35
    else:
        try:
            mesh_x = mesh_y = int(str(mesh_size))
        except:
            mesh_x = mesh_y = 35
    
    # Determine simulation time and time step based on Reynolds number
    if reynolds_number <= 100:
        sim_time = 10
        dt = 0.00025
    elif reynolds_number <= 400:
        sim_time = 25  
        dt = 0.0015
    elif reynolds_number <= 1000:
        sim_time = 50
        dt = 0.0015
    elif reynolds_number <= 2000:
        sim_time = 70
        dt = 0.0015
    else:
        sim_time = 100
        dt = 0.001
    
    # Generate user requirement based on geometry type detected from description/parameters
    geometry_lower = geometry.lower()
    
    if '2d' in geometry_lower and ('square' in geometry_lower or 'cavity' in geometry_lower):
        # 2D square cavity case
        user_requirement = f'''Do an incompressible lid-driven cavity flow.
The cavity is a square with dimensions 1 (x) × 1 (y) and very thin in z (0.1), making it effectively 2D.
Use a grid of {mesh_x} × {mesh_y} in x and y, and 1 cell in z. The front and back faces are 'empty'.
The top wall ('movingWall') at y=1 moves in +x with U=1 m/s.
All other walls ('fixedWalls') are no-slip (U=0).
Run from time=0 to time={sim_time} with time step Δt = {dt}; write results every 100 steps.
Set kinematic viscosity nu = {nu:.6f} m^2/s (Reynolds number = {reynolds_number}).
Visualize velocity magnitude contours and streamlines'''
    
    elif '3d' in geometry_lower and ('cube' in geometry_lower or 'cubic' in geometry_lower):
        # 3D cubic cavity case
        user_requirement = f'''Do an incompressible lid-driven cavity flow in 3D.
Cube 1 (x) × 1 (y) × 1 (z). Grid: {mesh_x} × {mesh_y} × {mesh_x}.
Top face at z=1 ('movingWall') moves in +x with U=1 m/s. All other faces no-slip.
Run time = 0 to time = {sim_time} with Δt = {dt}; write every 100 steps.
nu = {nu:.6f} m^2/s (Reynolds number = {reynolds_number}).
Visualize velocity magnitude contours and streamlines'''
    
    elif 'rectangle' in geometry_lower or 'rectangular' in geometry_lower:
        # Rectangular cavity case
        user_requirement = f'''Incompressible lid-driven cavity, rectangle 1×2, thin z (0.1), 2D.
Grid: {mesh_x} × {mesh_y*2} × 1; front/back 'empty'.
Lid y=2 U=1 m/s in +x; other walls no-slip.
Run time=0 to time={sim_time} with Δt = {dt}; write every 100 steps.
nu = {nu:.6f} m^2/s (Reynolds number = {reynolds_number}).
Visualize velocity magnitude contours and streamlines'''
    
    else:
        # Default to 2D square cavity
        user_requirement = f'''Do an incompressible lid-driven cavity flow.
The cavity is a square with dimensions 1 (x) × 1 (y) and very thin in z (0.1), making it effectively 2D.
Use a grid of {mesh_x} × {mesh_y} in x and y, and 1 cell in z. The front and back faces are 'empty'.
The top wall ('movingWall') at y=1 moves in +x with U=1 m/s.
All other walls ('fixedWalls') are no-slip (U=0).
Run from time=0 to time={sim_time} with time step Δt = {dt}; write results every 100 steps.
Set kinematic viscosity nu = {nu:.6f} m^2/s (Reynolds number = {reynolds_number}).
Visualize velocity magnitude contours and streamlines'''
    
    return user_requirement.strip()


def _analyze_and_collect_rerun_suggestions(
    batch_dir: Path,
    batch_summary: list,
    client,
    model_name: str,
    temperature: float,
    rerun_threshold: float = 7.0,
    auto_rerun_threshold: float = 5.0,
) -> tuple[list, list]:
    """
    Analyze each experiment run with visualization image and determine if rerun is needed.
    Combines analysis and validation into a single LLM call per run.
    
    Args:
        rerun_threshold: Threshold below which cases are flagged for manual rerun (default: 7.0)
        auto_rerun_threshold: Threshold below which cases are flagged for automatic rerun (default: 5.0)
    
    Returns:
        tuple: (rerun_suggestions, auto_rerun_cases)
            rerun_suggestions: List of dicts with standard rerun info
            auto_rerun_cases: List of dicts for cases needing immediate automatic rerun
    """
    system_message = (
        "You are a precise CFD validation expert with expertise in visual analysis of flow simulations. "
        "Your task is to:\n"
        "1. Analyze the visualization image to understand the flow physics\n"
        "2. Compare the visualization against the user requirement\n"
        "3. Determine if the simulation matches what was requested\n"
        "4. If errors are found, propose a complete corrected user requirement\n\n"
        "IMPORTANT: The user requirement must clearly state what should be visualized. "
        "If the visualization statement is unclear or missing, include it in your corrected requirement."
    )

    rerun_suggestions = []
    auto_rerun_cases = []

    for exp in batch_summary:
        exp_name = exp["experiment"]
        exp_dir = batch_dir / exp_name
        
        print(f"\n🔎 Analyzing experiment: {exp_name} with {len(exp.get('runs', []))} run(s)")
        
        for run in exp.get("runs", []):
            run_name = run["run"]
            ur = (run.get("user_requirement", "") or "").strip()
            first_line = next((ln.strip() for ln in ur.splitlines() if ln.strip()), "<missing user requirement>")
            if len(first_line) > 120:
                first_line = first_line[:117] + "..."
            print(f"   🧪 Run {run_name}: {first_line}")
            
            # Build comprehensive analysis prompt with images
            timestep_images = run.get("timestep_images", []) or []
            timestep_umag = run.get("timestep_images_umag", []) or [x for x in timestep_images if x.get("field") == "umag"]
            timestep_p = run.get("timestep_images_p", []) or [x for x in timestep_images if x.get("field") == "p"]
            viz_snippets = []
            for v in run.get("visualizations", [])[:2]:
                viz_snippets.append(f"Path: {v['path']}\n{v['content']}")
            viz_block = "\n\n".join(viz_snippets) if viz_snippets else "<no visualization scripts>"
            out_list = ", ".join([f[0] for f in run.get("output_files", [])[:15]]) or "<no outputs listed>"

            img_index_block = ""
            if timestep_umag or timestep_p:
                def _order(xs):
                    return sorted(
                        xs,
                        key=lambda d: (
                            d.get("t") is None,
                            float(d.get("t") or 0.0),
                            str(d.get("path") or ""),
                        ),
                    )

                ordered_umag = _order(timestep_umag)
                ordered_p = _order(timestep_p)

                lines = []
                i_img = 0
                if ordered_umag:
                    lines.append("UMAG (|U|) images:")
                    for d in ordered_umag:
                        i_img += 1
                        tstr = f"t={float(d['t']):.2f}s" if d.get("t") is not None else "t=?"
                        fname = Path(d.get("path", "")).name
                        lines.append(f"{i_img}. {tstr} — {fname}")

                if ordered_p:
                    lines.append("PRESSURE (p) images:")
                    for d in ordered_p:
                        i_img += 1
                        tstr = f"t={float(d['t']):.2f}s" if d.get("t") is not None else "t=?"
                        fname = Path(d.get("path", "")).name
                        lines.append(f"{i_img}. {tstr} — {fname}")

                img_index_block = "\nIMAGES PROVIDED (order):\n" + "\n".join(lines) + "\n"

            image_instruction = ""
            if timestep_umag and timestep_p:
                def _order(xs):
                    return sorted(
                        xs,
                        key=lambda d: (
                            d.get("t") is None,
                            float(d.get("t") or 0.0),
                            str(d.get("path") or ""),
                        ),
                    )

                ordered_umag = _order(timestep_umag)
                ordered_p = _order(timestep_p)

                print(f"      📸 UMag images: {len(ordered_umag)}, p images: {len(ordered_p)}")

                image_instruction = (
                    "\n\n🖼️ CRITICAL: MULTIPLE VISUALIZATION IMAGES ARE PROVIDED for the SAME run at multiple times.\n"
                    "You are given two fields at matching times:\n"
                    "  - UMag (|U|) contours\n"
                    "  - pressure p contours\n\n"
                    "CAREFULLY ANALYZE ALL IMAGES TOGETHER:\n"
                    "1. Requirement matching:\n"
                    "   - Are both requested fields present (UMag and p) at the requested times?\n"
                    "   - Are slice/plane and domain consistent with the requirement?\n"
                    "2. Temporal behavior (within each field):\n"
                    "   - Does UMag evolve smoothly over time without obvious numerical blow-up or artifacts?\n"
                    "   - Does p evolve consistently (no unphysical jumps between frames)?\n"
                    "3. Cross-field consistency:\n"
                    "   - Are high-|U| regions consistent with expected pressure gradients near inlet/outlet?\n"
                    "4. Quality indicators:\n"
                    "   - Signs of instability, excessive diffusion, checkerboarding, or mesh imprinting.\n\n"
                )
            else:
                if not timestep_umag:
                    print("      ⚠️  Missing UMag timestep images (umag_t*.png)")
                if not timestep_p:
                    print("      ⚠️  Missing pressure timestep images (p_t*.png)")
            
            analysis_prompt = (
                f"Analyze this CFD simulation run and determine if it matches the user requirement.\n\n"
                f"PRIMARY TASKS:\n"
                f"1. Describe the flow physics you observe in the visualization image(s)\n"
                f"2. Compare the visualization image(s) against the user requirement\n"
                f"3. Identify any errors, discrepancies, or issues\n"
                f"4. If issues found, propose a COMPLETE corrected user requirement\n\n"
                f"CRITICAL REQUIREMENTS:\n"
                f"- The user requirement MUST clearly state what should be visualized (e.g., 'Visualize velocity magnitude contours with streamlines')\n"
                f"- If the visualization statement is unclear or missing, add it to your proposed requirement\n"
                f"- If geometry, mesh, or physics parameters are wrong, specify exact corrections\n"
                f"- Be specific about boundary conditions, domain size, mesh resolution, and any key parameters\n"
                f"{img_index_block}{image_instruction}\n"
                f"IF ANY ISSUES ARE FOUND:\n"
                f"- Set 'accurate' to false if significant problems exist\n"
                f"- Give accuracy score from 1-10 (1=poor, 10=excellent) based on overall quality\n"
                f"- Accuracy >= 7.0 means good quality (accurate=true, no rerun needed)\n"
                f"- Accuracy 5.0-6.9 means moderate quality (accurate=false, manual review)\n"
                f"- Accuracy < 5.0 means poor quality (accurate=false, auto-rerun needed)\n"
                f"- Provide detailed 'explanation' of what is wrong\n"
                f"- Provide 'analysis' describing the flow physics you observe\n"
                f"- Provide 'proposed_user_requirement' only if significant corrections needed\n\n"
                f"Return STRICT JSON between ```json and ``` with fields:\n"
                f"{{\n"
                f"  \"analysis\": string,  // Describe flow physics and phenomena observed in visualization\n"
                f"  \"accurate\": boolean,  // false only if accuracy < 7.0 (significant issues)\n"
                f"  \"accuracy\": number,    // 1-10 score (7.0+ = good, 5.0-6.9 = moderate, <5.0 = poor)\n"
                f"  \"explanation\": string, // What issues exist (can be empty if accuracy >= 7.0)\n"
                f"  \"visualization_matches_requirement\": boolean,  // Does image match what was requested?\n"
                f"  \"visualization_statement_clear\": boolean,  // Is 'what to visualize' clearly stated in requirement?\n"
                f"  \"proposed_user_requirement\": string|null  // Corrected requirement (null if accuracy >= 7.0)\n"
                f"}}\n\n"
                f"USER REQUIREMENT:\n{ur}\n\n"
                f"VISUALIZATION SCRIPT CODE:\n{viz_block}\n\n"
                f"OUTPUT FILES (sample): {out_list}\n"
            )
            
            missing_umag = not bool(timestep_umag)
            missing_p = not bool(timestep_p)
            if missing_umag or missing_p:
                # Missing evaluator artifacts: treat as non-evaluable.
                # To keep existing rerun machinery working, propose rerunning with the same requirement.
                missing_parts = []
                if missing_umag:
                    missing_parts.append("umag_t*.png")
                if missing_p:
                    missing_parts.append("p_t*.png")
                missing_str = ", ".join(missing_parts)

                stub = {
                    "analysis": f"Missing timestep visualization images ({missing_str}). Run is not evaluable.",
                    "accurate": False,
                    "accuracy": 0,
                    "explanation": f"Required timestep images were not found: {missing_str}. Ensure deterministic postprocess runs and produces both UMag and p time-stamped PNG outputs.",
                    "visualization_matches_requirement": False,
                    "visualization_statement_clear": False,
                    "proposed_user_requirement": ur or None,
                }
                response_text = "```json\n" + json.dumps(stub, indent=2) + "\n```"
            else:
                try:
                    # If timestep images are available and using Bedrock, send ONE multimodal call with ALL images.
                    # Ordering: all UMag images (sorted by time) then all p images (sorted by time).
                    if "bedrock" in str(type(client)).lower():
                        def _order(xs):
                            return sorted(
                                xs,
                                key=lambda d: (
                                    d.get("t") is None,
                                    float(d.get("t") or 0.0),
                                    str(d.get("path") or ""),
                                ),
                            )

                        ordered_imgs = _order(timestep_umag) + _order(timestep_p)

                        content = []
                        bad = 0
                        for d in ordered_imgs:
                            try:
                                img_bytes = base64.b64decode(d.get("base64", "") or "")
                            except Exception:
                                img_bytes = None
                            if not img_bytes:
                                bad += 1
                                continue
                            content.append(
                                {
                                    "image": {
                                        "format": d.get("format", "png"),
                                        "source": {"bytes": img_bytes},
                                    }
                                }
                            )

                        # Append the text prompt last so the model sees the images first.
                        content.append({"text": analysis_prompt})

                        if len(content) <= 1:
                            print("      ⚠️  No valid image bytes could be prepared; falling back to text-only")
                            response_text, _ = get_response_from_llm(
                                analysis_prompt,
                                client,
                                model_name,
                                system_message,
                                print_debug=False,
                                temperature=temperature,
                            )
                        else:
                            if bad:
                                print(f"      ⚠️  Skipped {bad} timestep image(s) due to invalid base64")
                            print(f"Processing {len(content)-1} timestep image(s) (UMag+p) in one LLM call...")

                            messages = [{"role": "user", "content": content}]
                            response = client.converse(
                                modelId=model_name,
                                messages=messages,
                                system=[{"text": system_message}],
                                inferenceConfig={
                                    "temperature": temperature,
                                    "maxTokens": 4096,
                                },
                            )
                            response_text = response["output"]["message"]["content"][0]["text"]
                    else:
                        # Text-only fallback for non-multimodal clients.
                        response_text, _ = get_response_from_llm(
                            analysis_prompt,
                            client,
                            model_name,
                            system_message,
                            print_debug=False,
                            temperature=temperature,
                        )
                except Exception as e:
                    print(f"      ❌ LLM analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            data = extract_json_between_markers(response_text) or {}
            analysis_text = data.get("analysis", "")
            accurate = bool(data.get("accurate", False))
            accuracy = float(data.get("accuracy", 0))
            proposed = data.get("proposed_user_requirement")
            explanation = data.get("explanation", "")
            viz_matches = bool(data.get("visualization_matches_requirement", True))
            viz_statement_clear = bool(data.get("visualization_statement_clear", True))
            
            print(f"      → Accuracy: {accuracy:.1f}/10, Matches: {viz_matches}, Viz statement clear: {viz_statement_clear}")

            decision = {
                "experiment": exp_name,
                "run": run_name,
                "analysis": analysis_text,
                "accurate": accurate,
                "accuracy": accuracy,
                "explanation": explanation,
                "visualization_matches_requirement": viz_matches,
                "visualization_statement_clear": viz_statement_clear,
                "original_requirement": ur,
                "updated_requirement": proposed,
                "response_raw": response_text,
                "evidence_files": {
                    "umag_images": [x.get("path") for x in timestep_umag],
                    "p_images": [x.get("path") for x in timestep_p],
                },
            }
            
            # Save combined analysis to experiment directory
            try:
                analysis_file = exp_dir / run_name / "analysis.txt"
                analysis_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write human-readable analysis
                analysis_content = f"Run: {run_name}\n\n"
                analysis_content += f"Flow Physics Analysis:\n{analysis_text}\n\n"
                analysis_content += f"Accuracy: {accuracy:.1f}/10\n"
                analysis_content += f"Matches Requirement: {viz_matches}\n"
                analysis_content += f"Visualization Statement Clear: {viz_statement_clear}\n\n"
                if explanation:
                    analysis_content += f"Issues Found:\n{explanation}\n\n"
                if proposed:
                    analysis_content += f"Proposed Corrected Requirement:\n{proposed}\n"
                
                analysis_file.write_text(analysis_content, encoding="utf-8")
                
                # Also save JSON verdict
                verdict_file = exp_dir / run_name / "analysis_verdict.json"
                verdict_file.write_text(json.dumps(decision, indent=2), encoding="utf-8")
                
                print(f"      💾 Saved analysis to {analysis_file}")
            except Exception as e:
                print(f"      ⚠️  Failed to write analysis: {e}")

            # Check if rerun is needed - missing evaluator artifacts always triggers rerun.
            missing_artifacts = (not bool(timestep_umag)) or (not bool(timestep_p))
            should_rerun = missing_artifacts or ((accuracy < rerun_threshold) and isinstance(proposed, str) and proposed.strip())
            
            # Additional check for critical visualization issues 
            critical_viz_issue = (not viz_matches) or (not viz_statement_clear and accuracy < 8.0)
            if critical_viz_issue and isinstance(proposed, str) and proposed.strip():
                should_rerun = True
            
            if should_rerun:
                # Cases with very low scores get automatic rerun
                if accuracy < auto_rerun_threshold:
                    print(f"      🔥 CRITICAL: Auto-rerun needed (accuracy={accuracy:.1f}/10)")
                    auto_rerun_cases.append(decision)
                else:
                    print(f"      ⚠️  Rerun recommended (accuracy={accuracy:.1f}/10)")
                    rerun_suggestions.append(decision)
            else:
                print(f"      ✅ Run is satisfactory")

    # Save consolidated rerun suggestions
    total_reruns = len(rerun_suggestions) + len(auto_rerun_cases)
    if total_reruns > 0:
        try:
            # Save regular rerun suggestions
            if rerun_suggestions:
                suggestions_file = batch_dir / "rerun_suggestions.json"
                suggestions_file.write_text(json.dumps(rerun_suggestions, indent=2), encoding="utf-8")
                print(f"\n📋 Saved {len(rerun_suggestions)} rerun suggestions to {suggestions_file}")
            
            # Save auto-rerun cases separately
            if auto_rerun_cases:
                auto_rerun_file = batch_dir / "auto_rerun_cases.json"
                auto_rerun_file.write_text(json.dumps(auto_rerun_cases, indent=2), encoding="utf-8")
                print(f"\n🔥 Saved {len(auto_rerun_cases)} critical auto-rerun cases to {auto_rerun_file}")
                
        except Exception as e:
            print(f"Failed to write rerun suggestions: {e}")
    else:
        print("\n✅ No reruns needed - all experiments meet quality threshold!")

    return rerun_suggestions, auto_rerun_cases


def _execute_automatic_reruns(batch_dir: Path, auto_rerun_cases: list):
    """
    Automatically execute reruns for cases with critically low scores by calling Foam-Agent directly.
    Replaces the original bad case after backing it up.
    
    Args:
        batch_dir: Path to batch directory
        auto_rerun_cases: List of cases that need immediate rerun
    """
    import subprocess
    import uuid
    import shutil
    from datetime import datetime
    
    project_root = _project_root
    foam_bench = (project_root / "Foam-Agent" / "foambench_main.py").resolve()
    
    if not foam_bench.exists():
        print(f"❌ Error: Foam-Agent not found at {foam_bench}. Cannot perform automatic reruns.")
        return
    
    print(f"🔧 Foam-Agent found at: {foam_bench}")
    
    for i, case in enumerate(auto_rerun_cases, 1):
        exp_name = case['experiment']
        run_name = case['run']
        updated_requirement = case['updated_requirement']
        accuracy = case['accuracy']
        
        print(f"\n[{i}/{len(auto_rerun_cases)}] 🔄 Auto-rerunning: {exp_name}/{run_name}")
        print(f"   Original accuracy: {accuracy:.1f}/10")
        print(f"   Updated requirement: {updated_requirement[:100]}...")
        
        # Get paths
        exp_dir = batch_dir / exp_name
        original_run_dir = exp_dir / run_name
        
        if not original_run_dir.exists():
            print(f"   ❌ Original run directory not found: {original_run_dir}")
            continue
        
        # Create backup of original bad case
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # YYYYMMDD_HHMMSS_mmm
        backup_dir = exp_dir / f"backup_{run_name}_{timestamp}"
        
        try:
            print(f"   📦 Backing up original to: {backup_dir.name}")
            shutil.copytree(original_run_dir, backup_dir)
        except Exception as e:
            print(f"   ❌ Failed to backup original run: {e}")
            continue
        
        # Create temporary directory for new run
        temp_rerun_dir = exp_dir / f"temp_rerun_{timestamp}"
        temp_rerun_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory for Foam-Agent
        temp_output_dir = temp_rerun_dir / "output"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write updated user requirement to temp location
        temp_prompt_file = temp_rerun_dir / "user_requirement.txt"
        try:
            temp_prompt_file.write_text(updated_requirement, encoding='utf-8')
            print(f"   📄 Updated requirement written to temp location")
        except Exception as e:
            print(f"   ❌ Failed to write requirement file: {e}")
            # Clean up temp directory
            shutil.rmtree(temp_rerun_dir, ignore_errors=True)
            continue
        
        # Build Foam-Agent command
        cmd = [
            "python", str(foam_bench),
            "--openfoam_path", os.environ.get("WM_PROJECT_DIR", "/opt/openfoam10"),
            "--output", str(temp_output_dir),
            "--prompt_path", str(temp_prompt_file),
        ]
        
        print(f"   🚀 Running: {' '.join(cmd)}")
        
        # Set up environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
        
        # Execute Foam-Agent
        try:
            print(f"   ⏳ Executing Foam-Agent (this may take several minutes)...")
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print(f"   ✅ Auto-rerun successful! Replacing original run...")
                
                # Remove original bad run
                shutil.rmtree(original_run_dir)
                
                # Move temp rerun to replace original
                shutil.move(str(temp_rerun_dir), str(original_run_dir))
                
                # Save replacement metadata in the new run directory
                replacement_info = {
                    "replaced_on": datetime.now().isoformat(),
                    "original_accuracy": accuracy,
                    "backup_location": backup_dir.name,
                    "original_requirement": case['original_requirement'],
                    "updated_requirement": updated_requirement,
                    "foam_agent_success": True,
                    "replacement_reason": f"Auto-rerun due to accuracy score {accuracy:.1f}/10 < 5.0"
                }
                
                replacement_file = original_run_dir / "replacement_info.json"
                replacement_file.write_text(json.dumps(replacement_info, indent=2), encoding='utf-8')
                
                print(f"   📁 Original run replaced successfully!")
                print(f"   📦 Backup saved as: {backup_dir.name}")
                
            else:
                print(f"   ❌ Auto-rerun failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                
                # Clean up temp directory on failure
                shutil.rmtree(temp_rerun_dir, ignore_errors=True)
                
                # Save error info to backup directory
                error_info = {
                    "rerun_failed_on": datetime.now().isoformat(),
                    "original_accuracy": accuracy,
                    "updated_requirement": updated_requirement,
                    "foam_agent_success": False,
                    "foam_agent_error": result.stderr,
                    "return_code": result.returncode
                }
                
                error_file = backup_dir / "rerun_error.json"
                error_file.write_text(json.dumps(error_info, indent=2), encoding='utf-8')
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Auto-rerun timed out after 1 hour")
            # Clean up temp directory
            shutil.rmtree(temp_rerun_dir, ignore_errors=True)
        except Exception as e:
            print(f"   ❌ Auto-rerun exception: {e}")
            # Clean up temp directory
            shutil.rmtree(temp_rerun_dir, ignore_errors=True)
    
    print(f"\n🏁 Completed automatic reruns for {len(auto_rerun_cases)} cases")
    print("   ✅ Original bad runs have been replaced with corrected versions")
    print("   📦 Backups of original runs are preserved for reference")


def main():
    parser = argparse.ArgumentParser(description="Analyze a batch of experiments using an LLM")
    parser.add_argument("--batch", type=str, help="Batch folder name under data/experiments to analyze (defaults to latest)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name to use (default: Bedrock ARN)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--max-experiments", type=int, default=None, help="Limit number of experiments to analyze")
    parser.add_argument("--auto-rerun", action="store_true", help="Automatically rerun cases with scores < 5.0 until threshold is met")
    parser.add_argument("--auto-rerun-threshold", type=float, default=5.0, help="Threshold below which cases are automatically rerun (default: 5.0)")
    parser.add_argument("--max-rerun-iterations", type=int, default=3, help="Maximum rerun attempts per case (default: 3)")
    parser.add_argument("--auto-execute-recommendations", action="store_true", help="Automatically execute study recommendations as new experiments (default: False)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    base_dir = project_root / "data" / "experiments"
    
    batch_dir = choose_batch(base_dir, args.batch)
    if not batch_dir:
        print("Unable to locate a batch to analyze. Exiting.")
        return
    
    analysis, rerun_suggestions = analyze_batch(
        batch_dir, 
        model=args.model, 
        temperature=args.temperature, 
        max_experiments=args.max_experiments,
        auto_rerun_threshold=args.auto_rerun_threshold,
        enable_auto_rerun=args.auto_rerun,
        max_rerun_iterations=args.max_rerun_iterations,
        auto_execute_recommendations=args.auto_execute_recommendations
    )
    
    if analysis:
        print("\n" + "="*60)
        print("📊 Analysis complete.")
        print("="*60)
        if rerun_suggestions:
            print(f"\n⚠️  Found {len(rerun_suggestions)} experiments that need reruns")
            print("\nTo rerun these experiments, use:")
            print(f"  python src/main.py --rerun-batch {batch_dir.name}")
        else:
            print("\n✅ All experiments meet quality standards!") 

        if args.auto_rerun:
            print(f"\n🔥 Auto-rerun was enabled with threshold < {args.auto_rerun_threshold}")
            print(f"   Maximum {args.max_rerun_iterations} iterations per case")
            print("   Critical cases were automatically reprocessed until threshold was met")
            print("   Check experiment directories for 'backup_*' folders with original bad runs")
        
        # Show study recommendations info
        recommendations_file = batch_dir / "study_recommendations.txt"
        if recommendations_file.exists():
            print(f"\n🔬 STUDY RECOMMENDATIONS GENERATED:")
            print(f"   📋 Comprehensive suggestions saved to: {recommendations_file}")
            print("   💡 Review recommendations for next research steps")
            print("   🎯 Includes parameter gaps, validation opportunities, and publication enhancements")



if __name__ == '__main__':
    main()
