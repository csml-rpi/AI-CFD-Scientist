import os
import json
import yaml
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.base_llm import create_client, get_response_from_llm

load_dotenv()

class HypothesisAgent:
    """
    Agent for converting experiment concepts from Ideation agent into 
    Foam-Agent user requirements. 
    """
    
    def __init__(self, model: str = None):
        self.user_requirements = []
        self.model = model
        self.client, self.model_id = create_client(model)  # Will use env var or default if None
        self.prompts = self._load_prompts()
        print(f"HypothesisAgent initialized with model: {self.model_id}")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from the external prompts.yaml file."""
        try:
            prompts_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.yaml")
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = yaml.safe_load(f)
            return prompts_data.get('HypothesisAgent', {})
        except Exception as e:
            print(f"Warning: Could not load prompts from file: {e}")
            return {}
    
    def idea_to_user_requirments(self, ideation_ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts all the experiments from one idea into individual Foam-Agent user requirements.
        e.g. If idea has 3 experiments, this returns 3 dictionaries, each containing exp name, 
        description, parameters, and the converted Foam-Agent user requirement. 

        Args:
            ideation_ideas: List of experiment concepts from one idea 
          
        Returns:
            List of dictionaries containing idea info and user requirements for each experiment
        """
        processed_results = []
        
        for idea in ideation_ideas:
            idea_name = idea.get('idea_name', 'Unknown Idea')
            short_hypothesis = idea.get('short_hypothesis', '')
            experiments = idea.get('experiments', [])
            
            print(f"\n{'='*60}")
            print(f"Processing idea: {idea_name}")
            print(f"Hypothesis: {short_hypothesis}")
            print(f"Number of experiments: {len(experiments)}")
            print(f"{'='*60}")
            
            idea_results = {
                'idea_name': idea_name,
                'title': idea.get('title', ''),
                'short_hypothesis': short_hypothesis,
                'related_work': idea.get('related_work', ''),
                'abstract': idea.get('abstract', ''),
                'experiments': []
            }
            
            for i, experiment in enumerate(experiments, 1):
                print(f"\n--- Processing Experiment {i}/{len(experiments)} ---")
                print(f"Experiment: {experiment.get('experiment_name')}")
                
                user_requirement = self.convert_experiment_to_user_requirement(idea, experiment)
                
                experiment_result = {
                    'experiment_name': experiment.get('experiment_name'),
                    'experiment_description': experiment.get('experiment_description'),
                    'experiment_parameters': experiment.get('experiment_parameters'),
                    'user_requirement': user_requirement
                }
                
                idea_results['experiments'].append(experiment_result)
                print(f"Generated user requirement: {user_requirement}")
            
            processed_results.append(idea_results)
        
        self.user_requirements = processed_results
        return processed_results
    
    def generate_user_requirement(self, idea: Dict[str, Any], simulation: Dict[str, Any]) -> str:
        """
        Generate a user requirement for a specific simulation using the hypothesis prompt.
        
        Args:
            idea: Complete idea dictionary containing all idea information
            simulation: Simulation dictionary from experiment.simulations array
            
        Returns:
            Simulation requirement string for Foam-Agent
        """
        hypothesis_system_prompt = self.prompts.get('hypothesis_system_prompt', '')
        hypothesis_user_prompt = self.prompts.get('hypothesis_user_prompt', '')
        if not hypothesis_system_prompt or not hypothesis_user_prompt:
            raise ValueError("Could not load hypothesis_system_prompt or hypothesis_user_prompt from prompts.yaml")
        
        case_data = simulation.get('case_data', {})
        
        # Create a complete experiment concept with all necessary data
        experiment_concept = {
            'study_id': idea.get('study_id', ''),
            'description': idea.get('description', ''),
            'case_name': simulation.get('case_name', ''),
            'simulation_id': simulation.get('simulation_id', ''),
            'parameter_value': simulation.get('parameter_value', ''),
            'simulation_description': simulation.get('description', ''),
            'visualization': simulation.get('visualization', ''),
            'case_dimensions': case_data.get('dimensions', []),
            'case_topology': case_data.get('topology', ''),
            'lid_axis': case_data.get('lid_axis', ''),
            'lid_speed': case_data.get('lid_speed', 0),
            'solver': idea.get('solver', 'icoFoam'),
            'target_CFL': idea.get('target_CFL', 0.5),
            'post_processing': idea.get('post', {})
        }
        
        try:
            formatted_system_prompt = hypothesis_system_prompt.replace(
                '{experiment_concept}', json.dumps(experiment_concept, indent=2)
            )
            formatted_user_prompt = hypothesis_user_prompt.replace(
                '{study_id}', experiment_concept['study_id']
            ).replace(
                '{description}', experiment_concept['description']
            ).replace(
                '{case_name}', experiment_concept['case_name']
            ).replace(
                '{simulation_id}', experiment_concept['simulation_id']
            ).replace(
                '{parameter_value}', str(experiment_concept['parameter_value'])
            ).replace(
                '{simulation_description}', experiment_concept['simulation_description']
            ).replace(
                '{visualization}', experiment_concept['visualization']
            )
            
            print(f"Debug - Successfully formatted prompts")

            print(f"Formatted system prompt: {formatted_system_prompt}")
            print(f"Formatted user prompt: {formatted_user_prompt}")
            
            content, msg_history = get_response_from_llm(
                prompt=formatted_user_prompt,
                client=self.client,
                model=self.model_id,
                system_message=formatted_system_prompt,
                temperature=0.3,
                print_debug=False
            )
            
            content = content.strip()
            
            return content
            
        except Exception as e:
            print(f"Error generating user requirement: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return f"Error generating requirement for {simulation.get('simulation_id', 'simulation')}"
    
    
    def validate_user_requirements(self, requirements: List[str]) -> List[Dict[str, Any]]:
        """
        Validate and correct a list of user requirement strings using the LLM model.

        For each requirement, the model checks for inconsistencies and provides corrections.
        Returns exactly what the model produces without any regex processing.

        Returns a list of dicts with the following keys:
        - original: original requirement string
        - corrected: corrected requirement string (exactly as returned by model)
        - issues: list of identified issues (empty if none)
        """
        results: List[Dict[str, Any]] = []

        system_message = """You are a CFD simulation expert. Your task is to validate user requirements for OpenFOAM simulations and correct any inconsistencies.

Check for these common issues:
1. Time ranges where end time < start time (reversed)
2. Timesteps (Δt or dt) that are too large compared to simulation time, the start time, end time, and time step must make sense
3. Invalid or missing timesteps
4. Check calcluations  
5. Any other physics or simulation parameter inconsistencies

For the requirement:
- If no issues found, return the EXACT original text unchanged
- If issues found, correct ONLY the problematic values while preserving all formatting, spacing, and other content exactly

Return your response in this exact format:
ISSUES: [list any issues found, or "None" if no issues]
CORRECTED_REQUIREMENT:
[The corrected requirement text, or exact original if no changes needed]"""

        for i, req in enumerate(requirements, 1):
            original = req.strip()
            
            user_prompt = f"""Validate this CFD user requirement for consistency:

USER REQUIREMENT {i}:
{original}

Check for timing inconsistencies, invalid parameters, and other issues. Return the corrected version or the exact original if no issues found."""

            try:
                response, _ = get_response_from_llm(
                    prompt=user_prompt,
                    client=self.client,
                    model=self.model_id,
                    system_message=system_message,
                    temperature=0.1,
                    print_debug=False
                )
                
                # Parse the response
                issues = []
                corrected = original
                
                if "ISSUES:" in response:
                    issues_part = response.split("ISSUES:")[1].split("CORRECTED_REQUIREMENT:")[0].strip()
                    if issues_part.lower() != "none":
                        issues.append(issues_part)
                
                if "CORRECTED_REQUIREMENT:" in response:
                    corrected_part = response.split("CORRECTED_REQUIREMENT:")[1].strip()
                    if corrected_part:
                        corrected = corrected_part
                
            except Exception as e:
                print(f"Error validating requirement {i}: {e}")
                issues = [f"Validation error: {e}"]
                corrected = original

            results.append({
                'original': original,
                'corrected': corrected,
                'issues': issues,
                'model_response': response if 'response' in locals() else None
            })

        return results
    
    def save_user_requirements(self, user_requirements: List[Dict[str, Any]], idea_filename: str = None) -> str:
        """
        Save generated user requirements to a JSON file.
        
        Args:
            idea_filename: Optional idea filename to use as base (will append timestamp if not provided)
            
        Returns:
            Path to saved file
        """
        if not self.user_requirements:
            print("No user requirements to save")
            return ""
        
        os.makedirs("data/user_requirements", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if idea_filename is None:
            filename = f"user_requirements_{timestamp}.json"  
        else:
            # Extract the base name from the idea filename (remove path and extension)
            base_name = os.path.splitext(os.path.basename(idea_filename))[0]
            # Remove timestamp if it exists in the original filename
            if '_' in base_name:
                # Try to remove timestamp pattern (YYYYMMDD_HHMMSS)
                parts = base_name.split('_')
                if len(parts) >= 2:
                    # Check if last two parts look like a timestamp
                    if len(parts[-2]) == 8 and len(parts[-1]) == 6:  
                        base_name = '_'.join(parts[:-2])
                    else:
                        base_name = '_'.join(parts[:-1])  # Remove just the last timestamp part
            
            
            filename = f"{base_name}_hypothesis_{timestamp}.json"
        
        filepath = f"data/user_requirements/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(user_requirements, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(user_requirements)} processed ideas to {filepath}")
        return filepath


def main():
    """
    Main function for running the HypothesisAgent from command line.
    
    Usage:
        python agents/hypotheis_ag.py --input data/ideas/ideas_file.json --model "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2"
        python agents/hypotheis_ag.py --input data/ideas/ideas_file.json --model "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2"
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert ideation ideas to Foam-Agent user requirements using HypothesisAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using default Bedrock model
  python agents/hypotheis_ag.py --input data/ideas/ideas_20251029.json
  
  # Using AWS Bedrock with specific ARN
  python agents/hypotheis_ag.py --input data/ideas/ideas_20251029.json --model "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2"
  
  # Using Bedrock with model ID
  python agents/hypotheis_ag.py --input data/ideas/ideas_20251029.json --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to JSON file containing ideation ideas'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None, 
        help='Model to use for generation. Supports Bedrock, OpenAI, and Anthropic models (default: uses CFD_SCIENTIST_MODEL env var or Bedrock ARN)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename for user requirements (default: auto-generated with timestamp)'
    )
    
    args = parser.parse_args()
    
    # Load ideas
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return
    
    with open(args.input, 'r', encoding='utf-8') as f:
        ideas = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"CFD Hypothesis Agent")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Number of ideas: {len(ideas)}")
    print(f"{'='*60}\n")
    
    # Initialize agent with specified model
    agent = HypothesisAgent(model=args.model)
    
    # Convert ideas to user requirements
    user_requirements = agent.idea_to_user_requirments(ideas)
    
    # Save user requirements
    filepath = agent.save_user_requirements(user_requirements, filename=args.output)
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully converted {len(ideas)} ideas to {len(user_requirements)} user requirements")
    print(f"📄 Saved to: {filepath}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

