import argparse
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils import LLMService
from config import Config


def load_jsonl_data(file_path: Path) -> List[Dict]:
    """Load data from JSONL file"""
    print(f"Loading jsonl data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def group_by_case_name(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by case_name"""
    grouped = defaultdict(list)
    for record in data:
        case_name = record.get('case_name', 'unknown')
        grouped[case_name].append(record)
    return dict(grouped)


def create_system_prompt() -> str:
    """Create the system prompt for generating user requirements"""
    return """You are an expert OpenFOAM simulation engineer. Your task is to analyze OpenFOAM case files and generate a realistic user requirement that a simulation engineer would specify when requesting such a simulation.

Based on the provided OpenFOAM case files, generate a user_requirement that follows these patterns:

STRUCTURE REQUIREMENTS:
1. Start with "do a [simulation type]" or "Perform a [simulation type]" or "Conduct a [simulation type]"
2. Include solver specification: "using [solver name] solver" or "Use [solver name] solver"
3. Specify geometric details with precise dimensions.
4. When reporting dimensions, report values as is in the geometry file without scaling using convertToMeters parameter. Further report the convertToMeters value seperatly. 
5. Define all boundary conditions for different patches/surfaces
6. Include time parameters (start time, end time, timestep, output frequency)
7. Specify physical properties (viscosity, density, temperature, pressure, etc.)
8. Mention grid/mesh details when relevant
9. Include algorithm details (PIMPLE, SIMPLE, etc.) when applicable
10. When reporting intial location of fluid, report their location in x,y,z coordinates. For example water occupies the region 0<=x<=1, 0<=y<=1, 0<=z<=1.
11. Detail the geometry of the domain as much as possible in a concise manner.

TECHNICAL ACCURACY:
- Use correct OpenFOAM terminology and solver names
- Include realistic engineering values with proper units
- Specify boundary condition types accurately (fixedValue, zeroGradient, noSlip, etc.)
- Include material properties relevant to the simulation type
- Mention turbulence models when applicable (k-epsilon, RAS, etc.)

FORMAT REQUIREMENTS:
- Generate a single, comprehensive sentence or paragraph
- Use technical language appropriate for CFD engineers
- Include specific numerical values extracted from the case files
- Maintain consistency with OpenFOAM naming conventions

EXAMPLES OF GOOD PATTERNS:
- "do a Reynolds-Averaged Simulation (RAS) pitzdaily simulation. Use PIMPLE algorithm. The domain is a 2D millimeter-scale channel geometry. Boundary conditions specify a fixed velocity of 10m/s at the inlet (left), zero gradient pressure at the outlet (right), and no-slip conditions for walls. Use timestep of 0.0001 and output every 0.01. Finaltime is 0.3. use nu value of 1e-5."

- "do an incompressible lid driven cavity flow. The cavity is a square with dimensions normalized to 1 unit on both the x and y axes and very thin in the z-direction (0.1 unit scaled down by a factor of 0.1, making it effectively 2D). Use a grid of 20X20 in x and y direction and 1 cell in z-direction(due to the expected 2D flow characteristics). The top wall ('movingWall') moves in the x-direction with a uniform velocity of 1 m/s. The 'fixedWalls' have a no-slip boundary condition (velocity equal to zero at the wall). The front and back faces are designated as 'empty'. The simulation runs from time 0 to 10 with a time step of 0.005 units, and results are output every 100 time steps. The viscosity (nu) is set as constant with a value of 1e-05 m^2/s."

Generate ONLY the user_requirement text as a single comprehensive statement, with no additional explanation, formatting, or metadata."""


def create_user_prompt(case_data: List[Dict]) -> str:
    """Create the user prompt with case file information"""
    case_info = case_data[0]  # Get case metadata from first record
    
    prompt = f"""Analyze this OpenFOAM case and generate a realistic user requirement:

CASE METADATA:
- Case Name: {case_info['case_name']}
- Domain: {case_info['case_domain']}
- Category: {case_info['case_category']}
- Solver: {case_info['case_solver']}

CASE FILES ANALYSIS:
"""
    
    # Group files by folder for better organization
    files_by_folder = defaultdict(list)
    for record in case_data:
        files_by_folder[record['folder_name']].append(record)
    
    # Add file contents organized by folder
    for folder_name, files in files_by_folder.items():
        prompt += f"\n=== {folder_name}/ ===\n"
        for record in files:
            file_name = record['file_name']
            file_content = record['file_content']
            prompt += f"\n--- {file_name} ---\n{file_content}\n"
    
    prompt += """

TASK:
Based on the case files above, extract the key simulation parameters and generate a realistic user_requirement that an engineer would specify when requesting this simulation. Focus on:

1. Simulation type and solver
2. Domain geometry and dimensions
3. Boundary conditions for all patches
4. Time settings (timestep, end time, output frequency)
5. Physical properties (viscosity, density, temperature, etc.)
6. Grid/mesh specifications
7. Algorithm settings (PIMPLE, SIMPLE, turbulence models, etc.)

Generate a single, comprehensive user_requirement statement that captures all essential simulation parameters."""
    
    return prompt


def process_cases(grouped_data: Dict[str, List[Dict]], llm_service: LLMService, output_path: Path):
    """Process each case and generate user requirements"""
    results = []
    total_cases = len(grouped_data)
    
    print(f"Processing {total_cases} cases...")
    
    for i, (case_name, case_data) in tqdm(enumerate(grouped_data.items(), 1)):
        print(f"Processing case {i}/{total_cases}: {case_name}")
        
        try:
            # Create prompts
            system_prompt = create_system_prompt()
            user_prompt = create_user_prompt(case_data)
            
            # Invoke LLM
            user_requirement = llm_service.invoke(
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            # Create result record
            case_info = case_data[0]  # Get metadata from first record
            result = {
                'case_name': case_name,
                'case_domain': case_info['case_domain'],
                'case_category': case_info['case_category'],
                'case_solver': case_info['case_solver'],
                'user_requirement': str(user_requirement).strip(),
                'file_count': len(case_data),
                'files': [{'folder_name': r['folder_name'], 'file_name': r['file_name']} for r in case_data]
            }
            
            results.append(result)
            
            # Save progress after each case
            with open(output_path, 'w', encoding='utf-8') as f:
                for result_record in results:
                    json.dump(result_record, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"  ✓ Generated user requirement for {case_name}")

        except Exception as e:
            print(f"  ❌ Error processing {case_name}: {str(e)}")
            continue
    
    return results


def main():

    input_path = Path(__file__).parent / 'data' / 'parsed_openfoam_cases.jsonl'
    output_path = Path(__file__).parent / 'data' / 'foamgpt_user_requirements.jsonl'
    
    print(f"📂 Input: {input_path}")
    print(f"💾 Output: {output_path}")
    print()
    
    print("Initializing LLM service...")

    config = Config()
    print(f"Config: {config}")
    llm_service = LLMService(config)
    
    # Load and process data
    print("Loading data...")
    data = load_jsonl_data(input_path)
    print(f"Loaded {len(data)} records")
    
    print("Grouping by case name...")
    grouped_data = group_by_case_name(data)
    print(f"Found {len(grouped_data)} unique cases")
    
    # Process cases
    results = process_cases(grouped_data, llm_service, output_path)
    
    # Print statistics
    stats = llm_service.get_stats()
    print(f"\n📊 Processing Summary:")
    print(f"   Total cases processed: {len(results)}")
    print(f"   LLM calls made: {stats['total_calls']}")
    print(f"   Failed calls: {stats['failed_calls']}")
    print(f"   Total tokens used: {stats['total_tokens']}")
    print(f"   Prompt tokens: {stats['total_prompt_tokens']}")
    print(f"   Completion tokens: {stats['total_completion_tokens']}")
    
    print(f"\n✅ Done! User requirements saved to: {output_path}")

if __name__ == "__main__":
    main()
