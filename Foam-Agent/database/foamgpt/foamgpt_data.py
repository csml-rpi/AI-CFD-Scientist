import argparse
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm


def load_jsonl_data(file_path: Path) -> List[Dict]:
    """Load data from JSONL file"""
    print(f"Loading jsonl data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    user_req_jsonl = load_jsonl_data(Path(__file__).parent / 'data' / 'foamgpt_user_requirements.jsonl')
    user_req_dict = {req['case_name']: req['user_requirement'] for req in user_req_jsonl}
    print(f"Loaded {len(user_req_jsonl)} user requirements")

    foamgpt_input_data = load_jsonl_data(Path(__file__).parent / 'data' / 'parsed_openfoam_cases.jsonl')
    print(f"Loaded {len(foamgpt_input_data)} input data")
    print(f"{foamgpt_input_data[0].keys()}")

    output_data = []

    for case_file_data in foamgpt_input_data:
        case_name = case_file_data['case_name']
        file_name = case_file_data['file_name']
        folder_name = case_file_data['folder_name']
        case_solver = case_file_data['case_solver']
        case_domain = case_file_data['case_domain']
        case_category = case_file_data['case_category']

        case_user_requirement = user_req_dict[case_name]


        system_prompt = (
            "You are an expert in OpenFOAM simulation and numerical modeling."
            f"Your task is to generate a complete and functional file named: <file_name>{file_name}</file_name> within the <folder_name>{folder_name}</folder_name> directory. "
            "Before finalizing the output, ensure:\n"
            "- Ensure units and dimensions are correct** for all physical variables.\n"
            f"- Ensure case solver settings are consistent with the user's requirements. Available solvers are: {case_solver}.\n"
            "Provide only the code—no explanations, comments, or additional text."
        )

        user_prompt = (
            f"User requirement: {case_user_requirement}\n"
            "Please ensure that the generated file is complete, functional, and logically sound."
            "Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."
            "When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."
        )

        output_data.append({
            "case_name": case_name,
            "file_name": file_name,
            "folder_name": folder_name,
            "case_solver": case_solver,
            "case_domain": case_domain,
            "case_category": case_category,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "file_content": case_file_data['file_content'],
            "user_requirement": case_user_requirement
        })

    with open(Path(__file__).parent / 'data' / 'foamgpt_all.jsonl', 'w', encoding='utf-8') as f:
        for data in output_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
        
        print(f"Saved {len(output_data)} data to {Path(__file__).parent / 'data' / 'foamgpt_all.jsonl'}")

if __name__ == "__main__":
    main()
    