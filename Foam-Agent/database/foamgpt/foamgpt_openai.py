#!/usr/bin/env python3
"""
Convert FoamGPT fine-tune data to OpenAI format for supervised fine-tuning.
"""

import json
import os
from pathlib import Path

def convert_to_openai_format(input_file, output_file):
    """
    Convert FoamGPT fine-tune data to OpenAI format.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the original data
                data = json.loads(line.strip())
                
                # Create OpenAI format
                openai_format = {
                    "messages": [
                        {
                            "role": "system",
                            "content": data['system_prompt']
                        },
                        {
                            "role": "user", 
                            "content": data['user_prompt']
                        },
                        {
                            "role": "assistant",
                            "content": data['file_content']
                        }
                    ]
                }
                
                # Write to output file
                outfile.write(json.dumps(openai_format, ensure_ascii=False) + '\n')
                converted_count += 1
                
                # Progress indicator
                if converted_count % 100 == 0:
                    print(f"Converted {converted_count} records...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                error_count += 1
                continue
    
    print(f"\nConversion completed!")
    print(f"Successfully converted: {converted_count} records")
    print(f"Errors encountered: {error_count} records")
    print(f"Output saved to: {output_file}")

def main():
    """Main function to run the conversion."""
    
    # Define input and output paths
    input_file = f"{Path(__file__).parent}/data/foamgpt_train.jsonl"
    output_file = f"{Path(__file__).parent}/data/foamgpt_openai_train.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print(f"Converting {input_file} to OpenAI format...")
    print(f"Output will be saved to: {output_file}")
    
    # Perform conversion
    convert_to_openai_format(input_file, output_file)

        # Define input and output paths
    input_file = f"{Path(__file__).parent}/data/foamgpt_test.jsonl"
    output_file = f"{Path(__file__).parent}/data/foamgpt_openai_test.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print(f"Converting {input_file} to OpenAI format...")
    print(f"Output will be saved to: {output_file}")
    
    # Perform conversion
    convert_to_openai_format(input_file, output_file)

if __name__ == "__main__":
    main()
