import os
import json
import yaml
from typing import List, Dict, Any
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

class IdeationAgent:
    """
    Agent for generating candidate CFD simulation ideas based on research papers.
    """
    
    def __init__(self, model: str = None):
        self.topic = "OpenFOAM CFD simulation research"
        self.ideas = []
        self.model = model
        self.client, self.model_id = create_client(model)  # Will use env var or default if None
        self.prompts = self._load_prompts()
        self.messages = []
        print(f"IdeationAgent initialized with model: {self.model_id}")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from the external prompts.yaml file."""
        try:
            prompts_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.yaml")
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = yaml.safe_load(f)
            # print(f"PROMPTS DATA: {prompts_data}")
            return prompts_data.get('IdeationAgent', {})
        except Exception as e:
            print(f"Warning: Could not load prompts from file: {e}")
            return {}
    
    def generate_candidates(self, num_calls) -> List[Dict[str, Any]]:
        '''
        generate num_calls number of ideas         
        '''
        initial_idea_prompt = self.prompts.get('initial_idea_prompt', '')
        if not initial_idea_prompt:
            raise ValueError("Could not load initial_idea_prompt from prompts.yaml")

        print(f"PROMPT in ideation agent: {initial_idea_prompt}")
        all_ideas = []
        
        for call_num in range(num_calls):
            print(f"Making API call {call_num + 1}/{num_calls}...")
            
            # Build system message from previous messages
            system_message = None
            user_messages = []
            
            for msg in self.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "user":
                    user_messages.append(msg["content"])
            
            # Combine all previous user messages with the current prompt
            full_prompt = "\n\n".join(user_messages + [initial_idea_prompt])
            
            try:
                content, msg_history = get_response_from_llm(
                    prompt=full_prompt,
                    client=self.client,
                    model=self.model_id,
                    system_message=system_message or "You are an expert CFD researcher generating innovative simulation ideas.",
                    temperature=0.65,
                    print_debug=False
                )
                
                # Update messages with the new exchange
                self.messages.append({"role": "user", "content": initial_idea_prompt})
                self.messages.append({"role": "assistant", "content": content})
                
                print(f"Raw response from call {call_num + 1}:")
                print(content)
                print("-" * 50)
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON object in the response (not array)
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = content[start_idx:end_idx]
                        print(f"Extracted JSON string: {json_str}")
                        idea_from_call = json.loads(json_str)
                    else:
                        # Fallback: try to parse the entire response
                        print(f"Could not find JSON braces, trying to parse entire response")
                        idea_from_call = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Attempted to parse: {content[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else content}")
                    raise ValueError(f"Invalid JSON response: {content}")
                
                # Add the single idea from this call to the total collection
                all_ideas.append(idea_from_call)
                    
                print(f"Call {call_num + 1} generated 1 idea")
                
            except Exception as e:
                print(f"Error in call {call_num + 1}: {e}")
                continue
        
        self.ideas = all_ideas
        print(f"Total ideas generated across {num_calls} calls: {len(all_ideas)}")
        
        return self.ideas
    
    def reflect_on_idea(self, original_idea: Dict[str, Any], relevant_papers: str = "") -> Dict[str, Any]:
        """
        Reflect on and refine a generated idea using message history and research papers.
        
        Args:
            original_idea: The original idea dictionary to refine
            relevant_papers: String containing relevant research papers
        Returns:
            Refined idea dictionary with improvements
        """
        # get reflection prompt template from yaml file 
        reflection_prompt_template = self.prompts.get('reflection_prompt', '')
        if not reflection_prompt_template:
            print("Warning: Could not load reflection prompt from prompts.yaml")
            return original_idea
        
        # format in the original idea and relevant papers 
        try:
            # Use replace instead of format to avoid conflicts with JSON braces
            reflection_prompt = reflection_prompt_template.replace(
                "{original_idea}", json.dumps(original_idea, indent=2)
            ).replace(
                "{papers}", relevant_papers
            )
        except Exception as e:
            print(f"Error formatting reflection prompt: {e}")
            return original_idea
        
        # print(f"PROMPT in reflection agent: {reflection_prompt}")

        # print(f"\n--- Reflecting on and refining idea: {original_idea.get('title', 'Untitled')} ---")
        
        # Build system message from previous messages
        system_message = None
        for msg in self.messages:
            if msg["role"] == "system":
                system_message = msg["content"]
                break
        
        try:
            content, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=self.client,
                model=self.model_id,
                system_message=system_message or "You are an expert CFD researcher refining simulation ideas.",
                temperature=0.3,
                print_debug=False
            )
            
            # Add reflection to message history
            self.messages.append({"role": "user", "content": reflection_prompt})
            self.messages.append({"role": "assistant", "content": content})
            
            # print(f"Raw reflection response:")
            # print(content)
            # print("-" * 50)
            
            # Try to extract JSON from the response
            try:
                # Look for JSON object in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx]
                    print(f"Extracted JSON string: {json_str}")
                    refined_idea = json.loads(json_str)
                else:
                    # Fallback: try to parse the entire response
                    print(f"Could not find JSON braces, trying to parse entire response")
                    refined_idea = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON decode error in reflection: {e}")
                print(f"Attempted to parse: {content[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else content}")
                print("Returning original idea due to parsing error")
                return original_idea
            
            # print(f"Successfully refined idea: {refined_idea.get('title', 'Untitled')}")
            if 'reflection_notes' in refined_idea:
                print(f"Reflection notes: {refined_idea['reflection_notes']}")
            
            return refined_idea
            
        except Exception as e:
            print(f"Error during reflection: {e}")
            return original_idea

    def generate_candidates_with_reflection(self, num_calls: int = 3, relevant_papers: str = "") -> List[Dict[str, Any]]:
        """
        Generate candidate ideas and then reflect on each one to improve them.
        
        Args:
            num_calls: number of ideas to generate 
            relevant_papers: relevant research papers for reflection
        Returns:
            List of refined idea dictionaries
        """

        original_ideas = self.generate_candidates(num_calls)

        print(f"Original ideas of format: {type(original_ideas)}")
        print(f"Original ideas: {original_ideas}")
        
        # refine each idea through reflection individually
        # comment out refining for now  
        return original_ideas
    
        refined_ideas = []
        for i, original_idea in enumerate(original_ideas, 1):
            print(f"\n{'='*60}")
            print(f"REFLECTING ON IDEA {i}/{len(original_ideas)}")
            print(f"{'='*60}")
            
            refined_idea = self.reflect_on_idea(original_idea, relevant_papers)
            refined_ideas.append(refined_idea)
        
        # update to refined ideas 
        self.ideas = refined_ideas
        print(f"\nCompleted reflection on {len(refined_ideas)} ideas")
        
        return refined_ideas
    
    
    '''
    util functions below 
    '''

    def save_ideas(self, ideas: List[Dict[str, Any]], filename: str = None):
        """
        Save generated ideas to a JSON file with a unique timestamp and topic in the filename.
        """
        os.makedirs("data/ideas", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            safe_topic = ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in self.topic).strip().replace(' ', '_')
            filename = f"{safe_topic}_{timestamp}.json"
            filepath = f"data/ideas/{filename}"
        else:
            # Handle filename that might already contain a path
            if '/' in filename or '\\' in filename:
                # If filename already contains a path, use it as is
                if filename.endswith('.json'):
                    base_name = filename[:-5]
                    filepath = f"{base_name}_{timestamp}.json"
                else:
                    filepath = f"{filename}_{timestamp}.json"
            else:
                # If filename is just a name, prepend the data/ideas path
                if filename.endswith('.json'):
                    base_name = filename[:-5]
                    filename = f"{base_name}_{timestamp}.json"
                else:
                    filename = f"{filename}_{timestamp}.json"
                filepath = f"data/ideas/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ideas, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(ideas)} ideas to {filepath}")
        return filepath


def main():
    """
    Main function for running the IdeationAgent from command line.
    
    Usage:
        python agents/idea_ag.py --num-ideas 3 --model "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2"
        python agents/idea_ag.py --num-ideas 5 --model "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2"
        python agents/idea_ag.py --num-ideas 3 --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate CFD simulation ideas using IdeationAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using default Bedrock model
  python agents/idea_ag.py --num-ideas 3
  
  # Using AWS Bedrock with specific ARN
  python agents/idea_ag.py --num-ideas 5 --model "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2"
  
  # Using Bedrock with model ID
  python agents/idea_ag.py --num-ideas 3 --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
  
  # Using OpenAI GPT models
  python agents/idea_ag.py --num-ideas 3 --model "gpt-4o"
        """
    )
    
    parser.add_argument(
        '--num-ideas',
        type=int,
        default=3,
        help='Number of ideas to generate (default: 3)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,  # Will use environment variable or Bedrock default from base_llm
        help='Model to use for generation. Supports Bedrock, OpenAI, and Anthropic models (default: uses CFD_SCIENTIST_MODEL env var or Bedrock ARN)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename for ideas (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--with-reflection',
        action='store_true',
        help='Enable reflection phase to refine generated ideas'
    )
    
    parser.add_argument(
        '--papers',
        type=str,
        default="",
        help='Path to file containing relevant research papers for reflection'
    )
    
    args = parser.parse_args()
    
    # Load papers if provided
    relevant_papers = ""
    if args.papers and os.path.exists(args.papers):
        with open(args.papers, 'r', encoding='utf-8') as f:
            relevant_papers = f.read()
    
    print(f"\n{'='*60}")
    print(f"CFD Ideation Agent")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Number of ideas: {args.num_ideas}")
    print(f"Reflection: {'Enabled' if args.with_reflection else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Initialize agent with specified model
    agent = IdeationAgent(model=args.model)
    
    # Generate ideas
    if args.with_reflection:
        ideas = agent.generate_candidates_with_reflection(
            num_calls=args.num_ideas,
            relevant_papers=relevant_papers
        )
    else:
        ideas = agent.generate_candidates(num_calls=args.num_ideas)
    
    # Save ideas
    filepath = agent.save_ideas(ideas, filename=args.output)
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully generated {len(ideas)} ideas")
    print(f"📄 Saved to: {filepath}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


