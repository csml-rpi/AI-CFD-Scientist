#!/usr/bin/env python3
"""
Simple test script to verify Bedrock model integration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.base_llm import create_client, get_response_from_llm

def test_bedrock_simple():
    """Test a simple Bedrock API call."""
    
    print("=" * 60)
    print("Testing Bedrock Model Integration")
    print("=" * 60)
    
    # Test with default Bedrock model (from env var or hardcoded)
    model = None  # Will use CFD_SCIENTIST_MODEL env var or default
    
    print(f"\n1. Creating client...")
    try:
        client, model_id = create_client(model)
        print(f"✅ Client created successfully!")
        print(f"   Model ID: {model_id}")
        print(f"   Client type: {type(client).__name__}")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return
    
    # test_prompt = "Which model are you, specifically which Claude model? You are Claude 4 Sonnet correct?"
    test_prompt = "What are the last ten digits of seven raised to the power seven billion seven hundred seventy-seven million seven hundred seventy-seven thousand seven hundred seventy-seven?"
    system_message = "You are a helpful AI assistant specializing in CFD."
    
    print(f"\n2. Testing API call...")
    print(f"   Prompt: {test_prompt}")
    print(f"   System: {system_message}")
    
    try:
        content, msg_history = get_response_from_llm(
            prompt=test_prompt,
            client=client,
            model=model_id,
            system_message=system_message,
            temperature=0.7,
            print_debug=False
        )
        
        print(f"\n✅ API call successful!")
        print(f"\n📝 Response:")
        print(f"   {content}")
        print(f"\n📊 Message history length: {len(msg_history)}")
        
    except Exception as e:
        print(f"\n❌ API call failed: {e}")
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()
        return
    
    print(f"\n{'=' * 60}")
    print("✅ All tests passed!")
    print("=" * 60)


def test_bedrock_specific_model():
    """Test with a specific Bedrock ARN."""
    
    print("\n" + "=" * 60)
    print("Testing Specific Bedrock ARN")
    print("=" * 60)
    
    # model = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2"
    model = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2"
    
    print(f"\n1. Creating client with ARN...")
    print(f"   ARN: {model}")
    
    try:
        client, model_id = create_client(model)
        print(f"✅ Client created successfully!")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return
    
    # Simple test
    test_prompt = "Explain Reynolds number in 10 words or less."
    
    print(f"\n2. Testing API call...")
    print(f"   Prompt: {test_prompt}")
    
    # get response ##############
    try:
        content, _ = get_response_from_llm(
            prompt=test_prompt,
            client=client,
            model=model_id,
            system_message="You are a CFD expert.",
            temperature=0.5
        )
        
        print(f"\n✅ Response: {content}")
    ##############################    
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return
    
    print(f"\n{'=' * 60}")
    print("✅ Specific model test passed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests
    test_bedrock_simple()
    
    # Uncomment to test specific ARN
    # test_bedrock_specific_model()
