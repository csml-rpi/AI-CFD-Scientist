# config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    max_loop: int = 10
    
    batchsize: int = 10
    searchdocs: int = 2
    run_times: int = 1  # current run number (for directory naming)
    database_path: str = Path(__file__).resolve().parent.parent / "database"
    run_directory: str = Path(__file__).resolve().parent.parent / "runs"
    case_dir: str = ""
    max_time_limit: int = 3600 # Max time limit after which the openfoam run will be terminated, in seconds
    # Input writer generation mode:
    # - "sequential_dependency": generate files sequentially; use already-generated files as context to enforce consistency.
    # - "parallel_no_context": generate files in parallel without cross-file context (faster, may need more reviewer iterations).
    input_writer_generation_mode: str = "sequential_dependency"
    model_provider: str = "bedrock"# [openai, ollama, bedrock]
    # model_version should be in ["gpt-5-mini", "deepseek-r1:32b-qwen-distill-fp16", "qwen2.5:32b-instruct"]
    model_version: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    temperature: float = 1
    
    # Embedding Configuration
    embedding_provider: str = "openai" # [openai, huggingface, ollama]
    embedding_model: str = "text-embedding-3-small" # e.g. "text-embedding-3-small", "text-embedding-3-large", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B"
