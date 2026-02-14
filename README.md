# CFD-Scientist
A Lifelong Agent for Closed-Loop CFD Experimentation with OpenFOAM

## Prerequisites

### OpenFOAM
CFDScientist requires OpenFOAM v10. Please follow the official installation guide for your operating system:

- Official installation: [https://openfoam.org/version/10/](https://openfoam.org/version/10/)

Verify your installation with:

```bash
echo $WM_PROJECT_DIR
```

The result should be something like:

```text
/opt/openfoam10
```

`WM_PROJECT_DIR` is an environment variable that comes with your OpenFOAM installation, indicating the location of OpenFOAM on your computer.

### conda
https://www.anaconda.com/docs/getting-started/miniconda/install

For a Linux server:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./Miniconda3-latest-Linux-x86_64.sh
```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/csml-rpi/cfd-scientist.git
cd cfd-scientist
```

2. **Set up Python environment:**
```bash
conda env create --file environment.yml
conda activate CFDScientist
```

3. **Set up Foam-Agent dependency:**
```bash
# Foam-Agent is included as a git submodule in this repo
git submodule update --init --recursive
```

4. **Configure environment variables:**
```bash
export OPENAI_API_KEY=your_openai_api_key_here # for embedding in FoamAgent  
```

## Configuration

### Model Selection
There are **two** model knobs in this project:

1) **CFD Scientist (this repo)**: used for ideation/hypothesis/validation/analysis.
Set via env var or CLI:
```bash
export CFD_SCIENTIST_MODEL="gpt-5.3-codex"
# or
python src/main.py --model "gpt-5.3-codex"
```

2) **Foam-Agent (submodule)**: used to generate/run the OpenFOAM case workflow.
Configure in `Foam-Agent/src/config.py` (`model_provider`, `model_version`).

Supported `CFD_SCIENTIST_MODEL` values include:
- AWS Bedrock: `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`, or an ARN (model IDs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)
- Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-5-sonnet-20240620` (models: https://docs.anthropic.com/en/docs/about-claude/models)
- OpenAI Platform: `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini` (models: https://platform.openai.com/docs/models)
- OpenAI Codex subscription: `gpt-5.2.codex`, `gpt-5.3-codex` (Codex: https://platform.openai.com/docs/models)
- Google: `gemini-2.5-flash`, `gemini-2.5-pro` (models: https://ai.google.dev/gemini-api/docs/models)

### Experiment Parameters
Edit `prompts/prompts.yaml` to customize:
- Agent behavior and prompting strategies
- Validation criteria for physics parameters
- Analysis evaluation metrics

## Quick Start

### 1. Basic Experiment Run
```bash
# Run with default settings (manual rerun prompts)
python src/main.py

# Run with automatic retries (analysis → rerun low-scoring cases → re-analyze)
python src/main.py --auto-rerun

# Control retry behavior (threshold + max retry iterations)
python src/main.py --auto-rerun --auto-rerun-threshold 5.0 --max-rerun-iterations 3

# Skip analysis phase
python src/main.py --skip-analysis
```

### 2. Advanced Usage
```bash
# Rerun a specific batch later (uses <batch>/rerun_suggestions.json)
python src/main.py --rerun-batch batch_20251216_140000_abc123

# Auto-execute study recommendations
python src/main.py --auto-execute-recommendations

# Analyze an existing batch with automatic retries
python agents/analysis_ag.py --batch batch_20251216_140000_abc123 --auto-rerun --auto-rerun-threshold 5.0 --max-rerun-iterations 3
```

### 3. Analysis and Paper Generation
```bash
# Generate research paper from experiment batch
python src/latexpaper.py --batch batch_20251216_140000_abc123

# Run analysis agent on existing results (no retries)
python agents/analysis_ag.py --batch batch_20251216_140000_abc123
```

## Workflow Overview

1. **Ideation Phase**: Generate research ideas from literature or user input
2. **Hypothesis Generation**: Convert ideas to OpenFOAM-compatible requirements
3. **Parameter Validation**: Automatically validate and correct simulation parameters
4. **Simulation Execution**: Run OpenFOAM via Foam-Agent integration
5. **Results Analysis**: LLM vision evaluation of simulation mesh convergence, physics, and quality
6. **Iterative Improvement**: Automatic rerun of failed experiments with corrections

## Output Structure

```
data/experiments/
├── batch_YYYYMMDD_HHMMSS_id/
│   ├── sim_001/
│   │   ├── run_001/
│   │   │   ├── user_requirement.txt
│   │   │   └── output/
│   │   └── analysis.txt
│   ├── sim_002/
│   └── analysis_summary.txt
└── ideas/
    └── generated_ideas.json
```

## Example User Requirements

### Basic Lid-Driven Cavity
```
Do an incompressible lid driven cavity flow.
The cavity is a square with dimensions normalized to 1 unit.
Use a 20x20 grid with the top wall moving at 1 m/s.
Run from time 0 to 10 with timestep 0.005.
```

### Advanced Turbulent Flow
```
Perform turbulent flow over a 2D diamond obstacle using pimpleFoam.
Domain: x=[0,15], y=[0,5], z=[-0.5,0.5] (2D with 1 cell in z).
Diamond obstacle centered at (2.5,2.5) with diagonal = 1 unit.
Inlet velocity: 1 m/s, kinematic viscosity: 2×10⁻⁶ m²/s.
```

