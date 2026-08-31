"""FoamAgent's own prompts, verbatim.

Not re-derived, not paraphrased. This is the same text that
``cfd-skills/cfd-foamagent/SKILL.md`` embeds from
``Foam-Agent/src/services/{plan,input_writer,review}.py`` for Mode B's
agent-native path — copied here so Mode A (this deepagents manager) can run
the exact same stages as first-class Python tools instead of needing the
vendored Foam-Agent package's ``services.*`` modules imported at runtime, and
instead of needing Foam-Agent vendored at all for the core plan/write/review
loop. Only RAG retrieval (``rag.py``) still benefits from — but degrades
gracefully without — the vendored FAISS indices.

If FoamAgent's own prompts change, update ``cfd-skills/cfd-foamagent/SKILL.md``
first (it's the documented source of truth for the verbatim text) and mirror
the change here.
"""

from __future__ import annotations

PARSE_SYSTEM_PROMPT = """Please transform the following user requirement into a standard case description using a structured format.
The key elements should include case name, case domain, case category, and case solver.
Note: case domain must be one of {case_domain_list}.
Note: case category must be one of {case_category_list}.
Note: case solver must be one of {case_solver_list}."""

PARSE_USER_PROMPT = """User requirement: {user_requirement}."""

DECOMPOSE_SYSTEM_PROMPT = """You are an experienced Planner specializing in OpenFOAM projects.
Your task is to break down the following user requirement into a series of smaller, manageable subtasks.
For each subtask, identify the file name of the OpenFOAM input file (foamfile) and the corresponding folder name where it should be stored.
Your final output must strictly follow the JSON schema below and include no additional keys or information:

{{
  "subtasks": [
    {{ "file_name": "<string>", "folder_name": "<string>" }}
  ]
}}

Make sure that your output is valid JSON and strictly adheres to the provided schema."""

DECOMPOSE_USER_PROMPT = """User Requirement: {user_requirement}

Reference Directory Structure (similar case): {dir_structure}

{dir_counts_str}

Make sure you generate all the necessary files for the user's requirements.
Do not include any gmsh files like .geo etc. in the subtasks.
Only include blockMesh or snappyHexMesh if the user hasn't requested for gmsh mesh or user isn't using an external uploaded custom mesh.
Please generate the output as structured JSON."""

# NOTE ON THE WRITE SYSTEM PROMPTS
# ---------------------------------
# Upstream FoamAgent names the target file in the system prompt. Here it is
# named only in the user prompt (which already ends with an explicit
# "Generate <file_name>X</file_name> in <folder_name>Y</folder_name>."), so
# the system prompt is identical for every file written in a case.
#
# That is a prompt-caching requirement, not a stylistic preference: a
# provider caches a *prefix* of the whole request, so a single varying token
# anywhere in the system prompt invalidates everything after it — including
# the large tutorial reference repeated in every user message. With the file
# identifiers moved, the system prompt plus the user message's stable head
# form a real reusable prefix (see llm/caching.py: cacheable_human_message).
# No instruction was dropped or reworded; only the per-file identifiers moved
# to the message that already stated them.
INITIAL_WRITE_SYSTEM_PROMPT = """You are an expert in OpenFOAM simulation and numerical modeling.
Your task is to generate a complete and functional OpenFOAM case file, named and located as the user message specifies.
Ensure all required values are present and match with the files content already generated.
Before finalizing the output, ensure:
- All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`).
- Cross-check field names between different files to avoid mismatches.
- Ensure units and dimensions are correct for all physical variables.
- Ensure case solver settings are consistent with the user's requirements. Available solvers are: {case_solver}.
Provide only the code—no explanations, comments, or additional text."""

INITIAL_WRITE_USER_PROMPT = """User requirement: {user_requirement}

Tutorial reference (similar case content):
{tutorial_reference}

Already-written files in this case (for cross-referencing):
{written_files_ctx}

Generate <file_name>{file_name}</file_name> in <folder_name>{folder_name}</folder_name>.
Output only the file body — no markdown fences, no explanations."""

EDIT_WRITE_SYSTEM_PROMPT = """You are an expert in OpenFOAM simulation and numerical modeling.
An existing OpenFOAM case was copied from a baseline; you must EDIT one file in place.
The file to edit is named in the user message.
Apply the MINIMAL changes needed to satisfy the user requirement. Preserve mesh-related entries
(e.g. polyMesh paths, fvSchemes/fvSolution structure) unless the requirement explicitly demands remeshing or solver changes that require it.
Solver context from planner: {case_solver}.
Return only the complete updated file body — no markdown fences, no commentary."""

EDIT_WRITE_USER_PROMPT = """File to edit: <file_name>{file_name}</file_name> in <folder_name>{folder_name}</folder_name>.

Required changes for this file: {changes}

Current file contents:
---
{current_content}
---

Other foamfiles in the case (for cross-referencing):
{written_files_ctx}

Return only the complete updated file body."""

# Where cacheable_human_message splits the rendered user prompt. Everything
# before the marker is identical for every file written in one case (the
# requirement and the retrieved tutorial); everything from the marker on
# changes per call. The marker text itself stays in the prompt — it is the
# literal heading already in the template, used as a split point, not as a
# new instruction.
WRITE_CACHE_SPLIT_MARKER = "Already-written files in this case (for cross-referencing):"
REVIEW_CACHE_SPLIT_MARKER = "<foamfiles>"


COMMAND_SYSTEM_PROMPT = """You are an expert in OpenFOAM. The user will provide a list of available commands.
Your task is to generate only the necessary OpenFOAM commands required to create an Allrun script for the given user case, based on the provided directory structure.
Return only the list of commands—no explanations, comments, or additional text."""

COMMAND_APPENDIX_COPIED_CASE = """
CRITICAL: The case was copied from a baseline and already contains constant/polyMesh (or equivalent mesh).
Do NOT list blockMesh, snappyHexMesh, surfaceFeatureExtract, cartesianMesh, or other mesh generators
unless the user requirement explicitly demands remeshing.
Prefer: optional checkMesh (non-fatal), then the solver from controlDict.
Do not run decomposePar unless the user requires parallel execution."""

COMMAND_APPENDIX_CUSTOM_MESH = """
If custom mesh commands are provided, include them in the appropriate order (typically after blockMesh or instead of blockMesh if custom mesh is used)."""

COMMAND_USER_PROMPT = """Available OpenFOAM commands for the Allrun script: {commands}
Case directory structure: {dir_structure}
User case information: {case_info}
Reference Allrun scripts from similar cases: {allrun_reference}
Generate only the required OpenFOAM command list — no extra text."""

REVIEWER_SYSTEM_PROMPT = """You are an expert in OpenFOAM simulation and numerical modeling.
Your task is to review the provided error logs and diagnose the underlying issues.
You will be provided with a similar case reference, which is a list of similar cases that are ordered by similarity. You can use this reference to help you understand the user requirement and the error.
When an error indicates that a specific keyword is undefined (for example, 'div(phi,(p|rho)) is undefined'), your response must propose a solution that simply defines that exact keyword as shown in the error log.
Do not reinterpret or modify the keyword (e.g., do not treat '|' as 'or'); instead, assume it is meant to be taken literally.
Propose ideas on how to resolve the errors, but do not modify any files directly.
Please do not propose solutions that require modifying any parameters declared in the user requirement, try other approaches instead. Do not ask the user any questions.
The user will supply all relevant foam files along with the error logs, and within the logs, you will find both the error content and the corresponding error command indicated by the log file name."""

REVIEWER_USER_PROMPT = """<similar_case_reference>{tutorial_reference}</similar_case_reference>

{similar_case_advice_block}

<foamfiles>{foamfiles_xml}</foamfiles>

<error_logs>{error_logs}</error_logs>

<user_requirement>{user_requirement}</user_requirement>

{history_text}"""

REWRITE_PLANNER_SYSTEM_PROMPT = """You are an OpenFOAM debugging planner.
Given current foam files, error logs and reviewer analysis, create a minimal rewrite plan.
Output MUST be strict JSON only, with this exact schema:
{{"target_files": [{{"file": "relative/path", "changes": "change1; change2"}}]}}.
Rules:
1) Do not use markdown, backticks, or comments.
2) Use double quotes for all strings.
3) In changes, use short plain text actions separated by semicolons.
4) Do not include parentheses, backticks, or quote characters inside changes text.
5) Do not include run steps; only file edits."""

REWRITE_PLANNER_USER_PROMPT = """<foamfiles>{foamfiles_xml}</foamfiles>
<error_logs>{error_logs}</error_logs>
<review_analysis>{review_analysis}</review_analysis>
<user_requirement>{user_requirement}</user_requirement>
Return strict JSON now with key target_files only."""

# Default case_domain/category/solver lists for Stage 1, used only when the
# caller doesn't supply a narrower FAISS-derived list (see rag.py /
# Foam-Agent/database/raw/openfoam_case_stats.json in the real deployment).
DEFAULT_CASE_DOMAIN_LIST = [
    "incompressible", "compressible", "multiphase", "heatTransfer", "combustion",
    "particleTracking", "stressAnalysis", "electromagnetics", "financial", "DNS",
]
DEFAULT_CASE_CATEGORY_LIST = ["basic", "solver", "verificationAndValidation"]
DEFAULT_CASE_SOLVER_LIST = [
    "simpleFoam", "pimpleFoam", "icoFoam", "interFoam", "rhoPimpleFoam",
    "buoyantSimpleFoam", "reactingFoam", "sonicFoam", "chtMultiRegionFoam",
]
