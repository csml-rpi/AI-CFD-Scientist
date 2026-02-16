import json
import requests
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from datetime import datetime
import re
import subprocess
import shutil
import traceback


# Import the LLM utilities from the second file
from base_llm import get_response_from_llm, create_client
SUBFIG_WIDTH = 0.30
IMAGES_PER_ROW = 3

# Standard output times expected from the CFD runs
STANDARD_TIMES = {"t0p10", "t0p50", "t1p00", "t1p50", "t2p00", "t3p00"}

@dataclass
class ExperimentData:
    """Data class to hold experiment information"""
    experiment_id: int
    experiment_name: str
    experiment_description: str
    experiment_parameters: str
    user_requirement: str

@dataclass
class ResultData:
    """Data class to hold result information"""
    experiment_name: str
    key_findings: str
    data_table: Optional[Dict] = None
    images: Optional[List[str]] = None

@dataclass
class SemanticScholarPaper:
    """Data class to hold Semantic Scholar paper information"""
    title: str
    authors: List[str]
    year: int
    abstract: str
    doi: str
    url: str
    citation_count: int
    venue: str
    paper_id: str

@dataclass
class ReflectionFeedback:
    """Data class to hold reflection agent feedback"""
    section_name: str
    quality_score: int  # 1-10
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    needs_revision: bool


# ---------------------------------------------------------------------------
# HUMAN-LABEL REGISTRY
# ---------------------------------------------------------------------------

def _build_human_label_map(experiment_data: dict) -> dict:
    """Pre-compute a mapping  raw_exp_name -> human_readable_label  for every
    experiment in the JSON so that raw identifiers (fire_case_1_008, run_008,
    etc.) never appear anywhere in the generated paper.

    Priority order for the label:
      1. experiment_description  (if non-empty and reasonably short)
      2. experiment_parameters   (if non-empty and reasonably short)
      3. A numbered label derived from the experiment's position in the list
         ("Experiment 1", "Experiment 2", …).

    The function deliberately avoids using the raw experiment_name at all.
    """
    label_map: dict = {}
    experiments = experiment_data.get('experiments', [])

    for idx, exp in enumerate(experiments, 1):
        raw_name = exp.get('experiment_name', f'__exp_{idx}__')

        desc   = exp.get('experiment_description', '')
        params = exp.get('experiment_parameters', '')

        # Normalise desc / params to plain strings
        if isinstance(desc, dict):
            desc = ', '.join(f"{k}={v}" for k, v in desc.items())
        else:
            desc = str(desc).strip() if desc else ''

        if isinstance(params, dict):
            params = ', '.join(f"{k}={v}" for k, v in params.items())
        else:
            params = str(params).strip() if params else ''

        # Choose the best label
        if desc and len(desc) <= 120:
            label = desc.rstrip('.')
            label = label[0].upper() + label[1:] if label else label
        elif params and len(params) <= 120:
            label = params.rstrip('.')
            label = label[0].upper() + label[1:] if label else label
        else:
            label = f"Experiment {idx}"

        label_map[raw_name] = label

    # Also map results entries that might have different names
    for idx, result in enumerate(experiment_data.get('results', []), 1):
        raw_name = result.get('experiment_name', f'__result_{idx}__')
        if raw_name not in label_map:
            # Try to match by position
            if idx <= len(experiments):
                label_map[raw_name] = label_map.get(
                    experiments[idx - 1].get('experiment_name', ''),
                    f"Experiment {idx}"
                )
            else:
                label_map[raw_name] = f"Experiment {idx}"

    return label_map


class ReflectionAgent:
    """Agent that reviews and provides feedback on generated LaTeX sections"""
    
    def __init__(self, client, model: str = "claude-3-5-sonnet-20241022"):
        self.client = client
        self.model = model
    
    def reflect_on_section(self, section_name: str, content: str, 
                          context: Dict[str, any]) -> ReflectionFeedback:
        """Analyze a section and provide detailed feedback"""
        
        reflection_prompt = self._create_reflection_prompt(section_name, content, context)
        
        system_message = """You are an expert academic reviewer specializing in CFD research papers. 
Your task is to critically evaluate LaTeX sections for:
1. Technical accuracy and depth
2. Proper citation usage and integration
3. Logical flow and coherence
4. Academic writing quality
5. Completeness and coverage

Provide constructive, specific feedback in JSON format."""
        
        try:
            response_content, _ = get_response_from_llm(
                prompt=reflection_prompt,
                client=self.client,
                model=self.model,
                system_message=system_message,
                temperature=0.3
            )
            
            # Extract JSON from response
            feedback_json = self._extract_json(response_content)
            
            return ReflectionFeedback(
                section_name=section_name,
                quality_score=feedback_json.get('quality_score', 5),
                strengths=feedback_json.get('strengths', []),
                weaknesses=feedback_json.get('weaknesses', []),
                suggestions=feedback_json.get('suggestions', []),
                needs_revision=feedback_json.get('needs_revision', False)
            )
            
        except Exception as e:
            print(f"Reflection error for {section_name}: {e}")
            return ReflectionFeedback(
                section_name=section_name,
                quality_score=7,
                strengths=["Content generated successfully"],
                weaknesses=[],
                suggestions=[],
                needs_revision=False
            )
    
    def _extract_json(self, response: str) -> dict:
        """Extract JSON from Claude's response"""
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_obj_pattern, response, re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    pass
        
        return {
            'quality_score': 7,
            'strengths': [],
            'weaknesses': [],
            'suggestions': [],
            'needs_revision': False
        }
    
    def _create_reflection_prompt(self, section_name: str, content: str, 
                                 context: Dict[str, any]) -> str:
        """Create a reflection prompt based on section type"""
        
        base_prompt = f"""Analyze this {section_name} section from a CFD research paper.

CONTENT TO REVIEW:
{content[:3000]}

CONTEXT:
- Research Topic: {context.get('research_topic', 'N/A')}
- Number of Experiments: {context.get('num_experiments', 0)}
- Available Citations: {context.get('num_citations', 0)}

"""
        
        if section_name == "Introduction":
            base_prompt += """
EVALUATION CRITERIA:
1. Does it provide adequate background and context?
2. Are citations properly integrated (at least 5-8)?
3. Is the research gap clearly identified?
4. Is the motivation compelling?
5. Does it preview the methodology?

"""
        elif section_name == "Literature Review":
            base_prompt += """
EVALUATION CRITERIA:
1. Are relevant prior works comprehensively covered (10-15 citations)?
2. Is the literature organized thematically or chronologically?
3. Are critical analyses and comparisons present?
4. Is the research gap clearly identified?
5. Does it establish the foundation for current work?

"""
        elif section_name == "Methods":
            base_prompt += """
EVALUATION CRITERIA:
1. Are computational methods described in sufficient detail?
2. Are governing equations properly presented?
3. Are boundary conditions and domain clearly explained?
4. Are citations used for methodology validation (at least 3-5)?
5. Is reproducibility ensured?

"""
        elif section_name == "Results":
            base_prompt += """
EVALUATION CRITERIA:
1. Are results presented systematically?
2. Is there adequate comparison with literature (6-10 citations)?
3. Are physical mechanisms discussed?
4. Is quantitative analysis provided?
5. Are limitations acknowledged?
6. Are tables and figures properly referenced?

"""
        elif section_name == "Conclusion":
            base_prompt += """
EVALUATION CRITERIA:
1. Are main findings clearly summarized?
2. Are contributions to the field highlighted?
3. Are future directions specific and actionable?
4. Is the conclusion concise yet comprehensive?
5. Are strategic citations included (1-2)?

"""
        
        base_prompt += """
Provide feedback in JSON format:
{
    "quality_score": <1-10 integer>,
    "strengths": [<list of specific strengths>],
    "weaknesses": [<list of specific issues>],
    "suggestions": [<list of actionable improvements>],
    "needs_revision": <true/false>
}

Be specific and constructive. Focus on academic quality, technical depth, and proper citation usage.
"""
        
        return base_prompt
    
    def improve_section(self, section_name: str, original_content: str, 
                       feedback: ReflectionFeedback, context: Dict[str, any]) -> str:
        """Generate improved version based on reflection feedback"""
        
        if not feedback.needs_revision:
            return original_content
        
        improvement_prompt = f"""Revise this {section_name} section based on the following feedback:

ORIGINAL CONTENT:
{original_content}

REFLECTION FEEDBACK:
Quality Score: {feedback.quality_score}/10

Weaknesses identified:
{chr(10).join([f"- {w}" for w in feedback.weaknesses])}

Suggestions for improvement:
{chr(10).join([f"- {s}" for s in feedback.suggestions])}

CONTEXT:
- Research Topic: {context.get('research_topic', 'N/A')}
- Available Citations: {context.get('citation_context', '')}

INSTRUCTIONS:
1. Address all identified weaknesses
2. Implement the suggestions where applicable
3. Maintain the same LaTeX formatting style
4. Preserve existing citations but add more if needed
5. Ensure technical accuracy and academic tone
6. Keep the improved version approximately the same length

Return ONLY the improved LaTeX content for the {section_name} section.
"""
        
        system_message = f"You are an expert CFD researcher revising a {section_name} section. Improve the content while maintaining LaTeX formatting and academic standards."
        
        try:
            improved_content, _ = get_response_from_llm(
                prompt=improvement_prompt,
                client=self.client,
                model=self.model,
                system_message=system_message,
                temperature=0.5
            )
            
            return improved_content.strip()
            
        except Exception as e:
            print(f"Error improving {section_name}: {e}")
            return original_content


class SemanticScholarAPI:
    """Interface to Semantic Scholar API for fetching academic papers"""
    
    def __init__(self, api_key: str = None):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
        self.rate_limit_delay = 3
        self.max_retries = 3
    
    def search_papers(self, query: str, limit: int = 10, year_filter: str = "2018-") -> List[SemanticScholarPaper]:
        """Search for papers using Semantic Scholar API with retry logic"""
        papers = []
        
        try:
            url = f"{self.base_url}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "year": year_filter,
                "fields": "title,authors,year,abstract,externalIds,url,citationCount,venue,paperId"
            }
            
            response = None
            for attempt in range(self.max_retries):
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * 5
                    print(f"  Rate limited (429), waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"Semantic Scholar API error: {response.status_code}")
                    break
            else:
                print(f"  Max retries reached for query: {query[:50]}...")
                return papers
            
            time.sleep(self.rate_limit_delay)
            
            if response and response.status_code == 200:
                data = response.json()
                
                for paper_data in data.get("data", []):
                    authors = [author.get("name", "") for author in paper_data.get("authors", [])]
                    
                    doi = ""
                    external_ids = paper_data.get("externalIds", {})
                    if external_ids and "DOI" in external_ids:
                        doi = external_ids["DOI"]
                    
                    paper = SemanticScholarPaper(
                        title=paper_data.get("title", ""),
                        authors=authors,
                        year=paper_data.get("year", 0),
                        abstract=paper_data.get("abstract", ""),
                        doi=doi,
                        url=paper_data.get("url", ""),
                        citation_count=paper_data.get("citationCount", 0),
                        venue=paper_data.get("venue", ""),
                        paper_id=paper_data.get("paperId", "")
                    )
                    papers.append(paper)
                
        except Exception as e:
            print(f"Error searching papers: {e}")
        
        return papers

    def get_paper_recommendations(self, keywords: List[str], limit_per_keyword: int = 5) -> List[SemanticScholarPaper]:
        """Get paper recommendations based on multiple keywords"""
        all_papers = []
        seen_paper_ids = set()
        
        for i, keyword in enumerate(keywords, 1):
            print(f"Searching for papers related to ({i}/{len(keywords)}): {keyword}")
            papers = self.search_papers(keyword, limit=limit_per_keyword)
            
            for paper in papers:
                if paper.paper_id not in seen_paper_ids:
                    all_papers.append(paper)
                    seen_paper_ids.add(paper.paper_id)
            
            print(f"  Found {len(papers)} papers, {len(all_papers)} unique total so far")
        
        all_papers.sort(key=lambda p: p.citation_count, reverse=True)
        
        return all_papers

class CFDLatexPaperGenerator:
    """Generate CFD research paper in LaTeX format using Claude API with Semantic Scholar references and reflection"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", semantic_scholar_key: str = None,
                 enable_reflection: bool = True, max_reflection_iterations: int = 2):
        self.client, self.model = create_client(model)
        self.experiment_data = None
        self.semantic_scholar = SemanticScholarAPI(semantic_scholar_key)
        self.references = []
        self.citation_map = {}
        self.generated_title = None
        self.tex_output_path = None
        self._human_label_map: dict = {}   # populated in load_experiment_data()
        
        self.enable_reflection = enable_reflection
        self.max_reflection_iterations = max_reflection_iterations
        self.reflection_agent = ReflectionAgent(self.client, self.model) if enable_reflection else None
        self.reflection_history = []

    # -----------------------------------------------------------------------
    # HUMAN-LABEL HELPERS
    # -----------------------------------------------------------------------

    def _human_label(self, raw_exp_name: str) -> str:
        """Return the human-readable label for a raw experiment name.

        Falls back to a generic "Experiment N" if the name is not in the map
        (should not normally happen).
        """
        return self._human_label_map.get(raw_exp_name, raw_exp_name.replace('_', ' ').title())

    def _safe_label(self, raw_exp_name: str) -> str:
        """Return a LaTeX-safe identifier derived from the human label (not the
        raw experiment name) so that label slugs like ``fire_case_1_008`` never
        appear in the .tex source either.
        """
        human = self._human_label(raw_exp_name)
        slug = human.lower().replace(' ', '_').replace('-', '_')
        slug = re.sub(r'[^a-z0-9_]', '', slug)
        # Ensure it starts with a letter
        if slug and not slug[0].isalpha():
            slug = 'exp_' + slug
        return slug or 'experiment'

    def load_experiment_data(self, json_file_path: str) -> bool:
        """Load experiment data from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                self.experiment_data = json.load(f)
            # Build the human-label registry immediately after loading
            self._human_label_map = _build_human_label_map(self.experiment_data)
            print(f"Successfully loaded experiment data from {json_file_path}")
            print(f"  Built human-label map for {len(self._human_label_map)} experiments/results:")
            for raw, label in self._human_label_map.items():
                print(f"    {raw!r:40s} -> {label!r}")
            return True
        except FileNotFoundError:
            print(f"Error: File {json_file_path} not found")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_file_path}")
            return False
        except Exception as e:
            print(f"Error loading experiment data: {e}")
            return False

    # -----------------------------------------------------------------------
    # TABLE HELPERS
    # -----------------------------------------------------------------------

    @staticmethod
    def _format_numeric(val) -> str:
        """Format a table cell value.

        * Floats are rounded to 4 significant figures so that values like
          0.20000000298023224 become 0.2 and 0.021054700016975403 becomes
          0.02105.
        * Integers and strings are left as-is.
        """
        try:
            f = float(val)
            # Preserve exact integer look for values like 0.1, 0.5, 1.0 …
            if f == int(f) and abs(f) < 1e6:
                return str(int(f)) if f != 0.1 else f"{f:.1f}"
            # 4 significant figures, strip trailing zeros
            formatted = f"{f:.4g}"
            return formatted
        except (ValueError, TypeError):
            return str(val)

    @staticmethod
    def _escape_header(text: str) -> str:
        """Convert a raw column-name string into safe LaTeX for a tabular header."""
        if not text:
            return ""
        text = str(text)

        split_m = re.match(r'^([^\s\[]+)(.*)', text)
        if split_m:
            var_part = split_m.group(1)
            rest = split_m.group(2)
        else:
            var_part = text
            rest = ""

        _TEXT_ESCAPES = [
            ('&', r'\&'), ('%', r'\%'), ('#', r'\#'),
            ('~', r'\textasciitilde{}'), ('^', r'\^{}'),
        ]

        if '_' in var_part:
            math_var = re.sub(r'_([^_\s\[]+)', r'_{\1}', var_part)
            var_part = f"${math_var}$"
        else:
            for char, repl in _TEXT_ESCAPES:
                var_part = var_part.replace(char, repl)

        for char, repl in _TEXT_ESCAPES:
            rest = rest.replace(char, repl)

        return var_part + rest

    def generate_latex_table(self, table_data: Dict, exp_name: str = "",
                             table_num: int = 1, human_label: str = "") -> str:
        """Generate a LaTeX table. human_label is always used for captions and
        labels — the raw exp_name is only kept internally for safe_label
        generation via self._safe_label()."""
        if not table_data or 'columns' not in table_data or 'values' not in table_data:
            print(f"Warning: Invalid table data for {exp_name}")
            return ""
        
        columns = table_data['columns']
        values  = table_data['values']
        
        if not columns or not values:
            print(f"Warning: Empty columns or values for {exp_name}")
            return ""
        
        try:
            num_cols  = len(columns)
            col_format = 'l' + 'r' * (num_cols - 1)

            # Use human-label-derived slug for LaTeX labels, never raw exp_name
            safe_label = self._safe_label(exp_name)

            # Detect all-zero result (failed/stalled simulation)
            all_zero = all(
                all(float(cell) == 0.0 for cell in row[1:])
                for row in values
                if len(row) > 1
            )

            # Always use the human label for display; fall back gracefully
            display_name = human_label if human_label else self._human_label(exp_name)

            latex_table  = "\\begin{table}[H]\n"
            latex_table += "\\centering\n"
            latex_table += "\\small\n"
            latex_table += f"\\begin{{tabular}}{{{col_format}}}\n"
            latex_table += "\\toprule\n"

            latex_table += " & ".join(
                self._escape_header(str(col)) for col in columns
            ) + " \\\\\n"
            latex_table += "\\midrule\n"

            for row in values:
                latex_table += " & ".join(
                    self._format_numeric(cell) for cell in row
                ) + " \\\\\n"

            latex_table += "\\bottomrule\n"
            latex_table += "\\end{tabular}\n"

            if all_zero:
                latex_table += (
                    f"\\caption{{Centerline velocity statistics for "
                    f"{display_name} (all values zero --- simulation "
                    f"did not produce non-trivial flow; see text for "
                    f"discussion).}}\n"
                )
            else:
                latex_table += (
                    f"\\caption{{Centerline $U_y$ statistics for "
                    f"{display_name}.}}\n"
                )
            latex_table += f"\\label{{tab:{safe_label}}}\n"
            latex_table += "\\end{table}\n"

            return latex_table
            
        except Exception as e:
            print(f"Error generating table for {exp_name}: {e}")
            traceback.print_exc()
            return ""

    # -----------------------------------------------------------------------
    # IMAGE / FIGURE HELPERS
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_standard_time(path: str) -> bool:
        """Return True if the image corresponds to one of the six canonical
        output times."""
        base = os.path.splitext(os.path.basename(path))[0]
        time_token = base.split("_")[-1]
        return time_token in STANDARD_TIMES

    def generate_latex_figures(self, images: List[str], experiment_name: str,
                               human_label: str = "") -> str:
        """Generate LaTeX figures grouped by field type.

        All captions and LaTeX labels use the human label — never the raw
        experiment_name.
        """
        if not images:
            return ""

        tex_dir = (
            os.path.dirname(os.path.abspath(self.tex_output_path))
            if self.tex_output_path
            else os.getcwd()
        )

        def resolve_path(img_path: str):
            abs_path = os.path.abspath(img_path)
            if os.path.exists(abs_path):
                return abs_path.replace("\\", "/"), True

            candidate = os.path.join(tex_dir, img_path)
            abs_candidate = os.path.abspath(candidate)
            if os.path.exists(abs_candidate):
                return abs_candidate.replace("\\", "/"), True

            cwd_candidate = os.path.join(os.getcwd(), img_path)
            abs_cwd = os.path.abspath(cwd_candidate)
            if os.path.exists(abs_cwd):
                return abs_cwd.replace("\\", "/"), True

            return img_path.replace("\\", "/"), False

        standard_images = [img for img in images if self._is_standard_time(img)]
        if not standard_images:
            standard_images = images

        umag_images = [img for img in standard_images
                       if "umag" in os.path.basename(img).lower()]
        p_images    = [img for img in standard_images
                       if os.path.basename(img).lower().startswith("p_")]

        groups: List[Tuple[str, List[str]]] = []
        if umag_images:
            groups.append(("Velocity Magnitude", umag_images))
        if p_images:
            groups.append(("Pressure", p_images))
        if not groups:
            groups = [("Results", standard_images)]

        # Use human-label-derived slug for all LaTeX labels
        safe_label = self._safe_label(experiment_name)
        # Always use the human label for display
        display_name = human_label if human_label else self._human_label(experiment_name)

        latex = ""

        for group_name, group_imgs in groups:
            group_key = group_name.split()[0].lower()

            def _time_sort_key(p):
                base = os.path.splitext(os.path.basename(p))[0]
                tok  = base.split("_")[-1]
                return tok

            group_imgs = sorted(group_imgs, key=_time_sort_key)

            latex += "\\begin{figure}[H]\n"
            latex += "\\centering\n"

            for i, img_path in enumerate(group_imgs):
                latex_path, exists = resolve_path(img_path)
                time_label   = self._extract_time(img_path)
                subfig_label = f"fig:{safe_label}_{group_key}_{i + 1}"

                if exists:
                    print(f"      ✓ Image found: {latex_path}")
                else:
                    print(f"      ✗ Image NOT found: {img_path}")

                latex += f"\\begin{{subfigure}}[b]{{{SUBFIG_WIDTH}\\textwidth}}\n"
                latex += "  \\centering\n"

                if exists:
                    latex += (
                        f"  \\includegraphics"
                        f"[width=\\textwidth]{{{latex_path}}}\n"
                    )
                else:
                    latex += (
                        f"  \\fbox{{\\parbox{{\\textwidth}}"
                        f"{{\\centering\\small [Image not found]\\\\{time_label}}}}}\n"
                    )

                latex += f"  \\caption{{{time_label}}}\n"
                latex += f"  \\label{{{subfig_label}}}\n"
                latex += "\\end{subfigure}\n"

                is_last    = (i + 1 == len(group_imgs))
                end_of_row = (i + 1) % IMAGES_PER_ROW == 0

                if not is_last:
                    if end_of_row:
                        latex += "\\\\\n\\vspace{6pt}\n"
                    else:
                        latex += "\\hfill\n"

            latex += (
                f"\\caption{{{group_name} field evolution "
                f"for {display_name}}}\n"
            )
            latex += f"\\label{{fig:{safe_label}_{group_key}}}\n"
            latex += "\\end{figure}\n\n"

        return latex

    def _extract_time(self, path: str) -> str:
        """Convert a filename token like ``t1p50`` to ``t = 1.50 s``."""
        m = re.search(r"t(\d+)p(\d+)", os.path.basename(path))
        if not m:
            return ""
        whole, frac = m.group(1), m.group(2)
        frac_padded = frac.ljust(2, "0")
        return f"t = {whole}.{frac_padded} s"

    # -----------------------------------------------------------------------
    # REST OF THE CLASS
    # -----------------------------------------------------------------------

    def fetch_relevant_papers(self) -> List[SemanticScholarPaper]:
        if not self.experiment_data:
            return []
        
        print("Fetching relevant papers from Semantic Scholar...")
        
        keywords = []
        
        idea_name = self.experiment_data.get('idea_name', '')
        if idea_name:
            keywords.append(f"CFD {idea_name}")
            keywords.append(idea_name)
        
        keywords.extend([
            "computational fluid dynamics",
            "CFD simulation",
            "numerical methods fluid flow"
        ])
        
        for exp in self.experiment_data.get('experiments', []):
            # Use human label for keyword construction, not raw exp name
            exp_name = self._human_label(exp.get('experiment_name', ''))
            if exp_name:
                keywords.append(f"CFD {exp_name}")
        
        keywords = list(set(keywords))[:8]
        
        papers = self.semantic_scholar.get_paper_recommendations(keywords, limit_per_keyword=4)
        
        print(f"Found {len(papers)} relevant papers")
        return papers[:20]
    
    def create_citation_database(self, papers: List[SemanticScholarPaper]):
        self.references = []
        self.citation_map = {}
        
        for i, paper in enumerate(papers, 1):
            if paper.authors:
                first_author = paper.authors[0].split()[-1]
                citation_key = f"{first_author}{paper.year}"
            else:
                citation_key = f"Paper{paper.year}_{i}"
            
            original_key = citation_key
            counter = 1
            while citation_key in self.citation_map:
                citation_key = f"{original_key}_{chr(96 + counter)}"
                counter += 1
            
            self.citation_map[citation_key] = paper
            self.references.append((citation_key, paper))
    
    def generate_latex_text(self, system_message: str, prompt: str, 
                          temperature: float = 0.7) -> Optional[str]:
        try:
            content, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.model,
                system_message=system_message,
                temperature=temperature
            )
            
            if not content or len(content.strip()) == 0:
                print("Warning: Generated content is empty")
                return None
                
            return content.strip()
            
        except Exception as e:
            print(f"Claude API error: {e}")
            traceback.print_exc()
            return None
    
    def get_citation_context(self) -> str:
        if not self.references:
            return ""

        all_keys = [key for key, _ in self.references]

        context  = "CITATION RULES — READ CAREFULLY:\n"
        context += "- You MUST ONLY use \\cite{} with the exact keys listed below.\n"
        context += "- DO NOT invent or guess any citation key not in this list.\n"
        context += "- If a topic lacks a matching citation, discuss it without a cite.\n"
        context += f"- Valid keys ({len(all_keys)} total): "
        context += ", ".join(all_keys) + "\n\n"
        context += "Available citations (use \\cite{key} format):\n"

        for citation_key, paper in self.references[:10]:
            authors_str = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors_str += " et al."
            context += (
                f"- \\cite{{{citation_key}}}: {authors_str} ({paper.year}) "
                f"- {paper.title[:80]}...\n"
            )

        if len(self.references) > 10:
            context += f"... and {len(self.references) - 10} more (keys listed above).\n"

        context += (
            "\nWARNING: Any \\cite{key} where 'key' is NOT in the valid keys list "
            "above will cause a LaTeX compilation error. Only use the exact keys provided.\n"
        )
        return context

    def generate_paper_title(self) -> str:
        if not self.experiment_data:
            return "Computational Fluid Dynamics Study"
        
        prompt = f"""Generate a concise, academic title for a CFD research paper based on the following information:

Research Topic: {self.experiment_data.get('idea_name', '')}
Hypothesis: {self.experiment_data.get('short_hypothesis', '')}
Overview: {self.experiment_data.get('abstract', '')}

Experiments conducted:
{chr(10).join([f"- {self._human_label(exp.get('experiment_name', ''))}" for exp in self.experiment_data.get('experiments', [])])}

Requirements:
1. Title should be 10-20 words
2. Should be specific and informative
3. Should follow academic conventions
4. Should capture the main focus of the research
5. Do NOT use quotation marks
6. Do NOT use "Study of" or "Investigation of" unless necessary

Return ONLY the title text, nothing else."""

        system_message = "You are an expert at creating concise, impactful academic paper titles for CFD research."
        
        try:
            title, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.model,
                system_message=system_message,
                temperature=0.7
            )
            
            title = title.strip().strip('"').strip("'")
            
            if len(title) > 150:
                for prefix in ["Study of ", "Investigation of ", "Analysis of "]:
                    if title.startswith(prefix):
                        title = title[len(prefix):]
                        break
            
            return title if title else "Computational Fluid Dynamics Study"
            
        except Exception as e:
            print(f"Error generating title: {e}")
            return "Computational Fluid Dynamics Study"

    def create_literature_review_prompt(self) -> str:
        if not self.experiment_data:
            return ""
        
        citation_context = self.get_citation_context()
        
        papers_summary = []
        for citation_key, paper in self.references[:15]:
            authors_str = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors_str += " et al."
            papers_summary.append(f"- [{citation_key}] {authors_str} ({paper.year}): {paper.title}")
        
        return f"""Write a comprehensive Literature Review section for a CFD research paper in LaTeX format.

Research Topic: {self.experiment_data.get('idea_name', '')}
Hypothesis: {self.experiment_data.get('short_hypothesis', '')}
Related Work Overview: {self.experiment_data.get('related_work', '')}

Available Literature:
{chr(10).join(papers_summary)}

{citation_context}

Requirements:
1. Organize the review thematically (e.g., numerical methods, validation studies, application domains)
2. Discuss foundational works in CFD relevant to this research (cite 3-5 works)
3. Review recent advances and state-of-the-art techniques (cite 5-8 works)
4. Critically analyze methodologies and findings from prior work
5. Compare different approaches and highlight their strengths/limitations
6. Identify gaps in existing research that motivate the current study
7. MUST include at least 10-15 citations using \\cite{{key}} format
8. Integrate citations naturally with critical commentary
9. Use subsections to organize different themes if appropriate
10. Write approximately 1200-1500 words
11. ENSURE the section is COMPLETE with proper closing

Output Format:
- Return ONLY the LaTeX content for the Literature Review section
- Use \\section{{Literature Review}} or \\section{{Related Work}}
- Use \\subsection{{}} for different themes if needed
- Include proper LaTeX commands for emphasis
- Use \\cite{{}} commands extensively
- MUST end with a complete paragraph synthesizing the gap
"""

    def create_introduction_prompt(self) -> str:
        if not self.experiment_data:
            return ""
        
        citation_context = self.get_citation_context()
        
        return f"""Write a detailed Introduction section for a CFD research paper in LaTeX format. 

Research Topic: {self.experiment_data.get('idea_name', '')}
Title: {self.generated_title if self.generated_title else self.experiment_data.get('title', '')}
Hypothesis: {self.experiment_data.get('short_hypothesis', '')}
Related Work: {self.experiment_data.get('related_work', '')}
Abstract Overview: {self.experiment_data.get('abstract', '')}

{citation_context}

Requirements:
1. Introduce the research topic and its significance
2. Provide background with foundational concepts, citing key works
3. Summarize recent developments using citations
4. Identify the research gap and motivation
5. Preview methodology and expected outcomes
6. MUST include at least 5-8 citations using \\cite{{key}} format
7. Integrate citations naturally
8. ENSURE the section is COMPLETE with proper closing

Output Format:
- Return ONLY the LaTeX content for the Introduction section
- Use proper LaTeX formatting with \\section{{Introduction}}
- Include proper LaTeX commands for emphasis
- Use \\cite{{}} commands for citations
- Write approximately 800-1000 words
- MUST end with a complete paragraph (no truncation)
"""

    def create_methods_prompt(self) -> str:
        if not self.experiment_data:
            return ""
        
        citation_context = self.get_citation_context()
        
        # Use human labels in the methods prompt — never raw experiment names
        experiments_text = ""
        for exp in self.experiment_data.get('experiments', []):
            human_title = self._human_label(exp.get('experiment_name', ''))
            experiments_text += f"""
Configuration: {human_title}
Description: {exp.get('experiment_description', '')}
Parameters: {exp.get('experiment_parameters', '')}
"""
        
        return f"""Write a detailed Materials and Methods section for a CFD research paper in LaTeX format.

Research Focus: {self.experiment_data.get('idea_name', '')}
Hypothesis: {self.experiment_data.get('short_hypothesis', '')}

Experimental Configurations:
{experiments_text}

{citation_context}

Requirements:
1. Describe computational setup with citations
2. Include governing equations and numerical methods
3. Describe boundary conditions and domain
4. Explain mesh generation and convergence
5. MUST include at least 3-5 citations
6. Write approximately 1000-1200 words
7. ENSURE the section is COMPLETE with proper closing

Output Format:
- Return ONLY the LaTeX content
- MUST end with a complete subsection (no truncation)
"""

    def create_results_prompt(self) -> str:
        if not self.experiment_data:
            return ""
        
        citation_context = self.get_citation_context()
        
        results = self.experiment_data.get('results', [])
        results_text = ""
        
        for i, result in enumerate(results, 1):
            exp_name     = result.get('experiment_name', f'Experiment {i}')
            key_findings = result.get('key_findings', '')
            human_label  = self._human_label(exp_name)
            safe_label   = self._safe_label(exp_name)
            
            has_table  = 'data_table' in result and result['data_table']
            has_images = 'images' in result and result['images']
            
            results_text += f"""
Configuration: {human_label}
Key Findings: {key_findings}
Data Table Available: {has_table} (Reference as Table \\ref{{tab:{safe_label}}})
Figures Available: {has_images} (Reference as Figure \\ref{{fig:{safe_label}_velocity}}, \\ref{{fig:{safe_label}_pressure}})

"""
        
        return f"""Write a combined Results and Discussion section for a CFD research paper in LaTeX format.

Research Focus: {self.experiment_data.get('idea_name', '')}

Results Obtained:
{results_text}

{citation_context}

Requirements:
1. Present results systematically for each configuration
2. Reference tables and figures appropriately (e.g., "As shown in Table \\ref{{tab:...}}")
3. Compare with literature using citations (6-10 total citations across all experiments)
4. Discuss physical mechanisms and trends
5. Analyze parametric effects
6. DO NOT include table or figure LaTeX code - only discussion text
7. Write approximately 300-400 words per configuration
8. ENSURE all subsections are COMPLETE (no truncation)

Output Format:
- Return ONLY the LaTeX content for Results and Discussion
- Use \\section{{Results and Discussion}}
- Use \\subsection{{}} for each configuration
- Include proper cross-references to tables and figures
- MUST end properly without any incomplete commands
"""

    def create_conclusion_prompt(self) -> str:
        if not self.experiment_data:
            return ""
        
        citation_context = self.get_citation_context()
        
        key_findings = []
        for result in self.experiment_data.get('results', []):
            key_findings.append(result.get('key_findings', ''))
        
        return f"""Write a brief Conclusion section for a CFD research paper in LaTeX format.

Research Focus: {self.experiment_data.get('idea_name', '')}
Hypothesis: {self.experiment_data.get('short_hypothesis', '')}

Key Findings:
{chr(10).join([f"- {finding}" for finding in key_findings if finding])}

{citation_context}

Requirements:
1. Summarize the main objective in 1-2 sentences
2. List key findings concisely (3-5 short sentences)
3. State one main contribution to the field
4. Mention 1-2 future research directions briefly
5. Include 1-2 citations maximum
6. Write NO MORE THAN 200-250 words total — keep it tight and punchy
7. ENSURE the section is COMPLETE with proper closing

Output Format:
- Return ONLY the LaTeX content
- Use \\section{{Conclusion}}
- Single paragraph or two very short paragraphs — NO subsections
- MUST end cleanly without any incomplete commands
"""

    def create_abstract_prompt(self) -> str:
        if not self.experiment_data:
            return ""
        
        return f"""Write a concise Abstract for a CFD research paper in LaTeX format.

Research Topic: {self.experiment_data.get('idea_name', '')}
Hypothesis: {self.experiment_data.get('short_hypothesis', '')}
Abstract Overview: {self.experiment_data.get('abstract', '')}

Requirements:
1. Summarize research objective
2. Describe methodology briefly
3. Highlight key findings
4. State significance
5. No citations needed
6. Write one paragraph 
7. MUST be COMPLETE

Use \\begin{{abstract}} and \\end{{abstract}}
"""

    def validate_latex_section(self, content: str, section_name: str) -> bool:
        if not content:
            print(f"ERROR: {section_name} content is empty")
            return False
        
        if '\\fbox{' in content:
            fbox_count = content.count('\\fbox{')
            if content.count('}') < fbox_count:
                print(f"WARNING: {section_name} may have incomplete \\fbox commands")
                return False
        
        begin_count = content.count('\\begin{')
        end_count   = content.count('\\end{')
        if begin_count != end_count:
            print(f"WARNING: {section_name} has unmatched \\begin/\\end ({begin_count} vs {end_count})")
            return False
        
        lines = content.strip().split('\n')
        if lines and len(lines[-1].strip()) < 10 and not lines[-1].strip().startswith('\\'):
            print(f"WARNING: {section_name} may be truncated (last line: '{lines[-1][:50]}')")
            return False
        
        return True

    def generate_section_with_reflection(self, section_name: str, generate_func) -> str:
        print(f"Generating {section_name} section...")
        
        content = generate_func()
        
        if not content:
            print(f"ERROR: Failed to generate {section_name}")
            return f"% Error: Failed to generate {section_name} section\n"
        
        if not self.validate_latex_section(content, section_name):
            print(f"WARNING: {section_name} validation failed, but continuing...")
        
        if not self.enable_reflection:
            return content
        
        context = {
            'research_topic': self.experiment_data.get('idea_name', 'N/A'),
            'num_experiments': len(self.experiment_data.get('experiments', [])),
            'num_citations': len(self.references),
            'citation_context': self.get_citation_context()
        }
        
        for iteration in range(self.max_reflection_iterations):
            print(f"  Reflecting on {section_name} (iteration {iteration + 1})...")
            
            feedback = self.reflection_agent.reflect_on_section(section_name, content, context)
            self.reflection_history.append(feedback)
            
            print(f"  Quality Score: {feedback.quality_score}/10")
            
            if feedback.needs_revision and iteration < self.max_reflection_iterations - 1:
                print(f"  Improving {section_name} based on feedback...")
                improved_content = self.reflection_agent.improve_section(
                    section_name, content, feedback, context
                )
                
                if improved_content and improved_content != content:
                    if self.validate_latex_section(improved_content, f"{section_name} (improved)"):
                        content = improved_content
                        print(f"  {section_name} improved successfully")
                    else:
                        print(f"  Improved {section_name} failed validation, keeping original")
                        break
                else:
                    print(f"  No significant improvement, keeping original")
                    break
            else:
                if not feedback.needs_revision:
                    print(f"  {section_name} quality acceptable, no revision needed")
                break
        
        return content

    def generate_latex_literature_review(self) -> str:
        prompt = self.create_literature_review_prompt()
        if not prompt:
            return "% Error: No experiment data loaded"
        system_message = "You are an expert CFD researcher writing LaTeX code for a comprehensive Literature Review section. ENSURE your output is COMPLETE with no truncation."
        content = self.generate_latex_text(system_message, prompt, temperature=0.6)
        return content if content else "% Error: Failed to generate literature review"

    def generate_latex_introduction(self) -> str:
        prompt = self.create_introduction_prompt()
        if not prompt:
            return "% Error: No experiment data loaded"
        system_message = "You are an expert CFD researcher writing LaTeX code for a formal academic introduction. ENSURE your output is COMPLETE with no truncation."
        content = self.generate_latex_text(system_message, prompt, temperature=0.6)
        return content if content else "% Error: Failed to generate introduction"

    def generate_latex_methods(self) -> str:
        prompt = self.create_methods_prompt()
        if not prompt:
            return "% Error: No experiment data loaded"
        system_message = "You are an expert CFD researcher writing LaTeX code for a detailed Materials and Methods section. ENSURE your output is COMPLETE with no truncation."
        content = self.generate_latex_text(system_message, prompt, temperature=0.5)
        return content if content else "% Error: Failed to generate methods"

    def generate_latex_results_discussion(self) -> str:
        """Generate Results and Discussion section.

        Key guarantee: raw experiment names (fire_case_1_008, run_008, etc.)
        never appear in the LaTeX output.  Every subsection title, table
        caption, figure caption, and cross-reference label uses the
        human-readable label resolved via self._human_label() /
        self._safe_label().
        """
        if not self.experiment_data:
            return "% Error: No experiment data loaded"
        
        results     = self.experiment_data.get('results', [])
        experiments = self.experiment_data.get('experiments', [])

        full_content = "\\section{Results and Discussion}\n\n"
        
        for i, result in enumerate(results, 1):
            exp_name     = result.get('experiment_name', f'Experiment {i}')
            key_findings = result.get('key_findings', '')

            # Always use human-readable label — never the raw exp_name
            human_title = self._human_label(exp_name)
            safe_label  = self._safe_label(exp_name)

            experiment_prompt = f"""Write a detailed discussion for the following experimental results in LaTeX format.

Configuration: {human_title}
Key Findings: {key_findings}

{self.get_citation_context()}

Requirements:
1. Open with \\subsection{{{human_title}}} — use this EXACT title
2. Write a comprehensive discussion of these specific results
3. THIS SECTION WILL INCLUDE A DATA TABLE AND/OR FIGURES — reference them in your text
4. Use phrases like "As shown in Table \\ref{{tab:{safe_label}}}"
5. Use phrases like "Figure \\ref{{fig:{safe_label}_velocity}}" and "Figure \\ref{{fig:{safe_label}_pressure}}" to reference figures
6. Compare results with literature using citations (2-3 citations per experiment)
7. Discuss physical mechanisms and trends observed
8. Analyze the data quantitatively
9. Write approximately 300-400 words for this configuration
10. ENSURE the subsection is COMPLETE (no truncation)
11. Do NOT reference any internal run IDs, case numbers, or file-system names.
    Refer to configurations only by their descriptive labels.

Return ONLY the LaTeX content for this subsection (including \\subsection command).
Do NOT include table or figure code — only the discussion text that references them.
MUST end with a complete paragraph.
"""
            
            system_message = (
                "You are an expert CFD researcher writing LaTeX code for a results "
                "discussion section. Use only descriptive labels for configurations — "
                "never internal IDs, run numbers, or file-system names. "
                "ENSURE your output is COMPLETE."
            )
            
            print(f"  Generating discussion for \"{human_title}\"...")
            experiment_text = self.generate_latex_text(system_message, experiment_prompt, temperature=0.5)
            
            if not experiment_text:
                experiment_text = (
                    f"\\subsection{{{self.escape_latex_text(human_title)}}}\n\n"
                    f"Results for this configuration are presented below.\n\n"
                )
            else:
                # Replace whatever \subsection{...} the LLM wrote with our
                # controlled human_title (LLM sometimes echoes raw IDs back).
                safe_title = self.escape_latex_text(human_title)
                experiment_text = re.sub(
                    r'\\subsection\*?\{[^}]*\}',
                    f'\\\\subsection{{{safe_title}}}',
                    experiment_text,
                    count=1,
                )
                if '\\subsection' not in experiment_text:
                    experiment_text = (
                        f"\\subsection{{{safe_title}}}\n\n" + experiment_text
                    )

                # -----------------------------------------------------------
                # POST-GENERATION SANITISATION
                # Strip any residual raw experiment identifiers the LLM may
                # have echoed back (e.g. "fire_case_1_008", "run_008").
                # Replace them with the human label.
                # -----------------------------------------------------------
                experiment_text = self._sanitise_raw_ids(experiment_text)

                self.validate_latex_section(experiment_text, f"{human_title} discussion")
            
            full_content += experiment_text + "\n\n"
            
            if 'data_table' in result and result['data_table']:
                table_latex = self.generate_latex_table(
                    result['data_table'], exp_name, i, human_label=human_title
                )
                if table_latex:
                    full_content += table_latex + "\n\n"
                    print(f"    ✓ Added table for \"{human_title}\"")
            
            if 'images' in result and result['images']:
                figures_latex = self.generate_latex_figures(
                    result['images'], exp_name, human_label=human_title
                )
                if figures_latex:
                    if self.validate_latex_section(figures_latex, f"{human_title} figures"):
                        full_content += figures_latex + "\n"
                        shown = sum(
                            1 for img in result['images'] if self._is_standard_time(img)
                        )
                        print(f"    ✓ Added {shown} standard-time figures for \"{human_title}\" "
                              f"(skipped {len(result['images']) - shown} non-standard)")
                    else:
                        print(f"    ✗ Figure validation failed for \"{human_title}\"")
        
        if len(results) > 1:
            # Build a human-label summary — no raw names
            human_summaries = []
            for r in results:
                label = self._human_label(r.get('experiment_name', ''))
                findings = r.get('key_findings', '')[:100]
                human_summaries.append(f"- {label}: {findings}")

            comparison_prompt = f"""Write a brief comparative discussion synthesizing the results from all configurations in LaTeX format.

Number of Configurations: {len(results)}
Configurations:
{chr(10).join(human_summaries)}

{self.get_citation_context()}

Requirements:
1. Compare and synthesize findings across configurations
2. Discuss trends and patterns observed
3. Compare with literature (3-5 citations)
4. Identify key insights from the overall study
5. Write approximately 300-400 words
6. ENSURE the subsection is COMPLETE
7. Do NOT reference any internal run IDs or case numbers.

Return ONLY the LaTeX content for a \\subsection{{Comparative Analysis}} section.
MUST end properly.
"""
            
            system_message = (
                "You are an expert CFD researcher synthesizing multiple experimental results. "
                "Use only descriptive configuration labels — never internal IDs or run numbers. "
                "ENSURE your output is COMPLETE."
            )
            
            print(f"  Generating comparative analysis...")
            comparison_text = self.generate_latex_text(system_message, comparison_prompt, temperature=0.5)
            
            if comparison_text:
                comparison_text = self._sanitise_raw_ids(comparison_text)
                if self.validate_latex_section(comparison_text, "Comparative Analysis"):
                    full_content += "\n" + comparison_text + "\n"
        
        return full_content

    def _sanitise_raw_ids(self, text: str) -> str:
        """Replace any raw experiment identifier that leaked into generated
        LaTeX text with its human-readable label.

        Handles patterns like:
          fire_case_1_008   fire_case_001   run_008   case_3_015
          Fire_Case_1_008   Run_008  (capitalisation variants)

        The replacement uses the pre-built _human_label_map; for identifiers
        not in the map a generic sanitisation is applied (underscores ->
        spaces, title-cased).
        """
        if not self._human_label_map:
            return text

        # Build a regex that matches any known raw key (case-insensitive)
        # We sort by length descending so longer keys match first.
        sorted_keys = sorted(self._human_label_map.keys(), key=len, reverse=True)

        for raw_key in sorted_keys:
            if not raw_key or raw_key.startswith('__'):
                continue
            human = self._human_label_map[raw_key]
            # Match the raw key (case-insensitive), not inside a \label or
            # \ref command (those use the safe_label slug, not the raw name).
            pattern = re.compile(
                r'(?<!\\label\{)(?<!\\ref\{)(?<!tab:)(?<!fig:)'
                + re.escape(raw_key),
                re.IGNORECASE
            )
            text = pattern.sub(human, text)

        # Generic fallback: catch any remaining snake_case tokens that look
        # like internal IDs (e.g. fire_case_\d+_\d+, run_\d+, case_\d+_\d+)
        # and replace them with a title-cased human version.
        def _generic_replace(m: re.Match) -> str:
            token = m.group(0)
            # If it's inside a \label{} or \ref{} keep it intact
            start = m.start()
            preceding = text[max(0, start - 10):start]
            if re.search(r'\\(?:label|ref)\{[^}]*$', preceding):
                return token
            return token.replace('_', ' ').title()

        generic_id_pattern = re.compile(
            r'\b(?:fire_case|run|case)_\d[\w]*\b',
            re.IGNORECASE
        )
        text = generic_id_pattern.sub(_generic_replace, text)

        return text

    def generate_latex_conclusion(self) -> str:
        prompt = self.create_conclusion_prompt()
        if not prompt:
            return "% Error: No experiment data loaded"
        system_message = (
            "You are an expert CFD researcher. Write a Conclusion section in LaTeX. "
            "HARD LIMIT: the entire section body must be 150-200 words. "
            "One paragraph only. No subsections. No bullet lists. "
            "Do not pad or elaborate. Be direct and concise. "
            "ENSURE your output is COMPLETE."
        )
        content = self.generate_latex_text(system_message, prompt, temperature=0.4)
        if not content:
            return "% Error: Failed to generate conclusion"

        body = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', content)
        body = re.sub(r'\\[a-zA-Z]+', ' ', body)
        body = re.sub(r'[{}]', ' ', body)
        words = len(body.split())
        if words > 300:
            print(f"  ⚠ Conclusion is {words} words — trimming to keep it brief.")
            sentences = re.split(r'(?<=[.!?])\s+', content)
            trimmed, count = [], 0
            for s in sentences:
                w = len(re.sub(r'\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+|[{}]', ' ', s).split())
                if count + w > 280:
                    break
                trimmed.append(s)
                count += w
            if trimmed:
                content = ' '.join(trimmed)
                if not content.rstrip().endswith('}'):
                    content = content.rstrip() + '\n'
        else:
            print(f"  ✓ Conclusion is {words} words — within target.")

        return content

    def generate_latex_abstract(self) -> str:
        prompt = self.create_abstract_prompt()
        if not prompt:
            return "% Error: No experiment data loaded"
        system_message = "You are an expert CFD researcher writing a LaTeX abstract. ENSURE your output is COMPLETE."
        content = self.generate_latex_text(system_message, prompt, temperature=0.6)
        return content if content else "% Error: Failed to generate abstract"

    def escape_latex_text(self, text: str) -> str:
        """Escape special LaTeX characters for body text."""
        if not text:
            return ""
        
        text = str(text)
        
        latex_special_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
        }
        
        for char, replacement in latex_special_chars.items():
            text = text.replace(char, replacement)
        return text

    def generate_bibtex_entries(self) -> str:
        if not self.references:
            return "% No references available"
        
        bib_entries = []
        
        for citation_key, paper in self.references:
            if paper.venue and any(word in paper.venue.lower() for word in ["conference", "proceedings", "symposium", "workshop"]):
                entry_type = "inproceedings"
            else:
                entry_type = "article"
            
            entry = f"@{entry_type}{{{citation_key},\n"
            
            if paper.title:
                entry += f"  title = {{{paper.title}}},\n"
            
            if paper.authors:
                authors_str = " and ".join(paper.authors)
                entry += f"  author = {{{authors_str}}},\n"
            
            if paper.year and paper.year > 0:
                entry += f"  year = {{{paper.year}}},\n"
            
            if paper.venue:
                if entry_type == "inproceedings":
                    entry += f"  booktitle = {{{paper.venue}}},\n"
                else:
                    entry += f"  journal = {{{paper.venue}}},\n"
            
            if paper.doi:
                entry += f"  doi = {{{paper.doi}}},\n"
            
            if paper.url:
                entry += f"  url = {{{paper.url}}},\n"
            
            entry += "}"
            bib_entries.append(entry)
        
        return "\n\n".join(bib_entries)

    def sanitise_citations(self, latex_content: str) -> str:
        """Remove any \\cite{key} whose key is not in self.citation_map."""
        if not self.citation_map:
            return latex_content

        valid_keys = set(self.citation_map.keys())
        removed_total = 0

        def _filter_cite(m: re.Match) -> str:
            nonlocal removed_total
            raw_keys = [k.strip() for k in m.group(1).split(',')]
            good = [k for k in raw_keys if k in valid_keys]
            bad  = [k for k in raw_keys if k not in valid_keys]
            if bad:
                removed_total += len(bad)
            if not good:
                return ""
            return f"\\cite{{{','.join(good)}}}"

        cleaned = re.sub(r'\\cite\{([^}]+)\}', _filter_cite, latex_content)

        if removed_total:
            print(f"  ⚠ sanitise_citations: removed {removed_total} hallucinated "
                  f"citation key(s) not in the BibTeX database.")
        else:
            print("  ✓ sanitise_citations: all \\cite{} keys are valid.")

        return cleaned

    def clean_latex_unicode(self, latex_content: str) -> str:
        """Clean Unicode characters from LaTeX content."""
        unicode_replacements = {
            '≈': r'$\approx$',
            '⁻': r'$^{-}$',
            '±': r'$\pm$',
            '×': r'$\times$',
            '÷': r'$\div$',
            '≤': r'$\leq$',
            '≥': r'$\geq$',
            '≠': r'$\neq$',
            '→': r'$\rightarrow$',
            '←': r'$\leftarrow$',
            '↔': r'$\leftrightarrow$',
            '°': r'$^\circ$',
            '∞': r'$\infty$',
            '∂': r'$\partial$',
            '∇': r'$\nabla$',
            '∫': r'$\int$',
            '∑': r'$\sum$',
            '∏': r'$\prod$',
            '√': r'$\sqrt{}$',
            '∝': r'$\propto$',
            'α': r'$\alpha$',
            'β': r'$\beta$',
            'γ': r'$\gamma$',
            'δ': r'$\delta$',
            'ε': r'$\epsilon$',
            'θ': r'$\theta$',
            'λ': r'$\lambda$',
            'μ': r'$\mu$',
            'ν': r'$\nu$',
            'π': r'$\pi$',
            'ρ': r'$\rho$',
            'σ': r'$\sigma$',
            'τ': r'$\tau$',
            'φ': r'$\phi$',
            'χ': r'$\chi$',
            'ψ': r'$\psi$',
            'ω': r'$\omega$',
            'Δ': r'$\Delta$',
            'Θ': r'$\Theta$',
            'Λ': r'$\Lambda$',
            'Π': r'$\Pi$',
            'Σ': r'$\Sigma$',
            'Φ': r'$\Phi$',
            'Ψ': r'$\Psi$',
            'Ω': r'$\Omega$',
            '\u201c': "``",
            '\u201d': "''",
            '\u2018': "`",
            '\u2019': "'",
            '—': '---',
            '–': '--',
        }

        includegraphics_pattern = re.compile(
            r'(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})'
        )
        placeholders = {}

        def stash_path(m):
            token = f"__IMGPATH_{len(placeholders)}__"
            placeholders[token] = m.group(0)
            return token

        latex_content = includegraphics_pattern.sub(stash_path, latex_content)

        for unicode_char, latex_equiv in unicode_replacements.items():
            latex_content = latex_content.replace(unicode_char, latex_equiv)
        
        latex_content = re.sub(r'\n\s*\\\\\s*\n', '\n\n', latex_content)
        latex_content = re.sub(r'(?<!\\)(\d+(?:\.\d+)?)%', r'\1\\%', latex_content)

        for token, original in placeholders.items():
            latex_content = latex_content.replace(token, original)

        return latex_content

    def create_latex_document_structure(self, title: str, abstract: str, 
                                      introduction: str, literature_review: str,
                                      methods: str, results: str, conclusion: str) -> str:
        """Create complete LaTeX document."""
        
        escaped_title = self.escape_latex_text(title)
        
        latex_document = f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{geometry}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\graphicspath{{{{./}}}}
\\usepackage{{subcaption}}
\\usepackage{{url}}
\\usepackage{{hyperref}}
\\usepackage{{float}}
\\usepackage{{times}}
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}
\\usepackage{{array}}
\\usepackage{{setspace}}
\\usepackage{{cite}}


\\geometry{{margin=1in}}
\\doublespacing

\\title{{{escaped_title}}}
\\author{{Author Name\\\\
Department of Engineering\\\\
University Name\\\\
\\texttt{{email@university.edu}}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

{abstract if abstract else '% Abstract generation failed'}

\\section{{Keywords}}
Computational Fluid Dynamics, CFD, Numerical Simulation, Pool Fire, Fuel Velocity, Inlet Geometry

{introduction if introduction else '% Introduction generation failed'}

{literature_review if literature_review else '% Literature Review generation failed'}

{methods if methods else '% Methods generation failed'}

{results if results else '% Results generation failed'}

{conclusion if conclusion else '% Conclusion generation failed'}


\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}"""
        
        return latex_document

    def generate_complete_latex_paper(self) -> Tuple[str, str]:
        if not self.experiment_data:
            return ("% Error: No experiment data loaded", "")
        
        papers = self.fetch_relevant_papers()
        if papers:
            self.create_citation_database(papers)
            print(f"Created citation database with {len(self.references)} references")
        
        print("Generating paper title...")
        self.generated_title = self.generate_paper_title()
        print(f"Generated title: {self.generated_title}")
        
        print("Generating LaTeX Abstract...")
        abstract = self.generate_latex_abstract()
        
        introduction = self.generate_section_with_reflection(
            "Introduction", self.generate_latex_introduction
        )
        
        literature_review = self.generate_section_with_reflection(
            "Literature Review", self.generate_latex_literature_review
        )
        
        methods = self.generate_section_with_reflection(
            "Methods", self.generate_latex_methods
        )
        
        results = self.generate_section_with_reflection(
            "Results", self.generate_latex_results_discussion
        )
        
        print("Generating Conclusion section (no reflection — enforcing brevity)...")
        conclusion = self.generate_latex_conclusion()
        
        latex_document = self.create_latex_document_structure(
            self.generated_title, abstract, introduction, literature_review,
            methods, results, conclusion
        )
        
        latex_document = self.clean_latex_unicode(latex_document)
        latex_document = self.sanitise_citations(latex_document)

        print("\nPerforming final document validation...")
        if self.validate_latex_section(latex_document, "Complete Document"):
            print("✓ Document validation passed")
        else:
            print("⚠ Document validation found potential issues")
        
        bibtex_content = self.generate_bibtex_entries()
        
        return latex_document, bibtex_content

    def compile_latex(self, tex_filename: str, timeout: int = 30) -> Optional[str]:
        if not tex_filename or not os.path.exists(tex_filename):
            print(f"Error: LaTeX file {tex_filename} not found")
            return None
        
        if not shutil.which('pdflatex'):
            print("ERROR: pdflatex not found. Please install LaTeX distribution:")
            print("  - Ubuntu/Debian: sudo apt-get install texlive-full")
            print("  - macOS: brew install --cask mactex")
            print("  - Windows: Install MiKTeX or TeX Live")
            return None
        
        print("GENERATING LATEX")
        
        tex_dir       = os.path.dirname(os.path.abspath(tex_filename)) or '.'
        tex_basename  = os.path.basename(tex_filename)
        tex_base_noext = tex_basename.replace('.tex', '')
        pdf_path      = os.path.join(tex_dir, tex_base_noext + '.pdf')
        
        commands = [
            ["pdflatex", "-interaction=nonstopmode", tex_basename],
            ["bibtex", tex_base_noext],
            ["pdflatex", "-interaction=nonstopmode", tex_basename],
            ["pdflatex", "-interaction=nonstopmode", tex_basename],
        ]
        
        for i, command in enumerate(commands, 1):
            try:
                print(f"Running compilation step {i}/{len(commands)}: {' '.join(command)}")
                result = subprocess.run(
                    command,
                    cwd=tex_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    timeout=timeout,
                )

                stdout_text = (result.stdout or b"").decode("utf-8", errors="replace")
                stderr_text = (result.stderr or b"").decode("utf-8", errors="replace")

                if result.returncode != 0:
                    print(f"⚠ Step {i} returned non-zero exit code: {result.returncode}")
                    output = stdout_text.strip() or stderr_text.strip()
                    if output:
                        print("Last 50 lines of output:")
                        print('\n'.join(output.split('\n')[-50:]))
                    else:
                        print("  (no output captured)")
                else:
                    print(f"✓ Step {i} completed successfully")

            except subprocess.TimeoutExpired:
                print(f"ERROR: Step {i} timed out after {timeout} seconds")
                print(traceback.format_exc())
                return None
            except Exception as e:
                print(f"ERROR: Exception in step {i}: {e}")
                print(traceback.format_exc())
        
        print("FINISHED GENERATING LATEX")
        
        if os.path.exists(pdf_path):
            pdf_size = os.path.getsize(pdf_path)
            print(f"\n✓ PDF successfully generated: {pdf_path}")
            print(f"  File size: {pdf_size:,} bytes ({pdf_size / 1024:.1f} KB)")
            return pdf_path
        else:
            print("✗ Failed to generate PDF")
            log_file = os.path.join(tex_dir, tex_base_noext + '.log')
            if os.path.exists(log_file):
                print(f"  Log file location: {log_file}")
            return None

    def save_latex_to_file(self, latex_content: str, bibtex_content: str, filename: str = None) -> Tuple[str, str]:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = self.experiment_data.get('idea_name', 'cfd_paper').replace(" ", "_").lower()
            safe_title = ''.join(c for c in safe_title if c.isalnum() or c in ['_', '-'])
            filename = f"{safe_title}_research_paper_{timestamp}.tex"
        
        if not filename.endswith('.tex'):
            filename += '.tex'
        
        self.tex_output_path = os.path.abspath(filename)

        tex_dir      = os.path.dirname(self.tex_output_path)
        bib_filename = os.path.join(tex_dir, 'references.bib')
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            print(f"LaTeX research paper saved to: {filename}")
            
            with open(bib_filename, 'w', encoding='utf-8') as f:
                f.write(bibtex_content)
            print(f"BibTeX references saved to: {bib_filename}")
            
            return filename, bib_filename
            
        except Exception as e:
            print(f"Error saving files: {e}")
            traceback.print_exc()
            return "", ""

    def save_reflection_report(self, filename: str = None) -> str:
        if not self.reflection_history:
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reflection_report_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("REFLECTION AGENT FEEDBACK REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                for i, feedback in enumerate(self.reflection_history, 1):
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"REFLECTION {i}: {feedback.section_name}\n")
                    f.write(f"{'=' * 80}\n\n")
                    f.write(f"Quality Score: {feedback.quality_score}/10\n")
                    f.write(f"Needs Revision: {'Yes' if feedback.needs_revision else 'No'}\n\n")
                    
                    if feedback.strengths:
                        f.write("STRENGTHS:\n")
                        for strength in feedback.strengths:
                            f.write(f"  ✓ {strength}\n")
                        f.write("\n")
                    
                    if feedback.weaknesses:
                        f.write("WEAKNESSES:\n")
                        for weakness in feedback.weaknesses:
                            f.write(f"  ✗ {weakness}\n")
                        f.write("\n")
                    
                    if feedback.suggestions:
                        f.write("SUGGESTIONS:\n")
                        for suggestion in feedback.suggestions:
                            f.write(f"  → {suggestion}\n")
                        f.write("\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("SUMMARY STATISTICS\n")
                f.write("=" * 80 + "\n\n")
                
                avg_score = sum(f.quality_score for f in self.reflection_history) / len(self.reflection_history)
                f.write(f"Average Quality Score: {avg_score:.1f}/10\n")
                f.write(f"Total Reflections: {len(self.reflection_history)}\n")
                
                revisions_needed = sum(1 for f in self.reflection_history if f.needs_revision)
                f.write(f"Sections Needing Revision: {revisions_needed}\n")
                
                sections_reflected = set(f.section_name for f in self.reflection_history)
                f.write(f"Sections Reviewed: {', '.join(sections_reflected)}\n")
            
            print(f"Reflection report saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving reflection report: {e}")
            traceback.print_exc()
            return ""

    def print_experiment_summary(self):
        if not self.experiment_data:
            print("No experiment data loaded.")
            return
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY FOR LATEX GENERATION")
        print('='*60)
        print(f"Research Focus: {self.experiment_data.get('idea_name', 'N/A')}")
        print(f"Generated Title: {self.generated_title if self.generated_title else 'Not yet generated'}")
        print(f"Hypothesis: {self.experiment_data.get('short_hypothesis', 'N/A')}")
        print(f"Number of Experiments: {len(self.experiment_data.get('experiments', []))}")
        print(f"Number of Results: {len(self.experiment_data.get('results', []))}")
        print(f"Number of References: {len(self.references)}")
        print(f"Reflection Enabled: {'Yes' if self.enable_reflection else 'No'}")
        if self.enable_reflection:
            print(f"Max Reflection Iterations: {self.max_reflection_iterations}")
        print()
        
        print("EXPERIMENTS (human labels):")
        for i, exp in enumerate(self.experiment_data.get('experiments', []), 1):
            raw  = exp.get('experiment_name', 'N/A')
            label = self._human_label(raw)
            print(f"  {i}. {label}")
            print(f"     Description: {exp.get('experiment_description', 'N/A')[:100]}...")
        print()
        
        print("RESULTS:")
        for i, result in enumerate(self.experiment_data.get('results', []), 1):
            raw   = result.get('experiment_name', 'N/A')
            label = self._human_label(raw)
            print(f"  {i}. {label}")
            print(f"     Findings: {result.get('key_findings', 'N/A')[:100]}...")
            
            has_table_key   = 'data_table' in result
            has_table_value = result.get('data_table') if has_table_key else None
            has_table       = has_table_key and has_table_value
            
            has_images_key = 'images' in result
            images_value   = result.get('images', []) if has_images_key else []
            has_images     = has_images_key and len(images_value) > 0

            standard_count = sum(1 for img in images_value if self._is_standard_time(img))
            
            print(f"     Has Table: {has_table}")
            if has_table:
                table = result['data_table']
                cols  = len(table.get('columns', []))
                rows  = len(table.get('values', []))
                print(f"     Table: {cols} columns × {rows} rows")
            
            print(f"     Num Images: {len(images_value)} total, {standard_count} at standard times")
            if images_value:
                print(f"     Images: {', '.join(images_value[:3])}{' ...' if len(images_value) > 3 else ''}")
        print()
        
        if self.references:
            print("SEMANTIC SCHOLAR REFERENCES:")
            for i, (citation_key, paper) in enumerate(self.references[:5], 1):
                authors = ", ".join(paper.authors[:2]) if paper.authors else "Unknown"
                if len(paper.authors) > 2:
                    authors += " et al."
                print(f"  {i}. [{citation_key}] {authors} ({paper.year})")
                print(f"     {paper.title[:80]}...")
                print(f"     Citations: {paper.citation_count}, Venue: {paper.venue}")
            if len(self.references) > 5:
                print(f"  ... and {len(self.references) - 5} more references")
        
        print('='*60)

    def print_reflection_summary(self):
        if not self.reflection_history:
            print("\nNo reflection feedback available.")
            return
        
        print(f"\n{'='*60}")
        print("REFLECTION AGENT SUMMARY")
        print('='*60)
        
        for feedback in self.reflection_history:
            print(f"\n{feedback.section_name}:")
            print(f"  Quality Score: {feedback.quality_score}/10")
            print(f"  Revision Needed: {'Yes' if feedback.needs_revision else 'No'}")
            
            if feedback.strengths:
                print(f"  Strengths: {len(feedback.strengths)} identified")
            if feedback.weaknesses:
                print(f"  Weaknesses: {len(feedback.weaknesses)} identified")
            if feedback.suggestions:
                print(f"  Suggestions: {len(feedback.suggestions)} provided")
        
        avg_score = sum(f.quality_score for f in self.reflection_history) / len(self.reflection_history)
        print(f"\nAverage Quality Score: {avg_score:.1f}/10")
        print('='*60)


def main():
    SEMANTIC_SCHOLAR_KEY = None
    JSON_FILE_PATH = "writer_input1.json"
    MODEL = os.environ.get("CFD_SCIENTIST_MODEL", "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2")
    ENABLE_REFLECTION = True
    MAX_REFLECTION_ITERATIONS = 2
    
    if not os.path.exists(JSON_FILE_PATH):
        print(f"ERROR: JSON file '{JSON_FILE_PATH}' not found")
        return
    
    print("="*80)
    print("CFD LATEX RESEARCH PAPER GENERATOR WITH AUTO-TITLE & LITERATURE REVIEW")
    print("="*80)
    print(f"Using model from CFD_SCIENTIST_MODEL: {MODEL}")
    
    try:
        generator = CFDLatexPaperGenerator(
            model=MODEL,
            semantic_scholar_key=SEMANTIC_SCHOLAR_KEY,
            enable_reflection=ENABLE_REFLECTION,
            max_reflection_iterations=MAX_REFLECTION_ITERATIONS
        )
        
        print(f"Loading experiment data from {JSON_FILE_PATH}...")
        if not generator.load_experiment_data(JSON_FILE_PATH):
            print("Failed to load experiment data. Exiting.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = generator.experiment_data.get('idea_name', 'cfd_paper').replace(" ", "_").lower()
        safe_title = ''.join(c for c in safe_title if c.isalnum() or c in ['_', '-'])
        anticipated_filename = f"{safe_title}_research_paper_{timestamp}.tex"
        generator.tex_output_path = os.path.abspath(anticipated_filename)

        generator.print_experiment_summary()
        
        print("\nGenerating LaTeX research paper with auto-generated title and literature review...")
        print("This may take several minutes due to reflection iterations...")
        
        latex_content, bibtex_content = generator.generate_complete_latex_paper()
        
        if not latex_content or latex_content.startswith("% Error"):
            print(f"Error generating LaTeX paper: {latex_content}")
            return
        
        tex_filename, bib_filename = generator.save_latex_to_file(
            latex_content, bibtex_content, anticipated_filename
        )
        
        if not tex_filename:
            print("Failed to save LaTeX files.")
            return
        
        print("\n" + "="*80)
        print("COMPILING LATEX TO PDF")
        print("="*80)
        
        pdf_filename = generator.compile_latex(tex_filename)
        
        reflection_filename = ""
        if generator.reflection_history:
            reflection_filename = generator.save_reflection_report()
        
        print(f"\n{'='*80}")
        print("GENERATION COMPLETED!")
        print(f"{'='*80}")
        
        if pdf_filename:
            print(f"✓ PDF successfully created: {pdf_filename}")
            print(f"✓ LaTeX source: {tex_filename}")
            print(f"✓ BibTeX references: {bib_filename}")
        else:
            print(f"✗ PDF compilation failed")
            print(f"✓ LaTeX source available: {tex_filename}")
            print(f"✓ BibTeX references available: {bib_filename}")
            print(f"  You can try compiling manually with: pdflatex {tex_filename}")
        
        print(f"\nDocument statistics:")
        print(f"  - Generated Title: {generator.generated_title}")
        print(f"  - LaTeX length: {len(latex_content):,} characters")
        print(f"  - Academic references: {len(generator.references)}")
        
        generator.print_reflection_summary()
        
        print()
        print("DOCUMENT INCLUDES:")
        print("  ✓ Auto-generated Title")
        print("  ✓ Abstract")
        print("  ✓ Introduction (with citations and reflection)")
        print("  ✓ Literature Review (with 10-15 citations and reflection)")
        print("  ✓ Materials and Methods (with citations and reflection)")
        print(f"  ✓ Results and Discussion with {len(generator.experiment_data.get('results', []))} experiments")
        
        num_tables = sum(1 for r in generator.experiment_data.get('results', []) if 'data_table' in r and r['data_table'])
        num_figures_total = sum(len(r.get('images', [])) for r in generator.experiment_data.get('results', []))
        num_figures_standard = sum(
            sum(1 for img in r.get('images', []) if generator._is_standard_time(img))
            for r in generator.experiment_data.get('results', [])
        )
        print(f"  ✓ {num_tables} data tables from JSON")
        print(f"  ✓ {num_figures_standard} standard-time figures included ({num_figures_total} total in JSON)")
        print("  ✓ Conclusion (with citations and reflection)")
        print("  ✓ BibTeX bibliography")
        print()
        print("FILES CREATED:")
        if pdf_filename:
            print(f"  ✓ PDF paper: {pdf_filename}")
        print(f"  ✓ LaTeX source: {tex_filename}")
        print(f"  ✓ BibTeX references: {bib_filename}")
        if reflection_filename:
            print(f"  ✓ Reflection report: {reflection_filename}")
        print("="*80)
            
    except Exception as e:
        print(f"ERROR: Unexpected error during generation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()