import requests
import json
import time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import openai
from abc import ABC, abstractmethod
import re

@dataclass
class Paper:
    title: str
    authors: List[str]
    year: int
    citation_count: int
    abstract: str
    paper_id: str
    url: str
    venue: str = ""

    def format_authors(self, style: str = "citation") -> str:
        if not self.authors:
            return "Unknown"

        if style == "citation":
            if len(self.authors) == 1:
                return self.authors[0]
            elif len(self.authors) == 2:
                return f"{self.authors[0]} and {self.authors[1]}"
            else:
                return f"{self.authors[0]} et al."
        elif style == "full":
            return ", ".join(self.authors)
        elif style == "apa":
            if len(self.authors) == 1:
                return self.authors[0]
            elif len(self.authors) <= 6:
                return ", ".join(self.authors[:-1]) + f", & {self.authors[-1]}"
            else:
                return f"{self.authors[0]} et al."

        return self.format_authors("citation")

    def get_citation(self) -> str:
        return f"({self.format_authors()}, {self.year})"

    def get_apa_reference(self) -> str:
        authors = self.format_authors("apa")
        title = self.title
        venue = f" {self.venue}." if self.venue else ""
        url = f" Retrieved from {self.url}" if self.url else ""
        return f"{authors} ({self.year}). {title}.{venue}{url}"

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class StudyConfiguration:
    """Data class to hold study configuration from JSON"""
    name: str
    long_description: str
    short_description: str
    hypothesis: str
    variables: Dict[str, List[str]]
    metric: str
    pilot: str
    example_prompt: str

    @classmethod
    def from_dict(cls, data: Dict) -> 'StudyConfiguration':
        return cls(
            name=data.get("name", ""),
            long_description=data.get("long_description", ""),
            short_description=data.get("short_description", ""),
            hypothesis=data.get("hypothesis", ""),
            variables=data.get("variables", {}),
            metric=data.get("metric", ""),
            pilot=data.get("pilot", ""),
            example_prompt=data.get("example_prompt", "")
        )

class APIClient(ABC):
    """Abstract base class for API clients"""
    
    def __init__(self):
        self.session = requests.Session()
        self._setup_headers()
    
    @abstractmethod
    def _setup_headers(self):
        """Setup API-specific headers"""
        pass
    
    def _make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Generic request handler with error handling"""
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API request error: {e}")
            return None

class SemanticScholarAPI(APIClient):
    """Handler for Semantic Scholar API interactions - Public access only"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    DEFAULT_FIELDS = ["title", "authors", "year", "citationCount", "abstract", "paperId", "url", "venue"]
    
    def _setup_headers(self):
        """Setup headers for public API access"""
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; Academic Research Tool)"
        })
    
    def search_papers(self, query: str, limit: int = 20, fields: List[str] = None) -> List[Dict]:
        """Search for papers using Semantic Scholar API - public access with rate limiting"""
        fields = fields or self.DEFAULT_FIELDS
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),  # Public API limits
            "fields": ",".join(fields)
        }
        
        # Respect rate limits for public API
        print("Making API request to Semantic Scholar (public access)...")
        time.sleep(1)
        
        result = self._make_request("GET", url, params=params)
        if result and "data" in result:
            print(f"Successfully retrieved {len(result['data'])} papers from API")
            return result["data"]
        else:
            print("Failed to retrieve data from Semantic Scholar API")
            return []
    
    def get_paper_details(self, paper_id: str, fields: List[str] = None) -> Optional[Dict]:
        """Get detailed information about a specific paper"""
        fields = fields or self.DEFAULT_FIELDS
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {"fields": ",".join(fields)}
        
        # Rate limiting for public access
        time.sleep(1)
        return self._make_request("GET", url, params=params)


class OpenAIClient:
    """Handler for OpenAI GPT-4o interactions"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_text(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
        """Generate text using GPT-4o"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

class LaTeXFormatter:
    @staticmethod
    def escape_latex(text: str) -> str:
        """Escape LaTeX special characters in plain text, but not LaTeX commands."""
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

    @staticmethod
    def format_title(title: str) -> str:
        return LaTeXFormatter.escape_latex(title)

    @staticmethod
    def format_authors(authors: List[str]) -> str:
        if not authors:
            return "Unknown Author"
        escaped_authors = [LaTeXFormatter.escape_latex(author) for author in authors]
        if len(escaped_authors) == 1:
            return escaped_authors[0]
        elif len(escaped_authors) == 2:
            return f"{escaped_authors[0]} and {escaped_authors[1]}"
        else:
            return ", ".join(escaped_authors[:-1]) + f", and {escaped_authors[-1]}"

    @staticmethod
    def format_text_content(text: str) -> str:
        if not text:
            return ""

        text = re.sub(r'^#{1,6}\s+(.+)$', r'\\textbf{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        citation_pattern = r'\(([^,)]+),\s*(\d{4})\)'
        text = re.sub(citation_pattern, r'\\cite{\1\2}', text)
        text = LaTeXFormatter.escape_latex(text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

class PaperFilter:
    """Utility class for filtering and sorting papers"""
    
    @staticmethod
    def filter_quality_papers(papers_data: List[Dict], min_year: int = 2015, min_citations: int = 5) -> List[Paper]:
        """Filter papers based on quality criteria"""
        quality_papers = []
        
        for data in papers_data:
            # Handle missing or None values gracefully
            year = data.get("year") or 0
            citation_count = data.get("citationCount") or 0
            abstract = data.get("abstract") or ""
            
            if (year >= min_year and 
                citation_count >= min_citations and
                abstract.strip()):
                
                authors = []
                if data.get("authors"):
                    authors = [author.get("name", "") for author in data["authors"] if author.get("name")]
                
                paper = Paper(
                    title=data.get("title", ""),
                    authors=authors,
                    year=year,
                    citation_count=citation_count,
                    abstract=abstract,
                    paper_id=data.get("paperId", ""),
                    url=data.get("url", ""),
                    venue=data.get("venue", "")
                )
                quality_papers.append(paper)
        
        return quality_papers
    
    @staticmethod
    def categorize_papers(papers: List[Paper], current_year: int = None) -> Dict[str, List[Paper]]:
        """Categorize papers by different criteria"""
        if current_year is None:
            current_year = datetime.now().year
        
        return {
            "foundational": [p for p in papers if p.year <= current_year - 4],
            "recent": [p for p in papers if p.year > current_year - 4],
            "highly_cited": sorted(papers, key=lambda p: p.citation_count, reverse=True),
            "chronological": sorted(papers, key=lambda p: p.year, reverse=True)
        }

class CFDIntroductionAgent:
    """AI Agent for generating CFD research paper introductions using GPT-4o"""
    
    def __init__(self, openai_key: Optional[str] = None):
        self.semantic_api = SemanticScholarAPI()
        self.openai_client = OpenAIClient(openai_key) if openai_key else None
        self.papers: List[Paper] = []
        self.paper_filter = PaperFilter()
        self.latex_formatter = LaTeXFormatter()
        self.study_config: Optional[StudyConfiguration] = None
        
    def load_study_configuration(self, json_file_path: str) -> bool:
        """Load study configuration from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single configuration and array of configurations
            if isinstance(data, list):
                if len(data) > 0:
                    self.study_config = StudyConfiguration.from_dict(data[0])
                    print(f"Loaded study configuration: {self.study_config.name}")
                    if len(data) > 1:
                        print(f"Note: Multiple configurations found. Using the first one. Found {len(data)} configurations.")
                else:
                    print("Error: Empty configuration array")
                    return False
            elif isinstance(data, dict):
                self.study_config = StudyConfiguration.from_dict(data)
                print(f"Loaded study configuration: {self.study_config.name}")
            else:
                print("Error: Invalid JSON structure")
                return False
                
            return True
        except FileNotFoundError:
            print(f"Error: File {json_file_path} not found")
            return False
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return False
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def generate_search_query_from_config(self) -> str:
        """Generate search query based on study configuration"""
        if not self.study_config:
            return "computational fluid dynamics"
        
        # Extract key terms from the configuration
        key_terms = []
        
        # Add terms from short description
        if self.study_config.short_description:
            key_terms.append(self.study_config.short_description)
        
        # Add specific terms from long description
        long_desc = self.study_config.long_description.lower()
        cfd_terms = []
        
        if "wave" in long_desc:
            cfd_terms.append("wave")
        if "heat transfer" in long_desc:
            cfd_terms.append("heat transfer")
        if "turbulence" in long_desc:
            cfd_terms.append("turbulence")
        if "multiphase" in long_desc:
            cfd_terms.append("multiphase")
        if "openfoam" in long_desc:
            cfd_terms.append("openfoam")
        if "overset" in long_desc or "mesh" in long_desc:
            cfd_terms.append("overset mesh")
        if "energy converter" in long_desc:
            cfd_terms.append("energy converter")
        
        # Combine terms
        if cfd_terms:
            return f"computational fluid dynamics {' '.join(cfd_terms[:3])}"  # Limit to 3 main terms
        else:
            return f"computational fluid dynamics {self.study_config.name.replace('-', ' ')}"
    
    def fetch_cfd_papers(self, max_papers: int = 15) -> List[Paper]:
        """Fetch relevant CFD papers based on study configuration"""
        if not self.study_config:
            print("Error: No study configuration loaded. Please load a JSON configuration first.")
            return []
        
        query = self.generate_search_query_from_config()
        print(f"Searching for papers with query: '{query}'")
        
        paper_data = self.semantic_api.search_papers(query, limit=max_papers)
        
        if not paper_data:
            print("No papers found from API")
            return []
        
        # Filter and sort papers
        quality_papers = self.paper_filter.filter_quality_papers(paper_data)
        quality_papers.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
        
        self.papers = quality_papers[:max_papers]
        print(f"Found {len(self.papers)} relevant papers")
        return self.papers
    
    def generate_introduction(self, target_length: int = 300, use_gpt4o: bool = True) -> str:
        """Generate a research paper introduction with citations based on study configuration"""
        
        if not self.study_config:
            return "Error: No study configuration loaded. Please load a JSON configuration first."
        
        if not self.papers:
            return "No papers found. Please fetch papers first using fetch_cfd_papers()."
        
        if use_gpt4o and self.openai_client:
            return self._generate_gpt4o_introduction(target_length)
        else:
            return self._generate_template_introduction()
    
    def generate_materials_methods(self, use_gpt4o: bool = True) -> str:
        """Generate Materials and Methods section based on study configuration"""
        
        if not self.study_config:
            return "Error: No study configuration loaded. Please load a JSON configuration first."
        
        if use_gpt4o and self.openai_client:
            return self._generate_gpt4o_materials_methods()
        else:
            return self._generate_template_materials_methods()
    
    def _generate_gpt4o_materials_methods(self) -> str:
        """Generate Materials and Methods section using GPT-4o with study configuration"""
        
        # Prepare paper information for context
        paper_summaries = []
        for paper in self.papers[:5]:  # Limit to most relevant papers
            summary = {
                "title": paper.title,
                "authors": paper.format_authors("full"),
                "year": paper.year,
                "abstract_snippet": paper.abstract[:150] + "..." if len(paper.abstract) > 150 else paper.abstract,
                "citation": paper.get_citation()
            }
            paper_summaries.append(summary)
        
        # Create prompt for GPT-4o
        system_prompt = self._create_materials_methods_system_prompt()
        user_prompt = self._create_materials_methods_user_prompt(paper_summaries)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        materials_methods = self.openai_client.generate_text(messages, max_tokens=1200, temperature=0.5)
        
        if materials_methods:
            return materials_methods
        else:
            print("Failed to generate Materials and Methods with GPT-4o")
            return self._generate_template_materials_methods()
    
    def _generate_template_materials_methods(self) -> str:
        """Generate a template-based Materials and Methods section using study configuration"""
        if not self.study_config:
            return "Error: No study configuration available"
        
        template = f"""
        This computational study employs OpenFOAM, an open-source CFD software package, to investigate {self.study_config.short_description.lower()}.
        
        {self.study_config.long_description}
        
        The study methodology is designed to test the following hypothesis: {self.study_config.hypothesis}
        
        Independent variables include: {', '.join(self.study_config.variables.get('independent', []))}.
        Dependent variables to be measured are: {', '.join(self.study_config.variables.get('dependent', []))}.
        Control variables are maintained constant: {', '.join(self.study_config.variables.get('controls', []))}.
        
        Performance metrics: {self.study_config.metric}
        
        A pilot study approach will be implemented: {self.study_config.pilot}
        
        The simulations are based on the Navier-Stokes equations for fluid flow, solved using finite volume discretization in OpenFOAM. Appropriate boundary conditions and solver settings are applied based on the specific requirements outlined in the study configuration.
        """
        return template.strip()
    
    def _create_materials_methods_system_prompt(self) -> str:
        return """You are an expert in computational fluid dynamics (CFD) research and technical writing. Your task is to write a clear, detailed, and academically rigorous 'Materials and Methods' section for a research paper that uses OpenFOAM as the primary simulation tool.

        You should:
        - Use precise and formal academic language suitable for peer-reviewed journals.
        - Describe the simulation setup in enough detail for other researchers to reproduce it.
        - Explain choices made for modeling, solvers, boundary conditions, and validation.
        - Include relevant OpenFOAM utilities and terminology (e.g., solvers, mesh tools, parallelization).
        - Organize the section with clear subsections using descriptive paragraph headers.
        - Reference prior studies or data sources if validation or input parameters are based on literature.
        - Include information about variables, metrics, and methodology from the study configuration.
        """
    
    def _create_materials_methods_user_prompt(self, paper_summaries: List[Dict]) -> str:
        """Create user prompt for Materials and Methods generation using study configuration"""
        
        papers_text = "\n".join([
            f"- {p['title']} {p['citation']}: {p['abstract_snippet']}"
            for p in paper_summaries
        ])
        
        variables_text = ""
        if self.study_config.variables:
            variables_text = f"""
Study Variables:
- Independent variables: {', '.join(self.study_config.variables.get('independent', []))}
- Dependent variables: {', '.join(self.study_config.variables.get('dependent', []))}
- Control variables: {', '.join(self.study_config.variables.get('controls', []))}"""
        
        return f"""Write a detailed 'Materials and Methods' section for a computational fluid dynamics (CFD) research paper that uses OpenFOAM.

Study Configuration:
- Study Name: {self.study_config.name}
- Research Focus: {self.study_config.short_description}
- Detailed Description: {self.study_config.long_description}
- Hypothesis: {self.study_config.hypothesis}
{variables_text}
- Performance Metrics: {self.study_config.metric}
- Example Implementation: {self.study_config.example_prompt}

Use the following related papers as background context:
{papers_text}

Include relevant technical subsections (e.g., computational setup, governing equations, geometry and meshing, solver configuration, boundary conditions, post-processing, etc.). 
Clearly describe modeling decisions, solver settings, mesh characteristics, and boundary conditions based on the study configuration.
Mention any specific OpenFOAM solvers, utilities, or techniques mentioned in the example prompt.
Ensure you are always writing good compilable LaTeX code. Common mistakes that should be fixed include:
- LaTeX syntax errors (unenclosed math, unmatched braces, etc.).
- Duplicate figure labels or references.
- Unescaped special characters: & % $ # _ {{ }} ~ ^ \\
Use formal academic language suitable for a peer-reviewed journal and make sure the methodology is detailed enough for another researcher to reproduce the simulation using OpenFOAM.
"""
    
    def _generate_gpt4o_introduction(self, target_length: int) -> str:
        """Generate introduction using GPT-4o with study configuration"""
        categorized_papers = self.paper_filter.categorize_papers(self.papers)
        
        # Prepare paper information for GPT-4o
        paper_summaries = []
        for paper in self.papers[:8]:  # Limit to avoid token limits
            summary = {
                "title": paper.title,
                "authors": paper.format_authors("full"),
                "year": paper.year,
                "citations": paper.citation_count,
                "abstract_snippet": paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                "citation": paper.get_citation()
            }
            paper_summaries.append(summary)
        
        # Create prompt for GPT-4o
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(target_length, paper_summaries)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        introduction = self.openai_client.generate_text(messages, max_tokens=target_length*4)
        
        if introduction:
            return introduction
        else:
            print("Failed to generate introduction with GPT-4o, falling back to template method")
            return self._generate_template_introduction()
    
    def _generate_template_introduction(self) -> str:
        """Generate a template-based introduction when GPT-4o is not available using study configuration"""
        if not self.study_config:
            return "Error: No study configuration available"
        
        categorized = self.paper_filter.categorize_papers(self.papers)
        recent_papers = categorized["recent"][:3]
        foundational_papers = categorized["foundational"][:3]
        
        intro_parts = []
        
        # Opening with study context
        intro_parts.append(f"Computational Fluid Dynamics (CFD) has emerged as a powerful tool for analyzing {self.study_config.short_description.lower()}.")
        
        # Background with citations
        if foundational_papers:
            citations = ", ".join([p.get_citation() for p in foundational_papers[:2]])
            intro_parts.append(f"The foundational principles of CFD in this domain have been extensively developed and validated {citations}.")
        
        # Recent developments
        if recent_papers:
            citations = ", ".join([p.get_citation() for p in recent_papers[:2]])
            intro_parts.append(f"Recent advances in computational methods and numerical techniques have significantly enhanced CFD capabilities in this area {citations}.")
        
        # Research focus and hypothesis
        intro_parts.append(f"This research focuses on {self.study_config.long_description[:200]}...")
        intro_parts.append(f"The study hypothesis is: {self.study_config.hypothesis}")
        
        return " ".join(intro_parts)
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for GPT-4o"""
        return """You are an expert academic writer specializing in computational fluid dynamics (CFD) research. 
        Your task is to write a compelling, well-structured introduction for a CFD research paper that:
        1. Provides context and background on CFD
        2. Incorporates relevant citations naturally
        3. Identifies research gaps and motivations
        4. Uses formal academic language
        5. Flows logically from general concepts to specific research focus
        6. Writes in plain text without markdown formatting or special characters
        7. Incorporates the specific study configuration details provided
        
        Always include proper in-text citations using the format provided for each paper.
        Avoid using markdown headers (#), special formatting, or characters that might conflict with LaTeX."""
    
    def _create_user_prompt(self, target_length: int, paper_summaries: List[Dict]) -> str:
        """Create user prompt for GPT-4o using study configuration"""
        papers_text = "\n".join([
            f"- {p['title']} {p['citation']}: {p['abstract_snippet']}"
            for p in paper_summaries
        ])
        
        return f"""Write a {target_length}-word introduction for a CFD research paper based on the following study configuration:

Study Configuration:
- Study Name: {self.study_config.name}
- Research Focus: {self.study_config.short_description}
- Detailed Description: {self.study_config.long_description}
- Hypothesis: {self.study_config.hypothesis}
- Variables: Independent: {', '.join(self.study_config.variables.get('independent', []))}, Dependent: {', '.join(self.study_config.variables.get('dependent', []))}

Use these relevant papers and their citations:
{papers_text}

Structure the introduction with:
1. Introduction of the research topic and its significance in the context of the study
2. Background with foundational concepts and citations
3. Recent developments and advances relevant to the study focus
4. Research gap and motivation for the current work based on the hypothesis
5. Brief overview of the approach and expected outcomes

Ensure the introduction flows naturally and incorporates the specific research focus and hypothesis from the study configuration.
"""
    
    def generate_latex_document(self, 
                               authors: List[str] = ["Author Name"],
                               target_length: int = 1000,
                               include_materials_methods: bool = True) -> str:
        """Generate complete LaTeX document based on study configuration"""
        
        if not self.study_config:
            return "% Error: No study configuration loaded. Please load a JSON configuration first."
        
        if not self.papers:
            return "% Error: No papers available. Please fetch papers first."
        
        # Generate title from study configuration
        title = f"CFD Analysis: {self.study_config.short_description.title()}"
        
        # Generate content
        print("Generating introduction...")
        introduction = self.generate_introduction(target_length, use_gpt4o=True)
        
        materials_methods = ""
        if include_materials_methods:
            print("Generating Materials and Methods section...")
            materials_methods = self.generate_materials_methods(use_gpt4o=True)
        
        # Generate bibliography entries
        print("Generating bibliography...")
        bibliography = self.generate_latex_bibliography()
        
        # Format content for LaTeX
        latex_title = self.latex_formatter.format_title(title)
        latex_authors = self.latex_formatter.format_authors(authors)
        latex_introduction = self.latex_formatter.format_text_content(introduction)
        latex_materials_methods = self.latex_formatter.format_text_content(materials_methods)
        
        # Create LaTeX document
        latex_document = self._create_latex_document(
            latex_title, 
            latex_authors, 
            latex_introduction, 
            latex_materials_methods,
            bibliography
        )
        
        return latex_document
    
    def _create_latex_document(self, title: str, authors: str, introduction: str, materials_methods: str, bibliography: str) -> str:
        """Create complete LaTeX document structure with study configuration details"""
        
        # Create abstract from study configuration
        abstract = f"This paper presents a computational fluid dynamics (CFD) analysis using OpenFOAM to investigate {self.study_config.short_description.lower()}. {self.study_config.long_description[:200]}... The study tests the hypothesis that {self.study_config.hypothesis}"
        latex_abstract = self.latex_formatter.format_text_content(abstract)
        
        document = f"""\\documentclass[12pt, a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{geometry}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{cite}}
\\usepackage{{url}}
\\usepackage{{hyperref}}
\\usepackage{{float}}
\\usepackage{{times}}

\\geometry{{margin=1in}}

\\title{{{title}}}
\\author{{{authors}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{latex_abstract}
\\end{{abstract}}

\\section{{Introduction}}
{introduction}

"""
        
        if materials_methods:
            document += f"""\\section{{Materials and Methods}}
{materials_methods}

"""
        
        # Add study configuration details in appendix
        variables_text = ""
        if self.study_config.variables:
            indep_vars = ", ".join(self.study_config.variables.get('independent', []))
            dep_vars = ", ".join(self.study_config.variables.get('dependent', []))
            control_vars = ", ".join(self.study_config.variables.get('controls', []))
            variables_text = f"""
\\subsection{{Study Variables}}
\\textbf{{Independent Variables:}} {self.latex_formatter.escape_latex(indep_vars)}

\\textbf{{Dependent Variables:}} {self.latex_formatter.escape_latex(dep_vars)}

\\textbf{{Control Variables:}} {self.latex_formatter.escape_latex(control_vars)}
"""
        
        document += f"""\\section{{Results and Discussion}}
% Results section to be added based on simulation outcomes
This section will present the numerical results obtained from the OpenFOAM simulations, including flow field visualizations, performance metrics, and validation against the study hypothesis: {self.latex_formatter.escape_latex(self.study_config.hypothesis)}

{variables_text}

\\section{{Conclusion}}
% Conclusion section to be added based on results
The computational study demonstrates the effectiveness of CFD analysis in {self.latex_formatter.escape_latex(self.study_config.short_description.lower())}. The results provide valuable insights for engineering applications and future research directions in this domain.

\\section{{Acknowledgments}}
The authors acknowledge the use of OpenFOAM for computational fluid dynamics simulations and the Semantic Scholar API for literature review. This study was configured based on the research framework: {self.latex_formatter.escape_latex(self.study_config.name)}.

{bibliography}

\\end{{document}}"""
        
        return document
    
    def generate_latex_bibliography(self) -> str:
        """Generate LaTeX bibliography from papers"""
        if not self.papers:
            return "% No references available"
        
        bibliography = ["\\begin{thebibliography}{99}"]
        
        # Sort papers alphabetically by first author's last name
        sorted_papers = sorted(self.papers, key=lambda p: p.authors[0].split()[-1] if p.authors else "")
        
        for i, paper in enumerate(sorted_papers, 1):
            # Create citation key
            if paper.authors:
                first_author_last = paper.authors[0].split()[-1]
                cite_key = f"{first_author_last}{paper.year}"
            else:
                cite_key = f"Unknown{paper.year}"
            
            # Format authors for bibliography
            authors_formatted = self.latex_formatter.format_authors(paper.authors)
            title_formatted = self.latex_formatter.format_title(paper.title)
            venue_formatted = self.latex_formatter.escape_latex(paper.venue) if paper.venue else "Journal"
            
            # Create bibliography entry
            bib_entry = f"\\bibitem{{{cite_key}}} {authors_formatted}. ``{title_formatted},'' \\textit{{{venue_formatted}}}, {paper.year}."
            
            if paper.url:
                bib_entry += f" Available: \\url{{{paper.url}}}"
            
            bibliography.append(bib_entry)
        
        bibliography.append("\\end{thebibliography}")
        
        return "\n".join(bibliography)
    
    def generate_references(self) -> str:
        """Generate APA-style references"""
        if not self.papers:
            return "No references available."
        
        # Sort papers alphabetically by first author's last name
        sorted_papers = sorted(self.papers, key=lambda p: p.authors[0].split()[-1] if p.authors else "")
        
        references = ["References\n"]
        for paper in sorted_papers:
            references.append(paper.get_apa_reference())
        
        return "\n".join(references)
    
    def save_latex_document(self, latex_content: str, filename: str = None):
        """Save LaTeX document to file with automatic naming based on study configuration"""
        if filename is None and self.study_config:
            filename = f"{self.study_config.name.replace('-', '_')}_research_paper.tex"
        elif filename is None:
            filename = "cfd_research_paper.tex"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            print(f"LaTeX document saved as {filename}")
        except Exception as e:
            print(f"Error saving LaTeX document: {e}")
    
    def print_paper_summary(self):
        """Print summary of fetched papers"""
        if not self.papers:
            print("No papers available.")
            return
        
        print(f"\n=== CFD Papers Summary ({len(self.papers)} papers) ===")
        if self.study_config:
            print(f"Study: {self.study_config.name}")
            print(f"Focus: {self.study_config.short_description}")
            print()
        
        for i, paper in enumerate(self.papers, 1):
            print(f"{i}. {paper.title}")
            print(f"   Authors: {paper.format_authors('full')}")
            print(f"   Year: {paper.year}, Citations: {paper.citation_count}")
            if paper.venue:
                print(f"   Venue: {paper.venue}")
            if paper.url:
                print(f"   URL: {paper.url}")
            print()
    
    def print_study_configuration(self):
        """Print loaded study configuration details"""
        if not self.study_config:
            print("No study configuration loaded.")
            return
        
        print(f"\n=== Study Configuration ===")
        print(f"Name: {self.study_config.name}")
        print(f"Short Description: {self.study_config.short_description}")
        print(f"Hypothesis: {self.study_config.hypothesis}")
        print(f"Independent Variables: {', '.join(self.study_config.variables.get('independent', []))}")
        print(f"Dependent Variables: {', '.join(self.study_config.variables.get('dependent', []))}")
        print(f"Control Variables: {', '.join(self.study_config.variables.get('controls', []))}")
        print(f"Metrics: {self.study_config.metric}")
        print(f"Pilot Study: {self.study_config.pilot}")
        print(f"\nLong Description: {self.study_config.long_description}")
        print(f"\nExample Prompt: {self.study_config.example_prompt}")
        print()

def create_example_json(filename: str = "study_config.json"):
    """Create an example JSON configuration file"""
    example_config = [
        {
            "name": "wave-structure-interaction-optimization",
            "long_description": "This simulation study aims to evaluate the performance of the updated overset mesh solver in OpenFOAM v2212 for complex wave-structure interaction scenarios. The configuration will model a 3-D hemispherical wave energy converter subjected to focused wave interactions. Key geometrical features of the wave energy converter will be defined, and the simulation will investigate the efficiency and accuracy of the new hole-cutting procedure introduced in the solver. Boundary conditions will include wave generation at the inlet using a focused wave maker and a free surface condition at the outlet. Temporal discretization will be refined to ensure stability and accuracy, setting a timestep of 0.01 seconds with output written every 0.1 seconds. The study will particularly focus on the computational time and accuracy metrics of the results to understand the benefits brought by the improved overset solver.",
            "short_description": "Optimizing the overset solver in OpenFOAM for wave energy converter interactions.",
            "hypothesis": "The updated hole-cutting procedure will improve the accuracy of solutions for the wave energy converter, although computational time will remain unchanged.",
            "variables": {
                "independent": [
                    "wave height",
                    "wave period"
                ],
                "dependent": [
                    "computational time",
                    "solution accuracy"
                ],
                "controls": [
                    "mesh resolution",
                    "solver settings",
                    "timestep"
                ]
            },
            "metric": "Accuracy will be measured by comparing simulation results to analytical predictions and computational time will be logged during the simulation run.",
            "pilot": "Conduct a preliminary simulation with a simplified geometry of the wave energy converter to validate the accuracy of the overset mesh solver.",
            "example_prompt": "Set up a 3D model of a hemispherical wave energy converter in OpenFOAM with focused wave interactions. Apply the updated overset mesh solver with the new hole-cutting procedure. Use wave generation BC at the inlet and free surface BC at the outlet. Set timestep to 0.01s and write output every 0.1s. Adjust wave height and period as independent variables."
        }
    ]
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(example_config, f, indent=2)
        print(f"Example JSON configuration created: {filename}")
    except Exception as e:
        print(f"Error creating example JSON: {e}")

def main():
    # Initialize with OpenAI API key
    OPENAI_KEY = "key"  # Replace with your OpenAI API key
    agent = CFDIntroductionAgent(openai_key=OPENAI_KEY)
    
    # Create example JSON file (uncomment to create)
    # create_example_json("example_study_config.json")
    
    # Load study configuration from JSON file
    json_file_path = "ideas_OpenCFD.json"  # Change this to your JSON file path
    
    if not agent.load_study_configuration(json_file_path):
        print("Failed to load study configuration. Creating example file...")
        create_example_json("example_study_config.json")
        print("Please edit 'example_study_config.json' with your study details and run again.")
        return
    
    # Print loaded configuration
    agent.print_study_configuration()
    
    # Fetch papers based on study configuration
    papers = agent.fetch_cfd_papers(max_papers=12)
    
    if papers:
        # Print paper summary
        agent.print_paper_summary()
        
        # Generate LaTeX document using study configuration
        latex_content = agent.generate_latex_document(
            authors=["Author Name"],  # Replace with actual author names
            target_length=400,
            include_materials_methods=True
        )
        
        # Print LaTeX document
        print("\n" + "="*60)
        print("GENERATED LATEX DOCUMENT")
        print("="*60)
        print(latex_content)
        print("="*60)
        
        # Save to file with automatic naming
        agent.save_latex_document(latex_content)
        
        print(f"\nLaTeX document generated successfully for study: {agent.study_config.name}")
        print("You can now compile this .tex file using pdflatex or your preferred LaTeX compiler.")
        
    else:
        print("No papers found. This could be due to:")
        print("1. Network connectivity issues")
        print("2. Semantic Scholar API rate limiting")
        print("3. Query returning no results")
        print("Please check your connection and try again.")

if __name__ == "__main__":
    main()