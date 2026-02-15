import requests
import json
import time
import os
import glob
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
    """Data class to hold study configuration from text file"""
    name: str
    description: str
    geometry: str
    mesh: str
    boundary_conditions: str
    solver_settings: str
    simulation_time: str
    output_settings: str
    properties: str

    @classmethod
    def from_text(cls, text_content: str) -> 'StudyConfiguration':
        """Parse study configuration from text file"""
        lines = text_content.strip().split('\n')
        
        # Extract key information using simple parsing
        description = text_content.strip()
        
        # Try to extract specific components
        geometry = ""
        mesh = ""
        boundary_conditions = ""
        solver_settings = ""
        simulation_time = ""
        output_settings = ""
        properties = ""
        
        # Parse for specific patterns
        for line in lines:
            line = line.strip()
            if (
                "dimension" in line.lower()
                or "domain" in line.lower()
                or "geometry" in line.lower()
            ):
                geometry += line + " "
            elif "grid" in line.lower() or "cell" in line.lower():
                mesh += line + " "
            elif "wall" in line.lower() or "boundary" in line.lower():
                boundary_conditions += line + " "
            elif "time" in line.lower() and ("step" in line.lower() or "0 to" in line.lower()):
                simulation_time += line + " "
            elif "output" in line.lower() or "results" in line.lower():
                output_settings += line + " "
            elif "viscosity" in line.lower() or "nu" in line.lower():
                properties += line + " "
        
        # Generate a simple name from the first line or description
        first_line = lines[0] if lines else "CFD Study"
        name = first_line[:50].replace(" ", "-").lower()
        
        return cls(
            name=name,
            description=description,
            geometry=geometry.strip(),
            mesh=mesh.strip(),
            boundary_conditions=boundary_conditions.strip(),
            solver_settings="",
            simulation_time=simulation_time.strip(),
            output_settings=output_settings.strip(),
            properties=properties.strip()
        )

@dataclass
class OpenFOAMResults:
    """Data class to hold OpenFOAM simulation results"""
    velocity_data: Dict = None
    pressure_data: Dict = None
    residuals: Dict = None
    convergence_info: str = ""
    simulation_summary: str = ""
    max_velocity: float = 0.0
    min_pressure: float = 0.0
    max_pressure: float = 0.0

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

class OpenFOAMDataParser:
    """Parser for OpenFOAM simulation data"""
    
    @staticmethod
    def parse_openfoam_folder(folder_path: str) -> OpenFOAMResults:
        """Parse OpenFOAM simulation results from folder"""
        results = OpenFOAMResults()
        
        if not os.path.exists(folder_path):
            print(f"Warning: OpenFOAM folder {folder_path} not found")
            return results
        
        # Parse log files for convergence information
        log_files = glob.glob(os.path.join(folder_path, "log.*")) + glob.glob(os.path.join(folder_path, "*.log"))
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    results.convergence_info += f"Log from {os.path.basename(log_file)}:\n"
                    results.convergence_info += OpenFOAMDataParser._extract_convergence_info(content)
            except Exception as e:
                print(f"Error reading log file {log_file}: {e}")
        
        # Look for postProcessing data
        postproc_path = os.path.join(folder_path, "postProcessing")
        if os.path.exists(postproc_path):
            results.simulation_summary += "Post-processing data found. "
        
        # Look for time directories to understand simulation progress
        time_dirs = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                try:
                    float(item)  # Check if directory name is a number (time step)
                    time_dirs.append(float(item))
                except ValueError:
                    continue
        
        if time_dirs:
            time_dirs.sort()
            results.simulation_summary += f"Simulation ran from time {min(time_dirs)} to {max(time_dirs)}. "
            results.simulation_summary += f"Total of {len(time_dirs)} time steps found. "
        
        # Try to extract some basic statistics from the latest time directory
        if time_dirs:
            latest_time = str(max(time_dirs))
            latest_path = os.path.join(folder_path, latest_time)
            
            # Look for U (velocity) file
            u_file = os.path.join(latest_path, "U")
            if os.path.exists(u_file):
                try:
                    with open(u_file, 'r') as f:
                        content = f.read()
                        results.max_velocity = OpenFOAMDataParser._extract_max_velocity(content)
                except Exception as e:
                    print(f"Error reading velocity file: {e}")
            
            # Look for p (pressure) file
            p_file = os.path.join(latest_path, "p")
            if os.path.exists(p_file):
                try:
                    with open(p_file, 'r') as f:
                        content = f.read()
                        results.min_pressure, results.max_pressure = OpenFOAMDataParser._extract_pressure_range(content)
                except Exception as e:
                    print(f"Error reading pressure file: {e}")
        
        return results
    
    @staticmethod
    def _extract_convergence_info(log_content: str) -> str:
        """Extract convergence information from log file"""
        lines = log_content.split('\n')
        convergence_lines = []
        
        for line in lines:
            if 'GAMG:' in line or 'PCG:' in line or 'smoothSolver:' in line:
                if 'Initial residual' in line or 'Final residual' in line:
                    convergence_lines.append(line.strip())
            elif 'Time =' in line:
                convergence_lines.append(line.strip())
        
        return '\n'.join(convergence_lines[-20:])  # Last 20 convergence lines
    
    @staticmethod
    def _extract_max_velocity(u_content: str) -> float:
        """Extract maximum velocity magnitude from U file"""
        # Simple pattern matching for velocity values
        import re
        velocity_pattern = r'\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)'
        matches = re.findall(velocity_pattern, u_content)
        
        max_vel = 0.0
        for match in matches:
            try:
                ux, uy, uz = float(match[0]), float(match[1]), float(match[2])
                vel_mag = (ux**2 + uy**2 + uz**2)**0.5
                max_vel = max(max_vel, vel_mag)
            except (ValueError, IndexError):
                continue
        
        return max_vel
    
    @staticmethod
    def _extract_pressure_range(p_content: str) -> tuple:
        """Extract pressure range from p file"""
        import re
        pressure_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        matches = re.findall(pressure_pattern, p_content)
        
        pressures = []
        for match in matches:
            try:
                pressures.append(float(match))
            except ValueError:
                continue
        
        if pressures:
            return min(pressures), max(pressures)
        return 0.0, 0.0

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

class CFDResearchPaperGenerator:
    """Enhanced AI Agent for generating complete CFD research papers"""
    
    def __init__(self, openai_key: Optional[str] = None):
        self.semantic_api = SemanticScholarAPI()
        self.openai_client = OpenAIClient(openai_key) if openai_key else None
        self.papers: List[Paper] = []
        self.paper_filter = PaperFilter()
        self.latex_formatter = LaTeXFormatter()
        self.study_config: Optional[StudyConfiguration] = None
        self.openfoam_results: Optional[OpenFOAMResults] = None
        
    def load_study_configuration(self, text_file_path: str) -> bool:
        """Load study configuration from text file"""
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.study_config = StudyConfiguration.from_text(content)
            print(f"Loaded study configuration from: {text_file_path}")
            print(f"Study focus: {self.study_config.description[:100]}...")
            return True
            
        except FileNotFoundError:
            print(f"Error: File {text_file_path} not found")
            return False
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def load_openfoam_data(self, folder_path: str) -> bool:
        """Load OpenFOAM simulation data from folder"""
        try:
            self.openfoam_results = OpenFOAMDataParser.parse_openfoam_folder(folder_path)
            print(f"Loaded OpenFOAM data from: {folder_path}")
            if self.openfoam_results.simulation_summary:
                print(f"Simulation summary: {self.openfoam_results.simulation_summary}")
            return True
        except Exception as e:
            print(f"Error loading OpenFOAM data: {e}")
            return False
    
    def generate_search_query_from_config(self) -> str:
        """Generate search query based on study configuration"""
        if not self.study_config:
            return "computational fluid dynamics"
        
        # Extract key terms from the description
        description = self.study_config.description.lower()
        
        # Common CFD terms to look for
        cfd_terms = []
        # (Removed legacy case-specific query term)
        if "incompressible" in description:
            cfd_terms.append("incompressible flow")
        if "turbulence" in description:
            cfd_terms.append("turbulence")
        if "heat transfer" in description:
            cfd_terms.append("heat transfer")
        if "multiphase" in description:
            cfd_terms.append("multiphase")
        if "openfoam" in description:
            cfd_terms.append("openfoam")
        if "navier stokes" in description:
            cfd_terms.append("navier stokes")
        
        # Combine terms
        if cfd_terms:
            return f"computational fluid dynamics {' '.join(cfd_terms[:2])}"
        else:
            return "computational fluid dynamics"
    
    def fetch_cfd_papers(self, max_papers: int = 15) -> List[Paper]:
        """Fetch relevant CFD papers based on study configuration"""
        if not self.study_config:
            print("Error: No study configuration loaded. Please load a text file first.")
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
    
    def generate_introduction(self, target_length: int = 550, use_gpt4o: bool = True) -> str:
        """Generate a research paper introduction with citations"""
        if not self.study_config:
            return "Error: No study configuration loaded."
        
        if not self.papers:
            return "No papers found. Please fetch papers first."
        
        if use_gpt4o and self.openai_client:
            return self._generate_gpt4o_introduction(target_length)
        else:
            return self._generate_template_introduction()
    
    def generate_materials_methods(self, use_gpt4o: bool = True) -> str:
        """Generate Materials and Methods section"""
        if not self.study_config:
            return "Error: No study configuration loaded."
        
        if use_gpt4o and self.openai_client:
            return self._generate_gpt4o_materials_methods()
        else:
            return self._generate_template_materials_methods()
    
    def generate_results_discussion(self, use_gpt4o: bool = True) -> str:
        """Generate Results and Discussion section based on OpenFOAM data"""
        if not self.study_config:
            return "Error: No study configuration loaded."
        
        if use_gpt4o and self.openai_client:
            return self._generate_gpt4o_results_discussion()
        else:
            return self._generate_template_results_discussion()
    
    def generate_conclusion(self, use_gpt4o: bool = True) -> str:
        """Generate Conclusion section"""
        if not self.study_config:
            return "Error: No study configuration loaded."
        
        if use_gpt4o and self.openai_client:
            return self._generate_gpt4o_conclusion()
        else:
            return self._generate_template_conclusion()
    
    def _generate_gpt4o_results_discussion(self) -> str:
        """Generate Results and Discussion using GPT-4o"""
        # Prepare simulation results information
        results_info = ""
        if self.openfoam_results:
            results_info = f"""
Simulation Results:
- Maximum velocity magnitude: {self.openfoam_results.max_velocity:.4f} m/s
- Pressure range: {self.openfoam_results.min_pressure:.4f} to {self.openfoam_results.max_pressure:.4f} Pa
- Simulation summary: {self.openfoam_results.simulation_summary}
- Convergence info: {self.openfoam_results.convergence_info[:500]}...
"""
        
        system_prompt = """You are an expert CFD researcher writing a Results and Discussion section for a peer-reviewed journal. 
        Create a comprehensive section that presents simulation results clearly and discusses their implications.
        Use formal academic language and include references to figures and tables where appropriate.
        Discuss convergence, validation, flow patterns, and physical insights from the simulation."""
        
        user_prompt = f"""Write a detailed Results and Discussion section for a CFD study with the following details:

Study Configuration:
{self.study_config.description}

{results_info}

Structure the section to include:
1. Convergence and validation of the numerical solution
2. Flow field characteristics and patterns
3. Quantitative results (velocities, pressures, forces if applicable)
4. Physical interpretation of the results
5. Comparison with literature or analytical solutions where possible
6. Discussion of limitations and assumptions

Write in formal academic style suitable for a peer-reviewed journal.Make sure it is in depth an expores the results thoroughly"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        results_discussion = self.openai_client.generate_text(messages, max_tokens=1500, temperature=0.5)
        
        if results_discussion:
            return results_discussion
        else:
            return self._generate_template_results_discussion()
    
    def _generate_template_results_discussion(self) -> str:
        """Generate template Results and Discussion section"""
        results_text = f"""
The computational simulation successfully modeled the specified flow configuration: {self.study_config.description[:200]}...

Convergence Analysis:
"""
        
        if self.openfoam_results and self.openfoam_results.convergence_info:
            results_text += f"The simulation achieved convergence as indicated by the residual behavior. {self.openfoam_results.simulation_summary}\n\n"
        else:
            results_text += "The numerical solution converged within the specified tolerance criteria, ensuring reliable results.\n\n"
        
        results_text += """Flow Field Characteristics:
The velocity field shows the expected flow patterns for this configuration. """
        
        if self.openfoam_results and self.openfoam_results.max_velocity > 0:
            results_text += f"The maximum velocity magnitude reached {self.openfoam_results.max_velocity:.4f} m/s. "
        
        if self.openfoam_results and (self.openfoam_results.min_pressure != 0 or self.openfoam_results.max_pressure != 0):
            results_text += f"Pressure variations ranged from {self.openfoam_results.min_pressure:.4f} to {self.openfoam_results.max_pressure:.4f} Pa. "
        
        results_text += """

Discussion:
The results demonstrate the effectiveness of the OpenFOAM solver for this type of flow simulation. The numerical solution captures the essential physics of the problem and provides valuable insights into the flow behavior. These findings contribute to the understanding of the flow phenomena and validate the computational approach for similar applications."""
        
        return results_text.strip()
    
    def _generate_gpt4o_conclusion(self) -> str:
        """Generate Conclusion using GPT-4o"""
        system_prompt = """You are an expert CFD researcher writing a Conclusion section for a peer-reviewed journal.
        Summarize the key findings, contributions, and future work recommendations.
        Be concise but comprehensive, highlighting the significance of the study."""
        
        user_prompt = f"""Write a concise Conclusion section for a CFD study with the following details:

Study Configuration:
{self.study_config.description}

Key points to include:
1. Summary of the main objectives and what was achieved
2. Key findings from the simulation
3. Contributions to the field
4. Limitations of the current study
5. Recommendations for future work

Write in formal academic style."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        conclusion = self.openai_client.generate_text(messages, max_tokens=800, temperature=0.5)
        
        if conclusion:
            return conclusion
        else:
            return self._generate_template_conclusion()
    
    def _generate_template_conclusion(self) -> str:
        """Generate template Conclusion section"""
        return f"""
This study successfully implemented and validated a computational fluid dynamics simulation using OpenFOAM for the specified flow configuration. The investigation focused on {self.study_config.description[:100]}...

The key findings include successful convergence of the numerical solution, validation of the computational approach, and detailed characterization of the flow field. The results demonstrate the effectiveness of OpenFOAM for this class of problems and provide valuable insights into the underlying fluid mechanics.

The computational methodology developed in this work contributes to the broader understanding of CFD applications and provides a foundation for future investigations. The validated numerical approach can be extended to more complex geometries and flow conditions.

Future work should focus on extending the analysis to three-dimensional configurations, investigating parameter sensitivity, and exploring advanced turbulence modeling approaches. Additional validation against experimental data would further enhance the confidence in the computational predictions.
"""
    
    def _generate_gpt4o_introduction(self, target_length: int) -> str:
        """Generate introduction using GPT-4o"""
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
        """Generate a template-based introduction"""
        if not self.study_config:
            return "Error: No study configuration available"
        
        categorized = self.paper_filter.categorize_papers(self.papers)
        recent_papers = categorized["recent"][:3]
        foundational_papers = categorized["foundational"][:3]
        
        intro_parts = []
        
        # Opening with study context
        intro_parts.append(f"Computational Fluid Dynamics (CFD) has emerged as a powerful tool for analyzing complex fluid flow phenomena.")
        
        # Background with citations
        if foundational_papers:
            citations = ", ".join([p.get_citation() for p in foundational_papers[:2]])
            intro_parts.append(f"The foundational principles of CFD simulation have been extensively developed and validated {citations}.")
        
        # Recent developments
        if recent_papers:
            citations = ", ".join([p.get_citation() for p in recent_papers[:2]])
            intro_parts.append(f"Recent advances in computational methods and numerical techniques have significantly enhanced CFD capabilities {citations}.")
        
        # Research focus
        intro_parts.append(f"This research focuses on {self.study_config.description[:200]}...")
        intro_parts.append("The objective is to demonstrate the effectiveness of OpenFOAM for this class of problems and provide detailed analysis of the flow characteristics.")
        
        return " ".join(intro_parts)
    
    def _generate_gpt4o_materials_methods(self) -> str:
        """Generate Materials and Methods section using GPT-4o"""
        
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
        """Generate a template-based Materials and Methods section"""
        if not self.study_config:
            return "Error: No study configuration available"
        
        template = f"""
This computational study employs OpenFOAM, an open-source CFD software package, to investigate the specified flow configuration.

Problem Description:
{self.study_config.description}

Geometric Configuration:
{self.study_config.geometry if self.study_config.geometry else "The computational domain is defined according to the problem specifications."}

Mesh Generation:
{self.study_config.mesh if self.study_config.mesh else "A structured computational mesh is generated with appropriate resolution to capture the flow features."}

Boundary Conditions:
{self.study_config.boundary_conditions if self.study_config.boundary_conditions else "Appropriate boundary conditions are applied based on the physical problem requirements."}

Simulation Parameters:
{self.study_config.simulation_time if self.study_config.simulation_time else "The simulation is run for sufficient time to achieve steady-state or capture the required transient behavior."}

Material Properties:
{self.study_config.properties if self.study_config.properties else "Fluid properties are specified according to the problem requirements."}

The simulations are based on the Navier-Stokes equations for fluid flow, solved using finite volume discretization in OpenFOAM. The SIMPLE algorithm is employed for pressure-velocity coupling, and appropriate numerical schemes are selected for spatial and temporal discretization.
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
        """
    
    def _create_materials_methods_user_prompt(self, paper_summaries: List[Dict]) -> str:
        """Create user prompt for Materials and Methods generation"""
        
        papers_text = "\n".join([
            f"- {p['title']} {p['citation']}: {p['abstract_snippet']}"
            for p in paper_summaries
        ])
        
        return f"""Write a detailed 'Materials and Methods' section for a computational fluid dynamics (CFD) research paper that uses OpenFOAM.

Study Configuration:
{self.study_config.description}

Geometry: {self.study_config.geometry}
Mesh: {self.study_config.mesh}
Boundary Conditions: {self.study_config.boundary_conditions}
Simulation Time: {self.study_config.simulation_time}
Properties: {self.study_config.properties}

Use the following related papers as background context:
{papers_text}

Include relevant technical subsections (e.g., computational setup, governing equations, geometry and meshing, solver configuration, boundary conditions, post-processing, etc.). 
Clearly describe modeling decisions, solver settings, mesh characteristics, and boundary conditions.
Ensure you are writing good compilable LaTeX code by properly escaping special characters.
Use formal academic language suitable for a peer-reviewed journal and make sure the methodology is detailed enough for another researcher to reproduce the simulation using OpenFOAM.
"""
    
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
        """Create user prompt for GPT-4o"""
        papers_text = "\n".join([
            f"- {p['title']} {p['citation']}: {p['abstract_snippet']}"
            for p in paper_summaries
        ])
        
        return f"""Write a {target_length}-word introduction for a CFD research paper based on the following study configuration:

Study Configuration:
{self.study_config.description}

Use these relevant papers and their citations:
{papers_text}

Structure the introduction with:
1. Introduction of CFD and its significance
2. Background with foundational concepts and citations
3. Recent developments and advances relevant to the study focus
4. Research gap and motivation for the current work
5. Brief overview of the approach and expected outcomes

Ensure the introduction flows naturally and incorporates the specific research focus from the study configuration.
"""
    
    def generate_latex_document(self, 
                               authors: List[str] = ["Author Name"],
                               include_all_sections: bool = True) -> str:
        """Generate complete LaTeX document"""
        
        if not self.study_config:
            return "% Error: No study configuration loaded. Please load a text file first."
        
        if not self.papers:
            return "% Error: No papers available. Please fetch papers first."
        
        # Generate title from study configuration
        title = f"CFD Analysis of {self.study_config.name.replace('-', ' ').title()}"
        
        # Generate content
        print("Generating introduction...")
        introduction = self.generate_introduction(400, use_gpt4o=True)
        
        materials_methods = ""
        results_discussion = ""
        conclusion = ""
        
        if include_all_sections:
            print("Generating Materials and Methods section...")
            materials_methods = self.generate_materials_methods(use_gpt4o=True)
            
            print("Generating Results and Discussion section...")
            results_discussion = self.generate_results_discussion(use_gpt4o=True)
            
            print("Generating Conclusion section...")
            conclusion = self.generate_conclusion(use_gpt4o=True)
        
        # Generate bibliography entries
        print("Generating bibliography...")
        bibliography = self.generate_latex_bibliography()
        
        # Format content for LaTeX
        latex_title = self.latex_formatter.format_title(title)
        latex_authors = self.latex_formatter.format_authors(authors)
        latex_introduction = self.latex_formatter.format_text_content(introduction)
        latex_materials_methods = self.latex_formatter.format_text_content(materials_methods)
        latex_results_discussion = self.latex_formatter.format_text_content(results_discussion)
        latex_conclusion = self.latex_formatter.format_text_content(conclusion)
        
        # Create LaTeX document
        latex_document = self._create_latex_document(
            latex_title, 
            latex_authors, 
            latex_introduction, 
            latex_materials_methods,
            latex_results_discussion,
            latex_conclusion,
            bibliography
        )
        
        return latex_document
    
    def _create_latex_document(self, title: str, authors: str, introduction: str, 
                             materials_methods: str, results_discussion: str, 
                             conclusion: str, bibliography: str) -> str:
        """Create complete LaTeX document structure"""
        
        # Create abstract from study configuration
        abstract = f"This paper presents a computational fluid dynamics (CFD) analysis using OpenFOAM to investigate the specified flow configuration. {self.study_config.description[:200]}... The study demonstrates the effectiveness of CFD simulation for this class of problems and provides detailed analysis of the flow characteristics."
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
        
        if results_discussion:
            document += f"""\\section{{Results and Discussion}}
{results_discussion}

"""
        
        if conclusion:
            document += f"""\\section{{Conclusion}}
{conclusion}

"""
        
        # Add simulation details in appendix if OpenFOAM data is available
        if self.openfoam_results and self.openfoam_results.simulation_summary:
            document += f"""\\section{{Appendix: Simulation Details}}
\\subsection{{Simulation Summary}}
{self.latex_formatter.escape_latex(self.openfoam_results.simulation_summary)}

"""
            if self.openfoam_results.convergence_info:
                document += f"""\\subsection{{Convergence Information}}
The simulation convergence behavior is summarized as follows:
\\begin{{verbatim}}
{self.openfoam_results.convergence_info[:500]}
\\end{{verbatim}}

"""
        
        document += f"""\\section{{Acknowledgments}}
The authors acknowledge the use of OpenFOAM for computational fluid dynamics simulations and the Semantic Scholar API for literature review.

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
    
    def save_latex_document(self, latex_content: str, filename: str = None):
        """Save LaTeX document to file with automatic naming"""
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
            print(f"Description: {self.study_config.description[:100]}...")
            print()
        
        for i, paper in enumerate(self.papers, 1):
            print(f"{i}. {paper.title}")
            print(f"   Authors: {paper.format_authors('full')}")
            print(f"   Year: {paper.year}, Citations: {paper.citation_count}")
            if paper.venue:
                print(f"   Venue: {paper.venue}")
            print()
    
    def print_study_configuration(self):
        """Print loaded study configuration details"""
        if not self.study_config:
            print("No study configuration loaded.")
            return
        
        print(f"\n=== Study Configuration ===")
        print(f"Name: {self.study_config.name}")
        print(f"Description: {self.study_config.description}")
        print(f"Geometry: {self.study_config.geometry}")
        print(f"Mesh: {self.study_config.mesh}")
        print(f"Boundary Conditions: {self.study_config.boundary_conditions}")
        print(f"Simulation Time: {self.study_config.simulation_time}")
        print(f"Properties: {self.study_config.properties}")
        print()
    
    def print_openfoam_summary(self):
        """Print OpenFOAM simulation results summary"""
        if not self.openfoam_results:
            print("No OpenFOAM data loaded.")
            return
        
        print(f"\n=== OpenFOAM Results Summary ===")
        print(f"Simulation Summary: {self.openfoam_results.simulation_summary}")
        print(f"Maximum Velocity: {self.openfoam_results.max_velocity:.4f} m/s")
        print(f"Pressure Range: {self.openfoam_results.min_pressure:.4f} to {self.openfoam_results.max_pressure:.4f} Pa")
        if self.openfoam_results.convergence_info:
            print(f"Convergence Info Available: Yes ({len(self.openfoam_results.convergence_info)} characters)")
        print()

def main():
    # Initialize with OpenAI API key
    OPENAI_KEY = "key"  # Replace with your actual API key
    generator = CFDResearchPaperGenerator(openai_key=OPENAI_KEY)
    
    # Load study configuration from text file
    text_file_path = "sample_user_requirement.txt"  # Path to your text file
    openfoam_folder_path = "output"  # Path to your OpenFOAM data folder
    
    # Load configuration
    if not generator.load_study_configuration(text_file_path):
        print("Failed to load study configuration.")
        return
    
    # Load OpenFOAM data (optional)
    generator.load_openfoam_data(openfoam_folder_path)
    
    # Print loaded configuration
    generator.print_study_configuration()
    generator.print_openfoam_summary()
    
    # Fetch papers based on study configuration
    papers = generator.fetch_cfd_papers(max_papers=12)
    
    if papers:
        # Print paper summary
        generator.print_paper_summary()
        
        # Generate complete LaTeX document with all sections
        latex_content = generator.generate_latex_document(
            authors=["Author Name", "Co-Author Name"],  # Replace with actual author names
            include_all_sections=True
        )
        
        # Print LaTeX document
        print("\n" + "="*60)
        print("GENERATED LATEX DOCUMENT")
        print("="*60)
        print(latex_content)
        print("="*60)
        
        # Save to file with automatic naming
        generator.save_latex_document(latex_content)
        
        print(f"\nLaTeX document generated successfully for study: {generator.study_config.name}")
        print("You can now compile this .tex file using pdflatex or your preferred LaTeX compiler.")
        print("The document includes Introduction, Materials and Methods, Results and Discussion, and Conclusion sections.")
        
    else:
        print("No papers found. This could be due to:")
        print("1. Network connectivity issues")
        print("2. Semantic Scholar API rate limiting")
        print("3. Query returning no results")
        print("Please check your connection and try again.")

if __name__ == "__main__":
    main()