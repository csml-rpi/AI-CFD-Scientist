import json
import os
import re
from typing import Any, Optional, Tuple, List, Dict
from pathlib import Path

import requests

import anthropic
import backoff
import openai

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    # DeepSeek Models
    "deepseek-coder-v2-0724",
    "deepcoder-14b",
    # Llama 3 models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    "arn:aws:bedrock",  # Prefix for ARN-based models
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # Google Gemini models
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
]


class _CodexResponsesWrapper:
    """Wrapper for a Responses-compatible endpoint.

    Ported from Foam-Agent.

    For ChatGPT/Codex subscription access we call:
      https://chatgpt.com/backend-api/codex/responses

    Note: Codex subscription backend requires `stream=true`. We stream internally
    but return a final concatenated string to keep the rest of this repo simple.
    """

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def __init__(
        self,
        *,
        token: str,
        model: str,
        base_url: str = "https://chatgpt.com/backend-api/codex",
        account_id: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = 0.0,
        stream: bool = True,
    ):
        self._token = token
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._instructions = instructions
        self._temperature = temperature
        self._stream = bool(stream)

    @staticmethod
    def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            out.append({"role": role, "content": [{"type": "input_text", "text": content}]})
        return out

    @staticmethod
    def _extract_output_text(resp_json: dict) -> str:
        if isinstance(resp_json, dict) and isinstance(resp_json.get("output_text"), str):
            return resp_json["output_text"]

        texts: List[str] = []
        for item in resp_json.get("output", []) if isinstance(resp_json, dict) else []:
            for c in item.get("content", []) if isinstance(item, dict) else []:
                if isinstance(c, dict):
                    if c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                        texts.append(c["text"])
        return "\n".join(texts).strip()

    def _build_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._model,
            "input": self._to_responses_input(messages),
        }

        # OpenAI Platform supports temperature (not used for chatgpt.com backend).
        if "chatgpt.com" not in self._base_url:
            payload["temperature"] = self._temperature

        # ChatGPT/Codex subscription backend expects these extra keys.
        if "chatgpt.com" in self._base_url:
            payload.update(
                {
                    "instructions": self._instructions or "",
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "reasoning": {"summary": "auto"},
                    "store": False,
                    "stream": bool(self._stream),
                    "include": ["reasoning.encrypted_content"],
                }
            )

        return payload

    @staticmethod
    def _iter_sse_text(resp: requests.Response):
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            line = str(raw).strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            yield data

    def invoke(self, messages: List[Dict[str, Any]]) -> str:
        url = f"{self._base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not self._stream else "text/event-stream",
            "User-Agent": "cfd-scientist",
        }
        if self._account_id:
            headers["ChatGPT-Account-Id"] = self._account_id

        payload = self._build_payload(messages)

        r = requests.post(url, headers=headers, json=payload, timeout=60, stream=bool(self._stream))
        if not r.ok:
            raise requests.HTTPError(
                f"HTTP {r.status_code} for {url}. Body: {r.text[:2000]}", response=r
            )

        if not self._stream:
            return self._extract_output_text(r.json()).strip()

        # Streaming: accumulate text deltas.
        chunks: List[str] = []
        for s in self._iter_sse_text(r):
            try:
                j = json.loads(s)
            except Exception:
                continue

            if isinstance(j, dict):
                t = j.get("type")
                if t == "response.output_text.delta" and isinstance(j.get("delta"), str):
                    chunks.append(j["delta"])
                    continue
                if t == "response.output_text.done" and isinstance(j.get("text"), str):
                    if not chunks:
                        chunks.append(j["text"])
                    continue

            t2 = self._extract_output_text(j)
            if t2:
                chunks.append(t2)

        return "".join(chunks).strip()


def _load_codex_access_token_from_auth_json(auth_json_path: Path) -> str:
    data = json.loads(auth_json_path.read_text(encoding="utf-8"))

    candidates: List[str] = []

    def maybe_add(v):
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    if isinstance(data, dict):
        maybe_add(data.get("access_token"))
        maybe_add(data.get("token"))
        for k in ("auth", "credentials", "session"):
            v = data.get(k)
            if isinstance(v, dict):
                maybe_add(v.get("access_token"))
                maybe_add(v.get("token"))

    if not candidates:
        raise ValueError(
            f"Could not find an access token in {auth_json_path}. Expected keys like access_token/token."
        )

    return candidates[0]


def _load_codex_oauth_from_clawdbot_auth_profiles(auth_profiles_path: Path) -> Tuple[str, Optional[str]]:
    data = json.loads(auth_profiles_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON in {auth_profiles_path}")

    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"Missing 'profiles' in {auth_profiles_path}")

    preferred_keys = ["openai-codex:default", "openai-codex"]
    for k in preferred_keys:
        v = profiles.get(k)
        if isinstance(v, dict):
            token = v.get("access")
            account_id = v.get("accountId")
            if isinstance(token, str) and token.strip():
                return token.strip(), account_id if isinstance(account_id, str) else None

    for _, v in profiles.items():
        if isinstance(v, dict):
            token = v.get("access")
            account_id = v.get("accountId")
            if isinstance(token, str) and token.strip():
                return token.strip(), account_id if isinstance(account_id, str) else None

    raise ValueError(
        f"Could not find an 'access' token in {auth_profiles_path}. Expected profiles[*].access"
    )


def _load_codex_oauth() -> Tuple[str, Optional[str]]:
    candidates: List[Path] = []

    codex_home = os.getenv("CODEX_HOME")
    if codex_home:
        candidates.append(Path(codex_home) / "auth.json")

    candidates.append(Path.home() / ".codex" / "auth.json")
    candidates.append(Path.home() / ".clawdbot" / "agents" / "main" / "agent" / "auth-profiles.json")

    for p in candidates:
        if not p.exists():
            continue

        if p.name == "auth.json":
            return _load_codex_access_token_from_auth_json(p), None

        if p.name == "auth-profiles.json":
            return _load_codex_oauth_from_clawdbot_auth_profiles(p)

    raise FileNotFoundError(
        "Could not find a Codex/ChatGPT OAuth cache. "
        + "Looked for: "
        + ", ".join(str(x) for x in candidates)
    )


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    if "gpt" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif 'gemini' in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def make_llm_call(client, model, temperature, system_message, prompt):
    if "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "o1" in model or "o3" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *prompt,
            ],
            temperature=1,
            n=1,
            seed=0,
        )
    
    else:
        raise ValueError(f"Model {model} not supported.")


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    # Codex subscription backend (ChatGPT OAuth via /codex/responses)
    if isinstance(client, _CodexResponsesWrapper):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.extend(new_msg_history)

        content = client.invoke(messages)
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        return content, new_msg_history

    # Bedrock boto3 Client
    # set using environment variables 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'
    is_bedrock_boto3 = hasattr(client, 'invoke_model') and (model.startswith("arn:aws:bedrock") or "bedrock" in str(type(client)).lower())

    if (not isinstance(client, _CodexResponsesWrapper)) and is_bedrock_boto3:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        
        messages = []
        for msg_item in new_msg_history:
            content = msg_item["content"]
            if isinstance(content, str):
                formatted_content = [{"text": content}]
            elif isinstance(content, list):
                formatted_content = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            formatted_content.append({"text": item["text"]})
                        elif "type" in item and item["type"] == "text":
                            formatted_content.append({"text": item.get("text", "")})
                        else:
                            formatted_content.append({"text": str(item)})
                    else:
                        formatted_content.append({"text": str(item)})
            else:
                formatted_content = [{"text": str(content)}]
            
            messages.append({
                "role": msg_item["role"],
                "content": formatted_content
            })
        
        # Build converse API request
        converse_params = {
            "modelId": model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": MAX_NUM_TOKENS,
                "temperature": temperature,
            }
        }
        
        if system_message:
            converse_params["system"] = [{"text": system_message}]
        
        # Use the Converse API instead of InvokeModel
        response = client.converse(**converse_params)
        
        # Parse the response from Converse API
        content = response['output']['message']['content'][0]['text']
        
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        
    elif "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        # response = make_llm_call(client, model, temperature, system_message=system_message, prompt=new_msg_history)
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif "gpt" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "o1" in model or "o3" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepcoder-14b":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        try:
            response = client.chat.completions.create(
                model="agentica-org/DeepCoder-14B-Preview",
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
        except Exception as e:
            # Fallback to direct API call if OpenAI client doesn't work with HuggingFace
            import requests
            headers = {
                "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": {
                    "system": system_message,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in new_msg_history]
                },
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": MAX_NUM_TOKENS,
                    "return_full_text": False
                }
            }
            response = requests.post(
                "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                content = response.json()["generated_text"]
            else:
                raise ValueError(f"Error from HuggingFace API: {response.text}")

        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif 'gemini' in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None:
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model=None) -> tuple[Any, str]:
    # Use CFD_SCIENTIST_MODEL env var as default if model is not provided
    if model is None:
        model = os.environ.get("CFD_SCIENTIST_MODEL", "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2")
    if model.startswith("claude-"):
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        return anthropic.AnthropicBedrock(), client_model
    elif model.startswith("bedrock/") or model.startswith("arn:aws:bedrock"):
        if not BOTO3_AVAILABLE:
            raise ValueError("boto3 is required for Bedrock models. Please install it with: pip install boto3")
        if model.startswith("arn:aws:bedrock"):
            model_id = model
        else:
            model_id = model.split("/")[-1]
        try:
            import sys
            from pathlib import Path
            foam_agent_path = Path(__file__).resolve().parent.parent / "Foam-Agent" / "src"
            if str(foam_agent_path) not in sys.path:
                sys.path.insert(0, str(foam_agent_path))
            import tracking_aws
            bedrock_client = tracking_aws.new_default_client()
        except ImportError:
            bedrock_client = boto3.client('bedrock-runtime', region_name='us-west-2')
            
        return bedrock_client, model_id
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        return anthropic.AnthropicVertex(), client_model
    elif model.endswith("-codex"):
        # ChatGPT/Codex subscription backend (OAuth).
        token, account_id = _load_codex_oauth()

        # Keep instructions minimal; Codex backend requires non-empty instructions.
        instructions = "You are Codex, a coding assistant."

        return _CodexResponsesWrapper(
            token=token,
            account_id=account_id,
            model=model,
            temperature=0.0,
            base_url="https://chatgpt.com/backend-api/codex",
            instructions=instructions,
            stream=True,
        ), model
    elif "gpt" in model:
        return openai.OpenAI(), model
    elif "o1" in model or "o3" in model:
        return openai.OpenAI(), model
    elif model == "deepseek-coder-v2-0724":
        return (
            openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            ),
            model,
        )
    elif model == "deepcoder-14b":
        print(f"Using HuggingFace API with {model}.")
        if "HUGGINGFACE_API_KEY" not in os.environ:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        return (
            openai.OpenAI(
                api_key=os.environ["HUGGINGFACE_API_KEY"],
                base_url="https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
            ),
            model,
        )
    elif model == "llama3.1-405b":
        print(f"Using OpenAI API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ),
            "meta-llama/llama-3.1-405b-instruct",
        )
    elif 'gemini' in model:
        print(f"Using OpenAI API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            model,
        )
    else:
        raise ValueError(f"Model {model} not supported.")