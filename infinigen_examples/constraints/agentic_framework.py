from __future__ import annotations

import inspect
import logging
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
import traceback
from typing import Callable, List, Optional, Protocol, Sequence

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import usage_lookup
from infinigen_examples.constraints import util as cu
from infinigen_examples.constraints.semantics import home_asset_usage

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Simple protocol for pluggable LLM backends."""

    def complete(self, prompt: str) -> str: ...


class DummyLLM:
    """Fallback client that simply mirrors the in-context example."""

    def __init__(self, fallback_response: str):
        self.fallback_response = fallback_response

    def complete(self, prompt: str) -> str:
        logger.warning("DummyLLM in use – returning fallback response.")
        return self.fallback_response


class OpenAIChatClient:
    """LLM client backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "openai package is required to use OpenAIChatClient. Install via 'pip install openai'."
            ) from exc

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization

        self._client = OpenAI(**kwargs)
        self._model = model
        self._system_prompt = (
            system_prompt
            or "You are the InfiniBench constraint authoring agent."
        )
        self._temperature = temperature

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
        )
        return response.choices[0].message.content.strip()


class GeminiChatClient:
    """LLM client backed by the Google Gemini Generative Language API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "google-generativeai package is required to use GeminiChatClient. Install via 'pip install google-generativeai'."
            ) from exc

        effective_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not effective_key:
            raise RuntimeError(
                "Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY to use GeminiChatClient."
            )
        genai.configure(api_key=effective_key)

        model_kwargs = {}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt

        self._model = genai.GenerativeModel(model_name=model, **model_kwargs)
        self._generation_config = {"temperature": temperature}

    def complete(self, prompt: str) -> str:
        response = self._model.generate_content(prompt, generation_config=self._generation_config)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini response did not contain text content.")
        return text.strip()


@dataclass
class ConstraintProgram:
    """Container for machine-generated procedural constraints."""

    source_code: str
    entrypoint: str = "build_constraints"
    metadata: dict = field(default_factory=dict)

    def to_callable(self, extra_globals: Optional[dict] = None) -> Callable[[], cl.Problem]:
        """Compile the generated source into an executable builder."""
        exec_globals = {} if extra_globals is None else dict(extra_globals)
        exec_locals: dict = {}
        exec(self.source_code, exec_globals, exec_locals)
        if self.entrypoint not in exec_locals:
            raise ValueError(
                f"Entrypoint '{self.entrypoint}' not defined in generated program. "
                f"Available symbols: {list(exec_locals.keys())}"
            )
        return exec_locals[self.entrypoint]


@dataclass
class OptimizationFeedback:
    """Summarizes the optimizer's attempt to realize the generated constraints."""

    success: bool
    summary: str
    violation_report: dict[str, float] = field(default_factory=dict)
    bev_path: Optional[Path] = None
    raw_log: Optional[str] = None

    def to_prompt_fragment(self) -> str:
        status = "success" if self.success else "failure"
        report_lines = [
            f"- {name}: {value:.3f}" for name, value in self.violation_report.items()
        ]
        report = "\n".join(report_lines) if report_lines else "None"
        img_text = f"BEV: {self.bev_path}" if self.bev_path else "BEV: unavailable"
        return textwrap.dedent(
            f"""
            [Optimizer feedback: {status}]
            Summary: {self.summary}
            Violations:
            {report}
            {img_text}
            """
        )


class SceneLayoutOptimizer:
    """Adapter that executes a `ConstraintProgram` by calling into the existing solver stack."""

    def __init__(
        self,
        executor: Callable[[ConstraintProgram], OptimizationFeedback],
    ):
        self._executor = executor

    def run(self, program: ConstraintProgram) -> OptimizationFeedback:
        return self._executor(program)


@dataclass
class APIEntry:
    name: str
    description: str


class ProceduralAPIRegistry:
    """Tracks procedural APIs that the LLM agent is allowed to call."""

    def __init__(self):
        self._entries: list[APIEntry] = []

    def register(self, name: str, description: str):
        self._entries.append(APIEntry(name=name, description=description))

    def describe(self) -> str:
        return "\n".join(f"- `{entry.name}`: {entry.description}" for entry in self._entries)

    @classmethod
    def build_default(cls) -> "ProceduralAPIRegistry":
        reg = cls()
        reg.register(
            "cl.scene()[tags]",
            "Returns an object set filtered by semantic tags. Supports chaining relations.",
        )
        reg.register(
            "cl.count()/cl.in_range()/cl.mean()/cl.sum()",
            "Statistical operators over object sets, used to express quantity constraints.",
        )
        reg.register(
            "cl.StableAgainst / cl.SupportedBy / cl.CoPlanar",
            "Geometric relations that describe placement of child objects relative to parents.",
        )
        reg.register(
            "constraint_language.constants.RoomConstants",
            "Holds global parameters (unit sizes, wall thickness) used across constraints.",
        )
        reg.register(
            "usage_lookup / home_asset_usage",
            "Maps semantic intents to concrete asset factories.",
        )
        reg.register(
            "Scene Generator Moves",
            "addition, deletion, translate, rotate, plane_change, cluster_* moves enabled at runtime.",
        )
        return reg


def _load_home_constraint_example() -> str:
    from infinigen_examples.constraints import home

    source = inspect.getsource(home.home_furniture_constraints)
    return textwrap.dedent(source)


HOME_CONSTRAINT_EXAMPLE = _load_home_constraint_example()


class PromptBuilder:
    """Formats the system/user prompts that condition the LLM agent."""

    def __init__(
        self,
        api_registry: ProceduralAPIRegistry,
        examples: Sequence[str],
    ):
        self.api_registry = api_registry
        self.examples = examples

    def build(
        self,
        user_description: str,
        iteration_idx: int,
        feedback: Optional[OptimizationFeedback],
    ) -> str:
        feedback_block = (
            "\n## Optimizer Feedback\n" + feedback.to_prompt_fragment()
            if feedback is not None
            else ""
        )
        examples_block = "\n\n".join(
            f"### Example {i+1}\n```python\n{example}\n```"
            for i, example in enumerate(self.examples)
        )
        prompt = f"""
        You are the InfiniBench constraint authoring agent.
        Your job is to convert natural language scene descriptions into executable
        Infinigen constraint programs.

        ## Available APIs
        {self.api_registry.describe()}

        ## In-Context Examples
        {examples_block}

        ## Task Description (iteration {iteration_idx})
        "{user_description}"

        Requirements:
        - Return ONLY Python code defining a callable named `build_constraints()`.
        - Use the provided APIs. Avoid undefined helper functions.
        - Favor realistic, solvable layouts. Adjust parameters when surfaces are insufficient.
        - Document assumptions via concise comments when necessary.
        {feedback_block}
        """
        return textwrap.dedent(prompt)


class AgenticConstraintGenerator:
    """LLM-powered generator that iteratively proposes constraint programs."""

    def __init__(
        self,
        llm_client: LLMClient,
        api_registry: ProceduralAPIRegistry | None = None,
        examples: Sequence[str] | None = None,
    ):
        api_registry = api_registry or ProceduralAPIRegistry.build_default()
        examples = examples or (HOME_CONSTRAINT_EXAMPLE,)
        self.prompt_builder = PromptBuilder(api_registry, examples)
        self.llm_client = llm_client

    def generate(
        self,
        description: str,
        iteration_idx: int,
        feedback: Optional[OptimizationFeedback],
        entrypoint: str = "build_constraints",
    ) -> ConstraintProgram:
        prompt = self.prompt_builder.build(description, iteration_idx, feedback)
        source = self.llm_client.complete(prompt)
        return ConstraintProgram(source_code=source, entrypoint=entrypoint)


@dataclass
class GenerationHistoryItem:
    iteration: int
    program: ConstraintProgram
    feedback: OptimizationFeedback


@dataclass
class AgenticGenerationResult:
    success: bool
    final_program: Optional[ConstraintProgram]
    history: List[GenerationHistoryItem] = field(default_factory=list)


class AgenticSceneGenerator:
    """Wraps the entire scene generation process inside the agentic refinement loop."""

    def __init__(
        self,
        constraint_agent: AgenticConstraintGenerator,
        optimizer: SceneLayoutOptimizer,
        max_iterations: int = 3,
    ):
        self.constraint_agent = constraint_agent
        self.optimizer = optimizer
        self.max_iterations = max_iterations

    def generate_scene(
        self,
        description: str,
    ) -> AgenticGenerationResult:
        feedback: Optional[OptimizationFeedback] = None
        history: list[GenerationHistoryItem] = []

        for iteration in range(1, self.max_iterations + 1):
            program = self.constraint_agent.generate(
                description=description,
                iteration_idx=iteration,
                feedback=feedback,
            )

            optimization_feedback = self.optimizer.run(program)
            history.append(
                GenerationHistoryItem(
                    iteration=iteration,
                    program=program,
                    feedback=optimization_feedback,
                )
            )

            if optimization_feedback.success:
                logger.info("Agentic loop converged on iteration %d", iteration)
                return AgenticGenerationResult(
                    success=True,
                    final_program=program,
                    history=history,
                )

            feedback = optimization_feedback
            logger.info(
                "Agent iteration %d failed – violations: %s",
                iteration,
                optimization_feedback.violation_report,
            )

        logger.warning("Agentic loop exhausted without success")
        return AgenticGenerationResult(success=False, final_program=None, history=history)


def agentic_exec_globals(extra: Optional[dict] = None) -> dict:
    globals_dict = {
        "cl": cl,
        "cu": cu,
        "usage_lookup": usage_lookup,
        "home_asset_usage": home_asset_usage,
    }
    if extra:
        globals_dict.update(extra)
    return globals_dict


def compile_only_executor_factory(
    extra_globals: Optional[dict] = None,
) -> Callable[[ConstraintProgram], OptimizationFeedback]:
    """Builds a lightweight executor that simply compiles the program."""

    exec_globals = agentic_exec_globals(extra_globals)

    def _executor(program: ConstraintProgram) -> OptimizationFeedback:
        try:
            usage_lookup.initialize_from_dict(home_asset_usage())
            builder = program.to_callable(exec_globals)
            problem = builder()
            if not isinstance(problem, cl.Problem):
                raise TypeError(
                    f"build_constraints() must return cl.Problem, got {type(problem)}"
                )
            summary = "Constraint program compiled successfully."
            return OptimizationFeedback(success=True, summary=summary)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Constraint execution failed")
            return OptimizationFeedback(
                success=False,
                summary=f"Compilation failed: {exc}",
                raw_log=traceback.format_exc(),
            )

    return _executor


def resolve_llm_client_from_env() -> Optional[LLMClient]:
    """Instantiates a real LLM client based on environment variables."""

    provider = os.getenv("INFINIBENCH_AGENTIC_LLM", "").strip()
    if not provider:
        return None

    provider = provider.lower()
    if provider == "openai":
        model = os.getenv("INFINIBENCH_OPENAI_MODEL")
        if not model:
            raise RuntimeError(
                "Set INFINIBENCH_OPENAI_MODEL when INFINIBENCH_AGENTIC_LLM=openai."
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required to authenticate with OpenAI."
            )
        base_url = os.getenv("INFINIBENCH_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        organization = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION")
        return OpenAIChatClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    if provider == "gemini":
        model = os.getenv("INFINIBENCH_GEMINI_MODEL", "gemini-1.5-pro-latest")
        api_key = (
            os.getenv("INFINIBENCH_GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) when INFINIBENCH_AGENTIC_LLM=gemini."
            )
        return GeminiChatClient(
            model=model,
            api_key=api_key,
        )

    raise RuntimeError(f"Unsupported INFINIBENCH_AGENTIC_LLM provider: {provider}")


def build_default_agentic_generator(
    llm_client: Optional[LLMClient],
    executor: Optional[Callable[[ConstraintProgram], OptimizationFeedback]] = None,
    max_iterations: int = 3,
) -> AgenticSceneGenerator:
    """Convenience factory wiring the default API registry, examples, and optimizer."""

    if llm_client is None:
        llm_client = DummyLLM(fallback_response=HOME_CONSTRAINT_EXAMPLE)

    if executor is None:
        executor = compile_only_executor_factory()

    constraint_agent = AgenticConstraintGenerator(llm_client=llm_client)
    optimizer = SceneLayoutOptimizer(executor=executor)
    return AgenticSceneGenerator(
        constraint_agent=constraint_agent,
        optimizer=optimizer,
        max_iterations=max_iterations,
    )
