from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
from .templates.langgraph_random_agent import LangGraphRandom
from .templates.langgraph_thinking import LangGraphThinking
from .templates.curiosity_agent import Curiosity
from .templates.graph_explorer_agent import GraphExplorerAgent
from .templates.graph_explorer_jepa_agent import GraphExplorerJepaAgent
from .templates.graph_explorer_structural_agent import GraphExplorerStructuralAgent
from .templates.graph_explorer_learned_agent import GraphExplorerLearnedAgent
from .templates.hypothesis_agent import Hypothesis
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.memory_agent import Memory
from .templates.multimodal import MultiModalLLM
from .templates.press_once_agent import PressOnce
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent
# GraphExplorerJepaAgent subclasses GraphExplorerAgent, not Agent directly,
# so Agent.__subclasses__() above (only direct subclasses) misses it --
# same situation as ReasoningAgent, same fix.
AVAILABLE_AGENTS["graphexplorerjepaagent"] = GraphExplorerJepaAgent
AVAILABLE_AGENTS["graphexplorerstructuralagent"] = GraphExplorerStructuralAgent
AVAILABLE_AGENTS["graphexplorerlearnedagent"] = GraphExplorerLearnedAgent

__all__ = [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "PressOnce",
    "Curiosity",
    "Memory",
    "Hypothesis",
    "GraphExplorerAgent",
    "GraphExplorerJepaAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
    "MultiModalLLM",
]
