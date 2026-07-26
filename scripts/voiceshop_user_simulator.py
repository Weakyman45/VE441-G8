#!/usr/bin/env python3
"""VoiceShop shopping-user simulator settings.

This mirrors the tau2-style user setup in three layers:

1. global guidelines: stable rules for all simulated shopping users,
2. task-specific persona/scenario: what this user wants to buy and what they know,
3. runtime persona config: interruptiveness, patience, indecision, and style knobs.

The Realtime Talker ablation runners are still backend-API experiments; this
module only defines user settings and prompt text that cases can reference or
future dynamic user simulators can consume.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from interruption_eval_common import chat_completion

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ws_proxy import connect_openai_realtime, read_frame, write_frame  # noqa: E402


GLOBAL_SHOPPING_USER_GUIDELINES = """
# VoiceShop Shopping User Simulation Guidelines

You are simulating a real shopper speaking to a voice shopping assistant.

## Core behavior
- Speak one utterance at a time, as in a live voice conversation.
- Stay within the shopping scenario. Do not invent preferences, constraints,
  product knowledge, budget, brands, or personal details not provided by the
  scenario.
- Disclose information progressively. Start with the main need, then reveal
  constraints when the assistant asks or when they become relevant.
- Use natural spoken English. Short fragments, mild disfluencies, and quick
  corrections are allowed.
- If the assistant asks for information not specified in the scenario, say you
  are not sure or have no preference.
- Keep your goal in mind until it is satisfied: find or narrow down a product
  choice that fits the scenario.

## Interruption behavior
- You may interrupt while the assistant is speaking if your runtime persona
  config says you are likely to do so.
- A true interruption should express a current need, correction, objection, or
  follow-up question.
- A backchannel is not a new instruction. Use short phrases like "mm-hmm",
  "okay", "right", or "got it" only to acknowledge listening.
- If you change your mind, state the new constraint clearly and briefly.

## Completion
- If you have enough information to decide, say so naturally.
- If the assistant has not addressed the scenario goal, keep asking.
""".strip()


@dataclass(frozen=True)
class ShoppingScenario:
    """Task-specific scenario for one simulated shopper."""

    product_category: str
    reason_for_shopping: str
    known_info: str
    unknown_info: str = ""
    task_instructions: str = ""

    def to_prompt(self) -> str:
        lines = [
            f"Product category: {self.product_category}",
            f"Reason for shopping:\n\t{self.reason_for_shopping}",
            f"Known info:\n\t{self.known_info}",
        ]
        if self.unknown_info:
            lines.append(f"Unknown info:\n\t{self.unknown_info}")
        if self.task_instructions:
            lines.append(f"Task instructions:\n\t{self.task_instructions}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RuntimePersonaConfig:
    """Runtime behavior knobs for the simulated user."""

    name: str
    interruptiveness: float
    backchannel_rate: float
    mind_change_rate: float
    indecision: float
    patience: float
    verbosity: str
    speech_style: str
    interruption_style: str

    def to_prompt(self) -> str:
        return "\n".join(
            [
                f"Runtime persona name: {self.name}",
                f"Interruptiveness: {self.interruptiveness:.2f}",
                f"Backchannel rate: {self.backchannel_rate:.2f}",
                f"Mind-change rate: {self.mind_change_rate:.2f}",
                f"Indecision: {self.indecision:.2f}",
                f"Patience: {self.patience:.2f}",
                f"Verbosity: {self.verbosity}",
                f"Speech style: {self.speech_style}",
                f"Interruption style: {self.interruption_style}",
            ]
        )


@dataclass(frozen=True)
class UserPersona:
    """Task-specific persona plus default runtime config."""

    key: str
    label: str
    task_specific_persona: str
    runtime_config: RuntimePersonaConfig
    opening_templates: tuple[str, ...]
    true_interrupt_templates: tuple[str, ...]
    backchannel_templates: tuple[str, ...]
    mind_change_templates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["runtime_config"] = asdict(self.runtime_config)
        return data


PERSONAS: dict[str, UserPersona] = {
    "impatient": UserPersona(
        key="impatient",
        label="Impatient frequent interrupter",
        task_specific_persona=(
            "You are busy and easily frustrated. You want useful product options "
            "quickly. You often cut in when the assistant is too slow, too verbose, "
            "or starts describing something you no longer care about. You may change "
            "constraints mid-conversation if a better idea occurs to you."
        ),
        runtime_config=RuntimePersonaConfig(
            name="impatient",
            interruptiveness=0.85,
            backchannel_rate=0.15,
            mind_change_rate=0.55,
            indecision=0.25,
            patience=0.20,
            verbosity="short",
            speech_style="brisk, direct, occasionally annoyed",
            interruption_style="cuts in with corrections, budget changes, or pointed follow-up questions",
        ),
        opening_templates=(
            "I need {product_category}, but please be quick. {known_info}",
            "I'm trying to buy {product_category}. Keep it short: {known_info}",
        ),
        true_interrupt_templates=(
            "Wait, no, make that cheaper.",
            "Hold on, what about the second one?",
            "No, that's not what I meant. I care more about {constraint}.",
            "Actually, forget that part. I need {constraint} now.",
        ),
        backchannel_templates=("okay", "mm-hmm", "right"),
        mind_change_templates=(
            "Actually, change the budget.",
            "Wait, I changed my mind about {constraint}.",
        ),
    ),
    "hesitant": UserPersona(
        key="hesitant",
        label="Patient but hesitant shopper",
        task_specific_persona=(
            "You are polite and patient, but you second-guess trade-offs. You often "
            "ask clarifying questions because you are unsure which feature matters "
            "most. You rarely interrupt aggressively, but you may softly interject "
            "when confused."
        ),
        runtime_config=RuntimePersonaConfig(
            name="hesitant",
            interruptiveness=0.30,
            backchannel_rate=0.45,
            mind_change_rate=0.25,
            indecision=0.80,
            patience=0.75,
            verbosity="medium",
            speech_style="polite, tentative, reflective",
            interruption_style="soft interjections and clarification requests",
        ),
        opening_templates=(
            "I'm looking for {product_category}, but I'm not totally sure what to choose. {known_info}",
            "Could you help me compare {product_category}? I'm kind of torn. {known_info}",
        ),
        true_interrupt_templates=(
            "Sorry, can I ask about the second one?",
            "Wait, I'm not sure I follow. Which one is better for {constraint}?",
            "Could you pause there? I'm confused about {constraint}.",
        ),
        backchannel_templates=("mm-hmm", "okay", "got it", "right"),
        mind_change_templates=(
            "I'm not sure anymore. Maybe {constraint} matters more.",
            "Actually, I might prefer something with {constraint}.",
        ),
    ),
    "balanced": UserPersona(
        key="balanced",
        label="Balanced typical shopper",
        task_specific_persona=(
            "You are a typical shopper. You listen most of the time, ask concise "
            "follow-up questions, and interrupt only when the assistant is missing "
            "your main constraint or when a comparison point is unclear."
        ),
        runtime_config=RuntimePersonaConfig(
            name="balanced",
            interruptiveness=0.50,
            backchannel_rate=0.30,
            mind_change_rate=0.20,
            indecision=0.45,
            patience=0.55,
            verbosity="medium-short",
            speech_style="natural, practical, cooperative",
            interruption_style="brief follow-up questions or constraint reminders",
        ),
        opening_templates=(
            "I need help choosing {product_category}. {known_info}",
            "Can you compare some {product_category} options for me? {known_info}",
        ),
        true_interrupt_templates=(
            "Wait, what about the second one?",
            "Can you compare that with the first option?",
            "Hold on, I also need {constraint}.",
        ),
        backchannel_templates=("okay", "right", "got it"),
        mind_change_templates=(
            "Actually, I may care more about {constraint}.",
            "Could we adjust for {constraint}?",
        ),
    ),
}


def build_user_simulator_prompt(
    *,
    scenario: ShoppingScenario,
    persona_key: str,
    runtime_overrides: dict[str, Any] | None = None,
) -> str:
    """Build a tau2-style system prompt for a VoiceShop simulated shopper."""
    persona = get_persona(persona_key)
    runtime_config = _override_runtime_config(persona.runtime_config, runtime_overrides or {})
    return f"""
{GLOBAL_SHOPPING_USER_GUIDELINES}

<task_specific_persona>
{persona.task_specific_persona}
</task_specific_persona>

<runtime_persona_config>
{runtime_config.to_prompt()}
</runtime_persona_config>

<shopping_scenario>
{scenario.to_prompt()}
</shopping_scenario>
""".strip()


def build_case_user_simulator_metadata(
    *,
    scenario: ShoppingScenario,
    persona_key: str,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serializable metadata to attach to experiment cases."""
    persona = get_persona(persona_key)
    runtime_config = _override_runtime_config(persona.runtime_config, runtime_overrides or {})
    return {
        "global_guidelines": GLOBAL_SHOPPING_USER_GUIDELINES,
        "task_specific_persona": persona.task_specific_persona,
        "runtime_persona_config": asdict(runtime_config),
        "shopping_scenario": asdict(scenario),
        "system_prompt": build_user_simulator_prompt(
            scenario=scenario,
            persona_key=persona_key,
            runtime_overrides=runtime_overrides,
        ),
    }


def scenario_from_case(case: dict[str, Any]) -> ShoppingScenario:
    """Recover a shopping scenario from an experiment case."""
    scenario_data = case.get("shopping_scenario")
    if not isinstance(scenario_data, dict):
        simulator_data = case.get("user_simulator") or {}
        if isinstance(simulator_data, dict):
            scenario_data = simulator_data.get("shopping_scenario")
    if isinstance(scenario_data, dict):
        return ShoppingScenario(
            product_category=str(scenario_data.get("product_category") or "shopping products"),
            reason_for_shopping=str(
                scenario_data.get("reason_for_shopping")
                or scenario_data.get("reason")
                or "The shopper wants help choosing a suitable product."
            ),
            known_info=str(
                scenario_data.get("known_info")
                or scenario_data.get("known_preferences")
                or case.get("known_info")
                or "The shopper has not provided detailed constraints yet."
            ),
            unknown_info=str(scenario_data.get("unknown_info") or ""),
            task_instructions=str(scenario_data.get("task_instructions") or ""),
        )

    return ShoppingScenario(
        product_category=str(case.get("product_category") or "shopping products"),
        reason_for_shopping=str(
            case.get("reason_for_shopping")
            or case.get("shopping_goal")
            or "The shopper wants help choosing a suitable product."
        ),
        known_info=str(
            case.get("known_info")
            or case.get("known_preferences")
            or "The shopper has not provided detailed constraints yet."
        ),
        unknown_info=str(case.get("unknown_info") or ""),
        task_instructions=str(case.get("task_instructions") or ""),
    )


def build_prompt_from_case(
    case: dict[str, Any],
    *,
    persona_key: str | None = None,
) -> str:
    """Build a simulator prompt from a case and optional persona override."""
    resolved_persona = _resolve_persona_key(case, persona_key)
    return build_user_simulator_prompt(
        scenario=scenario_from_case(case),
        persona_key=resolved_persona,
        runtime_overrides=case.get("runtime_persona_config_overrides")
        or case.get("runtime_persona_overrides")
        or None,
    )


def generate_dynamic_user_opening(
    case: dict[str, Any],
    *,
    model: str | None,
    temperature: float,
    timeout: float,
    dry_run: bool = False,
    persona_key: str | None = None,
) -> str:
    """Generate the simulated shopper's first spoken utterance."""
    if dry_run:
        return _dry_run_opening(case, persona_key=persona_key)
    prompt = build_prompt_from_case(case, persona_key=persona_key)
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Generate only the shopper's first spoken utterance to start "
                "this voice shopping conversation. Use one short sentence or "
                "two. Do not include labels, quotation marks, or analysis."
            ),
        },
    ]
    text = _user_completion(
        messages,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
    return _clean_spoken_utterance(text)


def generate_dynamic_interrupt(
    case: dict[str, Any],
    *,
    assistant_heard_text: str,
    model: str | None,
    temperature: float,
    timeout: float,
    dry_run: bool = False,
    persona_key: str | None = None,
) -> str:
    """Generate what the simulated shopper says at the barge-in point."""
    interrupt_type = str(case.get("interrupt_type") or "true_interrupt")
    if dry_run:
        return _dry_run_interrupt(case, interrupt_type=interrupt_type, persona_key=persona_key)

    initial_user_text = str(case.get("user_text") or case.get("initial_user_text") or "").strip()
    prompt = build_prompt_from_case(case, persona_key=persona_key)
    type_instruction = _interrupt_type_instruction(interrupt_type)
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "The assistant is currently speaking and the shopper may cut in.\n\n"
                f"Shopper opening:\n{initial_user_text}\n\n"
                "Assistant words heard before the cut-in:\n"
                f"{assistant_heard_text}\n\n"
                f"Interruption type to generate: {interrupt_type}\n"
                f"{type_instruction}\n\n"
                "Generate only the shopper's next spoken utterance at this exact "
                "moment. Keep it brief and natural. Do not include labels, "
                "quotation marks, or analysis."
            ),
        },
    ]
    text = _user_completion(
        messages,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
    return _clean_spoken_utterance(text)


def get_persona(persona_key: str) -> UserPersona:
    try:
        return PERSONAS[persona_key]
    except KeyError as exc:
        choices = ", ".join(sorted(PERSONAS))
        raise ValueError(f"Unknown persona {persona_key!r}; choices: {choices}") from exc


def persona_manifest_json() -> str:
    return json.dumps(
        {
            "global_guidelines": GLOBAL_SHOPPING_USER_GUIDELINES,
            "personas": {key: persona.to_dict() for key, persona in PERSONAS.items()},
        },
        ensure_ascii=False,
        indent=2,
    )


def _override_runtime_config(
    config: RuntimePersonaConfig,
    overrides: dict[str, Any],
) -> RuntimePersonaConfig:
    if not overrides:
        return config
    data = asdict(config)
    for key, value in overrides.items():
        if key in data and value is not None:
            data[key] = value
    return RuntimePersonaConfig(**data)


def _user_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None,
    temperature: float,
    timeout: float,
) -> str:
    api = (os.environ.get("OPENAI_USER_API") or "realtime").strip().lower()
    if api in ("realtime", "real-time", "rt"):
        return _realtime_text_completion(
            messages,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
    if api in ("chat", "chat_completions", "chat-completions"):
        return chat_completion(
            messages,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported OPENAI_USER_API={api!r}; use realtime or chat")


def _realtime_text_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None,
    temperature: float,
    timeout: float,
) -> str:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for OPENAI_USER_API=realtime")
    realtime_model = (
        model
        or os.environ.get("OPENAI_USER_REALTIME_MODEL")
        or os.environ.get("OPENAI_REALTIME_MODEL")
        or os.environ.get("OPENAI_CHAT_MODEL")
        or "gpt-realtime"
    ).strip()
    if "/" in realtime_model:
        provider, realtime_model = realtime_model.split("/", 1)
        if provider.lower() not in ("openai", "gpt"):
            raise ValueError(
                f"OPENAI_USER_API=realtime only supports OpenAI models, got provider={provider!r}"
            )

    instructions = "\n\n".join(
        message["content"] for message in messages if message.get("role") == "system"
    ).strip()
    user_text = "\n\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in messages
        if message.get("role") != "system"
    ).strip()
    if not user_text:
        raise ValueError("Realtime user simulator requires a user prompt")

    sock = connect_openai_realtime(api_key, realtime_model, timeout=timeout)
    started = time.time()
    text = ""
    try:
        _send_realtime_json(
            sock,
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": instructions,
                    "output_modalities": ["text"],
                },
            },
        )
        _send_realtime_json(
            sock,
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                },
            },
        )
        _send_realtime_json(sock, {"type": "response.create"})
        while time.time() - started < timeout:
            opcode, payload = read_frame(sock)
            if opcode == 0x8:
                break
            if opcode == 0x9:
                write_frame(sock, 0xA, payload, mask=True)
                continue
            if opcode != 0x1:
                continue
            event = json.loads(payload.decode("utf-8"))
            event_type = event.get("type")
            if event_type in (
                "response.output_text.delta",
                "response.text.delta",
                "response.output_audio_transcript.delta",
            ):
                text += event.get("delta") or ""
            elif event_type == "response.done":
                break
            elif event_type == "error":
                message = ((event.get("error") or {}).get("message") or event.get("message") or "")
                raise RuntimeError(f"Realtime user simulator error: {message}")
        if not text.strip():
            raise RuntimeError("Realtime user simulator returned empty text")
        return text
    finally:
        try:
            write_frame(sock, 0x8, b"", mask=True)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def _send_realtime_json(sock: Any, obj: dict[str, Any]) -> None:
    write_frame(sock, 0x1, json.dumps(obj, ensure_ascii=False).encode("utf-8"), mask=True)


def configured_user_api() -> str:
    return (os.environ.get("OPENAI_USER_API") or "realtime").strip().lower()


def configured_user_model(model: str | None = None) -> str:
    if configured_user_api() in ("realtime", "real-time", "rt"):
        return (
            model
            or os.environ.get("OPENAI_USER_REALTIME_MODEL")
            or os.environ.get("OPENAI_REALTIME_MODEL")
            or "gpt-realtime"
        ).strip()
    return (model or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-5.5").strip()


def _resolve_persona_key(case: dict[str, Any], persona_key: str | None) -> str:
    if persona_key and persona_key != "case":
        return persona_key
    raw = str(case.get("user_persona") or "balanced")
    return raw if raw in PERSONAS else "balanced"


def _interrupt_type_instruction(interrupt_type: str) -> str:
    if interrupt_type in ("backchannel", "bystander", "noise"):
        return (
            "This is not a substantive new request. Output only a very short "
            "backchannel or background-like phrase such as okay, right, mm-hmm, "
            "or got it."
        )
    if interrupt_type in ("mind_change", "constraint_change"):
        return (
            "This should be a true interruption where the shopper changes a "
            "constraint, preference, budget, or priority."
        )
    return (
        "This should be a true interruption: a correction, follow-up question, "
        "or urgent clarification based only on what the shopper has heard."
    )


def _clean_spoken_utterance(text: str) -> str:
    cleaned = text.strip()
    for prefix in ("User:", "Shopper:", "Simulated user:", "Assistant:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    return cleaned or "Wait, can you clarify that?"


def _dry_run_opening(case: dict[str, Any], *, persona_key: str | None) -> str:
    persona = get_persona(_resolve_persona_key(case, persona_key))
    scenario = scenario_from_case(case)
    template = persona.opening_templates[0]
    return template.format(
        product_category=scenario.product_category,
        known_info=scenario.known_info,
        constraint=_constraint_hint(scenario),
    )


def _dry_run_interrupt(
    case: dict[str, Any],
    *,
    interrupt_type: str,
    persona_key: str | None,
) -> str:
    persona = get_persona(_resolve_persona_key(case, persona_key))
    scenario = scenario_from_case(case)
    if interrupt_type in ("backchannel", "bystander", "noise"):
        template = persona.backchannel_templates[0]
    elif interrupt_type in ("mind_change", "constraint_change"):
        template = persona.mind_change_templates[0]
    else:
        template = persona.true_interrupt_templates[0]
    return template.format(
        product_category=scenario.product_category,
        known_info=scenario.known_info,
        constraint=_constraint_hint(scenario),
    )


def _constraint_hint(scenario: ShoppingScenario) -> str:
    text = " ".join(
        part for part in (scenario.known_info, scenario.unknown_info, scenario.task_instructions) if part
    )
    return text.split(".")[0].strip() or "the main constraint"
