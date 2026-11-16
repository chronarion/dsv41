# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 reasoning and spaced DSML tool calls.

LOCAL OVERLAY (James Meadlock's lab, 2026-11-06): stock file + "structural closers".

DSML has no escaping, so a tool argument that contains a DSML closer (a file *about*
DSML) used to be cut at that closer: the argument was dropped or truncated, text after
it vanished, and the streaming path emitted unterminated JSON. Idea credited to
Hector Isaza's patch 0052 in MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold
(issue #14, reported by ecohash-co); this is an independent implementation for vLLM.

Rules (TOOL_ARGS state only; reasoning and content parsing are unchanged):
  * ``</parameter>`` ends a value only when what follows is whitespace and then a new
    ``<parameter name=`` head, or the end of the call's arguments.
  * ``</invoke>`` / ``</calls>`` end a call only when its arguments are closed (they end
    with ``</parameter>``, or hold no parameter). Otherwise the closer is held: it becomes
    structure if the next thing is ``</calls>``, a new ``<invoke``, or the end of the
    stream (the open parameter is then closed, not dropped), and literal text otherwise.
  * Every closer kept as text is logged.

Kill switch: ``VLLM_DSV41_STRUCTURAL_DSML=0`` restores stock behaviour. If the private
engine API this relies on is missing, the overlay logs an error and stays stock.
"""

import functools
import json
import os
import re
from dataclasses import replace

from vllm.logger import init_logger
from vllm.parser.deepseek_v4 import (
    DeepSeekV4Parser,
    _dsml_arg_converter,
    _param_patterns,
    deepseek_v4_config,
)
from vllm.parser.engine.parser_engine_config import ParserEngineConfig

logger = init_logger(__name__)

DSML_TOOL_START = "<｜DSML｜ calls>"
DSML_TOOL_END = "</｜DSML｜ calls>"
DSML_INVOKE_PREFIX = '<｜DSML｜ invoke name="'
DSML_INVOKE_END = "</｜DSML｜ invoke>"
DSML_PARAM_START = "<｜DSML｜ parameter"
DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"

_PARAM_RE, _PARTIAL_PARAM_RE = _param_patterns(DSML_PARAM_START, DSML_PARAM_CLOSE)

OVERLAY_ID = "dsv41-structural-dsml-v1"

# ── structural argument converter ──────────────────────────────────────────

_HEAD_RE = re.compile(
    rf'{re.escape(DSML_PARAM_START)}\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>'
)
# A complete next-parameter head start, as stock's missing-closer lookahead uses.
_NEXT_HEAD_RE = re.compile(rf"{re.escape(DSML_PARAM_START)}\s+name=")
# A parameter head still being written (partial mode only): starts with the tag and
# has not reached its closing '>' yet.
_HEAD_IN_PROGRESS_RE = re.compile(rf"{re.escape(DSML_PARAM_START)}(?:\s[^>]*)?$")


def _closer_is_structural(raw: str, after: int, partial: bool) -> bool:
    follower = raw[after:].lstrip()
    if not follower:
        return True
    if _NEXT_HEAD_RE.match(follower):
        return True
    if partial and (
        DSML_PARAM_START.startswith(follower) or _HEAD_IN_PROGRESS_RE.match(follower)
    ):
        # Multi-token deltas can end a streaming tick inside the next head. Treat the
        # closer as structure for this tick; if literal text follows later the value
        # only grows, which keeps the streamed JSON prefix-stable.
        return True
    return False


def _find_value_end(raw: str, vstart: int, partial: bool) -> tuple[int, int] | None:
    """Return (value_end, resume_pos) for the value starting at vstart, or None
    when the value is still open (no structural closer and no next head)."""
    pos = vstart
    head = _NEXT_HEAD_RE.search(raw, vstart)
    while True:
        c = raw.find(DSML_PARAM_CLOSE, pos)
        if head is not None and (c < 0 or head.start() < c):
            # Stock recovery: a value whose closer is missing ends at the next head.
            return head.start(), head.start()
        if c < 0:
            return None
        after = c + len(DSML_PARAM_CLOSE)
        if _closer_is_structural(raw, after, partial):
            return c, after
        pos = after  # literal closer inside the value; keep scanning


def _store(params: dict, name: str, is_str: str | None, value: str, partial: bool):
    if is_str == "true":
        params[name] = value
        return
    try:
        params[name] = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        if not partial:
            params[name] = value
        # Partial non-string values that do not parse yet are left out, as stock does.


def _structural_dsml_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}
    pos = 0
    while True:
        m = _HEAD_RE.search(raw_args, pos)
        if not m:
            break
        name, is_str = m.group(1), m.group(2)
        found = _find_value_end(raw_args, m.end(), partial)
        if found is None:
            if partial:
                value = raw_args[m.end():]
                if is_str == "true":
                    params[name] = value
                else:
                    try:
                        params[name] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass
            break
        vend, pos = found
        _store(params, name, is_str, raw_args[m.end():vend], partial=False)
    return json.dumps(params, ensure_ascii=False)


# ── engine guard ───────────────────────────────────────────────────────────

_ENABLED = os.environ.get("VLLM_DSV41_STRUCTURAL_DSML", "1") != "0"
_V41Engine = None

try:
    from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
    from vllm.parser.engine.parser_engine_config import ParserState

    _REQUIRED = (
        "_on_terminal",
        "_emit_for_state",
        "finish",
        "reset",
        "_process_scanner_items",
        "_process_lex_tokens",
        "_record_reasoning_tokens",
    )
    _missing = [a for a in _REQUIRED if not hasattr(StreamingParserEngine, a)]
    if _missing:
        raise AttributeError(f"StreamingParserEngine lacks {_missing}")

    class _V41StructuralEngine(StreamingParserEngine):
        """Keeps DSML closers that sit inside an open parameter value as text."""

        _v41_args = ""
        _v41_held = None
        _v41_held_ws = ""

        def reset(self, initial_state=None) -> None:
            super().reset(initial_state)
            self._v41_args = ""
            self._v41_held = None  # (terminal, value, token_count)
            self._v41_held_ws = ""

        def _v41_args_closed(self) -> bool:
            s = self._v41_args.rstrip()
            return (not s) or (DSML_PARAM_START not in s) or s.endswith(DSML_PARAM_CLOSE)

        def _v41_args_open_param(self) -> bool:
            """An opened parameter head with no structural closer after it."""
            s = self._v41_args
            i = s.rfind(DSML_PARAM_START)
            if i < 0:
                return False
            m = _HEAD_RE.match(s, i)
            if not m:
                return False
            return _find_value_end(s, m.end(), partial=False) is None

        def _emit_for_state(self, text: str, token_count: int = 0):
            if self.state != ParserState.TOOL_ARGS:
                return super()._emit_for_state(text, token_count)
            events = []
            if self._v41_held is not None:
                if not text.strip():
                    self._v41_held_ws += text
                    return events
                events.extend(self._v41_release_literal())
            self._v41_args += text
            events.extend(super()._emit_for_state(text, token_count))
            return events

        def _v41_release_literal(self):
            terminal, value, _ = self._v41_held
            text = value + self._v41_held_ws
            self._v41_held = None
            self._v41_held_ws = ""
            logger.info(
                "%s: DSML %s inside a parameter value kept as text (call %d)",
                OVERLAY_ID,
                terminal,
                self.tool_index,
            )
            self._v41_args += text
            return StreamingParserEngine._emit_for_state(self, text, 0)

        def _v41_resolve_structural(self):
            terminal, value, token_count = self._v41_held
            ws = self._v41_held_ws
            self._v41_held = None
            self._v41_held_ws = ""
            events = []
            if self._v41_args_open_param():
                # Model omitted </parameter>: close it so the value is kept.
                self._v41_args += DSML_PARAM_CLOSE
                events.extend(
                    StreamingParserEngine._emit_for_state(self, DSML_PARAM_CLOSE, 0)
                )
            events.extend(super()._on_terminal(terminal, value, token_count))
            if ws:
                events.extend(self._emit_for_state(ws, 0))
            return events

        def _on_terminal(self, terminal: str, value: str, token_count: int = 0):
            if self.state == ParserState.TOOL_ARGS:
                events = []
                if self._v41_held is not None:
                    if terminal in ("TOOL_END", "INVOKE_PREFIX"):
                        events.extend(self._v41_resolve_structural())
                        events.extend(self._on_terminal(terminal, value, token_count))
                        return events
                    events.extend(self._v41_release_literal())
                if terminal in ("INVOKE_END", "TOOL_END") and not self._v41_args_closed():
                    self._v41_held = (terminal, value, token_count)
                    self._v41_held_ws = ""
                    return events
                events.extend(self._v41_on_terminal_tracked(terminal, value, token_count))
                return events
            return self._v41_on_terminal_tracked(terminal, value, token_count)

        def _v41_on_terminal_tracked(self, terminal, value, token_count):
            before = self.state
            events = super()._on_terminal(terminal, value, token_count)
            if before != ParserState.TOOL_ARGS and self.state == ParserState.TOOL_ARGS:
                self._v41_args = ""
                self._v41_held = None
                self._v41_held_ws = ""
            return events

        def finish(self):
            if self._v41_held is None:
                return super().finish()
            events = self._process_scanner_items(self._scanner.flush_pending())
            events.extend(self._process_lex_tokens(self._lexer.flush()))
            if self._v41_held is not None and self.state == ParserState.TOOL_ARGS:
                events.extend(self._v41_resolve_structural())
            self._record_reasoning_tokens(events)
            events.extend(super().finish())
            return events

    _V41Engine = _V41StructuralEngine
except Exception as e:  # noqa: BLE001
    logger.error("%s: structural DSML overlay disabled (stock behaviour): %s", OVERLAY_ID, e)
    _V41Engine = None


@functools.cache
def deepseek_v41_config(thinking: bool = False) -> ParserEngineConfig:
    config = deepseek_v4_config(thinking=thinking)
    terminal_overrides = {
        "TOOL_START": DSML_TOOL_START,
        "TOOL_END": DSML_TOOL_END,
        "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
        "INVOKE_END": DSML_INVOKE_END,
        "PARAM_START": DSML_PARAM_START,
        "PARAM_CLOSE": DSML_PARAM_CLOSE,
    }
    if _ENABLED and _V41Engine is not None:
        converter = _structural_dsml_arg_converter
    else:
        converter = functools.partial(
            _dsml_arg_converter,
            param_re=_PARAM_RE,
            partial_param_re=_PARTIAL_PARAM_RE,
        )
    return replace(
        config,
        name="deepseek_v41",
        terminals={**config.terminals, **terminal_overrides},
        token_id_terminals={
            key: terminal_overrides.get(key, value)
            for key, value in config.token_id_terminals.items()
        },
        arg_converter=converter,
    )


class DeepSeekV41Parser(DeepSeekV4Parser):
    parser_config = staticmethod(deepseek_v41_config)

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        super().__init__(tokenizer, tools, **kwargs)
        if _ENABLED and _V41Engine is not None:
            self._engine = _V41Engine(
                self.parser_engine_config, tokenizer, vocab=self.vocab
            )


if _ENABLED and _V41Engine is not None:
    logger.info("%s: structural DSML closers active", OVERLAY_ID)
