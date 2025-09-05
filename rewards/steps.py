# grpo_single_policy_prm/rewards/steps.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Step:
    """
    Represents a single 'step' chunk of the model's answer.
    Character spans follow Python slicing convention [start, end).
    """
    text: str
    char_span: Tuple[int, int]  # inclusive-exclusive indices into the full answer text


@dataclass
class StepSplitResult:
    """
    Result of splitting an answer string into steps.
    The `steps` list is non-empty unless the answer is empty/whitespace.
    Coverage note:
      - We normalize line endings to '\n'.
      - We 'collapse' consecutive blank lines into a single separator for *step counting*,
        but we *assign* all trailing newline runs to the preceding non-empty step's span,
        so that char coverage is continuous and includes newline tokens.
      - If max_steps_per_response is hit, we merge the tail content into the last step.
    """
    steps: List[Step]
    normalized_answer: str
    had_overflow: bool  # True if we merged tail due to max_steps_per_response cap


def _normalize_line_endings(s: str) -> str:
    # Convert CRLF -> LF, and strip nothing else.
    return s.replace("\r\n", "\n").replace("\r", "\n")


def split_steps_by_newline(
    answer_text: str,
    max_steps_per_response: int = 64,
    collapse_blank: bool = True,
) -> StepSplitResult:
    """
    Split answer into steps using literal newline '\\n' boundaries.

    Args:
        answer_text: raw generated answer (no prompt).
        max_steps_per_response: cap for number of steps; tail is merged into last step.
        collapse_blank: treat consecutive blank lines as a single separator (no empty steps).

    Returns:
        StepSplitResult with step texts and their [start, end) char spans in the *normalized* answer.
    """
    text = _normalize_line_endings(answer_text)
    n = len(text)
    if n == 0:
        return StepSplitResult(steps=[], normalized_answer=text, had_overflow=False)

    # Find non-empty segments separated by one or more '\n'.
    segments: List[Tuple[int, int]] = []  # list of (start, end_exclusive) covering non-empty content
    i = 0
    while i < n:
        # Skip any leading newlines (potential blank run)
        while i < n and text[i] == "\n":
            i += 1
        if i >= n:
            break
        # Collect a non-newline run [seg_start, seg_end)
        seg_start = i
        while i < n and text[i] != "\n":
            i += 1
        seg_end = i  # exclusive
        segments.append((seg_start, seg_end))
        # Now consume the entire newline run that follows (possibly multiple \n)
        while i < n and text[i] == "\n":
            i += 1
        # If collapse_blank=True, we simply don't emit empty segments for blank runs

    had_overflow = False
    if len(segments) == 0:
        # All newlines or whitespace: assign a single empty step covering the full text (for coverage),
        # but empty steps carry no process reward downstream.
        return StepSplitResult(
            steps=[Step(text="", char_span=(0, n))],
            normalized_answer=text,
            had_overflow=False,
        )

    if len(segments) > max_steps_per_response:
        had_overflow = True
        # Merge tail into the last allowed segment
        head = segments[: max_steps_per_response - 1]
        tail_start = segments[max_steps_per_response - 1][0]
        tail_end = segments[-1][1]
        segments = head + [(tail_start, tail_end)]

    # Build Step objects. We extend each segment's end to include the *following* newline run,
    # so that coverage includes boundary newlines. The last step ends at the true end.
    steps: List[Step] = []
    for idx, (seg_start, seg_end) in enumerate(segments):
        # Find the immediate end by walking forward from seg_end while the next chars are '\n'
        extended_end = seg_end
        j = seg_end
        while j < n and text[j] == "\n":
            extended_end = j + 1
            j += 1
        # If this is not the last non-empty segment and extended_end == n,
        # we'll shrink at most to the next segment's start to avoid overlapping spans.
        if idx + 1 < len(segments):
            next_start = segments[idx + 1][0]
            if extended_end > next_start:
                extended_end = next_start
        step_text = text[seg_start:extended_end]
        steps.append(Step(text=step_text, char_span=(seg_start, extended_end)))

    return StepSplitResult(steps=steps, normalized_answer=text, had_overflow=had_overflow)
