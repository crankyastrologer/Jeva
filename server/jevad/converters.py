from __future__ import annotations

from jeva import Choice, Noul, Score

from . import schemas as S


def build_question(spec: S.QuestionSpec):
    if isinstance(spec, S.NoulSpec):
        return Noul(instructions=spec.instructions)
    if isinstance(spec, S.ChoiceSpec):
        return Choice(
            instructions=spec.instructions,
            criteria=spec.criteria,
            allow_null=spec.allow_null,
        )
    if isinstance(spec, S.ScoreSpec):
        return Score(instructions=spec.instructions, criteria=spec.criteria, values=spec.values)
    raise ValueError(f"Unsupported question type: {type(spec).__name__}")


def build_questions(specs: S.Questions):
    if isinstance(specs, dict):
        return {name: build_question(spec) for name, spec in specs.items()}
    return [build_question(spec) for spec in specs]
