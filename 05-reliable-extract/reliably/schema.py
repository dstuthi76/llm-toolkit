"""
A minimal, dependency-free validation layer inspired by Pydantic's validator
pattern, for environments where you can't/won't add the dependency, or to
show the underlying mechanism clearly. In production, prefer building this
on top of `pydantic` + `instructor` directly (see repair.py's docstring) --
this module re-implements just enough of the pattern to be self-contained
and testable without extra installs.

Usage:
    @dataclass
    class Event:
        name: str
        start_date: str
        end_date: str

    @field_validator(Event)
    def end_after_start(event: Event) -> str | None:
        if event.end_date < event.start_date:
            return f"end_date ({event.end_date}) is before start_date ({event.start_date})"
        return None  # None means "valid"
"""
from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from typing import Callable, Type, TypeVar, get_type_hints

T = TypeVar("T")

_VALIDATORS: dict[type, list[Callable]] = {}


def field_validator(cls: Type[T]) -> Callable[[Callable[[T], str | None]], Callable]:
    """Register a semantic validator function for a dataclass type."""
    def decorator(fn: Callable[[T], str | None]) -> Callable:
        _VALIDATORS.setdefault(cls, []).append(fn)
        return fn
    return decorator


@dataclass
class ValidationError:
    field_or_rule: str
    message: str


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError]

    def __bool__(self) -> bool:
        return self.valid


def validate(instance) -> ValidationResult:
    """Runs both structural checks (types match) and registered semantic validators."""
    errors: list[ValidationError] = []
    cls = type(instance)

    if is_dataclass(instance):
        hints = get_type_hints(cls)
        for f in fields(instance):
            expected_type = hints.get(f.name)
            value = getattr(instance, f.name)
            if expected_type in (str, int, float, bool) and not isinstance(value, expected_type):
                errors.append(ValidationError(
                    field_or_rule=f.name,
                    message=f"expected {expected_type.__name__}, got {type(value).__name__} ({value!r})",
                ))

    for validator_fn in _VALIDATORS.get(cls, []):
        try:
            result = validator_fn(instance)
        except Exception as e:
            # A semantic validator crashing usually means the structural
            # check above already caught the real problem (e.g. a field
            # has the wrong type) -- report it as a validator failure
            # rather than letting the crash propagate and hide the
            # structural error we already recorded.
            errors.append(ValidationError(
                field_or_rule=validator_fn.__name__,
                message=f"validator raised {type(e).__name__}: {e}",
            ))
            continue
        if result is not None:
            errors.append(ValidationError(field_or_rule=validator_fn.__name__, message=result))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def grounded_in(value: str, source_text: str) -> bool:
    """
    Semantic 'grounding' check: does this extracted value actually appear
    in the source text the model was given? Useful for catching hallucinated
    IDs, names, or quotes that pass type/format validation but were invented.
    """
    return value.lower() in source_text.lower()
