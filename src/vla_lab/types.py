"""Value types for serialized data."""

from __future__ import annotations

type JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
