"""Value types for serialized data."""

type JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
