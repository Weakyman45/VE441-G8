"""Optional Realtime function-calling tool schemas (Phase 2)."""

UPDATE_PREFERENCES_TOOL = {
    "type": "function",
    "name": "update_preferences",
    "description": "Update structured shopping preferences extracted from the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "budget": {"type": "integer"},
            "use_case": {"type": "string"},
            "platform": {"type": "string"},
        },
    },
}
