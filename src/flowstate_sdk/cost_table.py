COST_TABLE = {
    "openai": {
        "gpt-5.2": {
            "input": 0.00000175,
            "output": 0.000014,
        },
        "gpt-5.1": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gpt-5": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gpt-5-mini": {
            "input": 0.00000025,
            "output": 0.000002,
        },
        "gpt-5-nano": {
            "input": 0.00000005,
            "output": 0.0000004,
        },
        "gpt-5.2-chat-latest": {
            "input": 0.00000175,
            "output": 0.000014,
        },
        "gpt-5.1-chat-latest": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gpt-5-chat-latest": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gpt-5.2-codex": {
            "input": 0.00000175,
            "output": 0.000014,
        },
        "gpt-5.1-codex-max": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gpt-5.1-codex": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gpt-5-codex": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gpt-5.2-pro": {
            "input": 0.000021,
            "output": 0.000168,
        },
        "gpt-5-pro": {
            "input": 0.000015,
            "output": 0.00012,
        },
        "gpt-4.1": {
            "input": 0.000002,
            "output": 0.000008,
        },
        "gpt-4.1-mini": {
            "input": 0.0000004,
            "output": 0.0000016,
        },
        "gpt-4.1-nano": {
            "input": 0.0000001,
            "output": 0.0000004,
        },
        "gpt-4o": {
            "input": 0.0000025,
            "output": 0.00001,
        },
        "gpt-4o-2024-05-13": {
            "input": 0.000005,
            "output": 0.000015,
        },
        "gpt-4o-mini": {
            "input": 0.00000015,
            "output": 0.0000006,
        },
        "gpt-realtime": {
            "input": 0.000004,
            "output": 0.000016,
        },
        "gpt-realtime-mini": {
            "input": 0.0000006,
            "output": 0.0000024,
        },
        "gpt-4o-realtime-preview": {
            "input": 0.000005,
            "output": 0.00002,
        },
        "gpt-4o-mini-realtime-preview": {
            "input": 0.0000006,
            "output": 0.0000024,
        },
        "gpt-audio": {
            "input": 0.0000025,
            "output": 0.00001,
        },
        "gpt-audio-mini": {
            "input": 0.0000006,
            "output": 0.0000024,
        },
        "gpt-4o-audio-preview": {
            "input": 0.0000025,
            "output": 0.00001,
        },
        "gpt-4o-mini-audio-preview": {
            "input": 0.00000015,
            "output": 0.0000006,
        },
        "o1": {
            "input": 0.000015,
            "output": 0.00006,
        },
        "o1-pro": {
            "input": 0.00015,
            "output": 0.0006,
        },
        "o3-pro": {
            "input": 0.00002,
            "output": 0.00008,
        },
        "o3": {
            "input": 0.000002,
            "output": 0.000008,
        },
        "o3-deep-research": {
            "input": 0.00001,
            "output": 0.00004,
        },
        "o4-mini": {
            "input": 0.0000011,
            "output": 0.0000044,
        },
        "o4-mini-deep-research": {
            "input": 0.000002,
            "output": 0.000008,
        },
        "o3-mini": {
            "input": 0.0000011,
            "output": 0.0000044,
        },
        "o1-mini": {
            "input": 0.0000011,
            "output": 0.0000044,
        },
        "gpt-5.1-codex-mini": {
            "input": 0.00000025,
            "output": 0.000002,
        },
        "codex-mini-latest": {
            "input": 0.0000015,
            "output": 0.000006,
        },
        "gpt-5-search-api": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gpt-4o-mini-search-preview": {
            "input": 0.00000015,
            "output": 0.0000006,
        },
        "gpt-4o-search-preview": {
            "input": 0.0000025,
            "output": 0.00001,
        },
        "computer-use-preview": {
            "input": 0.000003,
            "output": 0.000012,
        },
        "gpt-image-1.5": {
            "input": 0.000005,
            "output": 0.00001,
        },
        "chatgpt-image-latest": {
            "input": 0.000005,
            "output": 0.00001,
        },
        "text-embedding-3-small": {
            "input": 0.00000002,
            "output": 0,
        },
        "text-embedding-3-large": {
            "input": 0.00000013,
            "output": 0,
        },
        "text-embedding-ada-002": {
            "input": 0.0000001,
            "output": 0,
        },
    },
    "anthropic": {
        "claude-opus-4-5": {
            "input": 0.000015,
            "output": 0.000075,
        },
        "claude-opus-4-5-20251101": {
            "input": 0.000015,
            "output": 0.000075,
        },
        "claude-opus-4-1-20250805": {
            "input": 0.000015,
            "output": 0.000075,
        },
        "claude-opus-4-20250514": {
            "input": 0.000015,
            "output": 0.000075,
        },
        "claude-sonnet-4-5": {
            "input": 0.000003,
            "output": 0.000015,
        },
        "claude-sonnet-4-5-20250929": {
            "input": 0.000003,
            "output": 0.000015,
        },
        "claude-sonnet-4-20250514": {
            "input": 0.000003,
            "output": 0.000015,
        },
        "claude-3-7-sonnet-20250219": {
            "input": 0.000003,
            "output": 0.000015,
        },
        "claude-haiku-4-5": {
            "input": 0.000001,
            "output": 0.000005,
        },
        "claude-haiku-4-5-20251001": {
            "input": 0.000001,
            "output": 0.000005,
        },
        "claude-3-5-haiku-latest": {
            "input": 0.0000008,
            "output": 0.000004,
        },
        "claude-3-5-haiku-20241022": {
            "input": 0.0000008,
            "output": 0.000004,
        },
        "claude-3-opus-20240229": {
            "input": 0.000015,
            "output": 0.000075,
        },
        "claude-3-haiku-20240307": {
            "input": 0.00000025,
            "output": 0.00000125,
        },
    },
    "google": {
        "gemini-3-pro-preview": {
            "input": 0.000002,
            "output": 0.000012,
        },
        "gemini-3-flash-preview": {
            "input": 0.0000005,
            "output": 0.000003,
        },
        "gemini-2.5-pro": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gemini-2.5-flash": {
            "input": 0.0000003,
            "output": 0.0000025,
        },
        "gemini-2.5-flash-lite": {
            "input": 0.0000001,
            "output": 0.0000004,
        },
        "gemini-2.5-flash-lite-preview-09-2025": {
            "input": 0.0000001,
            "output": 0.0000004,
        },
        "gemini-2.0-flash": {
            "input": 0.0000001,
            "output": 0.0000004,
        },
        "gemini-2.0-flash-lite": {
            "input": 0.000000075,
            "output": 0.0000003,
        },
        "gemini-2.5-computer-use-preview-10-2025": {
            "input": 0.00000125,
            "output": 0.00001,
        },
        "gemini-robotics-er-1.5-preview": {
            "input": 0.0000003,
            "output": 0.0000025,
        },
        "gemini-embedding-001": {
            "input": 0.00000015,
            "output": 0,
        },
        "gemini-2.5-flash-preview-tts": {
            "input": 0.0000005,
            "output": 0.00001,
        },
        "gemini-2.5-pro-preview-tts": {
            "input": 0.000001,
            "output": 0.00002,
        },
    },
}
