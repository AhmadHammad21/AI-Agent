from enum import Enum

class LLMEnums(Enum):
    OPENAI = "OPENAI"
    HUGGINGFACE = "HUGGINGFACE"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class HuggingFaceEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"