from .llms_enums import LLMEnums
from .providers import OpenAIProvider, HuggingFaceProvider

class LLMProviderFactory:
    def __init__(self, config: dict, settings: dict):
        self.config = config
        self.settings = settings

    def create(self, provider: str):
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key = self.config.OPENAI_API_KEY,
                input_max_characters=self.settings.INPUT_DAFAULT_MAX_CHARACTERS,
                max_output_tokens=self.settings.GENERATION_DAFAULT_MAX_TOKENS,
                temperature=self.settings.GENERATION_DAFAULT_TEMPERATURE,
                top_p=self.settings.TOP_P
            )

        if provider == LLMEnums.HUGGINGFACE.value:
            return HuggingFaceProvider(
                input_max_characters=self.settings.INPUT_DAFAULT_MAX_CHARACTERS,
                max_output_tokens=self.settings.GENERATION_DAFAULT_MAX_TOKENS,
                temperature=self.settings.GENERATION_DAFAULT_TEMPERATURE,
                top_k=self.settings.TOP_K,
                top_p=self.settings.TOP_P
            )

        return None