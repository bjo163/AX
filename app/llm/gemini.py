from google import genai

DESCRIPTION = "Gemini model for text generation"
VERSION = "2.0.0"
AUTHOR = "Google AI"
class GeminiClient:
    def __init__(self, api_key, model="gemini-2.0-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_content(self, prompt, system_instruction="", temperature=1.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
        """Fungsi untuk generate teks berdasarkan prompt"""
        # Konfigurasi yang sesuai dengan genai.Client
        config = {
            "system_instruction": system_instruction,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        # Hapus parameter yang tidak diterima
        config = {k: v for k, v in config.items() if v is not None}

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text