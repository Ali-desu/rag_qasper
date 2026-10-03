import httpx


class Generator:
    NOT_FOUND = "I couldn't find this in the documents."

    SYSTEM_PROMPT = (
        "You answer questions about a research paper using only the documents provided.\n"
        "Rules:\n"
        "- Use only the information inside <documents>. Do not use your own knowledge.\n"
        "- After each claim, cite the section it comes from, like [Section: Experiments].\n"
        "- If the documents only partly answer the question, answer what you can and say what is missing.\n"
        f'- If the documents do not contain the answer, reply exactly: "{NOT_FOUND}"\n'
        "- Keep the answer short: a few sentences at most."
    )

    def __init__(self, model="qwen3:4b-instruct-2507-q4_K_M", url="http://localhost:11434/api/chat", temperature=0.1):
        self.model = model
        self.url = url
        self.temperature = temperature

    def build_prompt(self, question, chunks):
        """Put the retrieved chunks first, then the question."""
        documents = []
        for chunk in chunks:
            section = chunk["section"].strip() or "Untitled"
            documents.append(f'<document section="{section}">\n{chunk["text"]}\n</document>')

        return (
            "<documents>\n"
            + "\n\n".join(documents)
            + "\n</documents>\n\n"
            + f"<question>\n{question}\n</question>"
        )

    def generate(self, question, chunks):
        """Send the prompt to Ollama and return the answer text and timing stats."""
        response = httpx.post(
            self.url,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.build_prompt(question, chunks)},
                ],
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": self.temperature, "num_ctx": 4096, "num_predict": 300},
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        stats = {
            "load_s": data["load_duration"] / 1e9,
            "prompt_tokens": data["prompt_eval_count"],
            "prefill_s": data["prompt_eval_duration"] / 1e9,
            "output_tokens": data["eval_count"],
            "decode_s": data["eval_duration"] / 1e9,
            "tok_per_s": data["eval_count"] / (data["eval_duration"] / 1e9),
        }
        return data["message"]["content"].strip(), stats
        