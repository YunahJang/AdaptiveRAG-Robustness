"""Thin parallel wrapper around the OpenAI chat-completions API.

Used by the query-variation generation scripts. Requires OPENAI_API_KEY.
"""
import os
import concurrent.futures
import backoff
from openai import OpenAI
from openai import (
    APIError,
    RateLimitError,
    APITimeoutError,
    BadRequestError,
    APIConnectionError,
    InternalServerError,
    AuthenticationError,
)

def encode_image(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class ParallelGPT:
    def __init__(self, model_id: str):
        self.model_id = model_id
        # API key is read from the environment (never hard-code it).
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Environment variable OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)

    @backoff.on_exception(
        backoff.expo,
        (
            RateLimitError,
            APIError,
            APITimeoutError,
            BadRequestError,
            APIConnectionError,
            InternalServerError,
            AuthenticationError,
        ),
        max_tries=5,
    )
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def _build_messages_text_only(self, t: str, system_prompt: str = None):
        assert isinstance(t, str) and len(t) > 0, "user content (text) must not be empty."
        msgs = [{"role": "user", "content": t}]
        # skip the system message when the prompt is falsy (None, "")
        if system_prompt:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        return msgs

    def _build_messages_with_image(self, t: str, img_b64: str, system_prompt: str = None):
        assert isinstance(t, str) and len(t) > 0, "user content (text) must not be empty."
        content = [
            {"type": "text", "text": t},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]
        msgs = [{"role": "user", "content": content}]
        if system_prompt:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        return msgs

    def generate(
        self,
        text,
        image=None,
        max_new_tokens=2048,
        temperature=1,
        num_return_sequences=1,
        system_prompt=None,
        batch_size=0,
        **kwargs,
    ):
        # normalize `text` into a list
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise TypeError(f"`text` must be str or List[str], got {type(text)}")

        if image is not None:
            if isinstance(image, str):
                image = [image]
            if len(text) != len(image):
                raise ValueError("text and image must have the same length.")

            def process_text_and_image(t, i, idx):
                base64_image = encode_image(i)
                completion = self.completion_with_backoff(
                    model=self.model_id,
                    messages=self._build_messages_with_image(t, base64_image, system_prompt),
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=num_return_sequences,
                    **kwargs,
                )
                return (completion, idx)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_text_and_image, t, i, idx) for idx, (t, i) in enumerate(zip(text, image))]
                completions = [future.result() for future in concurrent.futures.as_completed(futures)]

        else:
            def process_text(t, idx):
                completion = self.completion_with_backoff(
                    model=self.model_id,
                    messages=self._build_messages_text_only(t, system_prompt),
                    # max_tokens=max_new_tokens,   # uncomment to enforce a token limit
                    temperature=temperature,
                    n=num_return_sequences,
                    **kwargs,
                )
                return (completion, idx)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_text, t, idx) for idx, t in enumerate(text)]
                completions = [future.result() for future in concurrent.futures.as_completed(futures)]

        # restore the original order and unpack the responses
        completions_sorted = sorted(completions, key=lambda x: x[1])
        responses = [
            [c[0].choices[i].message.content for i in range(num_return_sequences)]
            for c in completions_sorted
        ]
        completions_only = [c[0] for c in completions_sorted]
        return {"responses": responses, "completions": completions_only}
