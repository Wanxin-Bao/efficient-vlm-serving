import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = Path("assets/000.png")

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY",
)


def encode_image(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


image_url = encode_image(IMAGE_PATH)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image concisely. "
                        "Identify the main objects, their appearance, "
                        "and the overall scene."
                    ),
                },
            ],
        }
    ],
    temperature=0,
    max_tokens=128,
)

print("\n=== SGLang VLM Response ===\n")
print(response.choices[0].message.content)
