import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

# 改成你自己的图片文件名
IMAGE_PATH = Path("assets/000.png")


def encode_image_to_data_url(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")

    mime_type, _ = mimetypes.guess_type(image_path.name)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_base64}"


def main():
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )

    image_data_url = encode_image_to_data_url(IMAGE_PATH)

    print("=== Input ===")
    print(f"Image path : {IMAGE_PATH.resolve()}")
    print(f"Image size : {IMAGE_PATH.stat().st_size / 1024:.2f} KB")
    print(f"Model      : {MODEL_NAME}")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
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
        temperature=0.0,
        max_tokens=128,
    )

    print("\n=== Model Response ===")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
