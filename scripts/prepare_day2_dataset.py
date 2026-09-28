from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SOURCE_IMAGE = Path("assets/000.png")
OUTPUT_DIR = Path("assets/bench_images")
DATASET_PATH = Path("assets/day2_benchmark.jsonl")

NUM_IMAGES = 128
IMAGE_SIZE = (640, 640)


def main():
    if not SOURCE_IMAGE.exists():
        raise FileNotFoundError(
            f"Source image not found: {SOURCE_IMAGE.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = Image.open(SOURCE_IMAGE).convert("RGB")

    # Keep every benchmark image at exactly the same resolution.
    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
    )

    dataset_lines = []

    for i in range(NUM_IMAGES):
        img = image.copy()

        # Make every image byte-wise different while keeping
        # visual content essentially unchanged.
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [0, 0, 12, 12],
            fill=(i % 256, (i * 7) % 256, (i * 13) % 256),
        )

        image_path = OUTPUT_DIR / f"image_{i:03d}.jpg"

        img.save(
            image_path,
            quality=95,
        )

        absolute_path = image_path.resolve()

        prompt = (
            "Describe this image concisely. "
            "Identify the main objects and the overall scene."
        )

        import json

        sample = {
            "prompt": prompt,
            "image_files": [str(absolute_path)],
        }

        dataset_lines.append(json.dumps(sample))

    DATASET_PATH.write_text(
        "\n".join(dataset_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Created {NUM_IMAGES} images")
    print(f"Resolution: {IMAGE_SIZE}")
    print(f"Dataset: {DATASET_PATH.resolve()}")


if __name__ == "__main__":
    main()
