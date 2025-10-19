"""Generate an advertising slogan for the Bee-8B model.

This example demonstrates how to use the Open-Bee/Bee-8B-SFT model to
produce a text slogan from an input image and prompt. It loads the
multimodal model, downloads the logo image from Hugging Face, and then
creates an advertising slogan based on that picture.

Requirements:
    * transformers
    * torch (with bfloat16 + CUDA support recommended)
    * pillow
    * requests

Running the script will print the generated slogan to standard output. A CUDA
GPU is recommended for performance, but the example falls back to CPU
execution if CUDA is unavailable.
"""

from __future__ import annotations

import io
from typing import List, TypedDict

import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_PATH = "Open-Bee/Bee-8B-SFT"
IMAGE_URL = (
    "https://huggingface.co/Open-Bee/Bee-8B-SFT/resolve/main/assets/logo.png"
)


class ChatMessageContentImage(TypedDict):
    type: str
    image: str


class ChatMessageContentText(TypedDict):
    type: str
    text: str


class ChatMessage(TypedDict):
    role: str
    content: List[ChatMessageContentImage | ChatMessageContentText]


def load_image(url: str) -> Image.Image:
    """Download an image from a URL and load it with Pillow."""
    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load the model
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)

    # Load the processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Define conversation messages
    messages: List[ChatMessage] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMAGE_URL,
                },
                {
                    "type": "text",
                    "text": (
                        "Based on this picture, write an advertising slogan about "
                        "Bee-8B (a Fully Open Multimodal Large Language Model)."
                    ),
                },
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Load image
    image = load_image(IMAGE_URL)

    # Process inputs
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    # Generate output
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=16384,
        temperature=0.6,
    )
    output_ids = generated_ids[0][len(inputs.input_ids[0]) :]

    # Decode output
    output_text = processor.decode(output_ids, skip_special_tokens=True)

    # Print result
    print(output_text)


if __name__ == "__main__":
    main()
