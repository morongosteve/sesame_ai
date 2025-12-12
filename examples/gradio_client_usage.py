"""Example showing how to call a Gradio Space with gradio_client.

This example is intentionally lightweight and demonstrates how you can
use ``gradio_client.Client`` to call a remote Gradio Space for inference.
Replace ``"your-username/your-space"`` and the ``api_name`` value with the
Space you want to target.
"""

from __future__ import annotations

from gradio_client import Client


def main() -> None:
    """Run a simple text generation call against a Gradio Space."""

    # Point this at the Space you want to call. You can find the Space
    # slug in the URL when viewing it on Hugging Face.
    space_id = "your-username/your-space"

    client = Client(space_id)

    # Every Space defines one or more endpoints. Replace "/predict" with
    # the endpoint you need and adjust the arguments to match the Space's
    # API. The example below assumes a text-only endpoint that accepts a
    # single string prompt.
    prompt = "Write a one sentence greeting for SesameAI users."
    response = client.predict(prompt, api_name="/predict")

    print("Response from Gradio Space:")
    print(response)


if __name__ == "__main__":
    main()
