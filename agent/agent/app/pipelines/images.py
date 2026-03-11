import base64
import json
from pathlib import Path

from app.services.llm_client import client, DEFAULT_MODEL

# what is the best way to understand the image using AI? My current approach is to encode the image in base64 and send it as a message with a system prompt that instructs the model to analyze the image and return a JSON with the relevant information. This way I can leverage the model's ability to understand multimodal inputs and generate a structured output that includes any readable text, a description of the image, and any identified objects or possible types.



def extract_text_and_description_from_image(file_path: Path) -> dict:
    suffix = file_path.suffix.lower()

    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }.get(suffix, "image/png")

    try:
        with open(file_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze images. "
                        "Return ONLY valid JSON. "
                        "Do not include markdown or extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image fully. "
                                "If there is readable text, extract it. "
                                "Also describe the image in plain language, list the main visible objects, "
                                "and identify the general type of image.\n\n"
                                "Return exactly this JSON:\n"
                                "{\n"
                                '  "text": "all readable text or empty string",\n'
                                '  "description": "short clear description of the whole image",\n'
                                '  "objects": ["object1", "object2"],\n'
                                '  "possible_type": "example: screenshot, document, car photo, street scene, product image"\n'
                                "}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_base64}"
                            },
                        },
                    ],
                },
            ],
        )

        content = completion.choices[0].message.content or ""
        parsed = json.loads(content)

        return {
            "status": "processed",
            "summary": "Image analyzed successfully",
            "text": parsed.get("text", ""),
            "description": parsed.get("description", ""),
            "objects": parsed.get("objects", []),
            "possible_type": parsed.get("possible_type", ""),
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": f"Image processing failed: {str(e)}",
            "text": "",
            "description": "",
            "objects": [],
            "possible_type": "",
        }