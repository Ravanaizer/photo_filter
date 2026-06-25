import base64
import json
import mimetypes
import os
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

import config

client = OpenAI(
    api_key="dummy",
    base_url=config.API_BASE_URL,
    max_retries=1,
    timeout=300.0,
)

# Load Haar Cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def encode_image(image_path: str) -> tuple:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ("image/jpeg", "image/png", "image/webp"):
        mime_type = "image/jpeg"
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded, mime_type


def extract_json_from_text(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def opencv_prefilter(image_path: str) -> dict:
    """
    OpenCV-based prefilter. Returns a dict with flags:
    - is_rotated: likely 90° rotation
    - is_document: likely a photo of a document
    - face_ratio: how much of the image is occupied by the detected face
    - background_clutter: estimated clutter level (0-1)
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Cannot read image"}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Check orientation: portrait photos should be taller than wide
    is_rotated = w > h * 1.3  # significantly wider than tall

    # 2. Detect document-like rectangle (large, sharp-edged contour)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_document = False
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        # Large rectangular contour that doesn't fill the whole image = document
        if len(approx) == 4 and area > (w * h * 0.3) and area < (w * h * 0.95):
            is_document = True

    # 3. Detect face and compute its ratio to the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    face_ratio = 0.0
    if len(faces) > 0:
        # Take the largest face
        largest_face = max(faces, key=lambda r: r[2] * r[3])
        face_area = largest_face[2] * largest_face[3]
        face_ratio = face_area / (w * h)

    # 4. Background clutter: high frequency content outside the face area
    # Mask out the face region, measure edge density in the rest
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    for x, y, fw, fh in faces:
        cv2.rectangle(mask, (x, y), (x + fw, y + fh), 0, -1)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    clutter_score = np.count_nonzero(masked_edges) / (w * h)
    background_clutter = min(1.0, clutter_score * 5)  # normalize

    return {
        "is_rotated": is_rotated,
        "is_document": is_document,
        "face_ratio": round(face_ratio, 3),
        "background_clutter": round(background_clutter, 3),
    }


def check_photo(image_path: str) -> dict:
    """
    Two-stage check: OpenCV prefilter + LLM final verdict.
    """
    # Stage 1: OpenCV prefilter
    prefilter = opencv_prefilter(image_path)
    if "error" in prefilter:
        return {
            "is_valid": False,
            "score": 0,
            "errors": [prefilter["error"]],
            "recommendations": "",
        }

    hard_errors = []
    if prefilter["is_rotated"]:
        hard_errors.append("rotated")
    if prefilter["is_document"]:
        hard_errors.append("document_photo")
    if prefilter["face_ratio"] < 0.15:
        hard_errors.append("small_face")
    if prefilter["background_clutter"] > 0.5:
        hard_errors.append("cluttered_background")

    # If OpenCV found hard issues, reject immediately without LLM call
    if hard_errors:
        return {
            "is_valid": False,
            "score": max(0, 50 - len(hard_errors) * 15),
            "errors": hard_errors,
            "recommendations": "Сделайте обычный портрет: лицо крупно, вертикально, на простом фоне без документов и вещей.",
            "opencv_debug": prefilter,
        }

    # Stage 2: LLM final check (only if OpenCV didn't find hard issues)
    try:
        base64_image, mime_type = encode_image(image_path)
    except Exception as e:
        return {
            "is_valid": False,
            "score": 0,
            "errors": [f"File error: {str(e)}"],
            "recommendations": "",
        }

    system_prompt = (
        "Проверь фото для базы данных. Это должен быть обычный портрет человека.\n\n"
        "ОТКАЗ, если:\n"
        "- Это фото документа/бумаги/карточки/рисунок (даже если там есть лицо).\n"
        "- Помехи на фото, которые перекрывают лицо (водяные знаки, рисунки).\n"
        "- Фото повернуто (лицо на боку или вверх ногами).\n"
        "- Освещение мешает сфокусироваться на лице и различить его, либо есть боковая тень.\n"
        "- Фон захламлен: полки, коробки, техника, документы, одежда, мешающие отличить лицо от остального изображения.\n"
        "- Лицо закрыто или слишком маленькое.\n\n"
        "Верни JSON: 'is_valid' (bool), 'score' (0-100), 'errors' (list), 'recommendations' (string)."
    )

    try:
        response = client.chat.completions.create(
            model="gemma-4-e4b-it",
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Проверь это фото для базы."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            temperature=0.2,
        )

        raw_content = response.choices[0].message.content
        result = extract_json_from_text(raw_content)

        return {
            "is_valid": bool(result.get("is_valid", False)),
            "score": int(result.get("score", 0)),
            "errors": result.get("errors", [])
            if isinstance(result.get("errors"), list)
            else [str(result.get("errors", ""))],
            "recommendations": str(result.get("recommendations", "")),
            "opencv_debug": prefilter,
        }

    except json.JSONDecodeError:
        return {
            "is_valid": False,
            "score": 0,
            "errors": ["Model returned invalid JSON."],
            "recommendations": "",
        }
    except Exception as e:
        return {
            "is_valid": False,
            "score": 0,
            "errors": [f"API Error: {str(e)}"],
            "recommendations": "",
        }


if __name__ == "__main__":
    input_dir = Path("./photo")
    photos = sorted(input_dir.glob("*.jpg"))

    for photo_path in photos:
        print(photo_path)
        if not os.path.exists(photo_path):
            print(
                f"Error: '{photo_path}' not found. Please provide a valid image path."
            )
        else:
            result = check_photo(str(photo_path))
            print(json.dumps(result, indent=4, ensure_ascii=False))

        print("=" * 50)
