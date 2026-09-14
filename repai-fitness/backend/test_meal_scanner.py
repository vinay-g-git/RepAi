"""
Pipeline test for the AI Meal Photo Scanner.

This sandbox has no live ANTHROPIC_API_KEY, so we cannot call the real model.
Instead we monkeypatch main._anthropic_client.messages.create to (a) assert the
ACTUAL uploaded image bytes are present in the outgoing request, and (b) return
a canned JSON response representative of what the real vision model would say
for each required test case, then run that through the full FastAPI endpoint
(parsing, pydantic validation, remaining-protein/calorie math, HTTP response).

This validates every part of the pipeline except the model's own visual
judgment, which requires a real API key to test live.
"""
import io
import os
import base64
import json
from types import SimpleNamespace

os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

client = TestClient(main.app)


def make_image_bytes(color, size=(120, 120), fmt="JPEG"):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def mock_response(payload_dict):
    text_block = SimpleNamespace(type="text", text=json.dumps(payload_dict))
    return SimpleNamespace(content=[text_block])


def run_case(name, image_bytes, canned_json, meal_type="Breakfast", protein_target=160, cal_target=2400, expect_status=200):
    captured = {}

    def fake_create(**kwargs):
        captured["kwargs"] = kwargs
        return mock_response(canned_json)

    main._anthropic_client.messages.create = fake_create

    files = {"file": ("meal.jpg", image_bytes, "image/jpeg")}
    data = {"meal_type": meal_type, "daily_protein_target": protein_target, "daily_calorie_target": cal_target}
    resp = client.post("/api/diet/analyze-meal-photo", files=files, data=data)

    print(f"\n=== {name} ===")
    print("HTTP status:", resp.status_code, "(expected", expect_status, ")")
    assert resp.status_code == expect_status, resp.text

    # Verify the ACTUAL uploaded image reached the model request payload.
    sent_kwargs = captured["kwargs"]
    sent_image_b64 = sent_kwargs["messages"][0]["content"][0]["source"]["data"]
    expected_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    assert sent_image_b64 == expected_b64, "Uploaded image bytes did NOT match what was sent to the vision model!"
    print("Image bytes verified in request payload: OK (", len(image_bytes), "bytes,",
          sent_kwargs["messages"][0]["content"][0]["source"]["media_type"], ")")

    body = resp.json()
    print(json.dumps(body, indent=2))
    return body


# ---------------------------------------------------------------------------
# 1. Chicken meal
# ---------------------------------------------------------------------------
run_case(
    "Chicken meal",
    make_image_bytes((180, 120, 80)),
    {
        "is_food": True,
        "detected_items": [
            {"name": "grilled chicken breast", "estimated_quantity": "1 breast", "estimated_weight_g": 180, "confidence": 0.82},
            {"name": "steamed broccoli", "estimated_quantity": "1 cup", "estimated_weight_g": 90, "confidence": 0.75},
        ],
        "nutrition": {"calories": 380, "protein_g": 42, "carbs_g": 12, "fat_g": 9, "fiber_g": 4},
        "uncertainties": [],
        "coach_feedback": "Solid high-protein plate with grilled chicken and steamed broccoli.",
    },
)

# ---------------------------------------------------------------------------
# 2. Egg meal
# ---------------------------------------------------------------------------
run_case(
    "Egg meal",
    make_image_bytes((240, 220, 150)),
    {
        "is_food": True,
        "detected_items": [
            {"name": "scrambled eggs", "estimated_quantity": "3 eggs", "estimated_weight_g": 150, "confidence": 0.88},
            {"name": "toast", "estimated_quantity": "2 slices", "estimated_weight_g": None, "confidence": 0.6},
        ],
        "nutrition": {"calories": 420, "protein_g": 24, "carbs_g": 35, "fat_g": 20, "fiber_g": 3},
        "uncertainties": ["Toast type (white vs whole wheat) is hard to tell from the photo, so carbs may vary."],
        "coach_feedback": "Good protein start; toast type is uncertain so carb estimate is rougher than usual.",
    },
)

# ---------------------------------------------------------------------------
# 3. Mixed meal (multiple distinct items)
# ---------------------------------------------------------------------------
run_case(
    "Mixed meal",
    make_image_bytes((150, 180, 100)),
    {
        "is_food": True,
        "detected_items": [
            {"name": "grilled salmon", "estimated_quantity": "1 fillet", "estimated_weight_g": 160, "confidence": 0.7},
            {"name": "white rice", "estimated_quantity": "1 cup", "estimated_weight_g": 158, "confidence": 0.65},
            {"name": "mixed greens salad", "estimated_quantity": "1 side portion", "estimated_weight_g": None, "confidence": 0.55},
        ],
        "nutrition": {"calories": 610, "protein_g": 38, "carbs_g": 55, "fat_g": 22, "fiber_g": 5},
        "uncertainties": ["Salad dressing amount not visible, so fat estimate could be higher."],
        "coach_feedback": "Balanced mixed plate: protein, carbs, and greens all represented.",
    },
)

# ---------------------------------------------------------------------------
# 4. Non-food image
# ---------------------------------------------------------------------------
run_case(
    "Non-food image",
    make_image_bytes((30, 30, 30)),
    {
        "is_food": False,
        "detected_items": [],
        "nutrition": {"calories": None, "protein_g": None, "carbs_g": None, "fat_g": None, "fiber_g": None},
        "uncertainties": ["The image does not appear to contain any food."],
        "coach_feedback": "This doesn't look like a meal photo — please upload a clear photo of your food.",
    },
)

# ---------------------------------------------------------------------------
# 5. Unclear / blurry image
# ---------------------------------------------------------------------------
run_case(
    "Unclear image",
    make_image_bytes((100, 100, 100)),
    {
        "is_food": True,
        "detected_items": [
            {"name": "unidentified food item", "estimated_quantity": None, "estimated_weight_g": None, "confidence": 0.25},
        ],
        "nutrition": {"calories": None, "protein_g": None, "carbs_g": None, "fat_g": None, "fiber_g": None},
        "uncertainties": ["Image is too blurry/dark to identify specific foods or estimate portions reliably."],
        "coach_feedback": "The photo is too unclear to analyze reliably — try retaking it in better light.",
    },
)

# ---------------------------------------------------------------------------
# 6. Malformed / non-JSON model output -> must surface as a clean error, not fake data
# ---------------------------------------------------------------------------
def fake_create_broken(**kwargs):
    return mock_response.__wrapped__ if False else SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Sorry, I can't help with that request.")]
    )

main._anthropic_client.messages.create = fake_create_broken
files = {"file": ("meal.jpg", make_image_bytes((10, 10, 10)), "image/jpeg")}
resp = client.post("/api/diet/analyze-meal-photo", files=files, data={"meal_type": "Lunch"})
print("\n=== Malformed model output ===")
print("HTTP status:", resp.status_code, "(expected 502)")
assert resp.status_code == 502
print(resp.json())

# ---------------------------------------------------------------------------
# 7. No API key configured -> must fail clearly, never silently fabricate data
# ---------------------------------------------------------------------------
main._anthropic_client = None
files = {"file": ("meal.jpg", make_image_bytes((10, 10, 10)), "image/jpeg")}
resp = client.post("/api/diet/analyze-meal-photo", files=files, data={"meal_type": "Lunch"})
print("\n=== No API key configured ===")
print("HTTP status:", resp.status_code, "(expected 503)")
assert resp.status_code == 503
print(resp.json())

# ---------------------------------------------------------------------------
# 8. Non-image file rejected before ever reaching the model
# ---------------------------------------------------------------------------
main._anthropic_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(AssertionError("model should never be called for a rejected file type"))))
resp = client.post(
    "/api/diet/analyze-meal-photo",
    files={"file": ("notes.txt", b"hello world", "text/plain")},
    data={"meal_type": "Lunch"},
)
print("\n=== Non-image file rejected ===")
print("HTTP status:", resp.status_code, "(expected 400)")
assert resp.status_code == 400
print(resp.json())

print("\nALL PIPELINE TESTS PASSED")
