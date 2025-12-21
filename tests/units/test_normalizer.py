from app.normalizer import normalize_arabic

def test_arabic_normalizer():
    text = "هٰذَا ـ هُوَ   الِاختِبار"
    normalized = normalize_arabic(text)
    assert "ـ" not in normalized
    assert "  " not in normalized
