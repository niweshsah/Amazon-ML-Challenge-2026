"""Small invariants for the matcher; run with `python3 test_core.py`."""

from __future__ import annotations

import numpy as np

from prepare import split_for
from text_features import make_record, normalize, pair_features
from train import macro_at_threshold


def main() -> None:
    assert normalize("  École—गुरु & Co. ") == "école गुरु and co"
    assert split_for("S1-123", 20260926) == split_for("S1-123", 20260926)
    left = make_record({"entity_id": "S1-a", "business_name": "Guru Private Limited", "business_address": "21 Main Rd", "country": "India"}, "S1")
    right = make_record({"entity_id": "S2-b", "business_name": "गुरु प्राइवेट लिमिटेड", "business_address": "21 Main Road", "country": "India"}, "S2")
    assert left.raw_name == "Guru Private Limited"
    assert right.name_script == "DEVANAGARI"
    assert right.transliterated_name != right.name
    assert "guru" in right.transliterated_name
    from collections import Counter
    frequency = {key: Counter() for key in ("name", "address", "name_token", "address_token")}
    features = pair_features(left, right, frequency, 1.0)
    assert features["address_number_containment"] == 1.0
    assert features["name_transliteration_similarity"] > 0
    # Query 0 is a correct singleton; query 1 has one true link and one false candidate.
    scores = np.array([0.1, 0.9, 0.8])
    labels = np.array([0, 1, 0])
    groups = np.array([0, 1, 1])
    truth = np.array([0, 1])
    result = macro_at_threshold(scores, labels, groups, truth, np.array([0, 1]), 0.85)
    assert result["macro_f0.5"] == 1.0
    result = macro_at_threshold(scores, labels, groups, truth, np.array([0, 1]), 0.5)
    assert result["macro_f0.5"] < 1.0
    print("core invariants PASS")


if __name__ == "__main__":
    main()
