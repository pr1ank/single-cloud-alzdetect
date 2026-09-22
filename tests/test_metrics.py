import pytest
from backend.metrics_service import classification_metrics, report_frame


def test_perfect_classification():
    result = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
        assert result[key] == 1
    assert sum(sum(row) for row in result["confusion_matrix"]) == 4


def test_known_imbalanced_predictions():
    result = classification_metrics([0, 0, 1, 2, 3], [0, 0, 0, 0, 0])
    assert result["accuracy"] == .4
    assert result["precision_macro"] == .1
    assert result["recall_macro"] == .25
    assert result["f1_macro"] == pytest.approx((4 / 7) / 4)
    assert result["f1_weighted"] == pytest.approx((4 / 7) * .4)


def test_missing_predicted_classes_are_explicit():
    result = classification_metrics([0, 1], [0, 0])
    assert len(result["confusion_matrix"]) == 4
    assert result["classification_report"]["ModerateDemented"]["recall"] == 0


def test_report_accuracy_row_has_actual_support():
    metrics = classification_metrics([0, 1, 2, 3], [0, 1, 1, 3])
    frame = report_frame(metrics)
    assert frame.loc["accuracy", "support"] == 4
    assert frame.loc["accuracy", "f1-score"] == .75
