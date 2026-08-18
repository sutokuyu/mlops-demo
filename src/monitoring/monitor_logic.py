def pick_best_detection(detections, identity_classes):
    """Return the highest-confidence identity-class detection."""
    if not detections:
        return None

    valid = [
        detection
        for detection in detections
        if detection.get("class_name") in identity_classes
    ]
    if not valid:
        return None

    return max(valid, key=lambda item: float(item.get("confidence", 0.0)))


def decide_cat_name(votes):
    """Choose the identity with the highest accumulated votes."""
    if not votes:
        return "UNKNOWN"
    return max(votes, key=votes.get).upper()


def should_confirm_exit(empty_probability, empty_threshold, empty_duration, empty_confirm_seconds):
    """Return True only when the toilet is empty with enough persistence."""
    return (
        float(empty_probability) >= float(empty_threshold)
        and float(empty_duration) >= float(empty_confirm_seconds)
    )
