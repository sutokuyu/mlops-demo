"""Shared detection post-processing for the cat monitoring scripts."""

Box = tuple[float, float, float, float]


def box_iou(first: Box, second: Box) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def boxes_from_result(result, expected_classes: dict[int, str]) -> list[tuple[int, float, Box]]:
    boxes = []
    for box in result.boxes or []:
        class_id = int(box.cls[0])
        if class_id not in expected_classes:
            continue
        boxes.append((class_id, float(box.conf[0]), tuple(float(v) for v in box.xyxy[0])))
    return boxes


def select_best_per_class(
    boxes: list[tuple[int, float, Box]],
    confidence_threshold: float,
    cross_class_iou_threshold: float = 0.90,
) -> list[tuple[int, float, Box]]:
    """Keep at most one box per identity class, then drop the weaker duplicate.

    Only one cat is ever in frame, so extra boxes for the same class are noise.
    When bagel and kurumi boxes overlap heavily the cat was double-detected as
    both identities, so only the higher-confidence box survives.
    """
    best_by_class: dict[int, tuple[int, float, Box]] = {}
    for class_id, confidence, box in boxes:
        if confidence < confidence_threshold:
            continue
        current = best_by_class.get(class_id)
        if current is None or confidence > current[1]:
            best_by_class[class_id] = (class_id, confidence, box)

    if len(best_by_class) == 2 and cross_class_iou_threshold:
        class_ids = list(best_by_class)
        _, first_conf, first_box = best_by_class[class_ids[0]]
        _, second_conf, second_box = best_by_class[class_ids[1]]
        if box_iou(first_box, second_box) >= cross_class_iou_threshold:
            weaker = class_ids[0] if first_conf < second_conf else class_ids[1]
            del best_by_class[weaker]

    return list(best_by_class.values())


def bottom_center(box: Box) -> tuple[float, float]:
    """Anchor point used for zone lookup; the box bottom is where the cat stands."""
    return (box[0] + box[2]) / 2, box[3]
