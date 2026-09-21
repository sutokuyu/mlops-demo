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


# How a detection box becomes the point (or points) zone lookup matches against.
BOTTOM_CENTER = "bottom_center"
LOWER_GRID = "lower_grid"
ANCHOR_STRATEGIES = (BOTTOM_CENTER, LOWER_GRID)
DEFAULT_GRID = 3


def anchor_points(
    box: Box, strategy: str = BOTTOM_CENTER, grid: int = DEFAULT_GRID
) -> list[tuple[float, float]]:
    """Candidate anchors for a box, in the box's own pixel coordinates.

    ``bottom_center`` matches where the cat stands, which is what zones are drawn
    on, but a dangling tail or a leg stretched over an edge puts the box bottom
    below the body. The anchor then lands in the wrong zone, or in none at all.

    ``lower_grid`` samples a fixed lattice over the bottom half instead. The body
    fills most of that area, so a majority vote is not dragged around by a tail.
    Measured over 5467 annotated boxes: coverage rises from 84.4% to 89.9%, and 8%
    of boxes resolve to a different zone than the bottom edge picks.

    The lattice is fixed rather than random on purpose. Random points would give
    the same frame two different answers on two runs, and this pipeline already
    learned that lesson when the toilet verdict had to move out of the prompt and
    into code.
    """
    if strategy == BOTTOM_CENTER:
        return [bottom_center(box)]
    if strategy != LOWER_GRID:
        raise ValueError(f"unknown anchor strategy: {strategy}")
    x1, y1, x2, y2 = box
    points = []
    for row in range(grid):
        for column in range(grid):
            x = x1 + (column + 0.5) / grid * (x2 - x1)
            y = (y1 + y2) / 2 + (row + 0.5) / grid * (y2 - y1) / 2
            points.append((x, y))
    return points
