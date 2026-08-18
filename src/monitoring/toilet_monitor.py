import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
from ultralytics import YOLO


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_config_path


# ============================================================
# Paths
# ============================================================

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")

# YOLO identity detection model: this should be a fine-tuned detector that sees
# the actual observed partial-cat classes directly (e.g. bagel / kurumi).
GENERIC_DETECTION_MODEL = CONFIG["models"]["detection_model"]
IDENTITY_DETECTION_MODEL = resolve_config_path(CONFIG["models"]["identity_model_path"])
OCCUPANCY_MODEL = resolve_config_path(CONFIG["models"]["occupancy_model_path"])

if IDENTITY_DETECTION_MODEL.exists():
    DETECTION_MODEL = IDENTITY_DETECTION_MODEL
else:
    print(
        "⚠️ Identity detection model not found; falling back to the generic YOLO model: "
        f"{GENERIC_DETECTION_MODEL}"
    )
    DETECTION_MODEL = resolve_config_path(GENERIC_DETECTION_MODEL)

if not OCCUPANCY_MODEL.exists():
    raise RuntimeError(
        f"Occupancy model not found: {OCCUPANCY_MODEL}"
    )

IDENTITY_CLASSES = CONFIG["cats"]["identity_classes"]
EMPTY_CLASS = CONFIG["cats"]["occupancy_empty_class"]
OCCUPANCY_CLASSES = CONFIG["cats"]["occupancy_classes"]
UNKNOWN_IDENTITY_LABEL = CONFIG["cats"]["identity_unknown_label"]


# ============================================================
# RTSP
# ============================================================

RTSP_URL = CONFIG["camera"]["rtsp_url"]


# ============================================================
# Detection settings
# ============================================================

# Minimum YOLO confidence required to detect a cat
CAT_DETECTION_CONF = CONFIG["monitoring"]["detection_confidence"]

# Process frames every N seconds
INTERVAL = CONFIG["monitoring"]["interval_seconds"]


# ============================================================
# Enter settings
# ============================================================

# Accumulate cat detections within this time window
ENTER_WINDOW_SECONDS = CONFIG["monitoring"]["enter_window_seconds"]

# Minimum detections required to confirm a cat entered
ENTER_MIN_DETECTIONS = CONFIG["monitoring"]["enter_min_detections"]


# ============================================================
# Exit settings
# ============================================================

# After how long without detecting a cat should the occupancy model be used to check for Empty
NO_CAT_SECONDS = CONFIG["monitoring"]["no_cat_seconds"]

# How long the Empty state must persist before confirming the cat has left
EMPTY_CONFIRM_SECONDS = 2.0

# Occupancy Empty probability threshold
EMPTY_THRESHOLD = 0.70


# ============================================================
# Sample settings
# ============================================================

# Wait how long after the cat leaves before saving a sample image
SAMPLE_DELAY_SECONDS = 2.0


# ============================================================
# Output
# ============================================================

EVENTS_DIR = PROJECT_ROOT / "toilet_events"

notification_callback = None


def register_notification_callback(callback):
    global notification_callback
    notification_callback = callback


try:
    from src.notification.notification_controller import send_notification as _send_notification
    register_notification_callback(_send_notification)
    print("🔔 Notification callback registered")
except Exception as exc:
    print(f"⚠️ Notification callback not registered: {exc}")


# ============================================================
# Load models
# ============================================================

print("Loading identity detection model...")

detection_model = YOLO(
    str(DETECTION_MODEL)
)

print("Detected identity classes:")
print(detection_model.names)

print()
print("Loading occupancy classifier...")

occupancy_model = YOLO(
    str(OCCUPANCY_MODEL)
)

print("Occupancy classes:")
print(occupancy_model.names)

print()


# ============================================================
# Validate model classes
# ============================================================

identity_names = set(detection_model.names.values())
if not set(IDENTITY_CLASSES).issubset(identity_names):
    raise RuntimeError(
        "❌ identity detection model must contain "
        f"{IDENTITY_CLASSES}. "
        f"Current classes: {detection_model.names}"
    )

occupancy_names = set(occupancy_model.names.values())
if not set(OCCUPANCY_CLASSES).issubset(occupancy_names):
    raise RuntimeError(
        "❌ occupancy model must contain "
        f"{OCCUPANCY_CLASSES}. "
        f"Current classes: {occupancy_model.names}"
    )


# ============================================================
# Open RTSP
# ============================================================

print("Opening RTSP stream...")

cap = cv2.VideoCapture(
    RTSP_URL
)

if not cap.isOpened():

    raise RuntimeError(
        "❌ Cannot open RTSP"
    )


print("✅ RTSP connected")
print("Press Ctrl+C to quit")
print()


# ============================================================
# State machine
# ============================================================

state = "WAITING"


# ============================================================
# Enter detection
# ============================================================

enter_window_start = None

enter_detection_count = 0


# ============================================================
# Occupied event
# ============================================================

event_start_time = None

last_cat_detected_time = None


# ============================================================
# Identity accumulation
# ============================================================

identity_scores = {
    class_name: 0.0
    for class_name in IDENTITY_CLASSES
}

identity_samples = 0


# ============================================================
# Empty confirmation
# ============================================================

empty_confirm_start = None


# ============================================================
# Exit / sample
# ============================================================

sample_save_time = None

current_event_dir = None

best_cat_crop = None
best_cat_conf = 0.0


# ============================================================
# Utility:
# get class id from class name
# ============================================================

def get_class_id(names, class_name):

    for class_id, name in names.items():

        if name == class_name:
            return class_id

    return None


# ============================================================
# YOLO detection
#
# Return:
#   has_cat
#   cat_box
#   detection_conf
# ============================================================

def detect_cat(frame):
    """Directly detect the observed cat identity classes, e.g. bagel / kurumi.

    This bypasses the generic "cat" detector and matches the real training data,
    which is made of partial cat crops instead of full cats.
    """

    results = detection_model(
        frame,
        conf=CAT_DETECTION_CONF,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:
        return (False, None, 0.0, None)

    best_box = None
    best_conf = 0.0
    best_class = None

    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names.get(cls_id, str(cls_id))

        if class_name not in IDENTITY_CLASSES:
            continue

        if confidence > best_conf:
            best_conf = confidence
            best_class = class_name
            best_box = box.xyxy[0].cpu().numpy().astype(int)

    if best_box is None:
        return (False, None, 0.0, None)

    return (True, best_box, best_conf, best_class)


# ============================================================
# occupancy classifier
#
# Input:
#   entire toilet frame
#
# Output:
#   occupancy probabilities for empty and non-empty
# ============================================================

def classify_occupancy(frame):

    results = occupancy_model(frame, verbose=False)
    result = results[0]

    # --------------------------------------------
    # Probabilities
    # --------------------------------------------

    probs = result.probs.data.cpu().numpy()
    names = result.names

    class_probs = {}
    for class_name in OCCUPANCY_CLASSES:
        class_id = get_class_id(names, class_name)
        if class_id is None:
            raise RuntimeError(
                f"occupancy model does not contain {class_name}"
            )
        class_probs[class_name] = float(probs[class_id])

    return class_probs


# ============================================================
# Main loop
# ============================================================

last_prediction_time = 0.0


try:

    while True:

        # ====================================================
        # Read RTSP frame
        # ====================================================

        ret, frame = cap.read()


        if not ret:

            print(
                "⚠️ Failed to read RTSP frame"
            )

            time.sleep(0.1)

            continue


        last_frame = frame.copy()


        now = time.time()


        # ====================================================
        # Sampling interval
        # ====================================================

        if (
            now - last_prediction_time
            < INTERVAL
        ):

            continue


        last_prediction_time = now


        # ====================================================
        # YOLO:
        # Is there actually a cat?
        # ====================================================

        (
            has_cat,
            cat_box,
            detection_conf,
            detected_class
        ) = detect_cat(frame)


        # ====================================================
        # WAITING
        #
        # Waiting for cat to enter
        # ====================================================

        if state == "WAITING":


            if has_cat:

                print(
                    f"🐱 Cat detected "
                    f"(detection="
                    f"{detection_conf:.2f})"
                )


                # --------------------------------------------
                # Start / continue enter window
                # --------------------------------------------

                if (
                    enter_window_start
                    is None
                ):

                    enter_window_start = now

                    enter_detection_count = 1

                    print(
                        "  → Possible "
                        "cat entering..."
                    )

                else:

                    enter_detection_count += 1


                # --------------------------------------------
                # Confirm entering
                # --------------------------------------------

                elapsed = (
                    now
                    - enter_window_start
                )


                if (
                    elapsed
                    <= ENTER_WINDOW_SECONDS
                    and
                    enter_detection_count
                    >= ENTER_MIN_DETECTIONS
                ):

                    state = "OCCUPIED"


                    event_start_time = now

                    last_cat_detected_time = now


                    # Reset identity
                    identity_scores = {
                        class_name: 0.0
                        for class_name in IDENTITY_CLASSES
                    }

                    identity_samples = 0

                    best_cat_crop = None
                    best_cat_conf = 0.0

                    # Reset empty
                    empty_confirm_start = None


                    # Reset enter detector
                    enter_window_start = None

                    enter_detection_count = 0


                    print()
                    print(
                        "🟢 CAT ENTERED"
                    )
                    print()


                elif (
                    elapsed
                    > ENTER_WINDOW_SECONDS
                ):

                    # Window expired.
                    # Start again.

                    enter_window_start = now

                    enter_detection_count = 1


            else:

                print(
                    "⬜ No cat"
                )


                # --------------------------------------------
                # Reset expired enter window
                # --------------------------------------------

                if (
                    enter_window_start
                    is not None
                    and
                    now
                    - enter_window_start
                    > ENTER_WINDOW_SECONDS
                ):

                    enter_window_start = None

                    enter_detection_count = 0


        # ====================================================
        # OCCUPIED
        #
        # Cat is considered inside
        # ====================================================

        elif state == "OCCUPIED":

            # =================================================
            # Identity detection while the cat is inside
            # =================================================

            if has_cat:
                last_cat_detected_time = now
                empty_confirm_start = None

                if cat_box is not None and detection_conf > best_cat_conf:
                    x1, y1, x2, y2 = cat_box
                    h, w = frame.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    cat_crop = frame[y1:y2, x1:x2]
                    if cat_crop.size > 0:
                        best_cat_conf = detection_conf
                        best_cat_crop = cat_crop.copy()

                identity = detected_class
                if identity is not None:
                    identity_samples += 1
                    identity_scores[identity] = identity_scores.get(identity, 0.0) + detection_conf

                average_scores = {
                    class_name: identity_scores.get(class_name, 0.0) / identity_samples
                    if identity_samples > 0 else 0.0
                    for class_name in IDENTITY_CLASSES
                }
                confidence = max(average_scores.values(), default=0.0)

                print(
                    f"🐱 {identity or 'UNKNOWN'} "
                    f"det={detection_conf:.2f} "
                    f"confidence={confidence:.2f} "
                    + "  ".join(
                        f"{class_name.capitalize()}={average_scores[class_name]:.2f}"
                        for class_name in IDENTITY_CLASSES
                    )
                )
            else:
                print("⬜ No identity detection")

            # =================================================
            # Always evaluate occupancy while occupied. Only leave
            # when the toilet is empty long enough.
            # ====================================================

            occupancy_scores = classify_occupancy(frame)
            empty = occupancy_scores.get(EMPTY_CLASS, 0.0)

            print(
                "   occupancy state: "
                + "  ".join(
                    f"{class_name.capitalize()}={occupancy_scores[class_name]:.3f}"
                    for class_name in OCCUPANCY_CLASSES
                )
            )

            if empty >= EMPTY_THRESHOLD:
                if empty_confirm_start is None:
                    empty_confirm_start = now
                    print("   → Possible cat leaving...")
                else:
                    empty_duration = now - empty_confirm_start
                    print(f"   → Empty for {empty_duration:.1f}s")

                    if empty_duration >= EMPTY_CONFIRM_SECONDS:
                        state = "CLEANUP"
                        duration = now - event_start_time

                        if identity_samples == 0:
                            average_scores = {
                                class_name: 0.0 for class_name in IDENTITY_CLASSES
                            }
                            cat_name = UNKNOWN_IDENTITY_LABEL
                        else:
                            average_scores = {
                                class_name: identity_scores[class_name] / identity_samples
                                for class_name in IDENTITY_CLASSES
                            }
                            cat_name = max(average_scores, key=average_scores.get).upper()

                        print()
                        print("🔴 CAT LEFT")
                        print(f"Cat: {cat_name}")
                        print(f"Duration: {duration:.1f}s")
                        print(f"Identity samples: {identity_samples}")

                        for class_name in IDENTITY_CLASSES:
                            print(
                                f"{class_name.capitalize()} score: "
                                f"{average_scores[class_name]:.3f}"
                            )

                        event_datetime = datetime.now()
                        date_dir = EVENTS_DIR / event_datetime.strftime("%Y-%m-%d")
                        event_name = event_datetime.strftime("%H-%M-%S") + "_" + cat_name
                        current_event_dir = date_dir / event_name
                        current_event_dir.mkdir(parents=True, exist_ok=True)

                        if best_cat_crop is not None:
                            cat_image_path = current_event_dir / "cat.jpg"
                            cv2.imwrite(str(cat_image_path), best_cat_crop)
                            print(f"📸 Cat image saved: {cat_image_path}")
                        else:
                            print("⚠️ No cat image available for this event")

                        sample_save_time = now + SAMPLE_DELAY_SECONDS
                        print(f"⏳ Waiting {SAMPLE_DELAY_SECONDS:.0f} seconds...")
                        print()
            else:
                empty_confirm_start = None
                print("   → Not empty yet")




        # ====================================================
        # CLEANUP
        #
        # Wait 2 seconds after confirmed exit,
        # then save a frame for manual poop/pee labeling.
        # ====================================================

        elif state == "CLEANUP":


            if (
                sample_save_time
                is not None
                and
                now
                >= sample_save_time
            ):


                if (
                    current_event_dir
                    is not None
                ):


                    cat_image_path = None
                    if best_cat_crop is not None:
                        cat_image_path = current_event_dir / "cat.jpg"

                    sample_path = (
                        current_event_dir
                        /
                        "after_2s.jpg"
                    )


                    cv2.imwrite(
                        str(sample_path),
                        frame
                    )


                    print(
                        "📸 Toilet sample saved: "
                        f"{sample_path}"
                    )

                    if (
                        notification_callback
                        is not None
                        and current_event_dir is not None
                    ):
                        attachment_paths = []
                        if cat_image_path is not None:
                            attachment_paths.append(
                                str(cat_image_path)
                            )
                        attachment_paths.append(
                            str(sample_path)
                        )

                        try:
                            notification_callback(
                                cat_name,
                                attachment_paths
                            )
                        except Exception as exc:
                            print(
                                "⚠️ Failed to send notification:",
                                exc
                            )

                    print(
                        "👉 Please manually label:"
                    )

                    print(
                        "   poop / pee / unknown"
                    )

                    print()


                # --------------------------------------------
                # Reset everything for next event
                # --------------------------------------------

                sample_save_time = None

                state = "WAITING"


                event_start_time = None

                last_cat_detected_time = None

                empty_confirm_start = None


                identity_scores = {
                    class_name: 0.0
                    for class_name in IDENTITY_CLASSES
                }

                identity_samples = 0


                current_event_dir = None


                enter_window_start = None

                enter_detection_count = 0


                print(
                    "🟦 Ready for next "
                    "toilet event"
                )

                print()


# ============================================================
# Shutdown
# ============================================================

except KeyboardInterrupt:

    print()
    print("Stopping...")


finally:

    cap.release()

    print(
        "RTSP closed."
    )