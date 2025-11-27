import cv2
import mediapipe as mp
import math
import time
import platform
from collections import deque

# ================== AYARLAR ==================

USE_CALIBRATION = True         # Açılışta kısa göz/head-pose kalibrasyonu
CALIBRATION_DURATION = 5.0     # sn

# Varsayılan eşikler (kalibrasyon olmazsa)
DEFAULT_EAR_THRESH = 0.23      # Göz kapalı eşiği
DEFAULT_MOUTH_THRESH = 0.70    # Ağız açık (esneme) eşiği (MAR)

FRAME_CHECK_EYE = 15           # Göz kaç frame kapalı -> uyku uyarısı
FRAME_CHECK_MOUTH = 20         # Ağız kaç frame çok açık -> esneme uyarısı

# Blink tespiti (kısa göz kapama)
BLINK_MAX_FRAMES = 6           # 1–6 frame arası kapanma blink sayılır

# Head pose / dikkat dağınıklığı
HEAD_YAW_THRESH = 0.10         # sağ/sol bakış eşiği (normalize)
HEAD_PITCH_THRESH = 0.15       # aşağı bakış eşiği (normalize)
HEAD_DISTRACT_FRAMES = 90      # ~3 sn yan/aşağı bakış

# Alarm cooldown
ALARM_COOLDOWN = 0.05          # sn

# Yorgunluk skoru için zaman penceresi
EVENT_WINDOW_SEC = 600         # 10 dakika

# ================== RENK PALETİ (BGR) ==================

COLOR_BG_DARK = (15, 15, 20)
COLOR_OVERLAY = (10, 10, 15)
COLOR_ACCENT = (60, 180, 255)
COLOR_ACCENT_SOFT = (100, 200, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_WARNING = (0, 165, 255)
COLOR_OK = (0, 200, 0)
COLOR_TEXT_MAIN = (240, 240, 240)
COLOR_TEXT_MUTED = (170, 170, 170)

# ================== SESLER ===================

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def sound_alarm_sleep():
    if HAS_WINSOUND and platform.system() == "Windows":
        winsound.Beep(2500, 700)
    else:
        print('\a[UYKU ALARMI]', end='', flush=True)


def sound_alarm_yawn():
    if HAS_WINSOUND and platform.system() == "Windows":
        winsound.Beep(1500, 300)
        winsound.Beep(1000, 300)
    else:
        print('\a[YORGUNLUK (ESNEME) ALARMI]', end='', flush=True)


def sound_alarm_distract():
    if HAS_WINSOUND and platform.system() == "Windows":
        winsound.Beep(2000, 300)
        winsound.Beep(2000, 300)
    else:
        print('\a[DIKKAT DAGINIKLIGI]', end='', flush=True)


# ================ HESAPLAYICILAR ================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def euclidean_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        pts.append((x, y))

    p1, p2, p3, p4, p5, p6 = pts
    A = euclidean_dist(p2, p6)
    B = euclidean_dist(p3, p5)
    C = euclidean_dist(p1, p4)

    ear = (A + B) / (2.0 * C + 1e-6)
    return ear, pts


def mouth_aspect_ratio(landmarks, image_w, image_h):
    idx_left = 78
    idx_right = 308
    idx_top = 13
    idx_bottom = 14

    lm_left = landmarks[idx_left]
    lm_right = landmarks[idx_right]
    lm_top = landmarks[idx_top]
    lm_bottom = landmarks[idx_bottom]

    left = (int(lm_left.x * image_w), int(lm_left.y * image_h))
    right = (int(lm_right.x * image_w), int(lm_right.y * image_h))
    top = (int(lm_top.x * image_w), int(lm_top.y * image_h))
    bottom = (int(lm_bottom.x * image_w), int(lm_bottom.y * image_h))  # ✅ DÜZGÜN

    horiz = euclidean_dist(left, right)
    vert = euclidean_dist(top, bottom)

    mar = vert / (horiz + 1e-6)
    pts = [left, right, top, bottom]
    return mar, pts, (left, right, top, bottom)



def cleanup_events(deq, now, window_sec):
    while deq and (now - deq[0] > window_sec):
        deq.popleft()


def compute_fatigue_score(
    sleep_events, yawn_events, distract_events, blink_events, now, window_sec
):
    for dq in (sleep_events, yawn_events, distract_events, blink_events):
        cleanup_events(dq, now, window_sec)

    c_sleep = len(sleep_events)
    c_yawn = len(yawn_events)
    c_distract = len(distract_events)
    c_blink = len(blink_events)

    minutes = window_sec / 60.0 if window_sec > 0 else 1.0
    blink_rate = c_blink / minutes

    sleep_score = min(50, c_sleep * 15)
    yawn_score = min(25, c_yawn * 5)
    head_score = min(15, c_distract * 5)

    blink_score = 0
    if blink_rate < 10:
        blink_score = min(10, int((10 - blink_rate) * 2))

    fatigue = int(min(100, sleep_score + yawn_score + head_score + blink_score))
    return fatigue, blink_rate


# ============== KALİBRASYON ==============

def calibrate_thresholds(cap, face_mesh, left_eye_idx, right_eye_idx):
    print("\n=== KALIBRASYON BASLIYOR ===")
    print("Lutfen kameraya bak, gozlerin acik, agzin kapali ve duz karsiya bak.")
    print(f"Yaklasik {CALIBRATION_DURATION} saniye surecek...\n")

    start_time = time.time()
    ear_values = []
    nose_yaw_norm_values = []
    mouth_eye_dy_values = []

    NOSE_TIP_IDX = 4
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    while True:
        if time.time() - start_time > CALIBRATION_DURATION:
            break

        ret, frame = cap.read()
        if not ret:
            print("Kalibrasyon icin kare okunamadi, varsayilan esikler kullanilacak.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        cv2.putText(
            frame,
            "Kalibrasyon: gozler acik, agiz kapali, duz karsiya bak",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_ACCENT_SOFT,
            2
        )

        kalan = max(0, int(CALIBRATION_DURATION - (time.time() - start_time)))
        cv2.putText(
            frame,
            f"Kalan sure: {kalan} sn",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_OK,
            2
        )

        if results.multi_face_landmarks:
            lm_list = results.multi_face_landmarks[0].landmark

            left_ear, _ = eye_aspect_ratio(lm_list, left_eye_idx, w, h)
            right_ear, _ = eye_aspect_ratio(lm_list, right_eye_idx, w, h)
            ear = (left_ear + right_ear) / 2.0

            mar, _, (left_m, right_m, top_m, bottom_m) = mouth_aspect_ratio(
                lm_list, w, h
            )

            lm_left_eye = lm_list[LEFT_EYE_OUTER]
            lm_right_eye = lm_list[RIGHT_EYE_OUTER]
            lx, ly = int(lm_left_eye.x * w), int(lm_left_eye.y * h)
            rx, ry = int(lm_right_eye.x * w), int(lm_right_eye.y * h)
            eye_center_y = (ly + ry) / 2.0
            eye_dist = euclidean_dist((lx, ly), (rx, ry)) + 1e-6

            mouth_center_y = (top_m[1] + bottom_m[1]) / 2.0
            dy_mouth_eye = mouth_center_y - eye_center_y

            nose = lm_list[NOSE_TIP_IDX]
            nx = int(nose.x * w)
            face_center_x = (lx + rx) / 2.0
            nose_offset = nx - face_center_x
            yaw_norm = nose_offset / eye_dist

            if mar < 0.4 and ear > 0.15:
                ear_values.append(ear)
                nose_yaw_norm_values.append(yaw_norm)
                mouth_eye_dy_values.append(dy_mouth_eye)

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}  MAR: {mar:.3f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT_MAIN,
                2
            )
        else:
            cv2.putText(
                frame,
                "Yuz gorunmuyor, biraz geri gel / aydinlik yere gec",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_DANGER,
                1
            )

        cv2.imshow("Kalibrasyon", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Kalibrasyon iptal. Varsayilan esikler kullanilacak.")
            break

    cv2.destroyWindow("Kalibrasyon")

    if not ear_values:
        print("Yeterli veri alinmadi, varsayilan esikler kullanilacak.")
        return DEFAULT_EAR_THRESH, None, None

    avg_ear = sum(ear_values) / len(ear_values)
    ear_thresh = avg_ear * 0.75

    if nose_yaw_norm_values:
        baseline_yaw = sum(nose_yaw_norm_values) / len(nose_yaw_norm_values)
    else:
        baseline_yaw = 0.0

    if mouth_eye_dy_values:
        baseline_dy = sum(mouth_eye_dy_values) / len(mouth_eye_dy_values)
    else:
        baseline_dy = None

    print(f"\nKalibrasyon tamamlandi.")
    print(f"Ortalama EAR: {avg_ear:.3f}  -> Esik: {ear_thresh:.3f}")
    print(f"Baseline yaw: {baseline_yaw:.3f}, baseline dy: {baseline_dy}\n")

    return ear_thresh, baseline_dy, baseline_yaw


# ================== MODERN UI ÇİZİMİ ==================

def draw_chip(frame, x, y, w, h, text, value, color, font_scale):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_OVERLAY, -1)
    frame[:] = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    radius = int(h * 0.25)
    cv2.circle(frame, (x + radius + 6, y + h // 2), radius, color, -1)

    cv2.putText(
        frame,
        text,
        (x + radius * 2 + 10, y + int(h * 0.45)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.8,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        value,
        (x + radius * 2 + 10, y + int(h * 0.80)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA
    )



def draw_modern_ui(
    frame,
    fatigue_score,
    blink_count,
    blink_rate,
    total_sleep_alerts,
    total_yawn_alerts,
    total_distract_alerts,
    eye_state,
    mouth_state,
    head_state,
    status_text
):
    h, w = frame.shape[:2]
    ui_scale = max(0.6, min(1.6, h / 720.0))

    # Arka planı yumuşak karart
    bg = frame.copy()
    bg[:] = COLOR_BG_DARK
    frame[:] = cv2.addWeighted(frame, 0.7, bg, 0.3, 0)

    # Üst bar
    top_h = int(0.12 * h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_h), (10, 10, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    cv2.putText(
        frame,
        "Surucu Yorgunluk Izleme Prototipi",
        (int(0.02 * w), int(0.06 * h)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * ui_scale,
        COLOR_TEXT_MAIN,
        2,
        cv2.LINE_AA
    )

    # Yorgunluk skoru barı
    if fatigue_score < 30:
        fatigue_color = COLOR_OK
    elif fatigue_score < 60:
        fatigue_color = (0, 255, 255)
    else:
        fatigue_color = COLOR_DANGER

    bar_w = int(0.32 * w)
    bar_x1 = w - bar_w - int(0.02 * w)
    bar_x2 = w - int(0.02 * w)
    bar_y1 = int(0.03 * h)
    bar_y2 = bar_y1 + int(0.05 * h)

    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 60), -1)
    inner_width = bar_x2 - bar_x1 - 6
    fill_width = int(inner_width * fatigue_score / 100.0)
    cv2.rectangle(
        frame,
        (bar_x1 + 3, bar_y1 + 3),
        (bar_x1 + 3 + fill_width, bar_y2 - 3),
        fatigue_color,
        -1
    )
    cv2.putText(
        frame,
        f"{fatigue_score}/100",
        (bar_x1 + 10, bar_y1 + int(0.035 * h)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * ui_scale,
        COLOR_TEXT_MAIN,
        2,
        cv2.LINE_AA
    )

    # Sol kartlar
    chip_w = int(0.28 * w)
    chip_h = int(0.06 * h)
    chip_x = int(0.02 * w)
    chip_y = top_h + int(0.03 * h)
    chip_gap = int(0.015 * h)
    font_scale = 0.7 * ui_scale

    draw_chip(frame, chip_x, chip_y, chip_w, chip_h, "Goz Durumu",
              eye_state, COLOR_OK if eye_state == "ACIK" else COLOR_DANGER, font_scale)

    draw_chip(frame, chip_x, chip_y + chip_h + chip_gap, chip_w, chip_h,
              "Agiz / Esneme",
              mouth_state,
              COLOR_WARNING if "ESNEME" in mouth_state else COLOR_OK,
              font_scale)

    head_color = COLOR_OK
    if "YAN" in head_state or "ASAGI" in head_state:
        head_color = COLOR_WARNING
    draw_chip(frame, chip_x, chip_y + 2 * (chip_h + chip_gap), chip_w, chip_h,
              "Bas Pozu", head_state, head_color, font_scale)

    # Ortada status
    cv2.putText(
        frame,
        status_text,
        (chip_x, chip_y + 3 * (chip_h + chip_gap) + int(0.03 * h)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9 * ui_scale,
        COLOR_TEXT_MAIN,
        2,
        cv2.LINE_AA
    )

    # Alt HUD
    hud_h = int(0.16 * h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (5, 5, 10), -1)
    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    hud_y = h - hud_h + int(0.045 * h)
    cv2.putText(
        frame,
        f"Uyku uyarisi: {total_sleep_alerts}   Esneme uyarisi: {total_yawn_alerts}   Dikkat uyarisi: {total_distract_alerts}",
        (int(0.02 * w), hud_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65 * ui_scale,
        COLOR_TEXT_MAIN,
        2,
        cv2.LINE_AA
    )
    hud_y += int(0.045 * h)
    cv2.putText(
        frame,
        f"Blink: {blink_count}   Son 10 dk blink orani: {blink_rate:.1f}/dk   |   'Q' ile cikis",
        (int(0.02 * w), hud_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * ui_scale,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA
    )


# ================= ANA UYGULAMA =================

def main():
    print("Kamera açılıyor...")
    print("Cikmak icin pencere aktifken 'q' ya bas.")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    NOSE_TIP_IDX = 4
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera acilamadi.")
        return

    # Gerekirse kamera çözünürlüğünü biraz yükselt (daha az pixel görünümü için)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if USE_CALIBRATION:
        EAR_THRESH, baseline_dy, baseline_yaw = calibrate_thresholds(
            cap, face_mesh, LEFT_EYE_IDX, RIGHT_EYE_IDX
        )
    else:
        EAR_THRESH = DEFAULT_EAR_THRESH
        baseline_dy = None
        baseline_yaw = 0.0

    MOUTH_THRESH = DEFAULT_MOUTH_THRESH

    eye_closed_frame_count = 0
    blink_count = 0
    open_mouth_frame_count = 0
    distract_frames = 0

    alarm_sleep_on = False
    alarm_yawn_on = False
    alarm_distract_on = False

    last_sleep_alarm_time = 0.0
    last_yawn_alarm_time = 0.0
    last_distract_alarm_time = 0.0

    total_sleep_alerts = 0
    total_yawn_alerts = 0
    total_distract_alerts = 0

    sleep_events = deque()
    yawn_events = deque()
    distract_events = deque()
    blink_events = deque()

    win_name = "Surucu Yorgunluk Izleme"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_head_state_label = "NORMAL"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status_text = "YUZ ALGILANMADI"
            eye_state_label = "BILINMIYOR"
            mouth_state_label = "BILINMIYOR"
            head_state_label = last_head_state_label

            now = time.time()

            if results.multi_face_landmarks:
                face_landmarks_obj = results.multi_face_landmarks[0]
                lm_list = face_landmarks_obj.landmark

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks_obj,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                ear, mar = None, None

                left_ear, left_pts = eye_aspect_ratio(lm_list, LEFT_EYE_IDX, w, h)
                right_ear, right_pts = eye_aspect_ratio(lm_list, RIGHT_EYE_IDX, w, h)
                ear = (left_ear + right_ear) / 2.0

                for p in left_pts + right_pts:
                    cv2.circle(frame, p, 1, (0, 255, 0), -1)

                mar, mouth_pts, (left_m, right_m, top_m, bottom_m) = mouth_aspect_ratio(
                    lm_list, w, h
                )
                for p in mouth_pts:
                    cv2.circle(frame, p, 2, (255, 0, 255), -1)

                # Göz
                if ear is not None and ear < EAR_THRESH:
                    eye_closed_frame_count += 1
                    eye_state_label = "KAPALI"
                    status_text = "GOZLER KAPALI"
                else:
                    if 1 <= eye_closed_frame_count <= BLINK_MAX_FRAMES:
                        blink_count += 1
                        blink_events.append(now)
                    eye_closed_frame_count = 0
                    alarm_sleep_on = False
                    eye_state_label = "ACIK"
                    status_text = "GOZLER ACIK"

                if eye_closed_frame_count >= FRAME_CHECK_EYE:
                    if (not alarm_sleep_on) or (now - last_sleep_alarm_time) > ALARM_COOLDOWN:
                        sound_alarm_sleep()
                        last_sleep_alarm_time = now
                        alarm_sleep_on = True
                        total_sleep_alerts += 1
                        sleep_events.append(now)
                    status_text = "UYKU TEHLIKESI!"

                if mar is not None and mar > MOUTH_THRESH and mar < 2.0:
                    open_mouth_frame_count += 1
                else:
                    open_mouth_frame_count = 0
                    alarm_yawn_on = False                
                # Ağız / esneme
                if mar is not None and mar > MOUTH_THRESH:
                    open_mouth_frame_count += 1
                    mouth_state_label = "ESNEME"
                else:
                    open_mouth_frame_count = 0
                    alarm_yawn_on = False
                    if mouth_state_label != "ESNEME":
                        mouth_state_label = "NORMAL"

                if open_mouth_frame_count >= FRAME_CHECK_MOUTH:
                    if (not alarm_yawn_on) or (now - last_yawn_alarm_time) > ALARM_COOLDOWN:
                        sound_alarm_yawn()
                        last_yawn_alarm_time = now
                        alarm_yawn_on = True
                        total_yawn_alerts += 1
                        yawn_events.append(now)
                    status_text = "YORGUNLUK (ESNEME)"

                # Head pose
                lm_left_eye = lm_list[LEFT_EYE_OUTER]
                lm_right_eye = lm_list[RIGHT_EYE_OUTER]
                lx, ly = int(lm_left_eye.x * w), int(lm_left_eye.y * h)
                rx, ry = int(lm_right_eye.x * w), int(lm_right_eye.y * h)
                eye_center_y = (ly + ry) / 2.0
                eye_dist = euclidean_dist((lx, ly), (rx, ry)) + 1e-6

                mouth_center_y = (top_m[1] + bottom_m[1]) / 2.0
                dy_mouth_eye = mouth_center_y - eye_center_y

                nose = lm_list[NOSE_TIP_IDX]
                nx = int(nose.x * w)
                face_center_x = (lx + rx) / 2.0
                nose_offset = nx - face_center_x
                yaw_norm = nose_offset / eye_dist

                if baseline_dy is None:
                    baseline_dy_local = dy_mouth_eye
                else:
                    baseline_dy_local = baseline_dy

                baseline_yaw_local = baseline_yaw if baseline_yaw is not None else 0.0

                yaw_dev = yaw_norm - baseline_yaw_local
                pitch_dev = (dy_mouth_eye - baseline_dy_local) / (abs(baseline_dy_local) + 1e-6)

                distracted = False
                if abs(yaw_dev) > HEAD_YAW_THRESH:
                    distracted = True
                    head_state_label = "YAN BAKIS"
                elif pitch_dev > HEAD_PITCH_THRESH:
                    distracted = True
                    head_state_label = "ASAGI BAKIS"
                else:
                    head_state_label = "NORMAL"

                if distracted:
                    distract_frames += 1
                else:
                    distract_frames = 0
                    alarm_distract_on = False

                if distract_frames >= HEAD_DISTRACT_FRAMES:
                    if (not alarm_distract_on) or (now - last_distract_alarm_time) > ALARM_COOLDOWN:
                        sound_alarm_distract()
                        last_distract_alarm_time = now
                        alarm_distract_on = True
                        total_distract_alerts += 1
                        distract_events.append(now)
                    status_text = f"DIKKAT DAGINIKLIGI ({head_state_label})"

                last_head_state_label = head_state_label

            else:
                eye_closed_frame_count = 0
                open_mouth_frame_count = 0
                distract_frames = 0
                alarm_sleep_on = alarm_yawn_on = alarm_distract_on = False
                status_text = "YUZ ALGILANMADI"

            fatigue_score, blink_rate = compute_fatigue_score(
                sleep_events, yawn_events, distract_events, blink_events, now, EVENT_WINDOW_SEC
            )

            draw_modern_ui(
                frame,
                fatigue_score,
                blink_count,
                blink_rate,
                total_sleep_alerts,
                total_yawn_alerts,
                total_distract_alerts,
                eye_state_label,
                mouth_state_label,
                head_state_label,
                status_text
            )

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
