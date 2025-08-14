import os
import sys
import cv2
import math
import time
import json
import argparse
import tempfile
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("mediapipe is required. Install from requirements.txt")

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LAND = mp_pose.PoseLandmark

def download_video(url, out_dir):
    if yt_dlp is None:
        raise RuntimeError("yt-dlp not installed; cannot download YouTube URLs")
    ydl_opts = {"outtmpl": os.path.join(out_dir, "input.%(ext)s"), "format": "mp4/best"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        p = ydl.prepare_filename(info)
    if not os.path.exists(p):
        raise RuntimeError("download failed")
    return p

def to_px(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h])

def angle(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def vector_angle_deg(v, ref):
    if np.linalg.norm(v) == 0 or np.linalg.norm(ref) == 0:
        return None
    cosang = np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref))
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def spine_lean_deg(hip_mid, sh_mid):
    v = sh_mid - hip_mid
    vertical = np.array([0, -1])
    a = vector_angle_deg(v, vertical)
    if a is None:
        return None
    return 90 - a

def head_over_knee_dx(head, knee):
    return head[0] - knee[0]

def foot_dir_deg(heel, toe):
    v = toe - heel
    ref = np.array([1, 0])
    a = vector_angle_deg(v, ref)
    return a

def pick_front_leg(sh_mid, hip_mid, lknee, rknee, lankle, rankle):
    cand = {"L": lankle[1], "R": rankle[1]}
    side = "L" if cand["L"] < cand["R"] else "R"
    return side

def draw_text(img, text, org, scale=0.6, thickness=2, color=(255,255,255), bg=(0,0,0)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def compute_scores(stats):
    res = {"Footwork": 0, "Head Position": 0, "Swing Control": 0, "Balance": 0, "Follow-through": 0}
    if len(stats) == 0:
        return res, {}
    elbows = [s["elbow_deg"] for s in stats if s["elbow_deg"] is not None]
    spines = [s["spine_lean"] for s in stats if s["spine_lean"] is not None]
    dxs = [abs(s["head_over_knee_dx_norm"]) for s in stats if s["head_over_knee_dx_norm"] is not None]
    feet = [s["front_foot_dir"] for s in stats if s["front_foot_dir"] is not None]
    smooth = [s["elbow_delta"] for s in stats if s["elbow_delta"] is not None]
    def avg(L):
        return sum(L)/len(L) if L else None
    ae, asn, adx, afd, asm = avg(elbows), avg(spines), avg(dxs), avg(feet), avg(smooth)
    fw = 10
    if afd is not None:
        d = abs((afd % 180) - 0)
        fw = max(1, 10 - int(d/10))
    hp = 10
    if adx is not None:
        hp = max(1, 10 - int((adx*100)/5))
    sc = 10
    if ae is not None:
        sc = max(1, min(10, int(10 - abs(ae - 115)/10)))
    bal = 10
    if asn is not None:
        bal = max(1, min(10, int(10 - abs(asn - 10)/5)))
    ft = 10
    if asm is not None:
        ft = max(1, 10 - int(asm/5))
    res["Footwork"], res["Head Position"], res["Swing Control"], res["Balance"], res["Follow-through"] = fw, hp, sc, bal, ft
    fb = {}
    fb["Footwork"] = "Front foot alignment close to crease axis." if fw >= 7 else "Adjust front foot direction to be more open towards cover."
    fb["Head Position"] = "Head stacks well over front knee." if hp >= 7 else "Bring head more over front knee at impact."
    fb["Swing Control"] = "Elbow elevation supports straight arc." if sc >= 7 else "Raise front elbow to maintain a high, straight bat path."
    fb["Balance"] = "Stable spine lean through contact." if bal >= 7 else "Reduce excessive lateral or backward lean."
    fb["Follow-through"] = "Smooth acceleration into follow-through." if ft >= 7 else "Aim for a more continuous, fluid finish."
    return res, fb

def analyze_video(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(out_dir, "annotated_video.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))
    pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    stats = []
    prev_elbow = None
    t0 = time.time()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        metrics = {"elbow_deg": None, "spine_lean": None, "head_over_knee_dx_norm": None, "front_foot_dir": None, "elbow_delta": None}
        cue_good = []
        cue_bad = []
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
            lm = res.pose_landmarks.landmark
            pts = {}
            needed = [LAND.LEFT_SHOULDER, LAND.RIGHT_SHOULDER, LAND.LEFT_ELBOW, LAND.RIGHT_ELBOW, LAND.LEFT_WRIST, LAND.RIGHT_WRIST, LAND.LEFT_HIP, LAND.RIGHT_HIP, LAND.LEFT_KNEE, LAND.RIGHT_KNEE, LAND.LEFT_ANKLE, LAND.RIGHT_ANKLE, LAND.NOSE, LAND.LEFT_HEEL, LAND.RIGHT_HEEL, LAND.LEFT_FOOT_INDEX, LAND.RIGHT_FOOT_INDEX]
            for k in needed:
                p = lm[k]
                pts[int(k)] = to_px(p, w, h)
            sh_mid = (pts[int(LAND.LEFT_SHOULDER)] + pts[int(LAND.RIGHT_SHOULDER)]) / 2.0
            hip_mid = (pts[int(LAND.LEFT_HIP)] + pts[int(LAND.RIGHT_HIP)]) / 2.0
            head = pts[int(LAND.NOSE)]
            lankle = pts[int(LAND.LEFT_ANKLE)]
            rankle = pts[int(LAND.RIGHT_ANKLE)]
            lknee = pts[int(LAND.LEFT_KNEE)]
            rknee = pts[int(LAND.RIGHT_KNEE)]
            side = pick_front_leg(sh_mid, hip_mid, lknee, rknee, lankle, rankle)
            if side == "L":
                sh = pts[int(LAND.LEFT_SHOULDER)]
                el = pts[int(LAND.LEFT_ELBOW)]
                wr = pts[int(LAND.LEFT_WRIST)]
                kf = lknee
                heel = pts[int(LAND.LEFT_HEEL)]
                toe = pts[int(LAND.LEFT_FOOT_INDEX)]
            else:
                sh = pts[int(LAND.RIGHT_SHOULDER)]
                el = pts[int(LAND.RIGHT_ELBOW)]
                wr = pts[int(LAND.RIGHT_WRIST)]
                kf = rknee
                heel = pts[int(LAND.RIGHT_HEEL)]
                toe = pts[int(LAND.RIGHT_FOOT_INDEX)]
            e = angle(sh, el, wr)
            slean = spine_lean_deg(hip_mid, sh_mid)
            dx = head_over_knee_dx(head, kf) / w
            fd = foot_dir_deg(heel, toe)
            metrics["elbow_deg"] = e
            metrics["spine_lean"] = slean
            metrics["head_over_knee_dx_norm"] = dx
            metrics["front_foot_dir"] = fd
            if prev_elbow is not None and e is not None:
                metrics["elbow_delta"] = abs(e - prev_elbow)
            prev_elbow = e if e is not None else prev_elbow
            if e is not None and e >= 100:
                cue_good.append("Good elbow elevation")
            elif e is not None:
                cue_bad.append("Raise front elbow")
            if slean is not None and slean >= 5:
                cue_good.append("Positive spine lean")
            elif slean is not None:
                cue_bad.append("Increase forward lean")
            if dx is not None and abs(dx) <= 0.03:
                cue_good.append("Head over front knee")
            elif dx is not None:
                cue_bad.append("Head not over front knee")
        draw_text(frame, f"FPS In: {fps_in:.1f}", (10, 30))
        y = 60
        if metrics["elbow_deg"] is not None:
            draw_text(frame, f"Elbow: {metrics['elbow_deg']:.0f} deg", (10, y)); y += 30
        if metrics["spine_lean"] is not None:
            draw_text(frame, f"Spine Lean: {metrics['spine_lean']:.1f} deg", (10, y)); y += 30
        if metrics["head_over_knee_dx_norm"] is not None:
            draw_text(frame, f"Head-Knee X: {metrics['head_over_knee_dx_norm']*100:.1f} %W", (10, y)); y += 30
        if metrics["front_foot_dir"] is not None:
            draw_text(frame, f"Foot Dir: {metrics['front_foot_dir']:.0f} deg", (10, y)); y += 30
        if cue_good:
            draw_text(frame, " ".join(["✅"] + cue_good), (10, h - 50), scale=0.7, bg=(0,80,0))
        if cue_bad:
            draw_text(frame, " ".join(["❌"] + cue_bad), (10, h - 15), scale=0.7, bg=(0,0,80))
        out.write(frame)
        stats.append(metrics)
        n += 1
    cap.release()
    out.release()
    dt = time.time() - t0
    fps_avg = n / dt if dt > 0 else 0
    scores, feedback = compute_scores(stats)
    evaluation = {"frames": n, "avg_fps": fps_avg, "scores": scores, "feedback": feedback}
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    return evaluation, out_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="https://youtube.com/shorts/vSX3IRxGnNY")
    p.add_argument("--output", default="output", help="output directory")
    args = p.parse_args()
    src = args.input
    out_dir = args.output
    if src.startswith("http"):
        with tempfile.TemporaryDirectory() as td:
            vpath = download_video(src, td)
            evaluation, out_path = analyze_video(vpath, out_dir)
    else:
        evaluation, out_path = analyze_video(src, out_dir)
    print(json.dumps(evaluation, indent=2))
    print(out_path)

if __name__ == "__main__":
    main()