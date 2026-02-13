import os
import re
import json
import time
import uuid
from typing import List, Dict

import streamlit as st
from pptx import Presentation

import numpy as np
import cv2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Config
# =========================
APP_TITLE = "Presentation Assessment (Streamlit MVP: PPTX + Video)"
MAX_VIDEO_SECONDS = 10 * 60
MAX_QA_SECONDS = 5 * 60

BASE_DIR = "data_sessions"
os.makedirs(BASE_DIR, exist_ok=True)


# =========================
# Helpers
# =========================
def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip())
    return name[:140] if name else "upload.bin"


def ensure_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    return st.session_state.session_id


def session_paths(session_id: str) -> Dict[str, str]:
    root = os.path.join(BASE_DIR, session_id)
    return {
        "root": root,
        "uploads": os.path.join(root, "uploads"),
        "answers": os.path.join(root, "answers"),
        "meta": os.path.join(root, "meta"),
    }


def init_dirs(paths: Dict[str, str]):
    for p in paths.values():
        os.makedirs(p, exist_ok=True)


def extract_slide_text(pptx_path: str) -> List[Dict]:
    prs = Presentation(pptx_path)
    slides_out = []

    for idx, slide in enumerate(prs.slides, start=1):
        blocks = []
        for shape in slide.shapes:
            if hasattr(shape, "text_frame"):
                txt = (shape.text or "").strip()
                if txt:
                    blocks.append(txt)

        title = ""
        bullets = []

        if blocks:
            first_lines = [ln.strip() for ln in blocks[0].splitlines() if ln.strip()]
            if first_lines:
                title = first_lines[0]

        for block in blocks:
            for ln in block.splitlines():
                ln2 = ln.strip()
                if not ln2:
                    continue
                if title and ln2 == title:
                    continue
                bullets.append(ln2)

        # de-dup bullets
        seen = set()
        bullets2 = []
        for b in bullets:
            if b not in seen:
                bullets2.append(b)
                seen.add(b)

        full_text = " ".join([title] + bullets2).strip()
        slides_out.append(
            {"slide": idx, "title": title, "bullets": bullets2, "full_text": full_text}
        )

    return slides_out


def generate_questions(slide_text: List[Dict], n_questions: int = 8) -> List[Dict]:
    candidates = []
    for s in slide_text:
        if s["title"]:
            candidates.append(
                {
                    "slide": s["slide"],
                    "question": f"Explain the main idea of slide {s['slide']}: {s['title']}",
                    "context": s["full_text"],
                }
            )
        for b in s["bullets"][:3]:
            candidates.append(
                {
                    "slide": s["slide"],
                    "question": f"Define or explain: “{b}” (from slide {s['slide']}).",
                    "context": s["full_text"],
                }
            )

    # Deduplicate
    seen = set()
    uniq = []
    for c in candidates:
        if c["question"] not in seen:
            uniq.append(c)
            seen.add(c["question"])

    return uniq[:n_questions]


def score_answers(qa_items: List[Dict]) -> Dict:
    vec = TfidfVectorizer(stop_words="english")
    scored = []

    for item in qa_items:
        context = (item.get("context") or "").strip()
        answer = (item.get("answer") or "").strip()

        if not answer:
            scored.append({**item, "score": 0.0, "reason": "No answer provided."})
            continue
        if not context:
            scored.append({**item, "score": 0.0, "reason": "No slide context available."})
            continue

        tfidf = vec.fit_transform([context, answer])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])
        pct = float(np.clip(sim * 100.0, 0.0, 100.0))

        scored.append(
            {
                **item,
                "score": pct,
                "reason": "Score reflects alignment with slide content (text similarity).",
            }
        )

    overall = float(np.mean([x["score"] for x in scored])) if scored else 0.0
    return {"overall": overall, "items": scored}


def get_video_duration_seconds(video_path: str) -> float:
    """
    Cloud-safe duration check using OpenCV.
    Works for most MP4/WebM. If metadata is missing, returns -1.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps and fps > 0 and frame_count and frame_count > 0:
        return float(frame_count / fps)
    return -1.0


# =========================
# UI State
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

session_id = ensure_session_id()
paths = session_paths(session_id)
init_dirs(paths)

st.session_state.setdefault("phase", "upload")  # upload -> review -> qa -> results
st.session_state.setdefault("pptx_path", None)
st.session_state.setdefault("video_path", None)
st.session_state.setdefault("slide_text", [])
st.session_state.setdefault("questions", [])
st.session_state.setdefault("slide_index", 0)
st.session_state.setdefault("qa_start", None)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("Session")
    st.write(f"Session ID: `{session_id}`")
    st.write(f"Phase: **{st.session_state.phase}**")
    st.write("---")
    st.write("Rules")
    st.write("- Video presentation ≤ 10 minutes")
    st.write("- Q&A ≤ 5 minutes")
    st.write("---")
    if st.button("Reset session"):
        # delete files + reset state
        try:
            import shutil
            shutil.rmtree(paths["root"], ignore_errors=True)
        except Exception:
            pass
        keep = {"session_id"}
        for k in list(st.session_state.keys()):
            if k not in keep:
                del st.session_state[k]
        st.rerun()


# =========================
# Phase 1: Upload
# =========================
if st.session_state.phase == "upload":
    st.markdown("## 1) Upload PPTX and your presentation video")

    col1, col2 = st.columns(2)

    with col1:
        pptx_file = st.file_uploader("Upload PPTX (.pptx)", type=["pptx"])

    with col2:
        video_file = st.file_uploader("Upload presentation video (mp4, webm, mov)", type=["mp4", "webm", "mov", "m4v"])

    if pptx_file is not None:
        pptx_name = safe_filename(pptx_file.name)
        pptx_path = os.path.join(paths["uploads"], pptx_name)
        with open(pptx_path, "wb") as f:
            f.write(pptx_file.getbuffer())
        st.session_state.pptx_path = pptx_path

        slide_text = extract_slide_text(pptx_path)
        st.session_state.slide_text = slide_text
        st.session_state.questions = generate_questions(slide_text, n_questions=8)

        with open(os.path.join(paths["meta"], "slides_text.json"), "w", encoding="utf-8") as f:
            json.dump(slide_text, f, indent=2, ensure_ascii=False)

        st.success("PPTX processed.")

    if video_file is not None:
        vid_name = safe_filename(video_file.name)
        vid_path = os.path.join(paths["uploads"], vid_name)
        with open(vid_path, "wb") as f:
            f.write(video_file.getbuffer())
        st.session_state.video_path = vid_path

        dur = get_video_duration_seconds(vid_path)
        if dur < 0:
            st.warning("Could not read video duration reliably. Try MP4 or WebM.")
        else:
            st.info(f"Detected video duration: {dur:.1f} seconds ({dur/60:.2f} minutes)")

            if dur > MAX_VIDEO_SECONDS:
                st.error("Video is longer than 10 minutes. Please upload a video that is 10 minutes or less.")
                st.session_state.video_path = None

    ready = (st.session_state.pptx_path is not None) and (st.session_state.video_path is not None)

    if ready:
        if st.button("Continue to review"):
            st.session_state.phase = "review"
            st.session_state.slide_index = 0
            st.rerun()
    else:
        st.info("Upload BOTH the PPTX and the video to continue.")


# =========================
# Phase 2: Review (Slides + Video)
# =========================
elif st.session_state.phase == "review":
    slides = st.session_state.slide_text
    video_path = st.session_state.video_path

    if not slides or not video_path:
        st.error("Missing PPTX or video. Reset and upload again.")
        st.stop()

    st.markdown("## 2) Review slides and video")

    left, right = st.columns([1, 1])

    # Slides (text slideshow)
    with left:
        idx = st.session_state.slide_index
        slide = slides[idx]
        total = len(slides)

        st.markdown(f"### Slide {slide['slide']} / {total}")
        st.markdown(f"**{slide['title'] or '(No title detected)'}**")

        if slide["bullets"]:
            for b in slide["bullets"]:
                st.write(f"- {b}")
        else:
            st.write("(No bullet text found on this slide.)")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Prev slide"):
                st.session_state.slide_index = max(0, idx - 1)
                st.rerun()
        with c2:
            if st.button("Next slide"):
                st.session_state.slide_index = min(total - 1, idx + 1)
                st.rerun()

    # Video
    with right:
        st.markdown("### Presentation video")
        with open(video_path, "rb") as f:
            st.video(f.read())

        st.caption("If the video doesn’t play, re-upload as MP4 (H.264) or WebM.")

    st.write("---")
    if st.button("Start Q&A (5 minutes)"):
        st.session_state.phase = "qa"
        st.session_state.qa_start = time.time()
        st.rerun()


# =========================
# Phase 3: Q&A (Timed)
# =========================
elif st.session_state.phase == "qa":
    st.markdown("## 3) Auto Questions (Answer within 5 minutes)")

    if st.session_state.qa_start is None:
        st.session_state.qa_start = time.time()

    elapsed = int(time.time() - st.session_state.qa_start)
    remaining = MAX_QA_SECONDS - elapsed

    st.metric("Time remaining", f"{max(0, remaining)//60:02d}:{max(0, remaining)%60:02d}")

    questions = st.session_state.questions or []
    if not questions:
        st.error("No questions generated from slides.")
        st.stop()

    answers_payload = []
    for i, q in enumerate(questions, start=1):
        st.markdown(f"**Q{i}.** {q['question']}")
        ans = st.text_area(f"Answer Q{i}", key=f"ans_{i}", height=90)
        answers_payload.append({**q, "answer": ans})

    os.makedirs(paths["answers"], exist_ok=True)
    with open(os.path.join(paths["answers"], "qa_answers.json"), "w", encoding="utf-8") as f:
        json.dump(answers_payload, f, indent=2, ensure_ascii=False)

    if remaining <= 0:
        st.warning("Time is up. Submitting answers.")
        st.session_state.phase = "results"
        st.rerun()

    if st.button("Submit now"):
        st.session_state.phase = "results"
        st.rerun()


# =========================
# Phase 4: Results
# =========================
elif st.session_state.phase == "results":
    st.markdown("## 4) Results and Auto-Scoring")

    ans_path = os.path.join(paths["answers"], "qa_answers.json")
    if not os.path.exists(ans_path):
        st.error("Answers not found.")
        st.stop()

    with open(ans_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    scoring = score_answers(qa_items)

    st.metric("Overall alignment score", f"{scoring['overall']:.1f} / 100")

    for i, item in enumerate(scoring["items"], start=1):
        st.markdown(f"### Q{i} (Slide {item.get('slide')})")
        st.write(item.get("question", ""))
        st.write("**Answer:**")
        st.write(item.get("answer", ""))
        st.write(f"**Score:** {item['score']:.1f} / 100")
        with st.expander("Slide context used for marking"):
            st.write(item.get("context", ""))

    st.success("Done. Files saved in the session folder.")

    if st.button("New session"):
        # soft reset
        keep = {"session_id"}
        for k in list(st.session_state.keys()):
            if k not in keep:
                del st.session_state[k]
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()
