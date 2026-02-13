import os
import re
import json
import time
import uuid
import shutil
from typing import List, Dict, Optional

import streamlit as st
from pptx import Presentation
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# App settings
# =========================
APP_TITLE = "Presentation Assessment Platform (Cloud-Safe MVP)"
MAX_PRESENTATION_SECONDS = 10 * 60
MAX_QA_SECONDS = 5 * 60

BASE_DIR = "data_sessions"
os.makedirs(BASE_DIR, exist_ok=True)


# =========================
# Utilities
# =========================
def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip())
    return name[:120] if name else "upload.pptx"


def ensure_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    return st.session_state.session_id


def session_paths(session_id: str) -> Dict[str, str]:
    root = os.path.join(BASE_DIR, session_id)
    return {
        "root": root,
        "uploads": os.path.join(root, "uploads"),
        "snapshots": os.path.join(root, "snapshots"),
        "answers": os.path.join(root, "answers"),
        "meta": os.path.join(root, "meta"),
    }


def init_dirs(paths: Dict[str, str]):
    for p in paths.values():
        os.makedirs(p, exist_ok=True)


def extract_slide_text(pptx_path: str) -> List[Dict]:
    """
    Extract a clean, slide-by-slide text structure.
    Works on Streamlit Cloud without extra binaries.
    """
    prs = Presentation(pptx_path)
    slides_out = []

    for idx, slide in enumerate(prs.slides, start=1):
        raw_blocks = []
        for shape in slide.shapes:
            if not hasattr(shape, "text_frame"):
                continue
            txt = (shape.text or "").strip()
            if txt:
                raw_blocks.append(txt)

        title = ""
        bullets = []
        if raw_blocks:
            first_lines = [ln.strip() for ln in raw_blocks[0].splitlines() if ln.strip()]
            if first_lines:
                title = first_lines[0]

        for block in raw_blocks:
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
    """
    Offline question generator.
    Produces short-answer questions anchored to slide numbers.
    """
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
        # Use up to 3 bullets per slide
        for b in s["bullets"][:3]:
            candidates.append(
                {
                    "slide": s["slide"],
                    "question": f"Define or explain: “{b}” (from slide {s['slide']}).",
                    "context": s["full_text"],
                }
            )

    # Deduplicate by question text
    seen = set()
    uniq = []
    for c in candidates:
        if c["question"] not in seen:
            uniq.append(c)
            seen.add(c["question"])

    return uniq[:n_questions]


def score_answers(qa_items: List[Dict]) -> Dict:
    """
    TF-IDF similarity between answer and slide context.
    Returns a 0–100 score per question plus overall.
    """
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


def stamp_image(img: Image.Image, stamp_text: str) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    bar_h = max(55, h // 12)

    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, bar_h // 2))
    except Exception:
        font = ImageFont.load_default()

    draw.text((12, h - bar_h + 10), stamp_text, fill=(255, 255, 255), font=font)
    return img


def save_snapshot(camera_file, snapshots_dir: str, session_id: str, slide_no: int, label: str) -> str:
    os.makedirs(snapshots_dir, exist_ok=True)
    img = Image.open(camera_file)
    ts = int(time.time())
    stamp = f"Session: {session_id} | Slide: {slide_no} | {time.strftime('%Y-%m-%d %H:%M:%S')} | {label}"
    out = stamp_image(img, stamp)
    out_path = os.path.join(snapshots_dir, f"snap_{ts}_slide{slide_no:03d}.jpg")
    out.save(out_path, quality=92)
    return out_path


def reset_everything(paths: Dict[str, str]):
    if os.path.exists(paths["root"]):
        shutil.rmtree(paths["root"], ignore_errors=True)
    keep = {"session_id"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]


# =========================
# Session state init
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

session_id = ensure_session_id()
paths = session_paths(session_id)
init_dirs(paths)

st.session_state.setdefault("phase", "upload")  # upload -> present -> qa -> results
st.session_state.setdefault("pptx_path", None)
st.session_state.setdefault("slide_text", [])
st.session_state.setdefault("questions", [])
st.session_state.setdefault("slide_index", 0)

st.session_state.setdefault("present_start", None)
st.session_state.setdefault("qa_start", None)

st.session_state.setdefault("snapshot_log", [])
st.session_state.setdefault("snapshot_taken_for_slide", False)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("Session")
    st.write(f"Session ID: `{session_id}`")
    st.write(f"Phase: **{st.session_state.phase}**")

    st.write("---")
    st.write("Limits")
    st.write("- Presentation: 10 minutes")
    st.write("- Q&A: 5 minutes")

    st.write("---")
    if st.button("Reset session (delete files)"):
        reset_everything(paths)
        st.rerun()


# =========================
# Phase: Upload
# =========================
if st.session_state.phase == "upload":
    st.markdown("## 1) Upload PowerPoint (.pptx)")
    uploaded = st.file_uploader("Upload PPTX", type=["pptx"])

    st.info(
        "This Cloud-safe version shows slides as text (title + bullets). "
        "It still enforces time limits and collects stamped camera evidence per slide."
    )

    if uploaded is not None:
        pptx_name = safe_filename(uploaded.name)
        pptx_path = os.path.join(paths["uploads"], pptx_name)
        with open(pptx_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.session_state.pptx_path = pptx_path

        with st.status("Reading PPTX...", expanded=True) as status:
            st.write("Extracting slide text...")
            slide_text = extract_slide_text(pptx_path)
            st.session_state.slide_text = slide_text

            st.write("Generating questions...")
            st.session_state.questions = generate_questions(slide_text, n_questions=8)

            # save meta
            os.makedirs(paths["meta"], exist_ok=True)
            with open(os.path.join(paths["meta"], "slides_text.json"), "w", encoding="utf-8") as f:
                json.dump(slide_text, f, indent=2, ensure_ascii=False)

            status.update(label="Ready", state="complete")

        if st.button("Start presentation"):
            st.session_state.phase = "present"
            st.session_state.present_start = time.time()
            st.session_state.slide_index = 0
            st.session_state.snapshot_taken_for_slide = False
            st.rerun()


# =========================
# Phase: Present
# =========================
elif st.session_state.phase == "present":
    slides = st.session_state.slide_text
    if not slides:
        st.error("No slides loaded. Go back and upload again.")
        st.stop()

    elapsed = int(time.time() - (st.session_state.present_start or time.time()))
    remaining = MAX_PRESENTATION_SECONDS - elapsed

    st.markdown("## 2) Presentation (max 10 minutes)")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Time remaining", f"{max(0, remaining)//60:02d}:{max(0, remaining)%60:02d}")
    with right:
        st.write("Rule: take at least one snapshot per slide before moving forward.")
        st.write("Snapshots are stamped with session id, slide number, timestamp.")

    # hard stop
    if remaining <= 0:
        st.warning("Time is up. Moving to Q&A.")
        st.session_state.phase = "qa"
        st.session_state.qa_start = time.time()
        st.rerun()

    # current slide
    idx = st.session_state.slide_index
    slide = slides[idx]
    slide_no = slide["slide"]
    total = len(slides)

    # Show slide content
    st.markdown(f"### Slide {slide_no} / {total}")
    if slide["title"]:
        st.markdown(f"**{slide['title']}**")
    else:
        st.markdown("**(No title detected)**")

    if slide["bullets"]:
        for b in slide["bullets"]:
            st.write(f"- {b}")
    else:
        st.write("(No bullet text found on this slide.)")

    st.write("---")

    # Camera capture
    st.markdown("### Camera snapshot (required)")
    shot = st.camera_input("Take snapshot for this slide")

    if shot is not None:
        saved = save_snapshot(
            shot,
            snapshots_dir=paths["snapshots"],
            session_id=session_id,
            slide_no=slide_no,
            label="SlideEvidence",
        )
        st.session_state.snapshot_log.append(
            {
                "slide": slide_no,
                "path": saved,
                "timestamp": int(time.time()),
            }
        )
        st.session_state.snapshot_taken_for_slide = True
        st.success("Snapshot saved and stamped.")

    # Navigation
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    with c1:
        if st.button("Prev"):
            st.session_state.slide_index = max(0, idx - 1)
            st.session_state.snapshot_taken_for_slide = False
            st.rerun()

    with c2:
        if st.button("Next"):
            if not st.session_state.snapshot_taken_for_slide:
                st.error("Take a snapshot before moving to the next slide.")
            else:
                st.session_state.slide_index = min(total - 1, idx + 1)
                st.session_state.snapshot_taken_for_slide = False
                st.rerun()

    with c3:
        if st.button("Finish early"):
            st.session_state.phase = "qa"
            st.session_state.qa_start = time.time()
            st.rerun()

    with c4:
        st.caption("Tip: If camera doesn’t open, allow browser camera permissions and refresh.")

    with st.expander("Snapshot log"):
        for it in st.session_state.snapshot_log[-15:]:
            st.write(f"- Slide {it['slide']} | {time.strftime('%H:%M:%S', time.localtime(it['timestamp']))} | {os.path.basename(it['path'])}")


# =========================
# Phase: Q&A
# =========================
elif st.session_state.phase == "qa":
    st.markdown("## 3) Auto Questions (answer within 5 minutes)")

    elapsed = int(time.time() - (st.session_state.qa_start or time.time()))
    remaining = MAX_QA_SECONDS - elapsed
    st.metric("Time remaining", f"{max(0, remaining)//60:02d}:{max(0, remaining)%60:02d}")

    if remaining <= 0:
        st.warning("Q&A time is up. Submitting answers.")
        st.session_state.phase = "results"
        st.rerun()

    questions = st.session_state.questions or []
    if not questions:
        st.error("No questions generated.")
        st.stop()

    answers_payload = []
    for i, q in enumerate(questions, start=1):
        st.markdown(f"**Q{i}.** {q['question']}")
        ans = st.text_area(f"Answer Q{i}", key=f"ans_{i}", height=90)
        answers_payload.append({**q, "answer": ans})

    os.makedirs(paths["answers"], exist_ok=True)
    with open(os.path.join(paths["answers"], "qa_answers.json"), "w", encoding="utf-8") as f:
        json.dump(answers_payload, f, indent=2, ensure_ascii=False)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Submit now"):
            st.session_state.phase = "results"
            st.rerun()
    with c2:
        st.info("Auto-submit happens at 5 minutes even if you don’t click submit.")


# =========================
# Phase: Results
# =========================
elif st.session_state.phase == "results":
    st.markdown("## 4) Results (Auto-scoring + Evidence)")

    ans_path = os.path.join(paths["answers"], "qa_answers.json")
    if not os.path.exists(ans_path):
        st.error("Answers file not found.")
        st.stop()

    with open(ans_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    scoring = score_answers(qa_items)
    st.metric("Overall alignment score", f"{scoring['overall']:.1f} / 100")

    st.write("---")
    for i, item in enumerate(scoring["items"], start=1):
        st.markdown(f"### Q{i} (Slide {item.get('slide')})")
        st.write(item.get("question", ""))
        st.write("**Answer**")
        st.write(item.get("answer", ""))

        st.write(f"**Score:** {item['score']:.1f} / 100")
        st.caption(item.get("reason", ""))

        with st.expander("Slide context used for marking"):
            st.write(item.get("context", ""))

    st.write("---")
    st.markdown("### Evidence snapshots")

    snaps = st.session_state.snapshot_log
    st.write(f"Snapshots captured: **{len(snaps)}**")

    if snaps:
        show_n = min(12, len(snaps))
        for it in snaps[-show_n:]:
            try:
                st.image(
                    it["path"],
                    caption=f"Slide {it['slide']} | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(it['timestamp']))}",
                    width=420,
                )
            except Exception:
                st.write(it["path"])

    st.success("Completed. Admin can review session files from the server storage.")

    if st.button("Start new session"):
        reset_everything(paths)
        # keep same session_id but reset app state fresh
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()
