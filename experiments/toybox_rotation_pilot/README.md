# Toybox in-context rotation pilot (one question, one call)

A single feasibility question: can GPT-6 Luna infer a shared rigid transformation (about 90° counter-clockwise,
in the image plane) from three before/after pairs of **real recorded frames** and apply it to a new object?
This is one example; it does not establish anything about the model's understanding of rigid motions.

## Files

| file | what |
|---|---|
| `question/images/` | the 11 model inputs (6 demo frames, 1 query, 4 candidates), recorded frames resized to 768 px wide; nothing rotated, flipped, aligned or generated |
| `question/composite.jpg` | readable sheet with neutral labels only (not sent to the model) |
| `question/prompt.txt`, `question/inputs.json` | exact prompt and the ordered, labelled image list that is sent |
| `metadata.json` | answer and provenance (videos, frames, timestamps, intended rotations, shuffle seed, uncertainty). Never sent to the model |
| `selection.json` | the hand-picked frames that `build_question.py` reads |
| `frames.py`, `angle.py`, `build_question.py` | frame reading and contact sheets; SIFT rotation check; question builder |
| `eval_once.py` | the single evaluation call (refuses to run twice) |
| `results/response.json`, `results/result_summary.json` | raw response and parsed result (created by the call) |

## Data

Toybox (Wang et al., 2018; <https://aivaslab.github.io/toybox/>) is distributed as three ~37 GB tar.gz archives on
Zenodo (vehicles: record 1290757, animals: record 1289936). The archives were **streamed**, and only the needed
videos were kept. In Toybox, `ry` videos are rotations in the image plane (`ryminus` = counter-clockwise as seen
by the camera). `rz` videos are rotations about the vertical axis, and `rx` videos tumble the object about a
horizontal axis.

Videos used:
* `car_18_pivothead_ryminus.mp4` (vehicles)
* `truck_28_pivothead_ryminus.mp4` (vehicles)
* `truck_24_pivothead_ryminus.mp4` (vehicles)
* `duck_12_pivothead_ryminus.mp4` (animals)

## How the frames were chosen

* Timestamped contact sheets (0.5 s and 1 s steps) of rotation videos from the four vehicle categories and four
  animal videos were inspected.
* Rejected:
  * airplanes, which often leave the frame or are thin;
  * the helicopter, which turns out of plane;
  * `rz` and `rx` videos, which are not in-plane rotations.
* Frames at roughly 0/90/180/270° were picked by eye. For the query object (`truck_24`), a SIFT + RANSAC similarity
  estimate confirms the candidates at about −2°, 93°, 179° and 273° relative to the query frame. For the
  demonstration objects the estimate had too few inliers to be used, so those are visual judgements only.
* The demonstrations start from different orientations (0°, 90°, 180°), so the rule cannot be inferred from a common
  final orientation.

## Reproduce

```bash
# 1. stream the archives and keep only the needed videos (slow: about 1.5 MB/s)
python - <<'PY'
import tarfile, urllib.request, os, shutil
need = {"car_18_pivothead_ryminus.mp4", "truck_28_pivothead_ryminus.mp4", "truck_24_pivothead_ryminus.mp4",
        "duck_12_pivothead_ryminus.mp4"}
os.makedirs("videos", exist_ok=True)
for rec, name in [(1290757, "vehicles"), (1289936, "animals")]:
    t = tarfile.open(fileobj=urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}/files/{name}.tar.gz/content"), mode="r|gz")
    for m in t:
        b = os.path.basename(m.name)
        if b in need:
            with open("videos/" + b, "wb") as fh: shutil.copyfileobj(t.extractfile(m), fh)
            need.discard(b)
        if not need or (name == "vehicles" and need == {"duck_12_pivothead_ryminus.mp4"}): break
PY
# 2. rebuild the question (deterministic; needs opencv-python-headless and pillow)
python build_question.py --videos videos
# 3. the single evaluation call (in Colab: notebooks/53_toybox_rotation_pilot.ipynb)
OPENROUTER_API_KEY=... python eval_once.py
```

## Batch 1: 30 questions (rotation / translation / rotation + translation)

Built by `make_questions.py` from `keyframes.json`, the visually checked keyframes at about 0/90/180/270° per object.
`orient.py` (accumulated SIFT tracking) proposes keyframes; they are manually chosen when the tracker drifts.

| task | n | transformation | candidates | objects | crop / shift |
|---|---|---|---|---|---|
| `R00-R09` | 10 | rotation by +90° (counter-clockwise), −90° or 180°, from recorded frames | 0 / 90 / 180 / 270° | car_18, truck_24, truck_28, duck_12 | 1344×756 |
| `T00-T09` | 10 | shift left / right / up / down | the four directions | same | 1344×756; 240 px / 150 px |
| `RT00-RT09` | 10 | ±90° rotation plus a shift | correct; rotation only; shift only; wrong rotation direction + shift | truck_20, car_21, truck_28, duck_12 | 1344×756; 330 px / 210 px |

Every question records its own crop, shift, objects and seed in `answers.json` (`config`).

* **Framing.** Every image is a crop of a recorded 1920×1080 frame, resized to 768 px wide. Nothing is padded,
  generated or digitally rotated.
* **Object-centred placement.** Crops are placed from the object centre in each keyframe (`keyframes.json`
  `centers`: automatic with `locate.py` and visually checked; manual for `truck_28`, whose rust colour is close to
  skin). Within a demonstration pair, and between the query and each candidate, the object's position changes by
  exactly the intended shift, or by 0 when there is no shift. This cancels the hand's own motion between recorded
  frames.
* **Translations.** Moving the crop window is equivalent to the camera panning.
* **RT rebuilt.** RT was rebuilt with smaller objects and a larger shift (25% / 28% of the crop), so the shift is
  easier to see. The garbage truck (`truck_24`) and the blue car (`car_18`) fill so much of the frame that a shift
  pushed them half out of view, so they are not used in RT. The rebuild command was:
  ```bash
  python make_questions.py --videos videos --out batch1 --tasks RT \
      --objects truck_20,car_21,truck_28,duck_12 --crop 1344x756 --shift 330,210 --merge
  ```
* **Objects.**
  * Only 4 objects per task, so every question uses all four and some demonstration pairs repeat across questions.
    The query object always differs from the three demonstration objects.
  * `car_21` is a clockwise (`ryplus`) video; its keyframes are still labelled counter-clockwise from frame 0.
* **Rotations.** Orientations are approximate (about ±15°).
* **Checks.** All 30 composites were inspected: rotation direction and shift direction are consistent within each
  question, the intended answer is right, and the four candidates differ visibly. Several questions were also
  checked numerically from the crop origins.
* **Viewer and answers.** `batch1/viewer.html` shows all questions with answers hidden (`python make_viewer.py`).
  Answers are in `batch1/answers.json`, which is only read when scoring.

Rebuild and evaluate:

```bash
python make_questions.py --videos videos --out batch1 --seed 0
OPENROUTER_API_KEY=... python eval_batch.py --batch batch1      # or notebooks/54_toybox_batch1.ipynb in Colab
```
