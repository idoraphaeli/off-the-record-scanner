# Off The Record — scratch analysis API

A small FastAPI service wrapping the classical-CV scratch detector. No machine
learning, no GPU: one request is about a second of CPU.

---

## 1. Run it on your machine first

From the `server` folder:

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open <http://localhost:8000/docs> — FastAPI generates a page where you can upload
a record photo and see the response, without writing any client code. Do this
before deploying; it is the fastest way to see the shape of the data.

---

## 2. Deploy on Render

The repo already contains `render.yaml` and `server/Dockerfile`, so Render needs
almost nothing from you.

1. Push the repo to GitHub (it already is).
2. On <https://render.com>: **New → Blueprint**, connect the repository, and
   Render reads `render.yaml` and offers to create the service.
3. Before the first deploy, set two environment variables in the dashboard:

   | variable | value |
   |---|---|
   | `ADMIN_TOKEN` | a long random string you invent. Anyone holding it can write feedback. |
   | `ALLOWED_ORIGINS` | your app's URL, e.g. `https://your-app.lovable.app`. Comma-separate several. |

   Leave `ALLOWED_ORIGINS` unset only while testing — the default `*` lets any
   website call your API.
4. Deploy, then check `https://<your-service>.onrender.com/health`.

**About the free plan:** the service sleeps after ~15 minutes idle, so the first
request after a quiet period takes 30–60 seconds while it wakes. Fine for
development; switch `plan: free` to `plan: starter` in `render.yaml` when you
demo it to anyone.

---

## 3. The API

### `POST /analyze`
Multipart form upload, field name `image`. Add `?overlay=false` to skip the
highlighted image — that response is roughly ten times smaller and noticeably
faster, which is what you want if the app only needs the numbers.

```json
{
  "analysis_id": "95087b54243f4737...",
  "mark_count": 6,
  "marks": [
    {
      "length_px": 52,
      "thickness_px": 5.9,
      "angle_to_groove_deg": 88.7,
      "radius_frac": 0.499,
      "angle_deg": 82.1
    }
  ],
  "grade": "Very Good Plus (VG+)",
  "damage_index": 1.44,
  "grade_is_calibrated": false,
  "coverage": { "judged_pct": 97.1, "unlit_pct": 1.3, "glare_pct": 1.6 },
  "disc": { "center_x": 797, "center_y": 613, "radius_px": 575, "found_by": "hough" },
  "warnings": [],
  "elapsed_ms": 1075,
  "overlay_png": "data:image/jpeg;base64,..."
}
```

Fields worth understanding before you build UI around them:

- **`coverage.judged_pct`** — how much of the playing surface was bright enough
  to assess. Read this before `mark_count`: a photo at 40% coverage that reports
  zero marks has not been shown to be clean, it has mostly not been looked at.
  Show it to the user.
- **`radius_frac` / `angle_deg`** — each mark's position on the *record*
  (fraction of the radius, and degrees around it), not in the photo. These stay
  comparable between two photos taken at different angles, which is what makes
  cross-checking and per-mark feedback possible.
- **`grade_is_calibrated: false`** — the grade thresholds were never fitted to
  human-graded records. Present the grade as a suggestion the seller confirms,
  never as a verdict. It also covers surface marks only: warps, edge chips and
  anything audible-but-invisible are out of scope.
- **`warnings`** — plain-language problems with the photo. Surface these
  verbatim; they tell the user how to get a better result.

Errors: `413` too large (limit 12 MB), `415` not an image, `422` unreadable.
The type is decided from the file's own bytes, not from its name.

### `POST /feedback` (admin)
Header `X-Admin-Token`. Records whether one detection was really a scratch:

```json
{ "analysis_id": "95087b...", "mark_index": 0, "is_scratch": false,
  "note": "lamp reflection", "record_id": "bowie-golden-years" }
```

### `GET /feedback` (admin)
Everything collected so far.

> **Feedback storage is temporary.** It writes to a file on the container's disk,
> which Render wipes on every deploy and every wake from sleep. That is enough to
> try the flow, not to accumulate a dataset. Point it at a database before the
> feedback is worth keeping — Supabase is the natural choice since the app
> already uses it.

---

## 4. Calling it from the Lovable app

```js
const form = new FormData();
form.append("image", file);                     // a File from an <input type="file">

const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
if (!res.ok) throw new Error(await res.text());
const result = await res.json();

// result.overlay_png goes straight into an <img src=...>
// result.coverage.judged_pct decides whether to ask for a better photo
// result.warnings is written for end users -- show it as-is
```

Do not set a `Content-Type` header yourself: the browser has to add the
multipart boundary, and setting it manually breaks the upload.

Keep `ADMIN_TOKEN` on your backend, never in the front-end bundle — anything
shipped to the browser is public. Have the app call your own backend, and let
that backend attach the token when it forwards to `/feedback`.

---

## 5. Promoting a new detector

`scanner/detector.py` is a deliberate snapshot of `experiments/detector.py`, not
an import of it. The experiments folder changes between runs; a deployed server
should not move because someone is mid-experiment. To ship an improvement, copy
the file across as its own commit and record what changed and what it measured.
