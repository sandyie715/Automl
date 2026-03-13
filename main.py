"""
AutoML Studio — FastAPI Backend  (MongoDB + GridFS edition)
All artefacts (model .pkl, drift .html, text report) stored in MongoDB.
Temp CSV uploads go to /tmp — safe for Vercel / any stateless platform.
"""

import os, uuid, pickle, traceback, io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── MongoDB ───────────────────────────────────────────────────────────────────
import pymongo
from pymongo import MongoClient
import gridfs

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error,
)

# ── Evidently ─────────────────────────────────────────────────────────────────
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    EVIDENTLY_OK = True
except Exception:
    EVIDENTLY_OK = False

# ── Groq ──────────────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_LIB_OK = True
except Exception:
    GROQ_LIB_OK = False

# ─── Load env ─────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MONGO_URI    = os.getenv("MONGO_URI")
MONGO_DB     = os.getenv("MONGO_DB", "automl_studio")

# ── Supported Groq models ─────────────────────────────────────────────────────
GROQ_MODELS: List[Dict[str, str]] = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 · 70B Versatile (Recommended)"},
    {"id": "llama-3.1-70b-versatile", "label": "Llama 3.1 · 70B Versatile"},
    {"id": "llama-3.1-8b-instant",    "label": "Llama 3.1 · 8B Instant (Fast)"},
    {"id": "gemma2-9b-it",            "label": "Gemma 2 · 9B IT"},
    {"id": "llama3-8b-8192",          "label": "Llama 3 · 8B (Legacy)"},
]
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoML Studio API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], expose_headers=["Content-Disposition"],
)

# ── /tmp for ephemeral CSV uploads (works on Vercel too) ─────────────────────
TMP_UPLOAD = Path("/tmp/automl_uploads")
TMP_UPLOAD.mkdir(parents=True, exist_ok=True)

# ── In-memory job progress (fast polling; persisted state goes to Mongo) ──────
JOBS: Dict[str, Dict[str, Any]] = {}

# ─── MongoDB helpers ──────────────────────────────────────────────────────────

def get_db():
    """Return (db, fs) — one connection per call, pooled by pymongo."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db     = client[MONGO_DB]
    fs     = gridfs.GridFS(db)
    return db, fs


def gridfs_put(fs: gridfs.GridFS, data: bytes, filename: str,
               content_type: str, job_id: str) -> str:
    """Store bytes in GridFS and return the string file_id."""
    fid = fs.put(
        data,
        filename=filename,
        content_type=content_type,
        metadata={"job_id": job_id},
    )
    return str(fid)


def gridfs_get(fs: gridfs.GridFS, file_id_str: str) -> bytes:
    from bson import ObjectId
    gf = fs.get(ObjectId(file_id_str))
    return gf.read()

# ─── Schemas ──────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    filename: str
    target_column: str
    job_id: str

class ChatRequest(BaseModel):
    job_id: str
    question: str
    groq_model: str = DEFAULT_MODEL

# ─── ML helpers ──────────────────────────────────────────────────────────────

def detect_task(series: pd.Series) -> str:
    if series.dtype == object or series.dtype.name == "category":
        return "classification"
    return "classification" if series.nunique() <= 20 else "regression"


def build_preprocessor(num_cols, cat_cols):
    t = []
    if num_cols:
        t.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num_cols))
    if cat_cols:
        t.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(t, remainder="drop")


def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42),
        "AdaBoost":            AdaBoostClassifier(algorithm="SAMME", random_state=42),
        "KNN":                 KNeighborsClassifier(),
        "Naive Bayes":         GaussianNB(),
        "Linear SVC":          CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
    }


def get_regressors():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(random_state=42),
        "Lasso":             Lasso(random_state=42),
        "Decision Tree":     DecisionTreeRegressor(random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=100, random_state=42),
        "AdaBoost":          AdaBoostRegressor(random_state=42),
        "KNN":               KNeighborsRegressor(),
        "Linear SVR":        LinearSVR(max_iter=2000, random_state=42),
    }


def setstatus(jid, step, msg, pct=None):
    if jid in JOBS:
        JOBS[jid]["step"]    = step
        JOBS[jid]["message"] = msg
        if pct is not None:
            JOBS[jid]["progress"] = pct


def make_groq_client():
    try:
        import httpx
        try:
            return Groq(api_key=GROQ_API_KEY, http_client=httpx.Client())
        except TypeError:
            pass
    except ImportError:
        pass
    return Groq(api_key=GROQ_API_KEY)

# ─── Pipeline worker ──────────────────────────────────────────────────────────

def run_pipeline(job_id: str, filepath: str, target_col: str):
    job = JOBS[job_id]
    db, fs = get_db()
    lines  = []

    def log(x=""): lines.append(x)

    try:
        # ── 1. Load ───────────────────────────────────────────────────────
        setstatus(job_id, "loading", "Loading dataset…", 5)
        df = pd.read_csv(filepath)
        nr, nc = df.shape

        # ── 2. Analyse columns ────────────────────────────────────────────
        setstatus(job_id, "analysis", "Analysing columns…", 15)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        fcols = [c for c in df.columns if c != target_col]
        ncols = df[fcols].select_dtypes(include=np.number).columns.tolist()
        ccols = df[fcols].select_dtypes(exclude=np.number).columns.tolist()
        task  = detect_task(df[target_col])

        # ── 3. Build text report ──────────────────────────────────────────
        setstatus(job_id, "report", "Generating data report…", 25)
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        log("=" * 70)
        log("  AUTOML STUDIO — DATA & MODEL REPORT")
        log(f"  Generated : {ts}  |  Job : {job_id}")
        log("=" * 70); log()

        log("── DATASET OVERVIEW ─────────────────────────────────────────────────")
        log(f"  File    : {Path(filepath).name}")
        log(f"  Rows    : {nr:,}  |  Columns : {nc}")
        log(f"  Memory  : {df.memory_usage(deep=True).sum()/1024:.1f} KB")
        log(f"  Task    : {task.upper()}  |  Target : {target_col}"); log()

        log("── COLUMN ANALYSIS ──────────────────────────────────────────────────")
        log(f"  Numeric ({len(ncols)}): {', '.join(ncols) or 'None'}")
        log(f"  Categorical ({len(ccols)}): {', '.join(ccols) or 'None'}"); log()

        log("── DATA TYPES ───────────────────────────────────────────────────────")
        for c, d in df.dtypes.items(): log(f"  {c:<35} {str(d)}")
        log()

        log("── MISSING VALUES ───────────────────────────────────────────────────")
        ms = df.isnull().sum()
        for c in df.columns:
            log(f"  {c:<35} {ms[c]:>6}  ({ms[c]/nr*100:.1f}%)")
        log()

        log("── STATISTICAL SUMMARY (NUMERIC) ────────────────────────────────────")
        if ncols:
            desc = df[ncols].describe().T
            log(f"  {'Column':<28}{'Count':>8}{'Mean':>12}{'Std':>12}{'Min':>12}{'Max':>12}")
            log("  " + "-"*86)
            for c, r in desc.iterrows():
                log(f"  {c:<28}{r['count']:>8.0f}{r['mean']:>12.4f}{r['std']:>12.4f}"
                    f"{r['min']:>12.4f}{r['max']:>12.4f}")
        else:
            log("  No numeric columns.")
        log()

        log("── CATEGORICAL SUMMARY ──────────────────────────────────────────────")
        for c in ccols:
            vc = df[c].value_counts()
            log(f"  {c} — {vc.shape[0]} unique values")
            for v, cnt in vc.head(8).items():
                log(f"    {str(v):<30} {cnt:>6}  ({cnt/nr*100:.1f}%)")
            if vc.shape[0] > 8: log(f"    … +{vc.shape[0]-8} more")
            log()

        log("── TARGET COLUMN ────────────────────────────────────────────────────")
        log(f"  Name:{target_col}  Dtype:{df[target_col].dtype}  "
            f"Unique:{df[target_col].nunique()}  Missing:{df[target_col].isnull().sum()}")
        if task == "classification":
            for cls, cnt in df[target_col].value_counts().items():
                log(f"    {str(cls):<30} {cnt:>6}  ({cnt/nr*100:.1f}%)")
        else:
            t = df[target_col].describe()
            log(f"  Mean:{t['mean']:.4f}  Std:{t['std']:.4f}  "
                f"Min:{t['min']:.4f}  Max:{t['max']:.4f}")
        log()

        log("── TOP CORRELATIONS ─────────────────────────────────────────────────")
        if len(ncols) >= 2:
            corr  = df[ncols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            for (c1, c2), val in upper.stack().sort_values(ascending=False).head(10).items():
                log(f"  {c1} ↔ {c2} : {val:.4f}")
        else:
            log("  Not enough numeric columns.")
        log()

        # ── 4. Splits ─────────────────────────────────────────────────────
        X = df[fcols]; y = df[target_col]
        if task == "classification" and y.dtype == object:
            le    = LabelEncoder()
            y_enc = pd.Series(le.fit_transform(y), name=target_col)
            job["label_map"] = {i: c for i, c in enumerate(le.classes_)}
        else:
            y_enc = y; job["label_map"] = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42,
            stratify=y_enc if task == "classification" else None,
        )

        # ── 5. Evidently drift ────────────────────────────────────────────
        setstatus(job_id, "drift", "Running drift analysis…", 40)
        log("── DATA DRIFT ANALYSIS (Evidently) ──────────────────────────────────")
        drift_fid = None
        drift_summary = {"html_available": False}

        if EVIDENTLY_OK:
            try:
                dreport = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
                dreport.run(reference_data=X_train.copy(), current_data=X_test.copy())

                # Save HTML to in-memory buffer then GridFS
                buf = io.StringIO()
                dreport.save_html(buf)
                drift_bytes = buf.getvalue().encode("utf-8")
                drift_fid = gridfs_put(
                    fs, drift_bytes,
                    filename=f"drift_report_{job_id}.html",
                    content_type="text/html",
                    job_id=job_id,
                )

                dd = dreport.as_dict()
                drifted, total_checked, col_drift = 0, 0, []
                for m in dd.get("metrics", []):
                    if m["metric"] == "DatasetDriftMetric":
                        drifted       = m["result"].get("number_of_drifted_columns", 0)
                        total_checked = m["result"].get("number_of_columns", 0)
                    if m["metric"] == "DataDriftTable":
                        for cn, ci in m["result"].get("drift_by_columns", {}).items():
                            col_drift.append({
                                "column":         cn,
                                "drift_detected": ci.get("drift_detected", False),
                                "drift_score":    ci.get("drift_score"),
                                "stattest":       ci.get("stattest_name", ""),
                            })

                log(f"  Train:{len(X_train)} rows  Test:{len(X_test)} rows")
                log(f"  Columns checked:{total_checked}  Drifted:{drifted}"); log()
                for r in col_drift:
                    sc   = f"{r['drift_score']:.4f}" if r["drift_score"] is not None else "N/A"
                    flag = "YES⚠" if r["drift_detected"] else "NO"
                    log(f"  {r['column']:<35} {flag:<8} {sc:>10}  {r['stattest']}")

                drift_summary = {
                    "html_available":  True,
                    "drift_file_id":   drift_fid,
                    "drifted":         drifted,
                    "total":           total_checked,
                    "column_results":  col_drift,
                }
            except Exception as e:
                log(f"  Drift failed: {e}")
                drift_summary = {"error": str(e), "html_available": False}
        else:
            log("  Evidently not installed — skipped.")
            drift_summary = {"error": "Evidently not available", "html_available": False}
        log()

        # ── 6. Train models ───────────────────────────────────────────────
        setstatus(job_id, "training", "Training models…", 50)
        prep   = build_preprocessor(ncols, ccols)
        mdict  = get_classifiers() if task == "classification" else get_regressors()

        log("── MODEL TRAINING ───────────────────────────────────────────────────")
        log(f"  Task:{task.upper()}  Train:{len(X_train)}  Test:{len(X_test)}"); log()

        best_score, best_name, best_pipe = -np.inf, None, None
        mresults = []

        for idx, (name, est) in enumerate(mdict.items()):
            setstatus(job_id, "training", f"Training {name}…",
                      50 + int(35 * (idx / len(mdict))))
            try:
                pipe = Pipeline([("pre", prep), ("mdl", est)])
                pipe.fit(X_train, y_train)
                yp = pipe.predict(X_test)

                if task == "classification":
                    met = {
                        "accuracy":  round(accuracy_score(y_test, yp), 4),
                        "f1":        round(f1_score(y_test, yp, average="weighted", zero_division=0), 4),
                        "precision": round(precision_score(y_test, yp, average="weighted", zero_division=0), 4),
                        "recall":    round(recall_score(y_test, yp, average="weighted", zero_division=0), 4),
                    }
                    pri = met["accuracy"]
                    log(f"  ✓ {name}  Acc={pri:.4f}  F1={met['f1']:.4f}")
                else:
                    r2, mse, mae = (
                        r2_score(y_test, yp),
                        mean_squared_error(y_test, yp),
                        mean_absolute_error(y_test, yp),
                    )
                    met = {"r2": round(r2,4), "mse": round(mse,4),
                           "mae": round(mae,4), "rmse": round(np.sqrt(mse),4)}
                    pri = r2
                    log(f"  ✓ {name}  R²={r2:.4f}  RMSE={np.sqrt(mse):.4f}")

                mresults.append({"name": name, "metrics": met,
                                 "primary": pri, "error": None})
                if pri > best_score:
                    best_score, best_name, best_pipe = pri, name, pipe
            except Exception as e:
                mresults.append({"name": name, "metrics": {},
                                 "primary": None, "error": str(e)})
                log(f"  ✗ {name}  ERROR: {e}")

        pk     = "Accuracy" if task == "classification" else "R²"
        ranked = sorted([r for r in mresults if r["primary"] is not None],
                        key=lambda x: x["primary"], reverse=True)
        log(); log("── MODEL LEADERBOARD ────────────────────────────────────────────────")
        for i, r in enumerate(ranked, 1):
            star = "  ★ BEST" if r["name"] == best_name else ""
            log(f"  {i}. {r['name']:<30} {r['primary']:.4f}{star}")
        log(f"\n  Winner: {best_name}  Score:{best_score:.4f} ({pk})")
        log(); log("=" * 70); log("  END OF REPORT"); log("=" * 70)

        # ── 7. Persist to MongoDB ─────────────────────────────────────────
        setstatus(job_id, "saving", "Saving to MongoDB…", 92)

        # 7a. Text report → GridFS
        report_text  = "\n".join(lines)
        report_bytes = report_text.encode("utf-8")
        report_fid   = gridfs_put(
            fs, report_bytes,
            filename=f"data_report_{job_id}.txt",
            content_type="text/plain",
            job_id=job_id,
        )

        # 7b. Best model (pickle) → GridFS
        model_buf = io.BytesIO()
        pickle.dump({
            "pipeline":        best_pipe,
            "task":            task,
            "target":          target_col,
            "label_map":       job.get("label_map"),
            "best_model_name": best_name,
            "best_score":      best_score,
        }, model_buf)
        model_bytes = model_buf.getvalue()
        model_fid   = gridfs_put(
            fs, model_bytes,
            filename=f"best_model_{job_id}.pkl",
            content_type="application/octet-stream",
            job_id=job_id,
        )

        # 7c. Job document → jobs collection
        job_doc = {
            "job_id":        job_id,
            "created_at":    datetime.utcnow(),
            "status":        "done",
            "task":          task,
            "target_col":    target_col,
            "best_model":    best_name,
            "best_score":    round(best_score, 4),
            "model_results": mresults,
            "n_rows":        nr,
            "n_cols":        nc,
            "num_cols":      ncols,
            "cat_cols":      ccols,
            "feature_cols":  fcols,
            "drift_summary": drift_summary,
            # GridFS file IDs stored as strings
            "report_file_id": report_fid,
            "model_file_id":  model_fid,
            "drift_file_id":  drift_fid,
        }
        db["jobs"].replace_one(
            {"job_id": job_id}, job_doc, upsert=True
        )

        # ── 8. Update in-memory state ─────────────────────────────────────
        JOBS[job_id].update({
            "status":          "done",
            "step":            "done",
            "progress":        100,
            "message":         "Pipeline complete! Saved to MongoDB.",
            "task":            task,
            "best_model":      best_name,
            "best_score":      round(best_score, 4),
            "model_results":   mresults,
            "n_rows":          nr,
            "n_cols":          nc,
            "num_cols":        ncols,
            "cat_cols":        ccols,
            "feature_cols":    fcols,
            "drift_summary":   drift_summary,
            "report_file_id":  report_fid,
            "model_file_id":   model_fid,
            "drift_file_id":   drift_fid,
        })

        # Clean up temp CSV
        try: Path(filepath).unlink()
        except: pass

    except Exception as e:
        tb = traceback.format_exc()
        JOBS[job_id].update({
            "status":    "error",
            "step":      "error",
            "message":   f"Pipeline failed: {e}",
            "traceback": tb,
        })
        try:
            db["jobs"].replace_one(
                {"job_id": job_id},
                {"job_id": job_id, "status": "error",
                 "message": str(e), "traceback": tb,
                 "created_at": datetime.utcnow()},
                upsert=True,
            )
        except: pass


# ─── Helper: load job from Mongo if not in memory ─────────────────────────────

def resolve_job(job_id: str) -> dict:
    """Return job dict — from memory first, then MongoDB fallback."""
    job = JOBS.get(job_id)
    if job and job.get("status") == "done":
        return job
    # Fallback: fetch from MongoDB (handles restarts / multi-instance)
    db, _ = get_db()
    doc   = db["jobs"].find_one({"job_id": job_id}, {"_id": 0})
    if doc:
        # Cache it back
        JOBS[job_id] = doc
        return doc
    return job  # may be None


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    try:
        db, _ = get_db()
        db.command("ping")
        mongo_status = "connected"
    except Exception as e:
        mongo_status = f"error: {e}"
    return {"status": "ok", "service": "AutoML Studio API",
            "version": "3.0", "mongodb": mongo_status}


@app.get("/api/models")
async def list_models():
    return {"models": GROQ_MODELS, "default": DEFAULT_MODEL}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")
    safe = f"{uuid.uuid4().hex}_{file.filename}"
    sp   = TMP_UPLOAD / safe
    sp.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()

    if len(content) > 20_000_000:
        raise HTTPException(400, "CSV too large (20MB max)")

    sp.write_bytes(content)
    try:
        df = pd.read_csv(sp)
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    return {
        "filename":      safe,
        "original_name": file.filename,
        "columns":       list(df.columns),
        "shape":         {"rows": len(df), "cols": len(df.columns)},
        "dtypes":        {c: str(d) for c, d in df.dtypes.items()},
        "missing":       {k: int(v) for k, v in df.isnull().sum().items()},
    }


@app.post("/api/start-pipeline")
async def start_pipeline(req: PipelineRequest, bg: BackgroundTasks):
    fp = str(TMP_UPLOAD / req.filename)
    if not Path(fp).exists():
        raise HTTPException(404, "Uploaded file not found. Please re-upload.")
    JOBS[req.job_id] = {
        "status": "running", "step": "starting",
        "progress": 0, "message": "Initialising pipeline…",
    }
    bg.add_task(run_pipeline, req.job_id, fp, req.target_column)
    return {"job_id": req.job_id, "status": "started"}


@app.get("/api/pipeline-status/{job_id}")
async def pipeline_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@app.get("/api/download-model/{job_id}")
async def download_model(job_id: str):
    job = resolve_job(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Model not ready.")
    fid = job.get("model_file_id")
    if not fid:
        raise HTTPException(404, "Model file ID missing.")
    try:
        _, fs = get_db()
        data  = gridfs_get(fs, fid)
    except Exception as e:
        raise HTTPException(500, f"Could not retrieve model from MongoDB: {e}")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=best_model.pkl"},
    )


@app.get("/api/download-report/{job_id}")
async def download_report(job_id: str):
    job = resolve_job(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Report not ready.")
    fid = job.get("report_file_id")
    if not fid:
        raise HTTPException(404, "Report file ID missing.")
    try:
        _, fs = get_db()
        data  = gridfs_get(fs, fid)
    except Exception as e:
        raise HTTPException(500, f"Could not retrieve report from MongoDB: {e}")
    return Response(
        content=data,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=data_report.txt"},
    )


@app.get("/api/download-drift-report/{job_id}")
async def download_drift_report(job_id: str):
    job = resolve_job(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Pipeline not complete.")
    fid = job.get("drift_file_id")
    if not fid:
        raise HTTPException(404, "Drift report not available — "
                                 "Evidently may not be installed or drift check failed.")
    try:
        _, fs = get_db()
        data  = gridfs_get(fs, fid)
    except Exception as e:
        raise HTTPException(500, f"Could not retrieve drift report: {e}")
    return Response(
        content=data,
        media_type="text/html",
        headers={"Content-Disposition": "inline; filename=drift_report.html"},
    )


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not GROQ_LIB_OK:
        raise HTTPException(500, "Groq library not installed.")
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set in .env")

    job = resolve_job(req.job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    fid = job.get("report_file_id")
    if not fid:
        raise HTTPException(404, "Report not available yet.")

    try:
        _, fs     = get_db()
        raw       = gridfs_get(fs, fid)
        report_txt = raw.decode("utf-8")
    except Exception as e:
        raise HTTPException(500, f"Could not load report: {e}")

    system = (
        "You are an expert data scientist AI assistant. "
        "The user ran an AutoML pipeline and the full report is below. "
        "Answer ONLY from the report. Be precise and concise. "
        "Use bullet points for lists.\n\n"
        f"=== PIPELINE REPORT ===\n{report_txt}\n=== END ==="
    )
    valid    = {m["id"] for m in GROQ_MODELS}
    model_id = req.groq_model if req.groq_model in valid else DEFAULT_MODEL
    try:
        resp = make_groq_client().chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": req.question},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return {"answer": resp.choices[0].message.content, "model_used": model_id}
    except Exception as e:
        raise HTTPException(500, f"Groq API error: {e}")


# ─── List all past jobs (bonus endpoint) ─────────────────────────────────────

@app.get("/api/jobs")
async def list_jobs():
    """Return all completed jobs stored in MongoDB."""
    db, _ = get_db()
    docs  = list(db["jobs"].find(
        {"status": "done"},
        {"_id": 0, "job_id": 1, "created_at": 1, "task": 1,
         "best_model": 1, "best_score": 1, "n_rows": 1,
         "target_col": 1},
    ).sort("created_at", -1).limit(50))
    # Convert datetime to string for JSON
    for d in docs:
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].strftime("%Y-%m-%d %H:%M UTC")
    return {"jobs": docs}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# app = FastAPI(title="AutoML Studio API", version="3.0")