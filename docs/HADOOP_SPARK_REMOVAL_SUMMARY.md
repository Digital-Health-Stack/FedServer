# Hadoop / Spark Removal — Summary & Server-Side Playbook

This document summarizes what was done in **this client repository** to remove Hadoop (HDFS) and Apache Spark, and how to apply a **similar migration on the server-side codebase** (or any sibling service that still depends on HDFS/Spark).

---

## Goals

- **Remove** all Hadoop/HDFS and Spark/PySpark dependencies, containers, env vars, and application code paths.
- **Replace** HDFS-backed storage with **local filesystem** storage under a configurable root directory.
- **Replace** Spark-based data processing with **pandas** (and scikit-learn where ML utilities were Spark-based).
- **Keep** Redis, federated learning flows, S3 (if used for QPD or other features), and unrelated services unchanged.

---

## What Was Deleted or Retired (Client / Reference Backend)

| Area | Action |
|------|--------|
| **Python modules** | Remove dedicated HDFS and Spark service layers (e.g. `hdfs_services.py`, `spark_services.py`) and any `testjob.py` or Spark-only harnesses. |
| **Dependencies** | Remove packages such as `pyspark`, `hdfs`, `py4j`, `findspark`, and any HDFS-only Arrow usage if applicable. |
| **Docker** | Remove `JAVA_HOME`, `SPARK_HOME`, Spark/HDFS image layers or startup scripts from `Dockerfile` / compose. |
| **Config** | Strip Hadoop/Spark/HDFS-related variables from `.env.example` and deployment docs; add something like `LOCAL_STORAGE_DIR=./storage` (or your chosen path). |

---

## Replacement Architecture

### Storage (was HDFS)

- Introduce a **single storage manager** (e.g. `LocalStorageManager`) that mirrors the **async method signatures** the old HDFS manager exposed (`save_file`, `delete_file`, `rename_file_or_folder`, `list_recent_uploads`, etc.) so route handlers change minimally.
- Root all file paths under one directory from env (e.g. `LOCAL_STORAGE_DIR`), with **flat or predictable layout** matching how uploads and derived files are named.
- Use standard library (`os`, `shutil`) and optional `tempfile` for uploads before moving into the storage root.

### Processing (was Spark)

- Introduce a **pandas-based processing manager** (e.g. `DataProcessingManager`) that replaces `SparkSessionManager`-style entry points: dataset overview/stats, preprocessing pipeline, and any JSON serialization helpers.
- Implement **`serialize_for_json`** (or equivalent) for NumPy/pandas/datetime types so API responses stay valid JSON.
- Rewrite column-level and “all columns” operations in a dedicated helper module using **pandas** (and **scikit-learn** only where Spark ML was used).

### API routes

- **File upload routes**: write to local storage instead of HDFS; trigger background jobs that call the new processing manager.
- **Preprocessing routes**: call local storage + pandas pipeline; remove any Spark session lifecycle.
- **Confidential / admin routers**: drop Spark/HDFS-only endpoints; keep DB and other non-Hadoop APIs.
- **Training / CRUD / QPD / federated services**: replace imports and calls from HDFS/Spark managers to local + pandas managers; update comments and env var names only where they referenced HDFS.

---

## Frontend Alignment (When the “Private” API Is Co-Evolved)

- Remove UI text and navigation for **HDFS/Spark status** or “processed vs raw” if product intent is **one “My Datasets” concept**.
- Point dataset lists and training dataset pickers at **one list endpoint** (e.g. raw/list-my-datasets) if the backend consolidates tables later.
- Remove routes like `/processed-dataset-overview/:filename` if there is no separate processed catalog in the product.

*(Exact file names in this repo: `ViewAllDatasets.jsx`, `Dashboard.jsx`, `ActionSection.jsx`, `privateService.ts`, `App.jsx`, `NavBar.jsx`.)*

---

## Database Note (Optional Second Phase)

Removing Spark/HDFS from **runtime** does not automatically merge **two ORM models** (e.g. “raw” vs “processed” datasets). If the product should have **one logical dataset**:

- Plan a migration: single table or unified list API, update CRUD, preprocessing job to update **one** row, and deprecate duplicate list/detail endpoints.
- This is **separate** from swapping HDFS → disk and Spark → pandas but avoids lingering “processed” APIs in code and docs.

---

## Verification Checklist (Repeat on Server)

1. **Grep** for: `hdfs`, `HDFS`, `pyspark`, `SparkSession`, `findspark`, `SPARK_HOME`, `JAVA_HOME` (for Spark only), `hdfs_services`, `spark_services`.
2. **Install** dependencies in a clean venv / image build; ensure no missing imports.
3. **Start** the API and hit: health, upload, list datasets, dataset details, preprocess (if applicable), training-adjacent paths that touched storage.
4. **Docker**: build image without Spark/HDFS base; confirm compose has no extra Hadoop services.
5. **E2E** (if available): upload → preprocess → training flow using local files only.

---

## Exclusions (Do Not Remove as Part of This Work)

- **Redis** (caching, sockets, sessions).
- **Federated learning** orchestration and related APIs.
- **S3** or other cloud object storage used for QPD or benchmarks (unless explicitly in scope of a different task).

---

## Quick “Server Repo” Task List

When applying this to the **server** codebase, mirror the above in order:

1. Inventory and delete Spark/HDFS modules; remove dead imports project-wide.
2. Add `LocalStorageManager` + `DataProcessingManager` (or equivalent) with APIs matching old call sites.
3. Rewrite processing helpers to pandas/sklearn; add JSON serialization for stats payloads.
4. Update upload, preprocessing, and any confidential routes that referenced HDFS/Spark.
5. Trim `requirements.txt` / `pyproject.toml` and Docker; refresh `.env.example`.
6. Run grep + smoke tests; optionally plan DB/API consolidation for “one dataset” product semantics.

---

## Reference: This Repository’s High-Level File Touchpoints

*(Use as a map when searching the server tree for equivalents.)*

- Backend: `requirements.txt`, `Dockerfile`, `main.py`, `api/*_routes.py`, `crud/datasets_crud.py`, `utility/*_services.py`, `utility/processing_helper_functions.py`, `utility/federated_services.py`
- Frontend (if server repo includes a UI): services, env examples, dataset/training components, routes

---

*Generated as a portable summary of the Hadoop/Spark removal and local/pandas replacement work performed in this project, for reuse on the server-side codebase.*
