# FedServer

FastAPI backend and Vite/React frontend for federated learning workflows.

## Prerequisites

- Python 3 with `venv`
- Node.js and npm
- A SQL database supported by SQLAlchemy (SQLite or PostgreSQL, etc.) — migrations expect `DB_URL` in `backend/.env`
- `concurrently` (e.g. `npm install -g concurrently`) — used by `start_local.sh`
- Redis on `localhost:6380` with password `123456` (matches `backend/utilities/redis.conf`; optional if you change app config, but `start_local.sh` starts Redis with that config)

On Debian/Ubuntu, `install_local.sh` installs the system `redis-server` package via `apt-get`.

## Replicate & install

```bash
git clone <repo-url> FedServer
cd FedServer
chmod +x install_local.sh start_local.sh
```

Create `backend/.env` with at least `DB_URL` (and any other variables your deployment needs). Create `frontend/.env` with API URLs such as `REACT_APP_SERVER_BASE_URL` pointing at your backend.

Then:

```bash
./install_local.sh
```

`install_local.sh` creates the Python virtualenv, installs dependencies, runs `alembic upgrade head`, and `npm install` in `frontend/`.

Edit `backend/.env` and `frontend/.env` if your API URLs, database, or storage paths differ.

### Database migrations (manual)

If you need to manage migrations outside the installer:

- `cd backend && source venv/bin/activate && alembic upgrade head`
- New revision: `alembic revision --autogenerate -m "description"` then `alembic upgrade head`

If migration history is inconsistent, you may need to fix `alembic/versions` per your environment before upgrading.

## Run

From the repo root:

```bash
./start_local.sh
```

- **Backend:** `http://0.0.0.0:8000` (reload via uvicorn)
- **Frontend dev server:** port **5173** (see `frontend/vite.config.js`)

Redis is started in the background using `backend/utilities/redis.conf` (port **6380**).
