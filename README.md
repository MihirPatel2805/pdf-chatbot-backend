# pdf-chatbot-backend

Setup

- Create venv: `python -m venv myenv`
- Activate venv: `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`)
- Install deps: `pip install -r requirements.txt`
- Run server: `uvicorn app.main:app --reload`

Features

- API docs (Swagger UI): open `http://127.0.0.1:8000/docs`
- ReDoc: open `http://127.0.0.1:8000/redoc`
- Health/root endpoint: `GET /` returns a simple JSON

Add a new route (example)

- Edit `app/main.py` and add another endpoint:
  - Example:
    - `@app.get("/ping")`
    - `def ping(): return {"pong": True}`

Admin page (using sqladmin) - instructions

- Install optional deps:
  - `pip install sqladmin sqlalchemy aiosqlite`
- Create a database model (example):
  - Create `app/models.py` with a SQLAlchemy model, e.g. `User`
- Initialize DB engine and admin in `app/main.py`:
  - Create SQLAlchemy engine: `create_async_engine("sqlite+aiosqlite:///./app.db")`
  - Create a `sqladmin.Admin` instance and register a `ModelView` for your models
- Start server and open `http://127.0.0.1:8000/admin` to access the admin panel

Notes

- Use `--reload` only for development
- Pin extra dependencies you adopt in `requirements.txt`

## Required Configuration

Before using the PDF upload functionality, you need to set up:

### 1. Google Gemini API Key

```bash
# Set environment variable
export GOOGLE_API_KEY=your_google_api_key_here

# Or create a .env file in project root
GOOGLE_API_KEY=your_google_api_key_here
```
