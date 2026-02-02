EVRAS Deployment Guide

Overview
- Backend: Express API that invokes the existing Python pipeline.
- Frontend: React (Vite) app with upload, LLM toggle, JSON view, and output download.
- Optional: Serve the built React app from the Express server.

Prerequisites
- Python 3.9+ with the existing virtual environment.
- Node.js 18+.
- (Optional) Ollama running locally for LLM explanations.

1) Python setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Backend setup
```
cd server
npm install
copy .env.example .env
npm run dev
```

3) Frontend setup (development)
```
cd web
npm install
npm run dev
```
Open `http://localhost:5173`.

4) Production build (single server)
```
cd web
npm run build
```
The Express server serves `web/dist` automatically if it exists. Start the server:
```
cd server
npm start
```
Open `http://localhost:5000`.

Notes
- The LLM explanation toggle controls the call to Ollama. When off, the pipeline skips it.
- Set `PYTHON_BIN` in `server/.env` if your Python binary is not `python`.
