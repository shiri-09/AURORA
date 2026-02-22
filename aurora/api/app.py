"""
AURORA API — FastAPI Application

Entry point for the AURORA web API.

Run with:
    uvicorn aurora.api.app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aurora.api.routes import router

app = FastAPI(
    title="AURORA API",
    description=(
        "Auditable Unlearning for Relational & Orchestrated Reasoning Architectures. "
        "Bounded multi-hop irrecoverable unlearning in transformer-based LLMs."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "AURORA",
        "description": "Auditable Unlearning for Relational & Orchestrated Reasoning Architectures",
        "version": "0.1.0",
        "docs": "/docs",
    }
