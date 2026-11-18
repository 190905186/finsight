from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.copilot import router as copilot_router
from app.api.routes.documents import router as documents_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.finance import router as finance_router
from app.api.routes.rag import router as rag_router
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    openapi_url=settings.openapi_url,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(copilot_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")
app.include_router(finance_router, prefix="")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}


@app.get("/")
async def root() -> dict:
    return {"message": "FinanceOps AI API is running"}
