from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_auth
from app.core.security import create_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
async def login() -> dict:
    token = create_token("finance_manager", role="finance_manager")
    return {"token": token, "token_type": "bearer"}


@router.post("/refresh")
async def refresh() -> dict:
    return {"token": create_token("finance_manager", role="finance_manager")}


@router.post("/logout")
async def logout() -> dict:
    return {"success": True, "message": "Logged out"}


@router.get("/me")
async def me(payload: dict = Depends(require_auth)) -> dict:
    return {"user": payload}
