from __future__ import annotations

from typing import Any, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import verify_token

bearer_scheme = HTTPBearer(auto_error=False)


async def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> Dict[str, Any]:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication")
    try:
        return verify_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
