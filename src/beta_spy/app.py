from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .live import StateHub


def create_app(hub: StateHub) -> FastAPI:
    app = FastAPI(title="Beta-spy", version="0.1.0")
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/state")
    async def state() -> dict:
        return hub.snapshot()

    @app.get("/api/health")
    async def health() -> dict:
        state = hub.snapshot()
        return {"ok": state.get("status") not in {"STOPPED", "ERROR"}, "status": state.get("status")}

    @app.websocket("/ws")
    async def websocket_state(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(hub.snapshot())
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return

    return app
