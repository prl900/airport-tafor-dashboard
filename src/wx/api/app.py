"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wx.api.routes import metar, stations, taf, verification


def create_app() -> FastAPI:
    app = FastAPI(
        title="Airport TAFOR Dashboard API",
        version="0.1.0",
        description="METAR observations, TAF forecasts and verification for Spanish airports.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # dev SPA; tighten for deployment
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(stations.router)
    app.include_router(metar.router)
    app.include_router(taf.router)
    app.include_router(verification.router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
