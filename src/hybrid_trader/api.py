from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api_schemas import (
    BacktestResource,
    CreateBacktestRequest,
    ErrorResource,
    HealthResource,
)
from .application import BacktestRunner, WalkForwardBacktestRunner


def _validation_error(request: Request, exception: RequestValidationError) -> JSONResponse:
    details = [
        {"type": error["type"], "location": list(error["loc"]), "message": error["msg"]}
        for error in exception.errors()
    ]
    content = ErrorResource(
        error="validation_error",
        message="Request validation failed",
        details=details,
        timestamp=datetime.now(UTC),
        path=request.url.path,
    ).model_dump(mode="json")
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=content)


def create_app(runner: BacktestRunner | None = None) -> FastAPI:
    backtest_runner = runner or WalkForwardBacktestRunner()
    app = FastAPI(
        title="Hybrid Trader Research API",
        version="1.0.0",
        description="Backtesting only. Live trading is intentionally unavailable.",
    )
    app.add_exception_handler(RequestValidationError, _validation_error)

    @app.get("/api/v1/health", response_model=HealthResource, tags=["system"])
    def health() -> HealthResource:
        return HealthResource(status="ready", live_trading=False)

    @app.post(
        "/api/v1/backtests",
        response_model=BacktestResource,
        status_code=status.HTTP_201_CREATED,
        tags=["backtests"],
    )
    def create_backtest(request: CreateBacktestRequest) -> BacktestResource:
        result = backtest_runner.run(request.symbol, request.to_frame())
        return BacktestResource(
            id=str(uuid4()),
            symbol=request.symbol,
            metrics=result.metrics,
            prediction_count=len(result.predictions),
            trade_count=len(result.trades),
        )

    return app


app = create_app()
