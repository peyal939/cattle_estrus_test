from fastapi import FastAPI, Request, HTTPException, Query, Depends, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from app.database import (
    connect_to_mongo,
    close_mongo_connection,
    fetch_sensor_data,
    fetch_daily_metrics,
    upsert_daily_metrics,
    get_all_tag_ids,
    get_latest_reading,
    get_reading_count
)
from app.analysis import (
    analyze_estrus,
    analyze_estrus_from_daily,
    compute_daily_metrics,
    get_current_activity,
)
from app.config import get_settings, format_bst, now_bst

import pytz
from datetime import datetime, timedelta
from typing import Optional
from app.models import SettingsUpdate, SettingsResponse
from app.auth import router as auth_router, get_current_active_user, get_admin_user
from app.websocket import manager, websocket_endpoint
from app.cache import TTLCache, make_key

settings = get_settings()

# Cache analysis results for short periods to keep the dashboard snappy on refresh.
analysis_cache = TTLCache(ttl_seconds=max(10, settings.ws_refresh_interval), max_items=512)


def _bst_date_range_to_utc(start_date: str = None, end_date: str = None):
    """Convert YYYY-MM-DD (BST) start/end to naive UTC datetimes suitable for Mongo queries."""
    if not start_date and not end_date:
        return None, None

    bst = pytz.timezone(settings.timezone)

    start_utc = None
    end_utc = None

    if start_date:
        start_bst = bst.localize(datetime.strptime(start_date, "%Y-%m-%d"))
        start_utc = start_bst.astimezone(pytz.utc).replace(tzinfo=None)

    if end_date:
        end_bst = bst.localize(datetime.strptime(end_date, "%Y-%m-%d"))
        # inclusive end-of-day
        end_bst = end_bst + timedelta(days=1) - timedelta(milliseconds=1)
        end_utc = end_bst.astimezone(pytz.utc).replace(tzinfo=None)

    return start_utc, end_utc

# Runtime settings (mutable, can be updated via API)
runtime_settings = {
    "rolling_activity_days": settings.rolling_activity_days,
    "estrus_baseline_days": settings.estrus_baseline_days,
    "relative_std_multiplier": settings.relative_std_multiplier,
    "absolute_activity_multiplier": settings.absolute_activity_multiplier,
    "walking_threshold": settings.walking_threshold,
    "dominance_ratio_threshold": settings.dominance_ratio_threshold,
    "ws_refresh_interval": settings.ws_refresh_interval,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    # Startup
    await connect_to_mongo()
    manager.start_background_task()
    yield
    # Shutdown
    manager.stop_background_task()
    await close_mongo_connection()


app = FastAPI(
    title="🐄 Cattle Estrus Detection System",
    description="Real-time cattle behavior monitoring and estrus detection with authentication",
    version="1.0.0",
    lifespan=lifespan
)

# Templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Include auth router
app.include_router(auth_router)


# ==============================
# PAGE ROUTES
# ==============================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to login or dashboard"""
    return RedirectResponse(url="/login")


@app.get("/favicon.ico")
async def favicon():
    """Avoid noisy 404s from browsers requesting a favicon."""
    return Response(status_code=204)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard page (requires authentication via JS)"""
    tag_ids = await get_all_tag_ids()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "tag_ids": tag_ids,
            "settings": runtime_settings,
            "timezone": settings.timezone
        }
    )


# ==============================
# API ROUTES - PUBLIC
# ==============================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "cattle-estrus-detection",
        "timezone": settings.timezone,
        "current_time": format_bst(now_bst())
    }


# ==============================
# API ROUTES - AUTHENTICATED
# ==============================

@app.get("/api/tags")
async def get_tags(current_user: dict = Depends(get_current_active_user)):
    """Get list of all tag IDs"""
    tag_ids = await get_all_tag_ids()
    return {"tag_ids": tag_ids, "total": len(tag_ids)}


@app.get("/api/tags/{tag_id}/latest")
async def get_tag_latest(
    tag_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Get the latest sensor reading for a specific tag"""
    reading = await get_latest_reading(tag_id=tag_id)
    if not reading:
        raise HTTPException(status_code=404, detail=f"No data for tag: {tag_id}")
    return reading


@app.get("/api/tags/{tag_id}/analysis")
async def get_tag_analysis(
    tag_id: str,
    days: int = Query(default=30, ge=0, le=365, description="Days to analyze (0 = all available)"),
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD) in BST"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD) in BST"),
    current_user: dict = Depends(get_current_active_user)
):
    """Get estrus analysis for a specific tag"""
    start_utc, end_utc = _bst_date_range_to_utc(start_date, end_date)
    
    # If date range is provided, ignore days parameter (fetch all in range)
    effective_days = 0 if (start_utc or end_utc) else days
    cache_key = make_key(
        "tag_analysis",
        str(tag_id),
        effective_days,
        start_utc.isoformat() if start_utc else None,
        end_utc.isoformat() if end_utc else None,
        tuple(sorted(runtime_settings.items())),
    )
    cached = analysis_cache.get(cache_key)
    if cached:
        return cached

    # Fast path: analyze from precomputed daily metrics.
    daily_df = await fetch_daily_metrics(
        tag_id=tag_id,
        start_date=start_date,
        end_date=end_date,
        days=effective_days,
    )

    # Fallback: compute daily metrics from raw data once (single-tag only), store, then analyze.
    df_recent = None
    if daily_df is None or daily_df.empty:
        df = await fetch_sensor_data(tag_id=tag_id, days=effective_days, start_utc=start_utc, end_utc=end_utc)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for tag: {tag_id}")

        daily_rows = compute_daily_metrics(df, config=runtime_settings)
        if daily_rows is not None and not daily_rows.empty:
            await upsert_daily_metrics(tag_id=tag_id, daily_rows=daily_rows.to_dict(orient="records"))
            daily_df = daily_rows

        # We already have raw data in memory; reuse a small tail for current activity.
        df_recent = df.tail(100)

    # Analyze from daily metrics if available; otherwise fall back to raw analysis (should be rare).
    if daily_df is not None and not daily_df.empty:
        result = analyze_estrus_from_daily(daily_df, config=runtime_settings)
    else:
        df = await fetch_sensor_data(tag_id=tag_id, days=effective_days, start_utc=start_utc, end_utc=end_utc)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for tag: {tag_id}")
        result = analyze_estrus(df, config=runtime_settings)

    result["tag_id"] = tag_id
    
    # Add latest reading info
    latest = await get_latest_reading(tag_id)
    if latest:
        result["battery_level"] = latest.get("battery_level")
        result["signal_strength"] = latest.get("rssi")
        result["last_reading_time"] = latest.get("time")
    
    # Get current activity (use small recent raw slice)
    if df_recent is None:
        df_recent = await fetch_sensor_data(tag_id=tag_id, days=1)
    result["current_activity"] = get_current_activity(df_recent)

    analysis_cache.set(cache_key, result)
    return result


@app.get("/api/analysis")
async def get_all_analysis(
    days: int = Query(default=30, ge=0, le=365, description="Days to analyze (0 = all available)"),
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD) in BST"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD) in BST"),
    current_user: dict = Depends(get_current_active_user)
):
    """Get estrus analysis for all tags"""
    tag_ids = await get_all_tag_ids()

    start_utc, end_utc = _bst_date_range_to_utc(start_date, end_date)
    
    # If date range is provided, ignore days parameter (fetch all in range)
    effective_days = 0 if (start_utc or end_utc) else days

    async def _analyze_one(tag_id: str):
        try:
            cache_key = make_key(
                "all_analysis_one",
                str(tag_id),
                effective_days,
                start_utc.isoformat() if start_utc else None,
                end_utc.isoformat() if end_utc else None,
                tuple(sorted(runtime_settings.items())),
            )
            cached = analysis_cache.get(cache_key)
            if cached:
                return cached

            # Daily-metrics fast path only for all-tags to prevent timeouts.
            daily_df = await fetch_daily_metrics(
                tag_id=tag_id,
                start_date=start_date,
                end_date=end_date,
                days=effective_days,
            )

            if daily_df is not None and not daily_df.empty:
                result = analyze_estrus_from_daily(daily_df, config=runtime_settings)
            else:
                # Do NOT fall back to raw data here; that recreates the original performance problem.
                result = {
                    "error": "Daily metrics not ready for this tag",
                    "daily_data": [],
                    "estrus_detected": False,
                    "estrus_date": None,
                    "total_days": 0,
                    "total_readings": 0,
                    "summary": {
                        "avg_activity_score": 0,
                        "avg_activity_score_raw": 0,
                        "avg_walking_fraction": 0,
                        "avg_resting_fraction": 0,
                        "max_activity_score": 0,
                        "min_activity_score": 0,
                        "max_activity_score_raw": 0,
                        "min_activity_score_raw": 0,
                    },
                    "last_updated": format_bst(now_bst()),
                }

            result["tag_id"] = tag_id

            latest = await get_latest_reading(tag_id)
            if latest:
                result["battery_level"] = latest.get("battery_level")
                result["signal_strength"] = latest.get("rssi")
                result["last_reading_time"] = latest.get("time")

            # Current activity from last 1 day only (lightweight)
            df_recent = await fetch_sensor_data(tag_id=tag_id, days=1)
            result["current_activity"] = get_current_activity(df_recent)
            analysis_cache.set(cache_key, result)
            return result
        except Exception:
            return {
                "error": "Failed to analyze tag",
                "tag_id": tag_id,
                "daily_data": [],
                "estrus_detected": False,
                "estrus_date": None,
                "total_days": 0,
                "total_readings": 0,
                "summary": {
                    "avg_activity_score": 0,
                    "avg_activity_score_raw": 0,
                    "avg_walking_fraction": 0,
                    "avg_resting_fraction": 0,
                    "max_activity_score": 0,
                    "min_activity_score": 0,
                    "max_activity_score_raw": 0,
                    "min_activity_score_raw": 0,
                },
                "last_updated": format_bst(now_bst()),
            }

    import asyncio

    analyzed = await asyncio.gather(*[_analyze_one(t) for t in tag_ids])
    results = analyzed
    estrus_alerts = [
        {
            "tag_id": r["tag_id"],
            "estrus_date": r.get("estrus_date"),
            "activity_score": r.get("summary", {}).get("max_activity_score"),
        }
        for r in results
        if r.get("estrus_detected")
    ]
    
    return {
        "analyses": results,
        "total_tags": len(tag_ids),
        "estrus_alerts": estrus_alerts,
        "alert_count": len(estrus_alerts),
        "last_updated": format_bst(now_bst())
    }


@app.get("/api/stats")
async def get_stats(current_user: dict = Depends(get_current_active_user)):
    """Get overall system statistics"""
    tag_ids = await get_all_tag_ids()
    
    stats = {
        "total_tags": len(tag_ids),
        "tags": []
    }
    
    for tag_id in tag_ids:
        count = await get_reading_count(tag_id=tag_id, days=30)
        latest = await get_latest_reading(tag_id)
        
        stats["tags"].append({
            "tag_id": tag_id,
            "reading_count_30d": count,
            "last_seen": latest.get("time") if latest else None,
            "battery_level": latest.get("battery_level") if latest else None
        })
    
    stats["current_time"] = format_bst(now_bst())
    return stats


# ==============================
# SETTINGS ROUTES (ADMIN ONLY)
# ==============================

@app.get("/api/settings", response_model=SettingsResponse)
async def get_current_settings(current_user: dict = Depends(get_current_active_user)):
    """Get current analysis settings"""
    return SettingsResponse(**runtime_settings)


@app.post("/api/settings", response_model=SettingsResponse)
async def update_settings(
    settings_update: SettingsUpdate,
    current_user: dict = Depends(get_admin_user)
):
    """Update analysis settings (admin only)"""
    update_data = settings_update.model_dump(exclude_unset=True)
    runtime_settings.update(update_data)
    
    # Update WebSocket refresh interval if changed
    if "ws_refresh_interval" in update_data:
        manager.update_refresh_interval(update_data["ws_refresh_interval"])
    
    return SettingsResponse(**runtime_settings)


# ==============================
# WEBSOCKET ROUTE
# ==============================

@app.websocket("/ws")
async def websocket_route(websocket: WebSocket, token: str = None):
    """WebSocket endpoint for real-time updates"""
    await websocket_endpoint(websocket, token)


# ==============================
# ERROR HANDLERS
# ==============================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    if request.url.path.startswith("/api/"):
        return {"error": "Not found", "detail": str(exc.detail)}
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Page not found"},
        status_code=404
    )
