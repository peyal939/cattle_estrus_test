import asyncio
import json
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from app.config import get_settings, format_bst, now_bst
from app.database import (
    fetch_sensor_data,
    fetch_daily_metrics,
    get_all_tag_ids,
    get_latest_reading,
)
from app.analysis import analyze_estrus_from_daily, get_current_activity
from app.auth import decode_token

settings = get_settings()


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        # All active connections
        self.active_connections: Dict[str, WebSocket] = {}
        # Connections subscribed to specific tags
        self.tag_subscriptions: Dict[str, Set[str]] = {}
        # Background task reference
        self.background_task: Optional[asyncio.Task] = None
        # Runtime settings (can be updated)
        self.refresh_interval: int = settings.ws_refresh_interval
        # Previous estrus states for alert detection
        self.previous_estrus: Dict[str, bool] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, token: str = None) -> bool:
        """Accept a new WebSocket connection with optional authentication"""
        # Verify token if provided
        user = None
        if token:
            token_data = decode_token(token)
            if token_data:
                from app.user_db import get_user_by_username
                user = get_user_by_username(token_data.username)
        
        if not user:
            await websocket.close(code=4001, reason="Authentication required")
            return False
        
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # Send connection confirmation
        await self.send_personal_message({
            "type": "connected",
            "data": {
                "client_id": client_id,
                "user": user["username"],
                "refresh_interval": self.refresh_interval
            },
            "timestamp": format_bst(now_bst())
        }, client_id)
        
        return True
    
    def disconnect(self, client_id: str):
        """Remove a connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from all subscriptions
        for tag_id in self.tag_subscriptions:
            self.tag_subscriptions[tag_id].discard(client_id)
    
    async def subscribe_to_tag(self, client_id: str, tag_id: str):
        """Subscribe a client to a specific tag's updates"""
        if tag_id not in self.tag_subscriptions:
            self.tag_subscriptions[tag_id] = set()
        self.tag_subscriptions[tag_id].add(client_id)
        
        # Send immediate update for this tag
        await self.send_tag_update(tag_id, [client_id])
    
    async def unsubscribe_from_tag(self, client_id: str, tag_id: str):
        """Unsubscribe a client from a tag"""
        if tag_id in self.tag_subscriptions:
            self.tag_subscriptions[tag_id].discard(client_id)
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to a specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception:
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)
    
    async def broadcast_to_tag_subscribers(self, tag_id: str, message: dict):
        """Send message to clients subscribed to a specific tag"""
        if tag_id not in self.tag_subscriptions:
            return
        
        disconnected = []
        for client_id in self.tag_subscriptions[tag_id]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception:
                    disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)
    
    async def send_tag_update(self, tag_id: str, client_ids: list = None):
        """Fetch and send analysis update for a specific tag"""
        try:
            daily_df = await fetch_daily_metrics(tag_id=tag_id, days=30)
            if daily_df is None or daily_df.empty:
                return

            analysis = analyze_estrus_from_daily(daily_df)
            analysis["tag_id"] = tag_id
            
            # Get latest reading info
            latest = await get_latest_reading(tag_id)
            if latest:
                analysis["battery_level"] = latest.get("battery_level")
                analysis["signal_strength"] = latest.get("rssi")
                analysis["last_reading_time"] = latest.get("time")
            
            # Get current activity from a small recent slice only
            df_recent = await fetch_sensor_data(tag_id=tag_id, days=1)
            analysis["current_activity"] = get_current_activity(df_recent)
            
            message = {
                "type": "update",
                "data": analysis,
                "timestamp": format_bst(now_bst())
            }
            
            # Check for new estrus detection
            if analysis.get("estrus_detected"):
                prev = self.previous_estrus.get(tag_id, False)
                if not prev:
                    # New estrus detected - send alert
                    alert_message = {
                        "type": "alert",
                        "data": {
                            "tag_id": tag_id,
                            "detected_date": analysis.get("estrus_date"),
                            "activity_score": analysis["summary"]["max_activity_score"],
                            "message": f"🚨 ESTRUS DETECTED for Tag {tag_id}!"
                        },
                        "timestamp": format_bst(now_bst())
                    }
                    await self.broadcast(alert_message)
            
            self.previous_estrus[tag_id] = analysis.get("estrus_detected", False)
            
            # Send to specific clients or subscribers
            if client_ids:
                for client_id in client_ids:
                    await self.send_personal_message(message, client_id)
            else:
                await self.broadcast_to_tag_subscribers(tag_id, message)
                
        except Exception as e:
            print(f"Error sending tag update: {e}")
    
    async def send_all_updates(self):
        """Send updates for all tags to all subscribers"""
        try:
            tag_ids = await get_all_tag_ids()
            
            for tag_id in tag_ids:
                await self.send_tag_update(tag_id)
            
            # Also broadcast a summary to all clients
            summary = {
                "type": "summary",
                "data": {
                    "total_tags": len(tag_ids),
                    "tags": tag_ids,
                    "estrus_alerts": [
                        tag for tag, status in self.previous_estrus.items() if status
                    ]
                },
                "timestamp": format_bst(now_bst())
            }
            await self.broadcast(summary)
            
        except Exception as e:
            print(f"Error in send_all_updates: {e}")
    
    async def background_refresh(self):
        """Background task that periodically sends updates"""
        while True:
            try:
                await asyncio.sleep(self.refresh_interval)
                
                if self.active_connections:
                    await self.send_all_updates()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Background refresh error: {e}")
    
    def start_background_task(self):
        """Start the background refresh task"""
        if self.background_task is None or self.background_task.done():
            self.background_task = asyncio.create_task(self.background_refresh())
    
    def stop_background_task(self):
        """Stop the background refresh task"""
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
    
    def update_refresh_interval(self, interval: int):
        """Update the refresh interval"""
        self.refresh_interval = interval


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, token: str = None):
    """WebSocket endpoint handler"""
    client_id = f"client_{id(websocket)}_{datetime.utcnow().timestamp()}"
    
    # Try to connect with authentication
    connected = await manager.connect(websocket, client_id, token)
    
    if not connected:
        return
    
    try:
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type")
            
            if msg_type == "subscribe":
                tag_id = message.get("tag_id")
                if tag_id:
                    await manager.subscribe_to_tag(client_id, tag_id)
            
            elif msg_type == "unsubscribe":
                tag_id = message.get("tag_id")
                if tag_id:
                    await manager.unsubscribe_from_tag(client_id, tag_id)
            
            elif msg_type == "refresh":
                # Manual refresh request
                tag_id = message.get("tag_id")
                if tag_id:
                    await manager.send_tag_update(tag_id, [client_id])
                else:
                    await manager.send_all_updates()
            
            elif msg_type == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": format_bst(now_bst())
                }, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(client_id)
