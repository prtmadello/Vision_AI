"""
Main API Endpoints - Unified REST API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

from utils.config_loader import APIConfigLoader
from utils.logger import setup_logger
from services.kafka_service import KafkaService
from services.data_stream_service import DataStreamService

import importlib.util

logger = setup_logger(__name__)

# Initialize configuration
config_loader = APIConfigLoader()
config = config_loader.config

# Initialize services
kafka_service = KafkaService(config_loader.get_kafka_config())
data_stream_service = DataStreamService(config_loader.get_data_stream_config())

# Dynamically load CoreAIService from AI Layer even though the directory has a space
def _load_core_ai_service():
    try:
        # Use absolute path to avoid issues with spaces in directory names
        ai_layer_dir = Path("/home/prithiviraj/Vision_AI/AI_Layer").resolve()
        ai_utils_dir = ai_layer_dir / "utils"
        
        # Temporarily modify sys.path to prioritize AI Layer imports
        original_path = sys.path.copy()
        
        # Remove API Layer utils from path temporarily to avoid conflicts
        api_utils_path = str(Path(__file__).parent / "utils")
        if api_utils_path in sys.path:
            sys.path.remove(api_utils_path)
        
        # Also remove the parent directory path that includes API Layer utils
        parent_dir_path = str(Path(__file__).parent.parent.parent)
        if parent_dir_path in sys.path:
            sys.path.remove(parent_dir_path)
        
        # Remove any other API Layer paths that might conflict
        api_layer_path = str(Path(__file__).parent)
        if api_layer_path in sys.path:
            sys.path.remove(api_layer_path)
        
        # Add AI Layer paths at the beginning (highest priority)
        if str(ai_utils_dir) not in sys.path:
            sys.path.insert(0, str(ai_utils_dir))
        if str(ai_layer_dir) not in sys.path:
            sys.path.insert(0, str(ai_layer_dir))
        
        try:
            # Debug: Check what's in sys.path
            logger.debug(f"AI Layer sys.path after manipulation: {sys.path[:5]}")
            
            # First load the config_loader module from AI Layer
            config_loader_path = ai_layer_dir / 'utils' / 'config_loader.py'
            config_spec = importlib.util.spec_from_file_location("ai_config_loader", str(config_loader_path))
            config_module = importlib.util.module_from_spec(config_spec)
            config_spec.loader.exec_module(config_module)
            
            # Then load the core_ai_service module
            module_path = ai_layer_dir / 'core_ai_service.py'
            spec = importlib.util.spec_from_file_location("core_ai_service", str(module_path))
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            return module.CoreAIService
        finally:
            # Restore original sys.path
            sys.path[:] = original_path
            
    except Exception as e:
        logger.warning(f"Failed to load CoreAIService, using stub: {e}")
        _err_msg = str(e)
        class _StubAI:
            def __init__(self, *args, **kwargs):
                self._error = _err_msg
            def get_service_status(self):
                return {"available": False, "error": self._error}
        return _StubAI

# Load AI service lazily to avoid import conflicts at module load time
ai_service = None

def get_ai_service():
    """Get AI service instance, initializing it lazily if needed"""
    global ai_service
    if ai_service is None:
        try:
            # Load CoreAIService class lazily each time
            CoreAIService = _load_core_ai_service()
            ai_service = CoreAIService(config_loader.get_ai_layer_config().get('config_path'))
        except Exception as e:
            logger.error(f"Failed to initialize AI service: {e}")
            # Create a stub service
            class _StubAI:
                def __init__(self, *args, **kwargs):
                    self._error = str(e)
                def get_service_status(self):
                    return {"available": False, "error": self._error}
            ai_service = _StubAI()
    return ai_service

# Initialize FastAPI app
app = FastAPI(
    title=config_loader.get('api.title', 'Paarvai Vision AI API'),
    version=config_loader.get('api.version', '1.0.0'),
    description=config_loader.get('api.description', 'AI-powered face detection and recognition API')
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",  # Alternative localhost
        "http://0.0.0.0:5173",    # Network access
        "http://localhost:3000",  # Alternative React dev server
        "http://127.0.0.1:3000",  # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class VideoProcessingRequest(BaseModel):
    video_path: str
    max_frames: Optional[int] = None
    frame_skip: Optional[int] = 1
    save_annotated_video: Optional[bool] = True
    save_csv: Optional[bool] = True

class PersonCountResponse(BaseModel):
    known_persons: int
    unknown_persons: int
    total_detections: int
    unique_known_persons: int
    unique_unknown_persons: int
    last_updated: str

class FaceAnalysisResponse(BaseModel):
    total_detections: int
    unique_tracks: int
    unique_persons: int
    known_persons: int
    unknown_persons: int
    confidence_stats: Dict[str, float]
    detection_timeline: List[Dict[str, Any]]
    person_summary: List[Dict[str, Any]]
    track_summary: List[Dict[str, Any]]

class CSVAnalysisRequest(BaseModel):
    csv_file_path: str
    limit: Optional[int] = 1000

class PeopleInsideRequest(BaseModel):
    total_count: int

class PeopleInsideResponse(BaseModel):
    status: str
    received: str

class LineCountRequest(BaseModel):
    count: int

class LineCountResponse(BaseModel):
    status: str
    received: str

class LineEventRequest(BaseModel):
    event_type: str
    person_id: int
    timestamp: str
    frame_id: int

class AlertRequest(BaseModel):
    Person: int
    Message: str

class AlertResponse(BaseModel):
    status: str
    received: Dict[str, Any]

class PeopleCrossingRequest(BaseModel):
    In: int
    Out: int
    Current_Inside: int

class PeopleCrossingResponse(BaseModel):
    status: str
    received_count: Dict[str, Any]

# Global state
app.state.kafka_service = kafka_service
app.state.data_stream_service = data_stream_service
app.state.get_ai_service = get_ai_service  # Store the function instead of the service
app.state.people_count = 0
app.state.line_events = []
app.state.alerts = []
app.state.crossing_data = {
    "In": 0,
    "Out": 0,
    "Current_Inside": 0
}
app.state.recent_detections = []
app.state.max_detections = 1000

# Alert deduplication tracking
app.state.alert_deduplication = {
    "blocked_persons": {},  # person_name -> last_alert_time
    "long_standing": {},    # person_name -> last_alert_time
    "normal_alerts": {}     # person_name -> last_alert_time
}
app.state.alert_cooldown = 300  # 5 minutes cooldown between same type alerts for same person

def should_send_alert(person_name: str, alert_type: str, timestamp: float) -> bool:
    """
    Check if an alert should be sent based on deduplication rules.
    
    Args:
        person_name: Name of the person
        alert_type: Type of alert (blocked_person, long_standing, normal)
        timestamp: Current timestamp
        
    Returns:
        bool: True if alert should be sent, False if it should be suppressed
    """
    if not person_name or person_name.strip() == "":
        return True  # Allow alerts for unknown persons
    
    person_name = person_name.strip().lower()
    current_time = timestamp
    
    # Get the appropriate deduplication dictionary
    dedup_dict = app.state.alert_deduplication.get(alert_type, {})
    
    # Check if we've sent this type of alert for this person recently
    if person_name in dedup_dict:
        last_alert_time = dedup_dict[person_name]
        time_since_last_alert = current_time - last_alert_time
        
        if time_since_last_alert < app.state.alert_cooldown:
            logger.info(f"Suppressing duplicate {alert_type} alert for {person_name} "
                       f"(last sent {time_since_last_alert:.1f}s ago)")
            return False
    
    # Update the last alert time for this person and alert type
    dedup_dict[person_name] = current_time
    app.state.alert_deduplication[alert_type] = dedup_dict
    
    return True

# Message handler for Kafka and data stream
def handle_message(message: Dict[str, Any]):
    """Handle messages from Kafka and data stream"""
    try:
        msg_type = message.get("type", "unknown")
        
        if msg_type == "count":
            app.state.people_count = message.get("total_inside", 0)
            logger.info(f"Updated people count: {app.state.people_count}")
            
        elif msg_type == "alert":
            track_id = message.get("track_id")
            timestamp = message.get("timestamp")
            msg_text = message.get("message", "")
            person_name = message.get("person_name", "")
            
            # Determine alert type based on message content and person status
            msg_text_lower = message.get("message", "").lower()
            person_status = message.get("person_status", "").lower()
            
            if "stayed more than" in msg_text_lower or "polygon" in msg_text_lower:
                alert_type = "long_standing"
            elif person_status == "blocked" or "suspected person" in msg_text_lower:
                alert_type = "blocked_person"
            else:
                alert_type = "normal"
            
            # Check if we should send this alert (deduplication)
            if not should_send_alert(person_name, alert_type, timestamp):
                logger.info(f"Suppressed duplicate alert for {person_name} (type: {alert_type})")
                return  # Skip processing this alert
            
            logger.warning(f"ALERT: Track {track_id} at {timestamp}: {msg_text}")
            
            # Use person_label from message if available, otherwise determine from person_status
            person_label = message.get("person_label")
            if not person_label:
                # Fallback: determine person label based on person_status
                person_status = message.get("person_status", "").lower()
                if person_status == "employee":
                    person_label = "employee"
                elif person_status == "active" or person_status == "known":
                    person_label = "customer"
                else:
                    person_label = "customer"  # Default to customer for other statuses
            
            logger.info(f"Person: {person_name}, Status: {message.get('person_status')}, Label: {person_label}")
            
            # Store alert in app state
            alert_data = {
                "track_id": track_id,
                "person_id": message.get("person_id"),
                "person_name": person_name,
                "person_status": message.get("person_status"),
                "person_label": person_label,
                "alert_type": alert_type,
                "recognition_type": message.get("recognition_type"),
                "match_confidence": message.get("match_confidence"),
                "detection_confidence": message.get("detection_confidence"),
                "timestamp": timestamp,
                "message": msg_text
            }
            app.state.alerts.append(alert_data)
            
            # Keep only recent alerts (last 100)
            if len(app.state.alerts) > 100:
                app.state.alerts = app.state.alerts[-100:]
                
        elif msg_type == "line_event":
            event_type = message.get("event_type")
            person_id = message.get("person_id")
            person_name = message.get("person_name")
            y_position = message.get("y_position")
            timestamp = message.get("timestamp")
            logger.info(f"LINE EVENT: {person_name} ({person_id}) {event_type.upper()}ED at y={y_position}")
            
            # Store line event in app state
            event_data = {
                "event_type": event_type,
                "person_id": person_id,
                "person_name": person_name,
                "person_status": message.get("person_status"),
                "recognition_type": message.get("recognition_type"),
                "match_confidence": message.get("match_confidence"),
                "detection_confidence": message.get("detection_confidence"),
                "y_position": y_position,
                "timestamp": timestamp
            }
            app.state.line_events.append(event_data)
            
            # Keep only recent events (last 1000)
            if len(app.state.line_events) > 1000:
                app.state.line_events = app.state.line_events[-1000:]
            
        elif msg_type == "detection":
            detection_data = message.get("data") or {}
            if not detection_data and any(k in message for k in [
                'frame_id','detection_id','track_id','bbox_x1','bbox_y1','bbox_x2','bbox_y2','timestamp']
            ):
                # Support raw detection records without wrapping in data
                detection_data = message
            if detection_data:
                logger.debug(f"Detection data received: {detection_data}")
                # Route through DataStreamService to emit derived alerts/counts
                app.state.data_stream_service.process_detection_record(detection_data, handle_message)
            else:
                logger.debug("Detection message missing data payload")
                
        elif msg_type in ["people_tracking", "person_recognition"]:
            # Handle new message types for people tracking and person recognition
            detection_data = message.get("data") or {}
            if detection_data:
                # Add message type to detection data
                detection_data['type'] = msg_type
                # Store in app state for new endpoints
                app.state.recent_detections.append(detection_data)
                
                # Keep only recent detections (last 1000)
                if len(app.state.recent_detections) > app.state.max_detections:
                    app.state.recent_detections = app.state.recent_detections[-app.state.max_detections:]
                
                logger.info(f"Stored {msg_type} data: track_id={detection_data.get('track_id')}, person_id={detection_data.get('person_id')}")
            else:
                logger.debug(f"{msg_type} message missing data payload")
            
    except Exception as e:
        logger.error(f"Error handling message: {e}")

# Start services
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Start Kafka consumer if enabled
        if config_loader.get('kafka.enabled', True):
            kafka_service.start_consumer(handle_message)
            logger.info("Kafka consumer started")
        
        # Start data stream monitoring based on source
        if config_loader.get('data_stream.source', 'csv') == 'csv':
            data_stream_service.start_csv_monitoring(handle_message)
            logger.info("Data stream (CSV) monitoring started")
        else:
            # For Kafka mode, we need to start the data stream service
            # The Kafka service will call data_stream_service.process_detection_record
            data_stream_service.running = True
            logger.info("Data stream (Kafka) processing enabled")
        
        logger.info("API services initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        kafka_service.stop_consumer()
        data_stream_service.stop_csv_monitoring()
        kafka_service.close()
        logger.info("API services shutdown successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Paarvai Vision AI API",
        "version": config_loader.get('api.version', '1.0.0'),
        "status": "running",
        "endpoints": {
            # Core API v1 endpoints
            "process_video": "POST /api/v1/process-video",
            "person_counts": "GET /api/v1/person-counts",
            "face_stream": "GET /api/v1/face-stream",
            "analyze_csv": "POST /api/v1/analyze-csv",
            "health": "GET /api/v1/health",
            "stats": "GET /api/v1/stats",
            
            # People counting endpoints
            "people_inside_post": "POST /PeopleInside",
            "people_inside_get": "GET /PeopleInside",
            "count_post": "POST /Count",
            "count_get": "GET /Count",
            
            # Line counting endpoints
            "line_count_post": "POST /LineCount",
            "line_count_get": "GET /LineCount",
            "line_event_post": "POST /LineEvent",
            "line_events_get": "GET /LineEvents",
            
            # Alert endpoints
            "alert_post": "POST /alert",
            "alert_get": "GET /alert",
            
            # People crossing endpoints
            "people_crossing_post": "POST /peoplecrossing",
            "people_crossing_get": "GET /peoplecrossing",
            
            # Analysis endpoints
            "analyze_face_stream": "GET /analyze-face-stream",
            "analyze_school_input_3": "GET /analyze-school-input-3",
            "csv_structure": "GET /csv-structure",
            
            # Video endpoints
            "videos": "GET /api/v1/videos",
            "video_file": "GET /api/v1/video/{filename}",
            
            # Demo data endpoints
            "demo_locations": "GET /api/v1/demo/locations",
            "demo_stats": "GET /api/v1/demo/stats",
            
            # Status endpoints
            "status": "GET /status"
        }
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    ai_service = app.state.get_ai_service()
    ai_status = ai_service.get_service_status()
    kafka_connected = kafka_service.is_connected()
    stream_running = data_stream_service.is_running()
    
    return {
        "status": "healthy",
        "service": "Paarvai Vision AI API",
        "version": config_loader.get('api.version', '1.0.0'),
        "services": {
            "ai_service": ai_status,
            "kafka_connected": kafka_connected,
            "data_stream_running": stream_running
        }
    }

@app.post("/api/v1/process-video")
async def process_video(request: VideoProcessingRequest, background_tasks: BackgroundTasks):
    """Process video for face detection and tracking"""
    try:
        # This would integrate with AI service for video processing
        # For now, return a placeholder response
        return {
            "status": "success",
            "message": "Video processing initiated",
            "video_path": request.video_path,
            "max_frames": request.max_frames,
            "frame_skip": request.frame_skip
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/person-counts", response_model=PersonCountResponse)
async def get_person_counts():
    """Get current person count statistics"""
    try:
        # Get data from CSV or Kafka
        csv_data = data_stream_service.get_csv_data(limit=1000)
        
        if not csv_data:
            return PersonCountResponse(
                known_persons=0,
                unknown_persons=0,
                total_detections=0,
                unique_known_persons=0,
                unique_unknown_persons=0,
                last_updated="No data available"
            )
        
        # Calculate statistics
        known_persons = len([d for d in csv_data if d.get('person_id') != 'Unknown'])
        unknown_persons = len([d for d in csv_data if d.get('person_id') == 'Unknown'])
        total_detections = len(csv_data)
        unique_known_persons = len(set([d['person_id'] for d in csv_data if d.get('person_id') != 'Unknown']))
        unique_unknown_persons = len(set([d['track_id'] for d in csv_data if d.get('person_id') == 'Unknown']))
        
        last_updated = csv_data[-1].get('timestamp', 'Unknown') if csv_data else 'Unknown'
        
        return PersonCountResponse(
            known_persons=known_persons,
            unknown_persons=unknown_persons,
            total_detections=total_detections,
            unique_known_persons=unique_known_persons,
            unique_unknown_persons=unique_unknown_persons,
            last_updated=str(last_updated)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/face-stream")
async def get_face_stream(limit: int = 100):
    """Get recent face stream data"""
    try:
        csv_data = data_stream_service.get_csv_data(limit=limit)
        
        return {
            "status": "success",
            "count": len(csv_data),
            "data": csv_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/people-tracking")
async def get_people_tracking(limit: int = 100):
    """Get recent people tracking data (human detection + ReID tracking)"""
    try:
        # Get people tracking data from app state
        people_data = [d for d in app.state.recent_detections if d.get('type') == 'people_tracking']
        
        if not people_data:
            return {"message": "No people tracking data available", "data": []}
        
        # Return last N records
        recent_data = people_data[-limit:] if len(people_data) > limit else people_data
        
        return {
            "status": "success",
            "message": f"Retrieved {len(recent_data)} people tracking records", 
            "data": recent_data,
            "total_tracks": len(set([d.get('track_id') for d in people_data if d.get('track_id') != 'N/A']))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/person-recognition")
async def get_person_recognition(limit: int = 100):
    """Get recent person recognition data (known/unknown face recognition)"""
    try:
        # Get person recognition data from app state
        person_data = [d for d in app.state.recent_detections if d.get('type') == 'person_recognition']
        
        if not person_data:
            return {"message": "No person recognition data available", "data": []}
        
        # Return last N records
        recent_data = person_data[-limit:] if len(person_data) > limit else person_data
        
        # Calculate statistics
        known_persons = [d for d in person_data if d.get('person_id') != 'Unknown']
        unknown_persons = [d for d in person_data if d.get('person_id') == 'Unknown']
        
        return {
            "status": "success",
            "message": f"Retrieved {len(recent_data)} person recognition records", 
            "data": recent_data,
            "statistics": {
                "total_detections": len(person_data),
                "known_persons": len(known_persons),
                "unknown_persons": len(unknown_persons),
                "unique_known": len(set([d.get('person_id') for d in known_persons])),
                "unique_unknown": len(set([d.get('track_id') for d in unknown_persons]))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze-csv", response_model=FaceAnalysisResponse)
async def analyze_csv(request: CSVAnalysisRequest):
    """Analyze CSV file for face detection data"""
    try:
        import pandas as pd
        
        csv_path = Path(request.csv_file_path)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")
        
        df = pd.read_csv(csv_path)
        if len(df) > request.limit:
            df = df.tail(request.limit)
        
        # Calculate statistics
        total_detections = len(df)
        unique_tracks = df['track_id'].nunique() if 'track_id' in df.columns else 0
        unique_persons = unique_tracks
        known_persons = len(df[df['person_id'] != 'Unknown']) if 'person_id' in df.columns else 0
        unknown_persons = total_detections - known_persons
        
        # Confidence statistics
        confidence_stats = {}
        if 'match_confidence' in df.columns:
            confidence_stats = {
                'mean': float(df['match_confidence'].mean()),
                'median': float(df['match_confidence'].median()),
                'min': float(df['match_confidence'].min()),
                'max': float(df['match_confidence'].max()),
                'std': float(df['match_confidence'].std())
            }
        
        # Detection timeline
        detection_timeline = []
        if 'frame_id' in df.columns:
            timeline_df = df.groupby('frame_id').agg({
                'detection_id': 'count',
                'track_id': 'nunique',
                'match_confidence': 'mean'
            }).reset_index()
            detection_timeline = timeline_df.to_dict('records')
        
        # Person summary
        person_summary = []
        if 'person_id' in df.columns:
            person_df = df.groupby('person_id').agg({
                'detection_id': 'count',
                'match_confidence': 'mean',
                'frame_id': 'nunique'
            }).reset_index()
            person_summary = person_df.to_dict('records')
        
        # Track summary
        track_summary = []
        if 'track_id' in df.columns:
            track_df = df.groupby('track_id').agg({
                'detection_id': 'count',
                'match_confidence': 'mean',
                'track_age': 'max',
                'track_hits': 'max'
            }).reset_index()
            track_summary = track_df.to_dict('records')
        
        return FaceAnalysisResponse(
            total_detections=total_detections,
            unique_tracks=unique_tracks,
            unique_persons=unique_persons,
            known_persons=known_persons,
            unknown_persons=unknown_persons,
            confidence_stats=confidence_stats,
            detection_timeline=detection_timeline,
            person_summary=person_summary,
            track_summary=track_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics"""
    try:
        ai_service = app.state.get_ai_service()
        ai_status = ai_service.get_service_status()
        kafka_stats = {
            "connected": kafka_service.is_connected(),
            "consumer_running": kafka_service.running
        }
        stream_stats = data_stream_service.get_statistics()
        
        return {
            "ai_service": ai_status,
            "kafka_service": kafka_stats,
            "data_stream": stream_stats,
            "current_people_count": app.state.people_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for backward compatibility
@app.post("/Count")
async def post_count_legacy(count_obj: Dict[str, int]):
    """Legacy endpoint for backward compatibility"""
    app.state.people_count = count_obj.get('count', 0)
    return {
        "Status": "Success",
        "Received": f"People count = {app.state.people_count}"
    }

@app.get("/Count")
async def get_count_legacy():
    """Legacy endpoint for backward compatibility"""
    return {"Received": f"People Count = {app.state.people_count}"}

# People Inside Box Endpoints
@app.post("/PeopleInside", response_model=PeopleInsideResponse)
async def post_people_inside(request: PeopleInsideRequest):
    """Update people count inside the monitored area"""
    try:
        app.state.people_count = request.total_count
        logger.info(f"Updated people count: {app.state.people_count}")
        
        return PeopleInsideResponse(
            status="success",
            received=f"People count = {app.state.people_count}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/PeopleInside")
async def get_people_inside():
    """Get current people count inside the monitored area"""
    return {"total_inside": app.state.people_count}

# Line Counting Endpoints
@app.post("/LineCount", response_model=LineCountResponse)
async def post_line_count(request: LineCountRequest):
    """Update line count"""
    try:
        logger.info(f"Line count updated: {request.count}")
        
        return LineCountResponse(
            status="success",
            received=f"Line count = {request.count}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/LineCount")
async def get_line_count():
    """Get current line count"""
    return {"count": app.state.people_count}

@app.post("/LineEvent")
async def post_line_event(request: LineEventRequest):
    """Record a line crossing event"""
    try:
        event_data = {
            "event_type": request.event_type,
            "person_id": request.person_id,
            "timestamp": request.timestamp,
            "frame_id": request.frame_id
        }
        app.state.line_events.append(event_data)
        
        # Keep only recent events (last 1000)
        if len(app.state.line_events) > 1000:
            app.state.line_events = app.state.line_events[-1000:]
        
        logger.info(f"Line event recorded: {event_data}")
        
        return {
            "status": "success",
            "message": "Line event recorded",
            "event": event_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/LineEvents")
async def get_line_events():
    """Get all line crossing events"""
    return {
        "events": app.state.line_events,
        "total_events": len(app.state.line_events)
    }

# Alert Endpoints
@app.post("/alert", response_model=AlertResponse)
async def post_alert(request: AlertRequest):
    """Send an alert"""
    try:
        alert_data = {
            "Person": request.Person,
            "Message": request.Message,
            "timestamp": str(datetime.now())
        }
        app.state.alerts.append(alert_data)
        
        # Keep only recent alerts (last 100)
        if len(app.state.alerts) > 100:
            app.state.alerts = app.state.alerts[-100:]
        
        logger.warning(f"ALERT: Person {request.Person} - {request.Message}")
        
        return AlertResponse(
            status="success",
            received=alert_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alert")
async def get_alerts():
    """Get all alerts"""
    return {
        "alerts": app.state.alerts,
        "total_alerts": len(app.state.alerts)
    }

@app.get("/alert/deduplication")
async def get_alert_deduplication():
    """Get alert deduplication settings and status"""
    return {
        "cooldown_seconds": app.state.alert_cooldown,
        "deduplication_status": app.state.alert_deduplication,
        "total_tracked_persons": sum(len(dict_obj) for dict_obj in app.state.alert_deduplication.values())
    }

@app.post("/alert/deduplication/cooldown")
async def update_alert_cooldown(cooldown_seconds: int):
    """Update alert cooldown period"""
    if cooldown_seconds < 0:
        raise HTTPException(status_code=400, detail="Cooldown must be non-negative")
    
    app.state.alert_cooldown = cooldown_seconds
    logger.info(f"Updated alert cooldown to {cooldown_seconds} seconds")
    
    return {
        "status": "success",
        "cooldown_seconds": app.state.alert_cooldown,
        "message": f"Alert cooldown updated to {cooldown_seconds} seconds"
    }

@app.post("/alert/deduplication/clear")
async def clear_alert_deduplication():
    """Clear alert deduplication history"""
    app.state.alert_deduplication = {
        "blocked_persons": {},
        "long_standing": {},
        "normal_alerts": {}
    }
    logger.info("Cleared alert deduplication history")
    
    return {
        "status": "success",
        "message": "Alert deduplication history cleared"
    }

# People Crossing Endpoints
@app.post("/peoplecrossing", response_model=PeopleCrossingResponse)
async def post_people_crossing(request: PeopleCrossingRequest):
    """Update people crossing data"""
    try:
        app.state.crossing_data = {
            "In": request.In,
            "Out": request.Out,
            "Current_Inside": request.Current_Inside
        }
        
        logger.info(f"People crossing updated: {app.state.crossing_data}")
        
        return PeopleCrossingResponse(
            status="success",
            received_count=app.state.crossing_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/peoplecrossing")
async def get_people_crossing():
    """Get people crossing data"""
    return {"Count": app.state.crossing_data}

# Status Endpoint
@app.get("/status")
async def get_status():
    """Get system status"""
    try:
        ai_service = app.state.get_ai_service()
        ai_status = ai_service.get_service_status()
        kafka_connected = kafka_service.is_connected()
        stream_running = data_stream_service.is_running()
        
        return {
            "status": "running",
            "service": "Paarvai Vision AI API",
            "version": config_loader.get('api.version', '1.0.0'),
            "services": {
                "ai_service": ai_status,
                "kafka_connected": kafka_connected,
                "data_stream_running": stream_running
            },
            "counts": {
                "people_inside": app.state.people_count,
                "line_events": len(app.state.line_events),
                "alerts": len(app.state.alerts)
            },
            "crossing_data": app.state.crossing_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CSV Structure Endpoint
@app.get("/csv-structure")
async def get_csv_structure():
    """Get CSV file structure information"""
    try:
        csv_data = data_stream_service.get_csv_data(limit=1)
        
        if not csv_data:
            return {
                "message": "No CSV data available",
                "structure": {}
            }
        
        # Get structure from first record
        sample_record = csv_data[0]
        structure = {
            "columns": list(sample_record.keys()),
            "sample_record": sample_record,
            "total_records": len(csv_data)
        }
        
        return {
            "message": "CSV structure retrieved",
            "structure": structure
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analyze School Input 3 Endpoint
@app.get("/analyze-school-input-3", response_model=FaceAnalysisResponse)
async def analyze_school_input_3(limit: int = 1000):
    """Analyze school input 3 data"""
    try:
        # This would analyze a specific school input file
        # For now, return analysis of current CSV data
        csv_data = data_stream_service.get_csv_data(limit=limit)
        
        if not csv_data:
            return FaceAnalysisResponse(
                total_detections=0,
                unique_tracks=0,
                unique_persons=0,
                known_persons=0,
                unknown_persons=0,
                confidence_stats={},
                detection_timeline=[],
                person_summary=[],
                track_summary=[]
            )
        
        # Calculate statistics
        total_detections = len(csv_data)
        unique_tracks = len(set([d.get('track_id', '') for d in csv_data]))
        unique_persons = len(set([d.get('person_id', '') for d in csv_data]))
        known_persons = len([d for d in csv_data if d.get('person_id', '') != 'Unknown'])
        unknown_persons = total_detections - known_persons
        
        # Confidence statistics
        confidences = [d.get('match_confidence', 0) for d in csv_data if 'match_confidence' in d]
        confidence_stats = {}
        if confidences:
            confidence_stats = {
                'mean': float(sum(confidences) / len(confidences)),
                'min': float(min(confidences)),
                'max': float(max(confidences))
            }
        
        return FaceAnalysisResponse(
            total_detections=total_detections,
            unique_tracks=unique_tracks,
            unique_persons=unique_persons,
            known_persons=known_persons,
            unknown_persons=unknown_persons,
            confidence_stats=confidence_stats,
            detection_timeline=[],
            person_summary=[],
            track_summary=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_default_location(filename: str) -> str:
    """Get default location based on filename"""
    default_locations = ['Chennai', 'Madurai', 'Trichy', 'Coimbatore', 'Salem']
    # Use hash of filename to consistently assign location
    hash_value = hash(filename) % len(default_locations)
    return default_locations[hash_value]

def _get_default_camera(filename: str) -> str:
    """Get default camera name based on filename"""
    default_cameras = ['Entrance Cam 1', 'Entrance Cam 2', 'Exit Cam', 'Parking Cam', 'Main Hall Cam', 'Cash Counter Cam']
    # Use hash of filename to consistently assign camera
    hash_value = hash(filename) % len(default_cameras)
    return default_cameras[hash_value]

@app.get("/api/v1/videos")
async def get_videos():
    """Get list of available videos from AI Layer output directory"""
    try:
        import os
        from datetime import datetime
        
        output_dir = Path("/home/prithiviraj/Vision_AI/AI_Layer/output")
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        videos = []
        
        if output_dir.exists():
            for file_path in output_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                    stat = file_path.stat()
                    videos.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "duration": "Unknown",  # Could be enhanced with video metadata
                        "location": _get_default_location(file_path.name),  # Default location
                        "camera": _get_default_camera(file_path.name)  # Default camera name
                    })
        
        # Sort by creation time (newest first)
        videos.sort(key=lambda x: x['created_time'], reverse=True)
        
        return {
            "status": "success",
            "count": len(videos),
            "videos": videos
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/video/{filename}")
async def get_video(filename: str):
    """Get specific video file"""
    try:
        from fastapi.responses import FileResponse
        
        output_dir = Path("/home/prithiviraj/Vision_AI/AI_Layer/output")
        video_path = output_dir / filename
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/demo/locations")
async def get_demo_locations():
    """Get demo locations and shops data"""
    try:
        locations = {
            "chennai": {
                "shops": ["tambaram", "velachery", "anna nagar", "t. nagar"],
                "display_name": "Chennai"
            },
            "madurai": {
                "shops": ["periyar", "gandhi nagar", "k.k. nagar"],
                "display_name": "Madurai"
            },
            "trichy": {
                "shops": ["central", "golden rock", "srirangam"],
                "display_name": "Trichy"
            },
            "coimbatore": {
                "shops": ["rs puram", "gandhipuram", "saibaba colony"],
                "display_name": "Coimbatore"
            },
            "salem": {
                "shops": ["four roads", "hasthampatti", "suramangalam"],
                "display_name": "Salem"
            }
        }
        
        return {
            "status": "success",
            "locations": locations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/demo/stats")
async def get_demo_stats():
    """Get demo statistics for dashboard"""
    try:
        import random
        from datetime import datetime, timedelta
        
        # Generate some realistic demo data
        stats = {
            "total_locations": 5,
            "total_shops": 20,
            "total_cameras": 60,
            "active_streams": random.randint(45, 60),
            "total_detections_today": random.randint(1500, 3000),
            "known_persons_today": random.randint(800, 1500),
            "unknown_persons_today": random.randint(200, 500),
            "alerts_today": random.randint(10, 50),
            "last_updated": datetime.now().isoformat(),
            "system_uptime": "2 days, 14 hours",
            "storage_used": f"{random.randint(60, 85)}%",
            "network_status": "Connected"
        }
        
        return {
            "status": "success",
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    api_config = config_loader.get_api_config()
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    debug = api_config.get('debug', False)
    
    uvicorn.run(app, host=host, port=port, debug=debug)
