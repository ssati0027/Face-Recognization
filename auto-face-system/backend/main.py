"""
Auto Face Recognition System
- Detects face → Auto-generates ID → Stores immediately
- Next time same person → Recognizes
- Multi-face support → Tracks multiple people
- Never misses anyone → Continuous monitoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import cv2
import numpy as np
import base64
import os
import json
from datetime import datetime
import uuid

app = FastAPI(title="Auto Face Recognition System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIGURATION
# =========================
RECOGNITION_THRESHOLD = 0.45  # Cosine distance threshold
MIN_FACE_QUALITY = 30  # Minimum blur threshold
MIN_FACE_SIZE = 80  # Minimum face size in pixels

# =========================
# LAZY LOAD MODEL
# =========================
face_app = None

def get_face_app():
    global face_app
    if face_app is None:
        print("Loading face recognition model...")
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if os.environ.get('GPU_ENABLED', 'false').lower() == 'true' else -1
        face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("Model loaded successfully")
    return face_app

# =========================
# DATABASE (In-memory for now)
# =========================
enrolled_database = {}
visit_history = []
STORAGE_FILE = "face_database.json"

def load_database():
    global enrolled_database, visit_history
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'r') as f:
                data = json.load(f)
                for person_id, person_data in data.get('enrolled', {}).items():
                    person_data['embedding'] = np.array(person_data['embedding'])
                    enrolled_database[person_id] = person_data
                visit_history = data.get('visits', [])
                print(f"Loaded {len(enrolled_database)} enrolled faces")
        except Exception as e:
            print(f"Error loading database: {e}")

def save_database():
    try:
        data = {
            'enrolled': {},
            'visits': visit_history
        }
        for person_id, person_data in enrolled_database.items():
            data['enrolled'][person_id] = {
                **person_data,
                'embedding': person_data['embedding'].tolist()
            }
        with open(STORAGE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving database: {e}")

load_database()

# =========================
# MODELS
# =========================

class ProcessFrameRequest(BaseModel):
    image: str
    auto_enroll: bool = True  # Auto-enroll new faces

class DetectedFace(BaseModel):
    person_id: str
    is_new: bool
    confidence: Optional[float]
    bbox: List[float]
    face_image: str
    timestamp: str

class ProcessFrameResponse(BaseModel):
    faces_detected: int
    faces: List[DetectedFace]
    message: str

class PersonInfo(BaseModel):
    person_id: str
    first_seen: str
    last_seen: str
    visit_count: int
    face_image: str

class StatsResponse(BaseModel):
    total_enrolled: int
    total_visits: int
    unique_today: int
    recent_visits: List[Dict]

# =========================
# UTILITIES
# =========================

def base64_to_image(base64_string: str):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

def normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def cosine_distance(vec1, vec2):
    from scipy.spatial.distance import cosine
    return cosine(vec1, vec2)

def generate_person_id():
    """Generate unique person ID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    unique = str(uuid.uuid4())[:8]
    return f"PERSON_{timestamp}_{unique}"

def check_face_quality(face, frame):
    """Check if face meets minimum quality standards + eyes/mask detection"""
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False, "Invalid bbox"
        
        face_w = x2 - x1
        face_h = y2 - y1
        
        # Check size
        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
            return False, "Face too small"
        
        # Check blur
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "Empty crop"
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur < MIN_FACE_QUALITY:
            return False, "Face too blurry"
        
        # ========== NEW: Check for landmarks ==========
        if not hasattr(face, 'kps') or face.kps is None or len(face.kps) < 5:
            return False, "No facial landmarks detected"
        
        kps = np.array(face.kps)
        
        # Convert to crop-relative coordinates
        left_eye = kps[0] - [x1, y1]
        right_eye = kps[1] - [x1, y1]
        nose = kps[2] - [x1, y1]
        left_mouth = kps[3] - [x1, y1]
        right_mouth = kps[4] - [x1, y1]
        
        # ========== NEW: EYES CLOSED DETECTION ==========
        eyes_closed = detect_eyes_closed(gray, left_eye, right_eye)
        if eyes_closed:
            return False, "Eyes closed - please open eyes"
        
        # ========== NEW: MASK DETECTION ==========
        mask_detected = detect_mask(gray, nose, left_mouth, right_mouth, face_h)
        if mask_detected:
            return False, "Mask detected - please remove mask"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def detect_eyes_closed(gray, left_eye, right_eye):
    """
    Detect if eyes are closed using variance and brightness analysis
    Closed eyes: low variance (smooth eyelid), darker, less edges
    Open eyes: high variance (iris/pupil/sclera), brighter, clear edges
    """
    try:
        eye_size = 16  # Focused region around eye
        
        def get_eye_patch(eye_pt):
            cx, cy = int(eye_pt[0]), int(eye_pt[1])
            x1 = max(0, cx - eye_size)
            x2 = min(gray.shape[1], cx + eye_size)
            y1 = max(0, cy - eye_size)
            y2 = min(gray.shape[0], cy + eye_size)
            return gray[y1:y2, x1:x2]
        
        left_patch = get_eye_patch(left_eye)
        right_patch = get_eye_patch(right_eye)
        
        if left_patch.size == 0 or right_patch.size == 0:
            return True  # Cannot validate = reject
        
        # Method 1: Variance Check
        # Open eyes have high variance (iris, sclera, pupil contrast)
        # Closed eyes are smooth (just eyelid skin)
        left_var = np.var(left_patch)
        right_var = np.var(right_patch)
        
        VARIANCE_THRESHOLD = 100  # Adjust: lower=stricter, higher=lenient
        
        if left_var < VARIANCE_THRESHOLD or right_var < VARIANCE_THRESHOLD:
            return True  # Eyes closed
        
        # Method 2: Brightness Peak Check
        # Open eyes have bright spots (sclera/reflection)
        left_max = np.max(left_patch)
        right_max = np.max(right_patch)
        left_mean = np.mean(left_patch)
        right_mean = np.mean(right_patch)
        
        left_peak_ratio = left_max / (left_mean + 1e-5)
        right_peak_ratio = right_max / (right_mean + 1e-5)
        
        PEAK_THRESHOLD = 1.3
        
        if left_peak_ratio < PEAK_THRESHOLD or right_peak_ratio < PEAK_THRESHOLD:
            return True  # Eyes closed
        
        # Method 3: Edge Density
        # Open eyes have clear vertical edges (eyelid boundaries)
        left_edges = cv2.Canny(left_patch, 30, 100)
        right_edges = cv2.Canny(right_patch, 30, 100)
        
        left_edge_density = np.sum(left_edges > 0) / (left_patch.size + 1e-5)
        right_edge_density = np.sum(right_edges > 0) / (right_patch.size + 1e-5)
        
        EDGE_THRESHOLD = 0.04
        
        if left_edge_density < EDGE_THRESHOLD or right_edge_density < EDGE_THRESHOLD:
            return True  # Eyes closed
        
        # All checks passed = eyes are open
        return False
        
    except Exception as e:
        print(f"Eyes closed detection error: {e}")
        return True  # On error, reject


def detect_mask(gray, nose, left_mouth, right_mouth, face_h):
    """
    Detect face mask by analyzing lower face region
    Mask creates: uniform texture, brightness drop, smooth surface
    """
    try:
        nose_y = int(nose[1])
        mouth_center_y = int((left_mouth[1] + right_mouth[1]) / 2)
        mouth_center_x = int((left_mouth[0] + right_mouth[0]) / 2)
        
        # Safety bounds
        if nose_y < 0 or mouth_center_y >= gray.shape[0]:
            return False
        
        # Extract mouth region (should show skin texture if no mask)
        mouth_width = int(abs(right_mouth[0] - left_mouth[0]) * 1.5)
        mouth_height = int(abs(mouth_center_y - nose_y) * 0.8)
        
        mx1 = max(0, mouth_center_x - mouth_width // 2)
        mx2 = min(gray.shape[1], mouth_center_x + mouth_width // 2)
        my1 = max(0, mouth_center_y - mouth_height // 2)
        my2 = min(gray.shape[0], mouth_center_y + mouth_height // 2)
        
        mouth_region = gray[my1:my2, mx1:mx2]
        
        if mouth_region.size == 0:
            return False
        
        # Mask indicators:
        # 1. Low variance (uniform fabric texture)
        mouth_variance = np.var(mouth_region)
        
        # 2. Check brightness compared to upper face
        upper_y1 = max(0, nose_y - mouth_height)
        upper_region = gray[upper_y1:nose_y, mx1:mx2]
        
        if upper_region.size == 0:
            return False
        
        mouth_brightness = np.mean(mouth_region)
        upper_brightness = np.mean(upper_region)
        brightness_ratio = mouth_brightness / (upper_brightness + 1e-5)
        
        # Mask detection thresholds
        # Masks typically show: low variance + similar/darker than upper face
        if mouth_variance < 120 and brightness_ratio < 1.15:
            return True  # Mask detected
        
        # Additional check: Edge strength (fabric pattern)
        edges = cv2.Canny(mouth_region, 50, 150)
        edge_density = np.sum(edges > 0) / (mouth_region.size + 1e-5)
        
        # Very smooth surface = likely mask
        if edge_density < 0.02 and mouth_variance < 150:
            return True  # Mask detected
        
        return False  # No mask
        
    except Exception as e:
        print(f"Mask detection error: {e}")
        return False

def extract_face_image(face, frame):
    """Extract face region from frame with padding"""
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        
        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        face_crop = frame[y1:y2, x1:x2]
        
        # Resize to standard size
        if face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (200, 200))
            return image_to_base64(face_crop)
        return None
        
    except Exception as e:
        print(f"Error extracting face: {e}")
        return None

def find_matching_person(embedding):
    """Find matching person in database"""
    best_match_id = None
    best_distance = float('inf')
    
    for person_id, person_data in enrolled_database.items():
        stored_embedding = person_data['embedding']
        distance = cosine_distance(embedding, stored_embedding)
        
        if distance < best_distance:
            best_distance = distance
            best_match_id = person_id
    
    if best_distance < RECOGNITION_THRESHOLD:
        confidence = (1 - best_distance) * 100
        return best_match_id, confidence
    
    return None, 0

def enroll_new_person(embedding, face_image):
    """Enroll a new person with auto-generated ID"""
    person_id = generate_person_id()
    
    enrolled_database[person_id] = {
        'person_id': person_id,
        'embedding': embedding,
        'face_image': face_image,
        'first_seen': datetime.utcnow().isoformat(),
        'last_seen': datetime.utcnow().isoformat(),
        'visit_count': 1
    }
    
    # Log first visit
    visit_history.append({
        'person_id': person_id,
        'timestamp': datetime.utcnow().isoformat(),
        'event': 'first_visit'
    })
    
    save_database()
    return person_id

def update_visit(person_id):
    """Update person's last visit"""
    if person_id in enrolled_database:
        enrolled_database[person_id]['last_seen'] = datetime.utcnow().isoformat()
        enrolled_database[person_id]['visit_count'] += 1
        
        visit_history.append({
            'person_id': person_id,
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'return_visit'
        })
        
        save_database()

# =========================
# ENDPOINTS
# =========================

@app.get("/")
async def root():
    return {
        "service": "Auto Face Recognition System",
        "features": [
            "Auto-enrollment on first detection",
            "Multi-face tracking",
            "Continuous monitoring",
            "No missed persons"
        ],
        "enrolled_count": len(enrolled_database)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "enrolled_count": len(enrolled_database),
        "total_visits": len(visit_history)
    }

@app.post("/api/process-frame", response_model=ProcessFrameResponse)
async def process_frame(request: ProcessFrameRequest):
    """
    Process a single frame:
    - Detect all faces
    - Recognize or auto-enroll each face
    - Track visits
    """
    try:
        model = get_face_app()
        
        img = base64_to_image(request.image)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Detect all faces in frame
        faces = model.get(img)
        
        if len(faces) == 0:
            return ProcessFrameResponse(
                faces_detected=0,
                faces=[],
                message="No faces detected"
            )
        
        detected_faces = []
        
        # Process each detected face
        for face in faces:
            # Check quality
            is_quality_ok, quality_msg = check_face_quality(face, img)
            if not is_quality_ok:
                continue  # Skip low quality faces
            
            # Extract embedding
            embedding = normalize(face.embedding)
            
            # Extract face image
            face_image = extract_face_image(face, img)
            if not face_image:
                continue
            
            # Try to find match
            match_id, confidence = find_matching_person(embedding)
            
            if match_id:
                # Recognized existing person
                update_visit(match_id)
                
                detected_faces.append(DetectedFace(
                    person_id=match_id,
                    is_new=False,
                    confidence=round(confidence, 2),
                    bbox=face.bbox.tolist(),
                    face_image=face_image,
                    timestamp=datetime.utcnow().isoformat()
                ))
            else:
                # New person - auto-enroll if enabled
                if request.auto_enroll:
                    new_id = enroll_new_person(embedding, face_image)
                    
                    detected_faces.append(DetectedFace(
                        person_id=new_id,
                        is_new=True,
                        confidence=None,
                        bbox=face.bbox.tolist(),
                        face_image=face_image,
                        timestamp=datetime.utcnow().isoformat()
                    ))
        
        message = f"Detected {len(detected_faces)} face(s)"
        if any(f.is_new for f in detected_faces):
            new_count = sum(1 for f in detected_faces if f.is_new)
            message += f" ({new_count} new)"
        
        return ProcessFrameResponse(
            faces_detected=len(detected_faces),
            faces=detected_faces,
            message=message
        )
        
    except Exception as e:
        print(f"Process frame error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/people", response_model=List[PersonInfo])
async def get_all_people():
    """Get all enrolled people"""
    return [
        PersonInfo(
            person_id=person_id,
            first_seen=data['first_seen'],
            last_seen=data['last_seen'],
            visit_count=data['visit_count'],
            face_image=data['face_image']
        )
        for person_id, data in enrolled_database.items()
    ]

@app.get("/api/person/{person_id}", response_model=PersonInfo)
async def get_person(person_id: str):
    """Get specific person details"""
    if person_id not in enrolled_database:
        raise HTTPException(status_code=404, detail="Person not found")
    
    data = enrolled_database[person_id]
    return PersonInfo(
        person_id=person_id,
        first_seen=data['first_seen'],
        last_seen=data['last_seen'],
        visit_count=data['visit_count'],
        face_image=data['face_image']
    )

@app.delete("/api/person/{person_id}")
async def delete_person(person_id: str):
    """Delete a person from database"""
    if person_id not in enrolled_database:
        raise HTTPException(status_code=404, detail="Person not found")
    
    del enrolled_database[person_id]
    save_database()
    
    return {
        "success": True,
        "message": f"Deleted {person_id}",
        "remaining_count": len(enrolled_database)
    }

@app.get("/api/stats", response_model=StatsResponse)
async def get_statistics():
    """Get system statistics"""
    today = datetime.utcnow().date().isoformat()
    
    # Count unique visitors today
    unique_today = len(set(
        v['person_id'] for v in visit_history 
        if v['timestamp'].startswith(today)
    ))
    
    # Recent visits (last 20)
    recent = sorted(visit_history, key=lambda x: x['timestamp'], reverse=True)[:20]
    
    return StatsResponse(
        total_enrolled=len(enrolled_database),
        total_visits=len(visit_history),
        unique_today=unique_today,
        recent_visits=recent
    )

@app.delete("/api/reset")
async def reset_database():
    """Reset entire database (use with caution!)"""
    global enrolled_database, visit_history
    enrolled_database = {}
    visit_history = []
    save_database()
    
    return {
        "success": True,
        "message": "Database reset successfully"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
