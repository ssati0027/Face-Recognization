// Auto Face Recognition System - Continuous Monitoring

class AutoFaceRecognitionApp {
    constructor() {
        this.apiUrl = localStorage.getItem('apiUrl') || 'http://localhost:8000';
        this.video = document.getElementById('webcam');
        this.overlay = document.getElementById('overlay');
        this.ctx = this.overlay.getContext('2d');
        
        this.stream = null;
        this.isMonitoring = false;
        this.processInterval = null;
        this.recentDetections = [];
        this.processDelay = 1000; // Process every 1 second
        
        this.init();
    }
    
    init() {
        document.getElementById('apiUrl').value = this.apiUrl;
        document.getElementById('apiUrl').addEventListener('change', (e) => {
            this.apiUrl = e.target.value;
            localStorage.setItem('apiUrl', this.apiUrl);
        });
        
        this.testConnection();
        this.loadPeople();
        
        // Auto-refresh stats every 5 seconds
        setInterval(() => {
            if (!this.isMonitoring) {
                this.loadStats();
            }
        }, 5000);
    }
    
    // =========================
    // API METHODS
    // =========================
    
    async testConnection() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            
            document.getElementById('connectionStatus').innerHTML = 
                `<span style="color: #10b981;">✅ Connected | Enrolled: ${data.enrolled_count} | Visits: ${data.total_visits}</span>`;
            
            this.loadStats();
            this.showToast('API connected', 'success');
        } catch (error) {
            document.getElementById('connectionStatus').innerHTML = 
                `<span style="color: #ef4444;">❌ Cannot connect to ${this.apiUrl}</span>`;
            this.showToast('API connection failed', 'error');
        }
    }
    
    async processFrame() {
        if (!this.isMonitoring || !this.stream) return;
        
        try {
            // Capture frame
            const canvas = document.createElement('canvas');
            canvas.width = this.video.videoWidth;
            canvas.height = this.video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(this.video, 0, 0);
            const base64 = canvas.toDataURL('image/jpeg', 0.85);
            
            // Send to backend
            const autoEnroll = document.getElementById('autoEnroll').checked;
            const response = await fetch(`${this.apiUrl}/api/process-frame`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: base64,
                    auto_enroll: autoEnroll
                })
            });
            
            const data = await response.json();
            
            // Update UI
            this.updateDetectionUI(data);
            
            // Draw bounding boxes
            this.drawDetections(data.faces);
            
        } catch (error) {
            console.error('Process frame error:', error);
        }
    }
    
    async loadStats() {
        try {
            const response = await fetch(`${this.apiUrl}/api/stats`);
            const data = await response.json();
            
            document.getElementById('totalPeople').textContent = data.total_enrolled;
            document.getElementById('totalVisits').textContent = data.total_visits;
            
        } catch (error) {
            console.error('Load stats error:', error);
        }
    }
    
    async loadPeople() {
        try {
            const response = await fetch(`${this.apiUrl}/api/people`);
            const people = await response.json();
            
            const grid = document.getElementById('peopleGrid');
            
            if (people.length === 0) {
                grid.innerHTML = '<p style="text-align: center; color: #999; padding: 40px;">No people enrolled yet</p>';
            } else {
                grid.innerHTML = people.map(person => `
                    <div class="person-card">
                        <img src="data:image/jpeg;base64,${person.face_image}" alt="${person.person_id}">
                        <div class="person-id">${person.person_id}</div>
                        <div class="person-visits">Visits: ${person.visit_count}</div>
                        <div style="font-size: 0.7rem; color: #999; margin-top: 5px;">
                            Last: ${new Date(person.last_seen).toLocaleString()}
                        </div>
                    </div>
                `).join('');
            }
            
            document.getElementById('totalPeople').textContent = people.length;
            
        } catch (error) {
            console.error('Load people error:', error);
        }
    }
    
    // =========================
    // MONITORING CONTROL
    // =========================
    
    async startMonitoring() {
        try {
            // Start camera if not already started
            if (!this.stream) {
                this.stream = await navigator.mediaDevices.getUserMedia({
                    video: { 
                        width: 1280, 
                        height: 720,
                        facingMode: 'user'
                    }
                });
                
                this.video.srcObject = this.stream;
                
                this.video.onloadedmetadata = () => {
                    this.overlay.width = this.video.videoWidth;
                    this.overlay.height = this.video.videoHeight;
                };
            }
            
            // Start continuous processing
            this.isMonitoring = true;
            document.getElementById('monitoringStatus').textContent = 'ACTIVE';
            document.getElementById('monitoringStatus').className = 'status-badge status-active';
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            // Process frames continuously
            this.processInterval = setInterval(() => {
                this.processFrame();
            }, this.processDelay);
            
            this.showToast('Monitoring started - System active', 'success');
            
        } catch (error) {
            console.error('Start monitoring error:', error);
            this.showToast('Failed to start camera', 'error');
        }
    }
    
    stopMonitoring() {
        this.isMonitoring = false;
        
        if (this.processInterval) {
            clearInterval(this.processInterval);
            this.processInterval = null;
        }
        
        document.getElementById('monitoringStatus').textContent = 'STOPPED';
        document.getElementById('monitoringStatus').className = 'status-badge status-inactive';
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        
        this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
        
        this.showToast('Monitoring stopped', 'info');
    }
    
    // =========================
    // UI UPDATES
    // =========================
    
    updateDetectionUI(data) {
        // Update detection info
        const info = document.getElementById('detectionInfo');
        if (data.faces_detected === 0) {
            info.textContent = '👀 Scanning... No faces';
            info.style.background = 'rgba(239, 68, 68, 0.8)';
        } else {
            info.textContent = `✅ ${data.faces_detected} face(s) detected`;
            info.style.background = 'rgba(16, 185, 129, 0.8)';
        }
        
        // Add to recent detections
        if (data.faces && data.faces.length > 0) {
            data.faces.forEach(face => {
                // Check if already in recent (within 5 seconds)
                const existing = this.recentDetections.find(d => 
                    d.person_id === face.person_id && 
                    Date.now() - d.timestamp < 5000
                );
                
                if (!existing) {
                    this.recentDetections.unshift({
                        ...face,
                        timestamp: Date.now()
                    });
                    
                    // Keep only last 10
                    this.recentDetections = this.recentDetections.slice(0, 10);
                    
                    // Show toast for new enrollments
                    if (face.is_new) {
                        this.showToast(`🆕 New person enrolled: ${face.person_id}`, 'success');
                        this.loadPeople(); // Refresh people grid
                    }
                }
            });
            
            this.updateRecentFaces();
            this.loadStats();
        }
    }
    
    updateRecentFaces() {
        const container = document.getElementById('recentFaces');
        
        if (this.recentDetections.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: #999; padding: 20px;">No faces detected yet</p>';
        } else {
            container.innerHTML = this.recentDetections.map(face => `
                <div class="face-item ${face.is_new ? 'new' : ''}">
                    <img src="data:image/jpeg;base64,${face.face_image}" alt="${face.person_id}">
                    <div class="face-info">
                        <div class="face-id">
                            ${face.person_id}
                            ${face.is_new ? '<span class="face-new-badge">NEW</span>' : ''}
                        </div>
                        <div class="face-status">
                            ${face.is_new ? 'First time detected' : `Confidence: ${face.confidence}%`}
                        </div>
                        <div class="face-status">
                            ${new Date(face.timestamp).toLocaleTimeString()}
                        </div>
                    </div>
                </div>
            `).join('');
        }
    }
    
    drawDetections(faces) {
        this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
        
        if (!faces || faces.length === 0) return;
        
        // Scale factors
        const scaleX = this.overlay.width / this.video.videoWidth;
        const scaleY = this.overlay.height / this.video.videoHeight;
        
        faces.forEach(face => {
            const [x1, y1, x2, y2] = face.bbox;
            
            // Scale coordinates
            const sx1 = x1 * scaleX;
            const sy1 = y1 * scaleY;
            const w = (x2 - x1) * scaleX;
            const h = (y2 - y1) * scaleY;
            
            // Draw bounding box
            this.ctx.strokeStyle = face.is_new ? '#10b981' : '#667eea';
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(sx1, sy1, w, h);
            
            // Draw label background
            const label = face.is_new ? 'NEW' : `${face.confidence}%`;
            this.ctx.font = 'bold 16px Arial';
            const labelWidth = this.ctx.measureText(label).width;
            
            this.ctx.fillStyle = face.is_new ? '#10b981' : '#667eea';
            this.ctx.fillRect(sx1, sy1 - 30, labelWidth + 20, 30);
            
            // Draw label text
            this.ctx.fillStyle = 'white';
            this.ctx.fillText(label, sx1 + 10, sy1 - 10);
        });
    }
    
    // =========================
    // UTILITIES
    // =========================
    
    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type} show`;
        
        setTimeout(() => {
            toast.className = `toast ${type}`;
        }, 3000);
    }
}

// Initialize app
let app;
window.addEventListener('DOMContentLoaded', () => {
    app = new AutoFaceRecognitionApp();
});
