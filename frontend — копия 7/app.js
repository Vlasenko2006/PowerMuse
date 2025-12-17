// MusicLab - Main Application JavaScript
// Handles audio upload, waveform visualization, pattern selection, and generation

class MusicLab {
    constructor() {
        this.tracks = {
            1: { audio: null, buffer: null, source: null, duration: 0, isPlaying: false, file: null },
            2: { audio: null, buffer: null, source: null, duration: 0, isPlaying: false, file: null }
        };
        
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.resultAudio = null;
        this.resultIsPlaying = false;
        
        this.init();
    }
    
    init() {
        this.setupFileUploads();
        this.setupDragAndDrop();
        this.checkGenerateReady();
    }
    
    // ========================================
    // File Upload Setup
    // ========================================
    setupFileUploads() {
        [1, 2].forEach(trackNum => {
            const fileInput = document.getElementById(`file-input-${trackNum}`);
            const clearBtn = document.getElementById(`clear-btn-${trackNum}`);
            
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e, trackNum));
            clearBtn.addEventListener('click', () => this.clearTrack(trackNum));
        });
    }
    
    setupDragAndDrop() {
        [1, 2].forEach(trackNum => {
            const uploadArea = document.getElementById(`upload-area-${trackNum}`);
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, this.preventDefaults, false);
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => {
                    uploadArea.classList.add('dragover');
                });
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => {
                    uploadArea.classList.remove('dragover');
                });
            });
            
            uploadArea.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.loadAudioFile(files[0], trackNum);
                }
            });
        });
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // ========================================
    // File Handling
    // ========================================
    handleFileSelect(event, trackNum) {
        const file = event.target.files[0];
        if (file) {
            this.loadAudioFile(file, trackNum);
        }
    }
    
    async loadAudioFile(file, trackNum) {
        // Check file type
        if (!file.type.startsWith('audio/')) {
            alert('Please upload an audio file (MP3, WAV, OGG, etc.)');
            return;
        }
        
        // Update status
        this.updateStatus(trackNum, 'uploading', 'Loading...');
        
        try {
            // Read file
            const arrayBuffer = await file.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            // Check duration (minimum 16 seconds)
            if (audioBuffer.duration < 16) {
                alert('Audio file must be at least 16 seconds long');
                this.updateStatus(trackNum, 'error', 'File too short');
                return;
            }
            
            // Store track data
            this.tracks[trackNum].buffer = audioBuffer;
            this.tracks[trackNum].duration = audioBuffer.duration;
            this.tracks[trackNum].file = file;
            
            // Show player
            document.getElementById(`upload-area-${trackNum}`).style.display = 'none';
            document.getElementById(`player-${trackNum}`).style.display = 'block';
            
            // Draw waveform
            this.drawWaveform(trackNum, audioBuffer);
            
            // Setup player controls
            this.setupPlayerControls(trackNum);
            
            // Setup selection controls
            this.setupSelectionControls(trackNum);
            
            // Update status
            this.updateStatus(trackNum, 'ready', `${file.name} (${this.formatTime(audioBuffer.duration)})`);
            
            // Check if can generate
            this.checkGenerateReady();
            
        } catch (error) {
            console.error('Error loading audio:', error);
            alert('Error loading audio file. Please try another file.');
            this.updateStatus(trackNum, 'error', 'Load failed');
        }
    }
    
    // ========================================
    // Waveform Visualization
    // ========================================
    drawWaveform(trackNum, audioBuffer) {
        const canvas = document.getElementById(`waveform-${trackNum}`);
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = canvas.offsetHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        
        // Get audio data
        const data = audioBuffer.getChannelData(0);
        const step = Math.ceil(data.length / width);
        const amp = height / 2;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw waveform
        ctx.beginPath();
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        
        for (let i = 0; i < width; i++) {
            const min = Math.min(...data.slice(i * step, (i + 1) * step));
            const max = Math.max(...data.slice(i * step, (i + 1) * step));
            
            if (i === 0) {
                ctx.moveTo(i, (1 + min) * amp);
            }
            ctx.lineTo(i, (1 + max) * amp);
            ctx.lineTo(i, (1 + min) * amp);
        }
        
        ctx.stroke();
        
        // Draw time markers
        this.drawTimeMarkers(trackNum, audioBuffer.duration);
        
        // Initialize selection overlay
        this.updateSelectionOverlay(trackNum, 0, 16);
    }
    
    drawTimeMarkers(trackNum, duration) {
        const container = document.getElementById(`markers-${trackNum}`);
        container.innerHTML = '';
        
        const numMarkers = 5;
        for (let i = 0; i <= numMarkers; i++) {
            const time = (duration / numMarkers) * i;
            const marker = document.createElement('span');
            marker.textContent = this.formatTime(time);
            marker.style.flex = '1';
            marker.style.textAlign = i === 0 ? 'left' : i === numMarkers ? 'right' : 'center';
            container.appendChild(marker);
        }
    }
    
    updateSelectionOverlay(trackNum, startTime, endTime) {
        const overlay = document.getElementById(`selection-${trackNum}`);
        const duration = this.tracks[trackNum].duration;
        
        const startPercent = (startTime / duration) * 100;
        const widthPercent = ((endTime - startTime) / duration) * 100;
        
        overlay.style.left = `${startPercent}%`;
        overlay.style.width = `${widthPercent}%`;
    }
    
    // ========================================
    // Player Controls
    // ========================================
    setupPlayerControls(trackNum) {
        const playBtn = document.getElementById(`play-btn-${trackNum}`);
        const volumeSlider = document.getElementById(`volume-${trackNum}`);
        const previewBtn = document.getElementById(`preview-btn-${trackNum}`);
        
        playBtn.addEventListener('click', () => this.togglePlay(trackNum));
        volumeSlider.addEventListener('input', (e) => this.setVolume(trackNum, e.target.value));
        previewBtn.addEventListener('click', () => this.previewSelection(trackNum));
        
        // Initialize duration display
        document.getElementById(`duration-${trackNum}`).textContent = 
            this.formatTime(this.tracks[trackNum].duration);
    }
    
    togglePlay(trackNum) {
        if (this.tracks[trackNum].isPlaying) {
            this.stopTrack(trackNum);
        } else {
            this.playTrack(trackNum);
        }
    }
    
    playTrack(trackNum, startTime = 0, endTime = null) {
        const track = this.tracks[trackNum];
        
        // Stop if already playing
        if (track.source) {
            track.source.stop();
        }
        
        // Create source
        track.source = this.audioContext.createBufferSource();
        track.source.buffer = track.buffer;
        
        // Create gain node for volume
        const gainNode = this.audioContext.createGain();
        const volume = document.getElementById(`volume-${trackNum}`).value / 100;
        gainNode.gain.value = volume;
        
        // Connect
        track.source.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        // Play
        const duration = endTime ? endTime - startTime : track.duration - startTime;
        track.source.start(0, startTime, duration);
        track.isPlaying = true;
        
        // Update UI
        this.updatePlayButton(trackNum, true);
        
        // Update time display
        const startTimestamp = Date.now();
        const updateTime = () => {
            if (track.isPlaying) {
                const elapsed = (Date.now() - startTimestamp) / 1000;
                document.getElementById(`current-time-${trackNum}`).textContent = 
                    this.formatTime(startTime + Math.min(elapsed, duration));
                requestAnimationFrame(updateTime);
            }
        };
        updateTime();
        
        // Handle end
        track.source.onended = () => {
            if (track.isPlaying) {
                this.stopTrack(trackNum);
            }
        };
    }
    
    stopTrack(trackNum) {
        const track = this.tracks[trackNum];
        
        if (track.source) {
            track.source.stop();
            track.source = null;
        }
        
        track.isPlaying = false;
        this.updatePlayButton(trackNum, false);
        document.getElementById(`current-time-${trackNum}`).textContent = '0:00';
    }
    
    updatePlayButton(trackNum, isPlaying) {
        const playBtn = document.getElementById(`play-btn-${trackNum}`);
        const playIcon = playBtn.querySelector('.icon-play');
        const pauseIcon = playBtn.querySelector('.icon-pause');
        
        if (isPlaying) {
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'block';
        } else {
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
        }
    }
    
    setVolume(trackNum, value) {
        // Volume will be applied on next play
    }
    
    previewSelection(trackNum) {
        const startTime = parseFloat(document.getElementById(`start-time-${trackNum}`).value);
        const endTime = parseFloat(document.getElementById(`end-time-${trackNum}`).value);
        
        this.stopTrack(trackNum);
        this.playTrack(trackNum, startTime, endTime);
    }
    
    // ========================================
    // Selection Controls
    // ========================================
    setupSelectionControls(trackNum) {
        const slider = document.getElementById(`slider-${trackNum}`);
        const startInput = document.getElementById(`start-time-${trackNum}`);
        const endInput = document.getElementById(`end-time-${trackNum}`);
        const durationDisplay = document.getElementById(`selection-duration-${trackNum}`);
        
        const duration = this.tracks[trackNum].duration;
        
        // Set slider max
        slider.max = duration - 16;
        slider.value = 0;
        
        // Set input constraints
        startInput.max = duration - 16;
        endInput.max = duration;
        endInput.min = 16;
        
        // Slider change
        slider.addEventListener('input', (e) => {
            const startTime = parseFloat(e.target.value);
            const endTime = startTime + 16;
            
            startInput.value = startTime.toFixed(1);
            endInput.value = endTime.toFixed(1);
            
            this.updateSelectionOverlay(trackNum, startTime, endTime);
            this.updateSelectionDuration(trackNum);
        });
        
        // Input changes
        startInput.addEventListener('change', () => {
            let startTime = parseFloat(startInput.value);
            const maxStart = duration - 16;
            
            startTime = Math.max(0, Math.min(startTime, maxStart));
            startInput.value = startTime.toFixed(1);
            
            const endTime = startTime + 16;
            endInput.value = endTime.toFixed(1);
            
            slider.value = startTime;
            this.updateSelectionOverlay(trackNum, startTime, endTime);
            this.updateSelectionDuration(trackNum);
        });
        
        endInput.addEventListener('change', () => {
            let endTime = parseFloat(endInput.value);
            endTime = Math.max(16, Math.min(endTime, duration));
            
            const startTime = endTime - 16;
            startInput.value = startTime.toFixed(1);
            endInput.value = endTime.toFixed(1);
            
            slider.value = startTime;
            this.updateSelectionOverlay(trackNum, startTime, endTime);
            this.updateSelectionDuration(trackNum);
        });
    }
    
    updateSelectionDuration(trackNum) {
        const startTime = parseFloat(document.getElementById(`start-time-${trackNum}`).value);
        const endTime = parseFloat(document.getElementById(`end-time-${trackNum}`).value);
        const duration = endTime - startTime;
        
        document.getElementById(`selection-duration-${trackNum}`).textContent = 
            `${duration.toFixed(1)} sec`;
    }
    
    // ========================================
    // Status Updates
    // ========================================
    updateStatus(trackNum, state, text) {
        const statusElement = document.getElementById(`status-track${trackNum}`);
        const indicator = statusElement.querySelector('.status-indicator');
        const statusText = statusElement.querySelector('.status-text');
        
        indicator.className = 'status-indicator';
        if (state === 'uploading') {
            indicator.classList.add('uploading');
        } else if (state === 'ready') {
            indicator.classList.add('ready');
        }
        
        statusText.textContent = text;
    }
    
    // ========================================
    // Clear Track
    // ========================================
    clearTrack(trackNum) {
        // Stop playback
        this.stopTrack(trackNum);
        
        // Clear data
        this.tracks[trackNum] = {
            audio: null,
            buffer: null,
            source: null,
            duration: 0,
            isPlaying: false,
            file: null
        };
        
        // Reset UI
        document.getElementById(`upload-area-${trackNum}`).style.display = 'block';
        document.getElementById(`player-${trackNum}`).style.display = 'none';
        document.getElementById(`file-input-${trackNum}`).value = '';
        
        this.updateStatus(trackNum, '', 'No file uploaded');
        this.checkGenerateReady();
    }
    
    // ========================================
    // Generate Music
    // ========================================
    checkGenerateReady() {
        const track1Ready = this.tracks[1].buffer !== null;
        const track2Ready = this.tracks[2].buffer !== null;
        
        console.log('Check Generate Ready:', { track1Ready, track2Ready });
        
        const generateSection = document.getElementById('generate-section');
        
        if (track1Ready && track2Ready) {
            console.log('Both tracks ready - showing Generate button');
            generateSection.style.display = 'block';
            this.setupGenerateButton();
        } else {
            generateSection.style.display = 'none';
        }
    }
    
    setupGenerateButton() {
        const generateBtn = document.getElementById('generate-btn');
        
        // Remove old listeners
        const newBtn = generateBtn.cloneNode(true);
        generateBtn.parentNode.replaceChild(newBtn, generateBtn);
        
        newBtn.addEventListener('click', () => this.generateMusic());
    }
    
    async generateMusic() {
        // Get selections
        const track1Start = parseFloat(document.getElementById('start-time-1').value);
        const track1End = parseFloat(document.getElementById('end-time-1').value);
        const track2Start = parseFloat(document.getElementById('start-time-2').value);
        const track2End = parseFloat(document.getElementById('end-time-2').value);
        
        // Show loading
        document.getElementById('generate-section').style.display = 'none';
        document.getElementById('loading-section').style.display = 'block';
        
        try {
            // Upload files and start generation
            await this.callGenerationAPI(track1Start, track1End, track2Start, track2End);
            
            // Show result
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('result-section').style.display = 'block';
            
            this.setupResultPlayer();
        } catch (error) {
            console.error('Generation failed:', error);
            alert('Failed to generate music. Please try again.');
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('generate-section').style.display = 'block';
        }
    }
    
    async callGenerationAPI(track1Start, track1End, track2Start, track2End) {
        const API_URL = 'http://localhost:8001';
        
        // Create FormData with files and parameters
        const formData = new FormData();
        formData.append('track1', this.tracks[1].file);
        formData.append('track2', this.tracks[2].file);
        formData.append('start_time_1', track1Start.toString());
        formData.append('end_time_1', track1End.toString());
        formData.append('start_time_2', track2Start.toString());
        formData.append('end_time_2', track2End.toString());
        
        // Submit generation request
        const response = await fetch(`${API_URL}/api/generate`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        this.jobId = data.job_id;
        
        // Poll for status
        await this.pollGenerationStatus();
    }
    
    async pollGenerationStatus() {
        const API_URL = 'http://localhost:8001';
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        while (true) {
            const response = await fetch(`${API_URL}/api/status/${this.jobId}`);
            const status = await response.json();
            
            // Update progress bar
            progressFill.style.width = `${status.progress}%`;
            progressText.textContent = `${status.progress}%`;
            
            if (status.status === 'completed') {
                this.outputPath = status.output_path;
                break;
            } else if (status.status === 'failed') {
                throw new Error(status.message || 'Generation failed');
            }
            
            // Wait 1 second before next poll
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    
    // ========================================
    // Result Player
    // ========================================
    async setupResultPlayer() {
        const playBtn = document.getElementById('play-btn-result');
        const downloadBtn = document.getElementById('download-btn');
        const newGenBtn = document.getElementById('new-generation-btn');
        
        playBtn.addEventListener('click', () => this.toggleResultPlay());
        downloadBtn.addEventListener('click', () => this.downloadResult());
        newGenBtn.addEventListener('click', () => this.resetForNewGeneration());
        
        // Load generated audio from API
        await this.loadGeneratedAudio();
        
        document.getElementById('duration-result').textContent = 
            this.formatTime(this.resultAudio.duration);
        
        this.drawResultWaveform();
    }
    
    async loadGeneratedAudio() {
        const API_URL = 'http://localhost:8001';
        
        try {
            // Fetch generated audio file
            const response = await fetch(`${API_URL}/api/download/${this.jobId}`);
            
            if (!response.ok) {
                throw new Error('Failed to download generated audio');
            }
            
            // Convert to audio buffer
            const arrayBuffer = await response.arrayBuffer();
            this.resultAudio = await this.audioContext.decodeAudioData(arrayBuffer);
            
            // Store blob for download
            this.resultBlob = new Blob([arrayBuffer], { type: 'audio/wav' });
            
        } catch (error) {
            console.error('Error loading generated audio:', error);
            // Fallback to track 1 for demo
            this.resultAudio = this.tracks[1].buffer;
        }
    }
    
    drawResultWaveform() {
        const canvas = document.getElementById('waveform-result');
        const ctx = canvas.getContext('2d');
        
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = canvas.offsetHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        
        const data = this.resultAudio.getChannelData(0);
        const step = Math.ceil(data.length / width);
        const amp = height / 2;
        
        ctx.clearRect(0, 0, width, height);
        ctx.beginPath();
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        
        for (let i = 0; i < width; i++) {
            const min = Math.min(...data.slice(i * step, (i + 1) * step));
            const max = Math.max(...data.slice(i * step, (i + 1) * step));
            
            if (i === 0) {
                ctx.moveTo(i, (1 + min) * amp);
            }
            ctx.lineTo(i, (1 + max) * amp);
            ctx.lineTo(i, (1 + min) * amp);
        }
        
        ctx.stroke();
    }
    
    toggleResultPlay() {
        if (this.resultIsPlaying) {
            this.stopResultPlay();
        } else {
            this.playResultAudio();
        }
    }
    
    playResultAudio() {
        const source = this.audioContext.createBufferSource();
        source.buffer = this.resultAudio;
        
        const gainNode = this.audioContext.createGain();
        gainNode.gain.value = document.getElementById('volume-result').value / 100;
        
        source.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        source.start(0);
        this.resultIsPlaying = true;
        this.resultSource = source;
        
        const playBtn = document.getElementById('play-btn-result');
        playBtn.querySelector('.icon-play').style.display = 'none';
        playBtn.querySelector('.icon-pause').style.display = 'block';
        
        source.onended = () => {
            this.stopResultPlay();
        };
        
        // Update time
        const startTime = Date.now();
        const updateTime = () => {
            if (this.resultIsPlaying) {
                const elapsed = (Date.now() - startTime) / 1000;
                document.getElementById('current-time-result').textContent = 
                    this.formatTime(Math.min(elapsed, this.resultAudio.duration));
                requestAnimationFrame(updateTime);
            }
        };
        updateTime();
    }
    
    stopResultPlay() {
        if (this.resultSource) {
            this.resultSource.stop();
            this.resultSource = null;
        }
        
        this.resultIsPlaying = false;
        
        const playBtn = document.getElementById('play-btn-result');
        playBtn.querySelector('.icon-play').style.display = 'block';
        playBtn.querySelector('.icon-pause').style.display = 'none';
        
        document.getElementById('current-time-result').textContent = '0:00';
    }
    
    downloadResult() {
        if (!this.resultBlob) {
            alert('No audio available to download');
            return;
        }
        
        // Create download link
        const url = URL.createObjectURL(this.resultBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `musiclab_generated_${Date.now()}.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    resetForNewGeneration() {
        this.stopResultPlay();
        document.getElementById('result-section').style.display = 'none';
        document.getElementById('generate-section').style.display = 'block';
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
    }
    
    // ========================================
    // Utility Functions
    // ========================================
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.musicLab = new MusicLab();
});
