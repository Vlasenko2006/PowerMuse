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
        this.resultBlob = null;
        this.jobId = null;
        
        // Try to restore last job ID from localStorage
        const savedJobId = localStorage.getItem('lastJobId');
        if (savedJobId) {
            console.log(`[DEBUG] Found saved job ID: ${savedJobId}`);
            this.jobId = savedJobId;
        }
        
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
        
        // Find max amplitude for normalization
        let maxAmp = 0;
        for (let i = 0; i < data.length; i++) {
            maxAmp = Math.max(maxAmp, Math.abs(data[i]));
        }
        const scale = maxAmp > 0 ? 0.95 / maxAmp : 1; // Scale to 95% of available height
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw waveform
        ctx.beginPath();
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        
        for (let i = 0; i < width; i++) {
            const min = Math.min(...data.slice(i * step, (i + 1) * step)) * scale;
            const max = Math.max(...data.slice(i * step, (i + 1) * step)) * scale;
            
            if (i === 0) {
                ctx.moveTo(i, (1 + min) * amp);
            }
            ctx.lineTo(i, (1 + max) * amp);
            ctx.lineTo(i, (1 + min) * amp);
        }
        
        ctx.stroke();
        
        // Draw time markers
        this.drawTimeMarkers(trackNum, audioBuffer.duration);
        
        // Restore current selection overlay (don't reset to 0-16)
        const startTime = parseFloat(document.getElementById(`start-time-${trackNum}`).value) || 0;
        const endTime = parseFloat(document.getElementById(`end-time-${trackNum}`).value) || 16;
        this.updateSelectionOverlay(trackNum, startTime, endTime);
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
        
        console.log(`[OVERLAY] Track ${trackNum} - Setting overlay: left=${startPercent.toFixed(2)}%, width=${widthPercent.toFixed(2)}% (start=${startTime.toFixed(2)}s, end=${endTime.toFixed(2)}s, duration=${duration.toFixed(2)}s)`);
        
        overlay.style.left = `${startPercent}%`;
        overlay.style.width = `${widthPercent}%`;
    }

    updateWaveformProgress(trackNum, progress) {
        const canvas = document.getElementById(`waveform-${trackNum}`);
        if (!canvas) return;

        const track = this.tracks[trackNum];
        if (!track.buffer) return;

        const ctx = canvas.getContext('2d');
        
        // Set canvas size (same as drawWaveform)
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = canvas.offsetHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        
        // Get audio data
        const data = track.buffer.getChannelData(0);
        const step = Math.ceil(data.length / width);
        const amp = height / 2;
        
        // Find max amplitude for normalization
        let maxAmp = 0;
        for (let i = 0; i < data.length; i++) {
            maxAmp = Math.max(maxAmp, Math.abs(data[i]));
        }
        const scale = maxAmp > 0 ? 0.95 / maxAmp : 1; // Scale to 95% of available height
        
        // Get time window info
        const startTime = track.startTimeOffset || 0;
        const duration = track.duration;
        const startPercent = startTime / duration;
        const playDuration = track.playDuration || duration;
        const endPercent = (startTime + playDuration) / duration;
        
        // Calculate pixel positions for time window
        const windowStartX = Math.floor(width * startPercent);
        const windowWidth = Math.floor(width * (endPercent - startPercent));
        const progressInWindow = Math.floor(windowWidth * progress);
        const progressAbsoluteX = windowStartX + progressInWindow;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw base waveform
        ctx.beginPath();
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        
        for (let i = 0; i < width; i++) {
            const min = Math.min(...data.slice(i * step, (i + 1) * step)) * scale;
            const max = Math.max(...data.slice(i * step, (i + 1) * step)) * scale;
            
            if (i === 0) {
                ctx.moveTo(i, (1 + min) * amp);
            }
            ctx.lineTo(i, (1 + max) * amp);
            ctx.lineTo(i, (1 + min) * amp);
        }
        
        ctx.stroke();
        
        // Draw progress overlay (from window start to current position)
        console.log(`[PROGRESS] Track ${trackNum} - progress=${(progress * 100).toFixed(1)}%, startTime=${startTime.toFixed(2)}s, playDuration=${playDuration.toFixed(2)}s, windowStartX=${windowStartX}, progressAbsoluteX=${progressAbsoluteX}`);
        
        if (progressInWindow > 0) {
            ctx.beginPath();
            ctx.strokeStyle = '#00d4aa';
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.6;
            
            for (let i = windowStartX; i < progressAbsoluteX; i++) {
                const min = Math.min(...data.slice(i * step, (i + 1) * step)) * scale;
                const max = Math.max(...data.slice(i * step, (i + 1) * step)) * scale;
                
                if (i === windowStartX) {
                    ctx.moveTo(i, (1 + min) * amp);
                }
                ctx.lineTo(i, (1 + max) * amp);
                ctx.lineTo(i, (1 + min) * amp);
            }
            
            ctx.stroke();
            ctx.globalAlpha = 1.0;
            
            // Draw progress line
            ctx.strokeStyle = '#00d4aa';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(progressAbsoluteX, 0);
            ctx.lineTo(progressAbsoluteX, height);
            ctx.stroke();
        }
        
        // Redraw selection overlay
        const selectionStart = parseFloat(document.getElementById(`start-time-${trackNum}`).value);
        const selectionEnd = parseFloat(document.getElementById(`end-time-${trackNum}`).value);
        console.log(`[REDRAW] Track ${trackNum} - Redrawing overlay after progress update: selectionStart=${selectionStart.toFixed(2)}s, selectionEnd=${selectionEnd.toFixed(2)}s`);
        this.updateSelectionOverlay(trackNum, selectionStart, selectionEnd);
    }
    
    // ========================================
    // Player Controls
    // ========================================
    setupPlayerControls(trackNum) {
        // Only setup listeners once per track to prevent duplicates
        if (!this.tracks[trackNum].listenersSetup) {
            const playBtn = document.getElementById(`play-btn-${trackNum}`);
            const volumeSlider = document.getElementById(`volume-${trackNum}`);
            const previewBtn = document.getElementById(`preview-btn-${trackNum}`);
            
            // Clone and replace to remove any old listeners
            const playBtnNew = playBtn.cloneNode(true);
            playBtn.parentNode.replaceChild(playBtnNew, playBtn);
            
            const volumeSliderNew = volumeSlider.cloneNode(true);
            volumeSlider.parentNode.replaceChild(volumeSliderNew, volumeSlider);
            
            const previewBtnNew = previewBtn.cloneNode(true);
            previewBtn.parentNode.replaceChild(previewBtnNew, previewBtn);
            
            // Add event listeners to new nodes
            document.getElementById(`play-btn-${trackNum}`).addEventListener('click', () => this.togglePlay(trackNum));
            document.getElementById(`volume-${trackNum}`).addEventListener('input', (e) => this.setVolume(trackNum, e.target.value));
            document.getElementById(`preview-btn-${trackNum}`).addEventListener('click', () => this.previewSelection(trackNum));
            
            this.tracks[trackNum].listenersSetup = true;
            console.log(`[DEBUG] Track ${trackNum} player controls setup`);
        }
        
        // Initialize duration display
        document.getElementById(`duration-${trackNum}`).textContent = 
            this.formatTime(this.tracks[trackNum].duration);
    }
    
    togglePlay(trackNum) {
        if (this.tracks[trackNum].isPlaying) {
            this.stopTrack(trackNum);
        } else {
            // Play full track from beginning
            this.playTrack(trackNum, 0, null);
        }
    }
    
    async playTrack(trackNum, startTime = 0, endTime = null) {
        // Resume AudioContext if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        
        const track = this.tracks[trackNum];
        
        // Stop if already playing
        if (track.source) {
            try {
                track.source.stop();
                track.source.disconnect();
            } catch (e) {}
        }
        
        // Calculate duration
        const duration = endTime ? endTime - startTime : track.duration - startTime;
        console.log(`[PLAY] Track ${trackNum} - startTime=${startTime.toFixed(2)}s, endTime=${endTime ? endTime.toFixed(2) : 'null'}, duration=${duration.toFixed(2)}s`);
        
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
        track.source.start(0, startTime, duration);
        track.isPlaying = true;
        
        // Update UI
        this.updatePlayButton(trackNum, true);
        
        // Update time display and waveform progress
        track.startTimestamp = Date.now();
        track.startTimeOffset = startTime;
        track.playDuration = duration;
        
        const updateTime = () => {
            if (!track.isPlaying) return;
            
            const elapsed = (Date.now() - track.startTimestamp) / 1000;
            const currentTime = track.startTimeOffset + Math.min(elapsed, track.playDuration);
            
            const currentTimeEl = document.getElementById(`current-time-${trackNum}`);
            if (currentTimeEl) {
                currentTimeEl.textContent = this.formatTime(currentTime);
            }
            
            // Update waveform progress
            const progress = Math.min(elapsed / track.playDuration, 1);
            this.updateWaveformProgress(trackNum, progress);
            
            track.animationFrame = requestAnimationFrame(updateTime);
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
            try {
                track.source.stop();
                track.source.disconnect();
            } catch (e) {}
            track.source = null;
        }
        
        console.log(`[STOP] Track ${trackNum} - Playback stopped`);
        
        track.isPlaying = false;
        
        // Cancel animation frame
        if (track.animationFrame) {
            cancelAnimationFrame(track.animationFrame);
            track.animationFrame = null;
        }
        
        this.updatePlayButton(trackNum, false);
        
        const currentTimeEl = document.getElementById(`current-time-${trackNum}`);
        if (currentTimeEl) {
            currentTimeEl.textContent = '0:00';
        }
        
        // Reset waveform to show no progress
        if (track.buffer) {
            this.drawWaveform(trackNum, track.buffer);
        }
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
        
        console.log(`[PREVIEW] Track ${trackNum} - Preview Selection clicked: startTime=${startTime.toFixed(2)}s, endTime=${endTime.toFixed(2)}s`);
        
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
            
            console.log(`[SLIDER] Track ${trackNum} - slider moved to startTime=${startTime.toFixed(2)}s, endTime=${endTime.toFixed(2)}s`);
            
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
        const statusMessage = document.getElementById('status-message');
        
        if (track1Ready && track2Ready) {
            console.log('Both tracks ready - showing Generate button');
            generateSection.style.display = 'block';
            if (statusMessage) statusMessage.style.display = 'none';
            this.setupGenerateButton();
        } else {
            generateSection.style.display = 'none';
            if (statusMessage) statusMessage.style.display = 'block';
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
        
        // Save job ID to localStorage for persistence across page refreshes
        localStorage.setItem('lastJobId', this.jobId);
        console.log(`[DEBUG] Job started with ID: ${this.jobId}`);
        
        // Poll for status
        await this.pollGenerationStatus();
    }
    
    async pollGenerationStatus() {
        const API_URL = 'http://localhost:8001';
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const loadingSubtitle = document.querySelector('.loading-subtitle');
        
        const startTime = Date.now();
        const timeout = 120000; // 2 minutes timeout
        
        while (true) {
            const elapsed = Date.now() - startTime;
            if (elapsed > timeout) {
                throw new Error('Generation timeout (2 minutes). Please try again with shorter audio files.');
            }
            
            try {
                const response = await fetch(`${API_URL}/api/status/${this.jobId}`, {
                    timeout: 5000
                });
                
                if (!response.ok) {
                    throw new Error(`Status check failed: ${response.status}`);
                }
                
                const status = await response.json();
                
                console.log('Generation status:', status);
                
                // Update progress bar
                progressFill.style.width = `${status.progress}%`;
                progressText.textContent = `${status.progress}%`;
                
                // Update status message
                if (loadingSubtitle && status.message) {
                    loadingSubtitle.textContent = status.message;
                }
                
                if (status.status === 'completed') {
                    this.outputPath = status.output_path;
                    console.log('Generation completed!', this.outputPath);
                    break;
                } else if (status.status === 'failed') {
                    throw new Error(status.message || 'Generation failed');
                }
                
                // Wait 1 second before next poll
                await new Promise(resolve => setTimeout(resolve, 1000));
            } catch (error) {
                console.error('Poll error:', error);
                // If it's a timeout error, throw it
                if (error.message.includes('timeout')) {
                    throw error;
                }
                // For other errors, wait and retry
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }
    
    // ========================================
    // Result Player
    // ========================================
    async setupResultPlayer() {
        const playBtn = document.getElementById('play-btn-result');
        const downloadBtn = document.getElementById('download-btn');
        const newGenBtn = document.getElementById('new-generation-btn');
        
        // Remove old event listeners by cloning and replacing nodes
        // This prevents duplicate listeners on regeneration
        if (!this.resultListenersSetup) {
            const playBtnNew = playBtn.cloneNode(true);
            playBtn.parentNode.replaceChild(playBtnNew, playBtn);
            
            const downloadBtnNew = downloadBtn.cloneNode(true);
            downloadBtn.parentNode.replaceChild(downloadBtnNew, downloadBtn);
            
            const newGenBtnNew = newGenBtn.cloneNode(true);
            newGenBtn.parentNode.replaceChild(newGenBtnNew, newGenBtn);
            
            // Add event listeners to new nodes
            document.getElementById('play-btn-result').addEventListener('click', () => this.toggleResultPlay());
            document.getElementById('download-btn').addEventListener('click', () => this.downloadResult());
            document.getElementById('new-generation-btn').addEventListener('click', () => this.resetForNewGeneration());
            
            this.resultListenersSetup = true;
            console.log('[DEBUG] Result player event listeners setup');
        }
        
        // Load generated audio from API
        await this.loadGeneratedAudio();
        
        document.getElementById('duration-result').textContent = 
            this.formatTime(this.resultAudio.duration);
        
        this.drawResultWaveform();
    }
    
    async loadGeneratedAudio() {
        const API_URL = 'http://localhost:8001';
        
        try {
            console.log(`[DEBUG] Fetching audio for job: ${this.jobId}`);
            
            // Fetch generated audio file
            const response = await fetch(`${API_URL}/api/download/${this.jobId}`);
            
            console.log(`[DEBUG] Download response status: ${response.status}`);
            console.log(`[DEBUG] Download response headers:`, response.headers.get('content-length'));
            
            if (!response.ok) {
                throw new Error('Failed to download generated audio');
            }
            
            // Convert to audio buffer
            const arrayBuffer = await response.arrayBuffer();
            console.log(`[DEBUG] ArrayBuffer size: ${arrayBuffer.byteLength} bytes`);
            
            // IMPORTANT: Create blob BEFORE decoding, as decodeAudioData consumes the ArrayBuffer
            this.resultBlob = new Blob([arrayBuffer], { type: 'audio/wav' });
            console.log(`[DEBUG] Blob created, size: ${this.resultBlob.size} bytes`);
            
            // Clone the arrayBuffer for decoding (since decodeAudioData transfers/consumes it)
            const arrayBufferCopy = arrayBuffer.slice(0);
            this.resultAudio = await this.audioContext.decodeAudioData(arrayBufferCopy);
            console.log(`[DEBUG] Audio decoded successfully, duration: ${this.resultAudio.duration}s`);
            
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
    
    async playResultAudio() {
        // Resume AudioContext if suspended (required by browser autoplay policy)
        if (this.audioContext.state === 'suspended') {
            console.log('[DEBUG] AudioContext suspended, resuming...');
            await this.audioContext.resume();
        }
        
        // Clean up any existing source
        if (this.resultSource) {
            try {
                this.resultSource.stop();
                console.log('[DEBUG] Stopped existing source');
            } catch (e) {
                console.log('[DEBUG] Could not stop existing source:', e.message);
            }
            this.resultSource.disconnect();
            this.resultSource = null;
        }
        
        console.log('[DEBUG] Playing result audio, buffer:', this.resultAudio);
        console.log('[DEBUG] AudioContext state:', this.audioContext.state);
        
        try {
            const source = this.audioContext.createBufferSource();
            source.buffer = this.resultAudio;
            
            const gainNode = this.audioContext.createGain();
            gainNode.gain.value = document.getElementById('volume-result').value / 100;
            
            source.connect(gainNode);
            gainNode.connect(this.audioContext.destination);
            
            source.start(0);
            console.log('[DEBUG] Audio source started successfully');
            
            this.resultIsPlaying = true;
            this.resultSource = source;
            
            const playBtn = document.getElementById('play-btn-result');
            playBtn.querySelector('.icon-play').style.display = 'none';
            playBtn.querySelector('.icon-pause').style.display = 'block';
            
            source.onended = () => {
                console.log('[DEBUG] Audio playback ended');
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
            
        } catch (error) {
            console.error('[ERROR] Failed to play audio:', error);
            alert('Failed to play audio. Please try again.');
            this.resultIsPlaying = false;
        }
    }
    
    stopResultPlay() {
        if (this.resultSource) {
            try {
                this.resultSource.stop();
                this.resultSource.disconnect();
                console.log('[DEBUG] Stopped and disconnected result source');
            } catch (e) {
                console.log('[DEBUG] Could not stop source:', e.message);
            }
            this.resultSource = null;
        }
        
        this.resultIsPlaying = false;
        
        const playBtn = document.getElementById('play-btn-result');
        playBtn.querySelector('.icon-play').style.display = 'block';
        playBtn.querySelector('.icon-pause').style.display = 'none';
        
        document.getElementById('current-time-result').textContent = '0:00';
    }
    
    downloadResult() {
        console.log(`[DEBUG] Download button clicked`);
        console.log(`[DEBUG] resultBlob exists: ${!!this.resultBlob}`);
        console.log(`[DEBUG] resultBlob size: ${this.resultBlob ? this.resultBlob.size : 'N/A'} bytes`);
        
        if (!this.resultBlob) {
            alert('No audio available to download');
            return;
        }
        
        if (this.resultBlob.size === 0) {
            alert('Generated audio file is empty (0 bytes). Please try generating again.');
            return;
        }
        
        // Create download link
        const url = URL.createObjectURL(this.resultBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `musiclab_generated_${Date.now()}.wav`;
        console.log(`[DEBUG] Creating download link: ${a.download}`);
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log(`[DEBUG] Download triggered successfully`);
    }
    
    resetForNewGeneration() {
        // Stop any playing audio
        this.stopResultPlay();
        
        // Clear previous result data
        if (this.resultSource) {
            try {
                this.resultSource.stop();
            } catch (e) {
                // Already stopped
            }
            this.resultSource = null;
        }
        
        this.resultIsPlaying = false;
        
        // Reset UI
        document.getElementById('result-section').style.display = 'none';
        document.getElementById('generate-section').style.display = 'block';
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        
        console.log('[DEBUG] Reset for new generation - cleared result state');
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

// ========================================
// MusicNote Chatbot
// ========================================

class MusicChatbot {
    constructor() {
        this.API_URL = 'http://localhost:8001';
        this.sessionId = this.generateSessionId();
        this.isOpen = false;
        
        this.trigger = document.getElementById('chatbot-trigger');
        this.window = document.getElementById('chatbot-window');
        this.closeBtn = document.getElementById('chatbot-close');
        this.messagesContainer = document.getElementById('chatbot-messages');
        this.input = document.getElementById('chatbot-input');
        this.sendBtn = document.getElementById('chatbot-send');
        
        this.setupEventListeners();
        this.setupBeforeUnload();
        
        console.log('[Chatbot] Initialized with session:', this.sessionId);
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    setupEventListeners() {
        this.trigger.addEventListener('click', () => this.toggle());
        this.closeBtn.addEventListener('click', () => this.close());
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }
    
    setupBeforeUnload() {
        // Clear chat history when user closes browser/tab
        window.addEventListener('beforeunload', () => {
            this.clearSession();
        });
    }
    
    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }
    
    open() {
        this.window.classList.add('active');
        this.isOpen = true;
        this.input.focus();
        console.log('[Chatbot] Opened');
    }
    
    close() {
        this.window.classList.remove('active');
        this.isOpen = false;
        console.log('[Chatbot] Closed');
    }
    
    async sendMessage() {
        const message = this.input.value.trim();
        if (!message) return;
        
        // Clear input
        this.input.value = '';
        
        // Add user message to UI
        this.addMessage(message, 'user');
        
        // Show typing indicator
        const typingIndicator = this.showTyping();
        
        try {
            // Send to API
            const formData = new FormData();
            formData.append('session_id', this.sessionId);
            formData.append('message', message);
            
            const response = await fetch(`${this.API_URL}/api/chat`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeTyping(typingIndicator);
            
            // Add bot response
            if (data.status === 'success') {
                this.addMessage(data.response, 'bot');
                console.log('[Chatbot] Exchange count:', data.history_length);
            } else {
                this.addMessage('Sorry, I encountered an error. Please try again! 🎵', 'bot');
            }
            
        } catch (error) {
            console.error('[Chatbot] Error:', error);
            this.removeTyping(typingIndicator);
            this.addMessage('Oops! Connection error. Make sure the server is running. 🎵', 'bot');
        }
    }
    
    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = content.replace(/\n/g, '<br>');
        
        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message bot-message';
        typingDiv.id = 'typing-indicator';
        
        const typingContent = document.createElement('div');
        typingContent.className = 'message-content chat-typing';
        typingContent.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        
        typingDiv.appendChild(typingContent);
        this.messagesContainer.appendChild(typingDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        
        return typingDiv;
    }
    
    removeTyping(typingElement) {
        if (typingElement && typingElement.parentNode) {
            typingElement.parentNode.removeChild(typingElement);
        }
    }
    
    async clearSession() {
        try {
            await fetch(`${this.API_URL}/api/chat/${this.sessionId}`, {
                method: 'DELETE',
                keepalive: true  // Allows request to complete during page unload
            });
            console.log('[Chatbot] Session cleared');
        } catch (error) {
            // Silently fail during page unload - this is expected behavior
            // The browser may cancel the request, which is okay
        }
    }
}

// About Modal Functionality
function initializeAboutModal() {
    const modal = document.getElementById('about-modal');
    const btn = document.getElementById('about-btn');
    const closeBtn = document.querySelector('.modal-close');
    const gotItBtn = document.querySelector('.modal-btn');

    if (!modal || !btn) return;

    // Open modal
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
    });

    // Close modal functions
    const closeModal = () => {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto'; // Restore scrolling
    };

    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (gotItBtn) gotItBtn.addEventListener('click', closeModal);

    // Close when clicking outside the modal content
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            closeModal();
        }
    });

    // Close with Escape key
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            closeModal();
        }
    });
}

// Examples Modal Player Class
class ExamplesPlayer {
    constructor() {
        this.audioContext = null;
        this.examples = {
            input: { buffer: null, source: null, gainNode: null, isPlaying: false },
            target: { buffer: null, source: null, gainNode: null, isPlaying: false },
            output: { buffer: null, source: null, gainNode: null, isPlaying: false }
        };
        this.animationFrames = {};
    }

    async initialize() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        await this.loadAllExamples();
        this.setupControlListeners();
    }

    async loadAllExamples() {
        const examples = [
            { id: 'input', name: 'input' },
            { id: 'target', name: 'target' },
            { id: 'output', name: 'output' }
        ];

        for (const example of examples) {
            try {
                const response = await fetch(`http://localhost:8001/api/examples/${example.name}`);
                const arrayBuffer = await response.arrayBuffer();
                const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
                this.examples[example.id].buffer = audioBuffer;
                
                // Draw waveform
                this.drawWaveform(example.id, audioBuffer);
                
                // Update duration
                const duration = this.formatTime(audioBuffer.duration);
                document.getElementById(`${example.id}-duration`).textContent = duration;
                document.getElementById(`${example.id}-total-time`).textContent = duration;
            } catch (error) {
                console.error(`Failed to load ${example.id}:`, error);
            }
        }
    }

    drawWaveform(exampleId, audioBuffer) {
        const canvas = document.getElementById(`${exampleId}-waveform`);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        const channelData = audioBuffer.getChannelData(0);
        const step = Math.ceil(channelData.length / width);
        const amp = height / 2;
        
        ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        for (let i = 0; i < width; i++) {
            let min = 1.0;
            let max = -1.0;
            for (let j = 0; j < step; j++) {
                const datum = channelData[i * step + j];
                if (datum < min) min = datum;
                if (datum > max) max = datum;
            }
            const yMin = (1 + min) * amp;
            const yMax = (1 + max) * amp;
            
            ctx.fillRect(i, yMin, 1, yMax - yMin);
        }
        ctx.stroke();
    }

    updateWaveformProgress(exampleId, progress) {
        const canvas = document.getElementById(`${exampleId}-waveform`);
        if (!canvas) return;

        const example = this.examples[exampleId];
        if (!example.buffer) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        const channelData = example.buffer.getChannelData(0);
        const step = Math.ceil(channelData.length / width);
        const amp = height / 2;
        
        // Draw base waveform
        ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
        
        for (let i = 0; i < width; i++) {
            let min = 1.0;
            let max = -1.0;
            for (let j = 0; j < step; j++) {
                const datum = channelData[i * step + j];
                if (datum < min) min = datum;
                if (datum > max) max = datum;
            }
            const yMin = (1 + min) * amp;
            const yMax = (1 + max) * amp;
            
            ctx.fillRect(i, yMin, 1, yMax - yMin);
        }
        
        // Draw progress overlay
        const progressX = Math.floor(width * progress);
        if (progressX > 0) {
            ctx.fillStyle = 'rgba(0, 212, 170, 0.4)';
            
            for (let i = 0; i < progressX; i++) {
                let min = 1.0;
                let max = -1.0;
                for (let j = 0; j < step; j++) {
                    const datum = channelData[i * step + j];
                    if (datum < min) min = datum;
                    if (datum > max) max = datum;
                }
                const yMin = (1 + min) * amp;
                const yMax = (1 + max) * amp;
                
                ctx.fillRect(i, yMin, 1, yMax - yMin);
            }
            
            // Draw progress line
            ctx.strokeStyle = '#00d4aa';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(progressX, 0);
            ctx.lineTo(progressX, height);
            ctx.stroke();
        }
    }

    setupControlListeners() {
        ['input', 'target', 'output'].forEach(id => {
            const playBtn = document.getElementById(`${id}-play-btn`);
            const volumeSlider = document.getElementById(`${id}-volume`);
            
            if (playBtn) {
                playBtn.addEventListener('click', () => this.togglePlayback(id));
            }
            
            if (volumeSlider) {
                volumeSlider.addEventListener('input', (e) => {
                    this.setVolume(id, e.target.value / 100);
                });
            }
        });
    }

    async togglePlayback(exampleId) {
        const example = this.examples[exampleId];
        
        if (example.isPlaying) {
            this.stopPlayback(exampleId);
        } else {
            // Stop other playing examples
            Object.keys(this.examples).forEach(id => {
                if (id !== exampleId && this.examples[id].isPlaying) {
                    this.stopPlayback(id);
                }
            });
            
            await this.startPlayback(exampleId);
        }
    }

    async startPlayback(exampleId) {
        const example = this.examples[exampleId];
        if (!example.buffer) return;

        // Resume audio context if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        // Create source and gain node
        example.source = this.audioContext.createBufferSource();
        example.source.buffer = example.buffer;
        
        example.gainNode = this.audioContext.createGain();
        const volumeSlider = document.getElementById(`${exampleId}-volume`);
        example.gainNode.gain.value = volumeSlider ? volumeSlider.value / 100 : 0.7;
        
        example.source.connect(example.gainNode);
        example.gainNode.connect(this.audioContext.destination);
        
        example.source.onended = () => {
            if (example.isPlaying) {
                this.stopPlayback(exampleId);
            }
        };
        
        example.source.start(0);
        example.isPlaying = true;
        example.startTime = this.audioContext.currentTime;
        
        // Update button to show pause icon
        this.updatePlayButton(exampleId, true);
        
        // Start time update animation
        this.updateTimeDisplay(exampleId);
    }

    stopPlayback(exampleId) {
        const example = this.examples[exampleId];
        
        if (example.source) {
            try {
                example.source.stop();
            } catch (e) {
                // Ignore if already stopped
            }
            example.source = null;
        }
        
        if (example.gainNode) {
            example.gainNode.disconnect();
            example.gainNode = null;
        }
        
        example.isPlaying = false;
        
        // Update button to show play icon
        this.updatePlayButton(exampleId, false);
        
        // Cancel animation frame
        if (this.animationFrames[exampleId]) {
            cancelAnimationFrame(this.animationFrames[exampleId]);
        }
        
        // Reset time display
        document.getElementById(`${exampleId}-current-time`).textContent = '0:00';
        
        // Reset waveform to show no progress
        if (example.buffer) {
            this.drawWaveform(exampleId, example.buffer);
        }
    }

    updatePlayButton(exampleId, isPlaying) {
        const button = document.getElementById(`${exampleId}-play-btn`);
        if (!button) return;
        
        const svg = button.querySelector('svg');
        if (isPlaying) {
            // Pause icon
            svg.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
        } else {
            // Play icon
            svg.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
        }
    }

    updateTimeDisplay(exampleId) {
        const example = this.examples[exampleId];
        if (!example.isPlaying || !example.buffer) return;
        
        const elapsed = this.audioContext.currentTime - example.startTime;
        const currentTimeEl = document.getElementById(`${exampleId}-current-time`);
        
        if (currentTimeEl) {
            currentTimeEl.textContent = this.formatTime(Math.min(elapsed, example.buffer.duration));
        }
        
        // Update waveform progress
        const progress = Math.min(elapsed / example.buffer.duration, 1);
        this.updateWaveformProgress(exampleId, progress);
        
        if (elapsed < example.buffer.duration) {
            this.animationFrames[exampleId] = requestAnimationFrame(() => this.updateTimeDisplay(exampleId));
        }
    }

    setVolume(exampleId, value) {
        const example = this.examples[exampleId];
        if (example.gainNode) {
            example.gainNode.gain.value = value;
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    stopAllPlayback() {
        Object.keys(this.examples).forEach(id => {
            if (this.examples[id].isPlaying) {
                this.stopPlayback(id);
            }
        });
    }
}

// Initialize Examples Modal
function initializeExamplesModal() {
    const modal = document.getElementById('examples-modal');
    const btn = document.getElementById('examples-btn');
    const closeBtn = document.querySelector('.modal-close-examples');
    const closeFooterBtn = document.querySelector('#examples-modal .modal-btn');
    let examplesPlayer = null;

    if (!modal || !btn) return;

    // Open modal
    btn.addEventListener('click', async function(e) {
        e.preventDefault();
        modal.style.display = 'flex';
        
        // Initialize player if not already done
        if (!examplesPlayer) {
            examplesPlayer = new ExamplesPlayer();
            await examplesPlayer.initialize();
        }
    });

    // Close modal - X button
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            modal.style.display = 'none';
            if (examplesPlayer) {
                examplesPlayer.stopAllPlayback();
            }
        });
    }

    // Close modal - footer button
    if (closeFooterBtn) {
        closeFooterBtn.addEventListener('click', function() {
            modal.style.display = 'none';
            if (examplesPlayer) {
                examplesPlayer.stopAllPlayback();
            }
        });
    }

    // Close when clicking outside modal content
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
            if (examplesPlayer) {
                examplesPlayer.stopAllPlayback();
            }
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && modal.style.display === 'flex') {
            modal.style.display = 'none';
            if (examplesPlayer) {
                examplesPlayer.stopAllPlayback();
            }
        }
    });
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.musicLab = new MusicLab();
    window.musicChatbot = new MusicChatbot();
    initializeAboutModal();
    initializeExamplesModal();
});
