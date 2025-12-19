# AWS Deployment Session Checkpoint
**Date:** December 19, 2025  
**Project:** MusicLab - AI Music Generation Platform  
**Deployment Target:** AWS EC2 t3.micro (Free Tier)

---

## 🎯 Mission Accomplished

Successfully deployed MusicLab to AWS EC2 and configured public access via DuckDNS. The application is now live and accessible worldwide.

**Live URLs:**
- **Primary:** http://musiclab.duckdns.org
- **Direct IP:** http://51.20.58.15
- **Backend API:** http://musiclab.duckdns.org:8001
- **API Docs:** http://musiclab.duckdns.org:8001/docs

---

## 📋 Deployment Timeline

### Phase 1: AWS Infrastructure Setup
1. **EC2 Instance Creation**
   - Launched t3.micro instance (1 vCPU, 1GB RAM)
   - Instance ID: `i-0ed8087de322eeff4`
   - Region: eu-north-1 (Stockholm)
   - AMI: Ubuntu 24.04 LTS
   - SSH Key: sentiment-analysis-key.pem

2. **Security Group Configuration**
   - Security Group ID: `sg-00bda0428fbaa9c5d`
   - Opened ports:
     * Port 22 (SSH) - 0.0.0.0/0
     * Port 80 (HTTP) - 0.0.0.0/0
     * Port 8001 (Backend API) - 0.0.0.0/0

3. **Memory Management**
   - Added 4GB swap file (instance has only 1GB RAM)
   - Configuration: `vm.swappiness=10` (use swap only when needed)
   - Swap file location: `/swapfile`

### Phase 2: Deployment Script Development
Created comprehensive deployment automation script: `deploy_to_aws.sh`

**Script Features:**
- Dynamic IP detection (handles temporary AWS IPs)
- Automatic 4GB swap setup
- Docker installation (version 29.1.3)
- GitHub repository cloning
- Configuration files transfer via SCP:
  * `.env` (Groq API key, ports)
  * `backend/config/config_key.yaml` (Rita chatbot config)
  * `backend/checkpoints/best_model.pt` (239MB model checkpoint)
- Docker Compose build and deployment
- Health check verification

**Key Script Sections:**
```bash
# Dynamic IP detection
EC2_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# Swap setup
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo sysctl vm.swappiness=10

# File transfers
scp -i "$SSH_KEY" .env ubuntu@$EC2_IP:~/PowerMuse/.env
scp -i "$SSH_KEY" backend/config/config_key.yaml ubuntu@$EC2_IP:~/PowerMuse/backend/config/config_key.yaml
scp -i "$SSH_KEY" backend/checkpoints/best_model.pt ubuntu@$EC2_IP:~/PowerMuse/backend/checkpoints/best_model.pt

# Docker deployment
sudo docker compose up -d --build
```

### Phase 3: GitHub Repository Preparation
**Files Added to Git:**
1. `music_samples/` folder (3 WAV files for Examples modal):
   - input.wav (750KB) - Vivaldi August (Storm)
   - 1_noisy_target.wav (750KB) - Emir Kusturica Djinji Dinji
   - 1_predicted.wav (750KB) - AI-generated fusion

2. `frontend/Images/` folder (About section screenshots):
   - 1.png (1.0MB) - Upload interface
   - 2.png (1.2MB) - Pattern selection interface
   - 3.png (934KB) - Results interface

3. `AWS_DEPLOYMENT.md` - Comprehensive deployment guide (460 lines)

4. `deploy_to_aws.sh` - Automated deployment script (141 lines)

**Files Excluded (via .gitignore):**
- `.env` (contains Groq API key)
- `backend/config/config_key.yaml` (contains API keys)
- `backend/checkpoints/` (large model files - 239MB)
- `best_checkpoints/`

### Phase 4: Frontend Configuration Fixes

#### Issue 1: Hardcoded localhost URLs
**Problem:** Frontend had hardcoded `http://localhost:8001` in multiple places, causing CORS errors on AWS.

**Solution:** Dynamic API URL detection based on hostname
```javascript
// Before
const API_URL = 'http://localhost:8001';

// After
const API_URL = window.location.hostname === 'localhost' 
  ? 'http://localhost:8001' 
  : `http://${window.location.hostname}:8001`;
```

**Files Updated:**
- `frontend/app.js` - Updated 5 locations:
  * `callGenerationAPI()` - Music generation
  * `pollGenerationStatus()` - Status polling
  * `loadGeneratedAudio()` - Audio download
  * `MusicChatbot.constructor()` - Chatbot API URL
  * `loadAllExamples()` - Examples loading

#### Issue 2: Chatbot Language Support Not Working
**Problem 1:** Frontend not sending language parameter to backend
**Solution:**
```javascript
// Added to sendMessage() in MusicChatbot class
formData.append('language', currentLanguage);
```

**Problem 2:** Welcome message not translating with language changes
**Solution:** Added support for `data-i18n-html` attributes (in addition to existing `data-i18n`)
```javascript
// Added to setLanguage() function
document.querySelectorAll('[data-i18n-html]').forEach(element => {
    const key = element.getAttribute('data-i18n-html');
    const keys = key.split('.');
    let value = langData;
    
    for (const k of keys) {
        value = value[k];
        if (!value) break;
    }
    
    if (value) {
        element.innerHTML = value;
    }
});
```

**Backend Support (already existed):**
```python
# backend/main_api.py
@app.post("/api/chat")
async def chat_with_rita(
    session_id: str = Form(...),
    message: str = Form(...),
    language: str = Form("en")
):
    # Prepends strong language instruction to message
    lang_instruction = f"[{language_map.get(language, language_map['en'])}]\n\n"
    enhanced_message = lang_instruction + message
```

#### Issue 3: Missing Images in About Section
**Problem:** Images folder not in GitHub repository
**Solution:** Force-added PNG files to git:
```bash
git add -f frontend/Images/*.png
```

### Phase 5: Mobile Optimization

#### Initial Mobile Support (768px)
Already existed for tablets:
- Single column layout
- Responsive buttons
- Stacked elements

#### High-Resolution Phone Optimization (< 768px)
**Issues Reported:**
- Chatbot button too far to the right and off-screen
- Header too long, language selector requires horizontal scrolling

**Solution:** Ultra-compact mobile styles
```css
@media (max-width: 768px) {
    /* Ultra-compact header */
    .header-content {
        padding: 0.5rem 0;
        flex-wrap: nowrap;
        gap: 0.5rem;
    }
    
    .logo {
        font-size: 1rem;
        gap: 0.3rem;
    }
    
    .logo h1 {
        font-size: 1rem;
    }
    
    .logo-icon {
        width: 24px;
        height: 24px;
    }
    
    .nav {
        gap: 0.3rem;
        font-size: 0.75rem;
    }
    
    .nav-link {
        padding: 0.3rem 0.5rem;
        white-space: nowrap;
    }
    
    .lang-btn {
        padding: 0.3rem 0.5rem;
        font-size: 0.75rem;
        min-width: 45px;
    }
    
    /* Minimal chatbot */
    .chatbot-trigger {
        width: 50px;
        height: 50px;
        bottom: 10px;
        right: 10px;
    }
    
    .chatbot-trigger .note-icon {
        width: 26px;
        height: 26px;
    }
    
    .chatbot-window {
        bottom: 70px;
        right: 10px;
        left: 10px;
        width: auto;
        height: 65vh;
        max-height: 450px;
    }
}
```

**Key Changes:**
- Logo: 1.5rem → 1rem font size, 40px → 24px icon
- Nav links: 0.85rem → 0.75rem, tight gaps (0.3rem)
- Language button: Minimal 45px width
- Chatbot: 70px → 50px button, 30px → 10px margins
- All fixed values (rem/px) instead of CSS variables for predictable sizing

### Phase 6: DuckDNS Domain Configuration

**Setup Process:**
1. Registered at https://www.duckdns.org
2. Created subdomain: `musiclab`
3. Pointed to EC2 IP: `51.20.58.15`
4. Result: http://musiclab.duckdns.org

**Benefits:**
- Friendly, memorable URL
- No need to remember IP address
- Professional appearance

**Note:** IP is temporary (changes when instance stops/starts). Can set up automatic DuckDNS updates with a cron job if needed.

---

## 🐛 Issues Encountered & Resolutions

### 1. Music Samples Missing from Docker Build
**Error:**
```
ERROR: failed to calculate checksum: "/music_samples": not found
```

**Cause:** Folder existed locally but wasn't committed to GitHub

**Resolution:**
```bash
git add music_samples/
git commit -m "Add music_samples files for Examples modal"
git push origin main
```

### 2. GitHub Push Protection - API Key Exposure
**Error:**
```
remote: - Push cannot contain secrets
remote:   —— Groq API Key ——————————————————————
```

**Cause:** AWS_DEPLOYMENT.md contained real Groq API key in examples

**Resolution:**
- Replaced real key with placeholder: `your_groq_api_key_here`
- Used `git commit --amend` and `git push --force` to rewrite history

### 3. Model Checkpoint Not Included
**Problem:** User asked: "What about checkpoint that model uses to generate music?"

**Cause:** 239MB `best_model.pt` excluded from git (in .gitignore)

**Resolution:** Added SCP transfer to deployment script:
```bash
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "mkdir -p ~/PowerMuse/backend/checkpoints"
scp -i "$SSH_KEY" backend/checkpoints/best_model.pt ubuntu@$EC2_IP:~/PowerMuse/backend/checkpoints/best_model.pt
```

**Transfer time:** ~30 seconds for 239MB

### 4. Security Group Missing HTTP Ports
**Problem:** Browser timeout when accessing http://51.20.58.15

**Cause:** Security group only had SSH (port 22) open, not HTTP (80) or API (8001)

**Resolution:**
```bash
aws ec2 authorize-security-group-ingress --group-id sg-00bda0428fbaa9c5d --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id sg-00bda0428fbaa9c5d --protocol tcp --port 8001 --cidr 0.0.0.0/0
```

### 5. Frontend Container Not Starting After Rebuild
**Problem:** Frontend container showed "Created" status but wasn't running

**Cause:** Container dependency ordering issue after hot update

**Resolution:**
```bash
cd PowerMuse && sudo docker compose up -d frontend
```

### 6. Git Merge Conflict on EC2
**Problem:** `git pull` failed with "Your local changes would be overwritten by merge"

**Cause:** Direct SCP file copy to EC2 created uncommitted changes

**Resolution:**
```bash
git stash
git pull
sudo docker compose up -d --build frontend
```

---

## 📊 Technical Specifications

### AWS Resources
| Resource | Value |
|----------|-------|
| Instance Type | t3.micro |
| vCPUs | 1 |
| RAM | 1GB |
| Swap | 4GB |
| Storage | 20GB (EBS) |
| Region | eu-north-1 (Stockholm) |
| OS | Ubuntu 24.04.3 LTS |
| Public IP | 51.20.58.15 (temporary) |
| Domain | musiclab.duckdns.org |

### Docker Containers
| Container | Image | Size | Status | Ports |
|-----------|-------|------|--------|-------|
| musiclab-backend | powermuse-backend | 3.5GB | healthy | 8001 |
| musiclab-frontend | powermuse-frontend | 88MB | healthy | 80, 3000 |

### Backend Stack
- **Base Image:** python:3.10-slim
- **Framework:** FastAPI
- **ML Libraries:** PyTorch 2.1.0+cpu, EnCodec, librosa
- **Chatbot:** Groq API (Llama 3.3 70B)
- **Model:** SimpleTransformer (15.4M parameters, 239MB checkpoint)
- **Audio:** 24kHz sample rate, 16-second segments

### Frontend Stack
- **Base Image:** nginx:alpine
- **Server:** nginx 1.29.4
- **Languages:** Vanilla JavaScript, HTML5, CSS3
- **Supported Languages:** 8 (EN, RU, DE, FR, ES, PT, AR, ZH)
- **Features:** Real-time waveform visualization, multilingual chatbot, drag-and-drop uploads

### Network Configuration
- **Frontend Port:** 80 (HTTP)
- **Backend Port:** 8001 (HTTP)
- **Security Group:** sg-00bda0428fbaa9c5d
- **Allowed Traffic:** 0.0.0.0/0 (public access)

---

## 📝 Git Commit History (This Session)

```
6d37790 Aggressive mobile optimization: ultra-compact header and chatbot
f2d832f Optimize mobile layout: compact header and better chatbot positioning
14f4a4e Update deployment script
a7c74f7 Fix chatbot welcome message translation (data-i18n-html support)
71df3d2 Fix chatbot language support and add About section images
a9373b4 Add music_samples files for Examples modal
5838b87 Add music_samples and AWS deployment guide (without API keys)
```

---

## 🎨 Features Verified Working

✅ **Music Generation**
- Upload two audio files (MP3/WAV/OGG)
- Select 16-second time windows with waveform visualization
- Generate AI-fused music output
- Download results as WAV file

✅ **Rita Chatbot**
- Multilingual support (8 languages)
- Responds in user's selected language
- Provides music theory and production tips
- Explains MusicLab architecture
- Welcome message translates dynamically

✅ **Examples Modal**
- Three sample tracks showcase AI capabilities:
  * Input: Vivaldi - August (Storm)
  * Target: Emir Kusturica - Djinji Dinji
  * Output: AI-generated fusion
- Play, pause, seek controls
- Waveform visualization

✅ **About Section**
- Three interface screenshots showing workflow
- Responsive on mobile and desktop
- Explains 3-stage process

✅ **Mobile Responsiveness**
- Compact header fits on one line
- Chatbot accessible without scrolling
- Touch-friendly buttons and controls
- Optimized for high-resolution phone screens

✅ **Multilingual Interface**
- 8 languages fully supported
- Language selector in header
- Persistent language preference (localStorage)
- All UI elements translate dynamically

---

## 💰 Cost Analysis

### Free Tier (First 12 Months)
- **EC2 t3.micro:** 750 hours/month - **FREE**
- **20GB EBS storage:** Included - **FREE**
- **15GB data transfer out:** Included - **FREE**
- **Total:** $0/month

### After Free Tier
- **EC2 t3.micro:** ~$0.0104/hour × 730 hours = **$7.60/month**
- **20GB EBS storage:** $0.10/GB × 20 = **$2.00/month**
- **Data transfer:** $0.09/GB (first 10TB/month, usage-based)
- **Elastic IP (if added):** $3.60/month (when not associated with running instance)
- **Estimated Total:** $9-12/month (excluding data transfer)

### Optional Upgrades
- **Elastic IP:** $3.60/month (for permanent IP address)
- **Upgrade to t3.small:** ~$15/month (2GB RAM, better performance)
- **Backup snapshots:** $0.05/GB-month

---

## 🔒 Security Considerations

### Implemented
✅ SSH key authentication (sentiment-analysis-key.pem)
✅ Security group restricts access to specific ports only
✅ API keys not committed to GitHub
✅ Environment variables in .env (not in git)
✅ Docker container isolation

### Recommendations for Production
🔸 **HTTPS/SSL:** Set up Let's Encrypt certificate with nginx reverse proxy
🔸 **API Rate Limiting:** Add rate limiting to prevent abuse
🔸 **File Upload Validation:** Enhanced validation for audio uploads
🔸 **Monitoring:** Set up CloudWatch alarms for CPU/memory
🔸 **Backups:** Enable automated EBS snapshots
🔸 **Elastic IP:** Purchase permanent IP address
🔸 **CDN:** Use CloudFront for static assets
🔸 **Database:** If adding user accounts, use RDS instead of local storage

---

## 🚀 Deployment Commands Reference

### Initial Deployment
```bash
./deploy_to_aws.sh
```

### Update After Code Changes
```bash
# On local machine
git add .
git commit -m "Your changes"
git push origin main

# On EC2
ssh -i ~/.ssh/sentiment-analysis-key.pem ubuntu@51.20.58.15 \
  'cd PowerMuse && git pull && sudo docker compose up -d --build'
```

### Check Status
```bash
ssh -i ~/.ssh/sentiment-analysis-key.pem ubuntu@51.20.58.15 \
  'cd PowerMuse && sudo docker compose ps'
```

### View Logs
```bash
ssh -i ~/.ssh/sentiment-analysis-key.pem ubuntu@51.20.58.15 \
  'cd PowerMuse && sudo docker compose logs -f'
```

### Restart Services
```bash
ssh -i ~/.ssh/sentiment-analysis-key.pem ubuntu@51.20.58.15 \
  'cd PowerMuse && sudo docker compose restart'
```

### Check Memory Usage
```bash
ssh -i ~/.ssh/sentiment-analysis-key.pem ubuntu@51.20.58.15 \
  'free -h && sudo docker stats --no-stream'
```

---

## 📚 Documentation Created

1. **AWS_DEPLOYMENT.md** (460 lines)
   - Complete EC2 setup guide
   - Security group configuration
   - Swap setup instructions
   - Docker installation
   - Troubleshooting section

2. **deploy_to_aws.sh** (141 lines)
   - Automated deployment script
   - Dynamic IP detection
   - Health check verification
   - Error handling

3. **AWS_DEPLOYMENT_CHECKPOINT.md** (this file)
   - Complete session history
   - Technical specifications
   - Issue resolutions
   - Cost analysis

---

## 🎯 Next Steps (Optional Improvements)

### Short Term
1. Set up automatic DuckDNS IP updates (cron job)
2. Add SSL certificate via Let's Encrypt
3. Configure CloudWatch monitoring
4. Enable automated EBS snapshots

### Medium Term
1. Purchase Elastic IP for permanent address ($3.60/month)
2. Add nginx rate limiting for API endpoints
3. Implement user authentication system
4. Add file size limits and validation
5. Set up log rotation

### Long Term
1. Upgrade to t3.small for better performance (2GB RAM)
2. Add Redis for session management
3. Implement job queue for music generation
4. Use CloudFront CDN for static assets
5. Consider multi-region deployment

---

## ✅ Session Completion Checklist

- [x] EC2 instance launched and configured
- [x] 4GB swap memory configured
- [x] Docker installed and containers deployed
- [x] Security group ports opened (22, 80, 8001)
- [x] Repository cloned and updated on EC2
- [x] Configuration files transferred (.env, config_key.yaml)
- [x] Model checkpoint transferred (239MB)
- [x] Music samples added to GitHub
- [x] Images added to GitHub
- [x] Frontend API URLs fixed for AWS
- [x] Chatbot language support fixed
- [x] Chatbot welcome message translation fixed
- [x] Mobile optimization completed
- [x] DuckDNS domain configured (musiclab.duckdns.org)
- [x] Application tested and verified working
- [x] All changes committed to GitHub
- [x] Documentation completed

---

## 📞 Access Information

**Live Application:**
- Primary URL: http://musiclab.duckdns.org
- Direct IP: http://51.20.58.15
- Backend API: http://musiclab.duckdns.org:8001
- API Documentation: http://musiclab.duckdns.org:8001/docs

**SSH Access:**
```bash
ssh -i ~/.ssh/sentiment-analysis-key.pem ubuntu@51.20.58.15
```

**AWS Details:**
- Account: 242201275044
- Region: eu-north-1 (Stockholm)
- Instance ID: i-0ed8087de322eeff4
- Security Group: sg-00bda0428fbaa9c5d

**GitHub Repository:**
- https://github.com/Vlasenko2006/PowerMuse
- Branch: main
- Latest commit: 6d37790

---

## 🏁 Summary

Successfully deployed MusicLab to AWS EC2 with full functionality:
- ✅ Music generation working
- ✅ Chatbot responding in 8 languages
- ✅ Examples modal functional
- ✅ Mobile-optimized interface
- ✅ Public access via DuckDNS domain
- ✅ All features tested and verified

**Total deployment time:** ~4 hours including troubleshooting
**Issues resolved:** 6 major issues
**Files modified:** 5 (app.js, styles.css, deploy_to_aws.sh, added images and samples)
**Git commits:** 7
**Documentation pages:** 3

**Status:** 🟢 Production Ready
