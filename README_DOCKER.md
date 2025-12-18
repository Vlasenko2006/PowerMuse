# MusicLab Docker Deployment Guide

**Quick reference for Docker deployment. See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for complete guide.**

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB free RAM (4GB instance + 4GB swap)
- ~20GB disk space (model checkpoints + images)
- Groq API key from https://console.groq.com

## Quick Start (Local)

1. **Configure Environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env and add your Groq API key
   nano .env
   # Add: GROQ_API_KEY=gsk_your_actual_key_here
   ```

2. **Build and Run**
   ```bash
   docker compose up --build -d
   ```

3. **Access the Application**
   - **Frontend:** http://localhost (port 80)
   - **Frontend (alt):** http://localhost:3000
   - **Backend API:** http://localhost:8001 (internal only)
   - **API Health:** http://localhost:8001/health
   - **API Docs:** http://localhost:8001/docs

4. **Verify Deployment**
   ```bash
   # Check container health
   docker ps
   # Both should show (healthy)
   
   # Test backend
   curl http://localhost:8001/health
   
   # Test frontend
   curl -I http://localhost
   ```

5. **View Logs**
   ```bash
   # All services
   docker compose logs -f
   
   # Specific service
   docker compose logs -f backend
   docker compose logs -f frontend
   
   # Last 50 lines
   docker compose logs --tail=50
   ```

6. **Stop Services**
   ```bash
   # Stop but keep data
   docker compose down
   
   # Stop and remove volumes (DELETES DATA!)
   docker compose down -v
   ```

## Production Deployment (AWS EC2)

### Option 1: EC2 with Docker Compose

1. **Launch EC2 Instance**
   - Instance Type: t3.medium (2 vCPU, 4GB RAM)
   - OS: Ubuntu 22.04 LTS
   - Storage: 20GB gp3
   - Security Group: Open ports 80, 443, 8001 (temporarily)

2. **Install Docker**
   ```bash
   # SSH into instance
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   
   # Install Docker Compose
   sudo apt-get update
   sudo apt-get install docker-compose-plugin
   ```

3. **Deploy Application**
   ```bash
   # Clone repository
   git clone https://github.com/Vlasenko2006/PowerMuse.git
   cd PowerMuse
   
   # Configure
   cp config/config_key-example.yaml config/config_key.yaml
   nano config/config_key.yaml  # Add your API key
   
   # Build and run
   docker-compose up -d
   ```

4. **Setup Nginx Reverse Proxy (Optional)**
   ```bash
   sudo apt install nginx
   sudo nano /etc/nginx/sites-available/musiclab
   ```
   
   Add configuration:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:3000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

5. **Enable SSL with Let's Encrypt**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

### Option 2: AWS ECS Fargate

1. **Create ECR Repositories**
   ```bash
   aws ecr create-repository --repository-name musiclab-backend
   aws ecr create-repository --repository-name musiclab-frontend
   ```

2. **Build and Push Images**
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and tag
   docker build -f Dockerfile.backend -t musiclab-backend .
   docker tag musiclab-backend:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/musiclab-backend:latest
   
   docker build -f Dockerfile.frontend -t musiclab-frontend .
   docker tag musiclab-frontend:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/musiclab-frontend:latest
   
   # Push
   docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/musiclab-backend:latest
   docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/musiclab-frontend:latest
   ```

3. **Create ECS Cluster and Task Definition** (via AWS Console or CLI)

4. **Setup Application Load Balancer**

## Environment Variables

Create `.env` file (copy from `.env.example`):

```bash
GROQ_API_KEY=your_actual_key_here
BACKEND_PORT=8001
FRONTEND_PORT=3000
```

## Troubleshooting

### Backend not starting
```bash
docker-compose logs backend
# Check if config/config_key.yaml exists
# Verify Groq API key is valid
```

### Frontend can't reach backend
```bash
# Check network
docker network inspect musiclab_musiclab-network
# Verify backend is healthy
curl http://localhost:8001/health
```

### Model checkpoint missing
```bash
# Ensure checkpoints directory exists and contains best_model.pt
ls -lh checkpoints/
```

## Resource Requirements

- **Development**: 2GB RAM, 2 CPU cores
- **Production**: 4GB RAM, 2-4 CPU cores
- **Storage**: ~2GB (containers + model checkpoint)

## Monitoring

Use Docker stats to monitor resource usage:
```bash
docker stats musiclab-backend musiclab-frontend
```

## Updates

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up --build -d

# View logs to confirm
docker-compose logs -f
```

## Security Notes

- Never commit `config/config_key.yaml` to git
- Use environment variables for sensitive data in production
- Keep Docker and system packages updated
- Use SSL/TLS in production (Let's Encrypt)
- Restrict security group rules (only necessary ports)

## Support

For issues, check:
1. Docker logs: `docker-compose logs`
2. Backend health: `curl http://localhost:8001/health`
3. GitHub Issues: https://github.com/Vlasenko2006/PowerMuse/issues
