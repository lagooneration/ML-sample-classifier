# Deployment Guide

## 🚀 Deployment Options

### Option 1: Heroku (Recommended)

1. **Install Heroku CLI**
2. **Login to Heroku:**
   ```bash
   heroku login
   ```

3. **Create Heroku app:**
   ```bash
   heroku create your-audio-classifier
   ```

4. **Deploy:**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

5. **Open app:**
   ```bash
   heroku open
   ```

### Option 2: Railway

1. **Connect GitHub repo to Railway**
2. **Railway will auto-detect Python and deploy**
3. **Environment variables are handled automatically**

### Option 3: Render

1. **Connect GitHub repo to Render**
2. **Select "Web Service"**
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `gunicorn app:app`

### Option 4: Docker (Any platform)

1. **Build image:**
   ```bash
   docker build -t audio-classifier .
   ```

2. **Run locally:**
   ```bash
   docker run -p 5000:5000 audio-classifier
   ```

3. **Deploy to any Docker-compatible platform**

## 📝 Environment Variables

For production deployment, set these environment variables:

- `PORT`: Port number (automatically set by most platforms)
- `FLASK_ENV`: Set to `production`

## 🔧 Files Created for Deployment

- `Procfile` - Heroku process definition
- `runtime.txt` - Python version specification
- `Dockerfile` - Container configuration
- `vercel.json` - Vercel/Railway configuration
- Updated `requirements.txt` - Added gunicorn
- Updated `app.py` - Production-ready settings

## ⚠️ Why Not Netlify?

Netlify is designed for static sites and JAMstack apps. Your Flask app needs:
- Python runtime
- Server-side processing
- File upload handling
- Machine learning model execution

These requirements make Netlify unsuitable for this application.

## 🎯 Recommended: Railway or Render

Both platforms offer:
- ✅ Easy GitHub integration
- ✅ Automatic deployments
- ✅ Python support
- ✅ Free tier available
- ✅ Built-in environment management
