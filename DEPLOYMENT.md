# 🚀 Deployment Guide - PredCA Dashboard

## Local Deployment

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Steps

1. **Clone/Download Project**
```bash
cd path/to/PredCA
```

2. **Create Virtual Environment** (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Application**
```bash
streamlit run app.py
```

5. **Access Dashboard**
- Open browser to `http://localhost:8501`
- Dashboard is now live!

---

## Cloud Deployment

### Option 1: Streamlit Cloud (Recommended)

**Easiest option - Free tier available**

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
- Go to https://share.streamlit.io
- Click "New app"
- Select your GitHub repo
- Choose branch and file (`app.py`)
- Click "Deploy"

3. **Share Link**
- Your app is now live!
- Share the public URL

**Pros:**
- Free tier available
- Easy deployment
- Automatic updates from GitHub
- Built-in SSL

**Cons:**
- Limited resources on free tier
- Requires GitHub account

---

### Option 2: Heroku

**Traditional cloud platform**

1. **Create Heroku Account**
- Sign up at https://www.heroku.com

2. **Install Heroku CLI**
```bash
# Windows
choco install heroku-cli

# macOS
brew tap heroku/brew && brew install heroku

# Linux
curl https://cli-assets.heroku.com/install.sh | sh
```

3. **Create Heroku App**
```bash
heroku login
heroku create your-app-name
```

4. **Create Procfile**
```
web: streamlit run app.py --logger.level=error
```

5. **Create setup.sh**
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = \$PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

6. **Deploy**
```bash
git push heroku main
```

**Pros:**
- Reliable platform
- Good documentation
- Flexible deployment options

**Cons:**
- Paid service (after free tier)
- More complex setup

---

### Option 3: AWS

**Enterprise-grade deployment**

1. **Create AWS Account**
- Sign up at https://aws.amazon.com

2. **Use EC2 Instance**
```bash
# SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install Python and dependencies
sudo yum install python3 python3-pip
pip3 install -r requirements.txt

# Run app
streamlit run app.py --server.port 80
```

3. **Configure Security Group**
- Allow inbound traffic on port 80 (HTTP)
- Allow inbound traffic on port 443 (HTTPS)

4. **Set Up Domain**
- Point domain to EC2 instance IP
- Configure SSL certificate

**Pros:**
- Highly scalable
- Enterprise features
- Full control

**Cons:**
- More complex setup
- Requires AWS knowledge
- Paid service

---

### Option 4: Docker Deployment

**Containerized deployment**

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

2. **Create .dockerignore**
```
.git
.gitignore
__pycache__
*.pyc
.streamlit
venv
```

3. **Build Docker Image**
```bash
docker build -t predca-dashboard .
```

4. **Run Container**
```bash
docker run -p 8501:8501 predca-dashboard
```

5. **Push to Docker Hub** (Optional)
```bash
docker tag predca-dashboard username/predca-dashboard
docker push username/predca-dashboard
```

**Pros:**
- Consistent environment
- Easy scaling
- Works anywhere Docker runs

**Cons:**
- Requires Docker knowledge
- Additional setup

---

## Performance Optimization

### 1. Data Caching
```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')
```

### 2. Reduce Computation
```python
# Use sampling for large datasets
df_sample = df.sample(n=10000)
```

### 3. Optimize Visualizations
```python
# Use Plotly for interactive charts
# Limit number of points in scatter plots
# Use aggregation for large datasets
```

### 4. Database Connection
```python
# Use connection pooling
# Cache database queries
# Use read replicas for analytics
```

---

## Monitoring & Maintenance

### Streamlit Cloud
- Built-in monitoring
- Automatic logs
- Performance metrics

### Self-Hosted
- Set up logging
- Monitor CPU/Memory
- Set up alerts
- Regular backups

### Useful Tools
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **ELK Stack**: Logging
- **New Relic**: APM

---

## Security Best Practices

1. **Environment Variables**
```python
import os
API_KEY = os.getenv('API_KEY')
```

2. **Secrets Management**
```bash
# Streamlit Cloud
# Add secrets in Settings > Secrets
```

3. **HTTPS/SSL**
- Always use HTTPS in production
- Use Let's Encrypt for free certificates

4. **Authentication**
```python
# Add authentication layer
# Use OAuth, JWT, or API keys
```

5. **Data Protection**
- Encrypt sensitive data
- Use secure connections
- Regular security audits

---

## Scaling Strategies

### Vertical Scaling
- Increase server resources
- More CPU/RAM
- Better performance

### Horizontal Scaling
- Multiple instances
- Load balancer
- Database replication

### Caching
- Redis for session data
- CDN for static files
- Database query caching

---

## Troubleshooting Deployment

| Issue | Solution |
|-------|----------|
| App crashes | Check logs, verify dependencies |
| Slow performance | Optimize code, increase resources |
| Data not loading | Check file paths, permissions |
| Port conflicts | Use different port number |
| Memory issues | Reduce data size, optimize code |

---

## Cost Estimation

### Streamlit Cloud
- **Free**: Limited resources
- **Pro**: $5-20/month

### Heroku
- **Free**: Deprecated
- **Hobby**: $7/month
- **Standard**: $25+/month

### AWS
- **EC2**: $5-50+/month
- **RDS**: $10-100+/month
- **S3**: $0.023/GB

### Docker (Self-hosted)
- **VPS**: $5-20/month
- **Dedicated**: $50+/month

---

## Maintenance Checklist

- [ ] Regular backups
- [ ] Update dependencies
- [ ] Monitor performance
- [ ] Check logs
- [ ] Security updates
- [ ] User feedback
- [ ] Performance optimization
- [ ] Documentation updates

---

## Support & Resources

- **Streamlit Docs**: https://docs.streamlit.io/deploy
- **Heroku Docs**: https://devcenter.heroku.com
- **AWS Docs**: https://docs.aws.amazon.com
- **Docker Docs**: https://docs.docker.com

---

## Next Steps

1. Choose deployment platform
2. Follow platform-specific guide
3. Test thoroughly
4. Monitor performance
5. Gather user feedback
6. Iterate and improve

---

**Happy Deploying! 🎉**
