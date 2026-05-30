#!/usr/bin/env bash
# deploy/ec2_setup.sh
#
# Bootstrap an AWS EC2 instance to run ARGUS.
# Tested on: Amazon Linux 2023, t3.medium (2 vCPU, 4GB RAM)
#
# Usage (from your local machine):
#   ssh -i your-key.pem ec2-user@<EC2-PUBLIC-IP> "bash -s" < deploy/ec2_setup.sh
#
# Or copy the repo to EC2 first:
#   scp -i your-key.pem -r . ec2-user@<EC2-PUBLIC-IP>:~/ARGUS
#   ssh -i your-key.pem ec2-user@<EC2-PUBLIC-IP> "cd ~/ARGUS && bash deploy/ec2_setup.sh"
#
# After running: the ARGUS API is accessible at http://<EC2-PUBLIC-IP>:8000
# Ensure port 8000 is open in your EC2 Security Group.

set -euo pipefail

echo "=== ARGUS EC2 Setup ==="
echo "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo 'unknown')"
echo "Region:   $(curl -s http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo 'unknown')"
echo ""

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/5] Installing system packages..."
sudo yum update -y -q
sudo yum install -y -q docker git python3-pip

# ---------------------------------------------------------------------------
# 2. Docker
# ---------------------------------------------------------------------------
echo "[2/5] Starting Docker..."
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker "$USER"
# Apply group without logout
newgrp docker <<DOCKERGRP

# ---------------------------------------------------------------------------
# 3. Environment variables
# ---------------------------------------------------------------------------
echo "[3/5] Configuring environment..."
if [ -z "${COHERE_API_KEY:-}" ]; then
    echo "WARNING: COHERE_API_KEY not set."
    echo "Set it with: export COHERE_API_KEY=your-key-here"
    echo "Or create a .env file in the repo root."
fi

# Write .env if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << EOF
COHERE_API_KEY=${COHERE_API_KEY:-}
MLFLOW_TRACKING_URI=mlruns
ARGUS_PORT=8000
EOF
    echo "Created .env file. Edit it to add your COHERE_API_KEY."
fi

# ---------------------------------------------------------------------------
# 4. Build and run Docker container
# ---------------------------------------------------------------------------
echo "[4/5] Building ARGUS Docker image..."
docker build -t argus-app:latest .

echo "Starting ARGUS container..."
docker stop argus-running 2>/dev/null || true
docker rm   argus-running 2>/dev/null || true

docker run -d \
    --name argus-running \
    --restart unless-stopped \
    -p 8000:8000 \
    --env-file .env \
    -v "$(pwd)/mlruns:/app/mlruns" \
    -v "$(pwd)/vectorstore:/app/vectorstore" \
    argus-app:latest

DOCKERGRP

# ---------------------------------------------------------------------------
# 5. Health check
# ---------------------------------------------------------------------------
echo "[5/5] Health check..."
sleep 5  # wait for container to start

MAX_RETRIES=10
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "ARGUS is running at http://localhost:8000"
        echo "API docs: http://localhost:8000/docs"
        PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<EC2-PUBLIC-IP>")
        echo "External: http://${PUBLIC_IP}:8000"
        break
    fi
    echo "Waiting for ARGUS to start ($i/$MAX_RETRIES)..."
    sleep 3
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Useful commands:"
echo "  docker logs -f argus-running       # tail logs"
echo "  docker exec -it argus-running bash # shell inside container"
echo "  docker restart argus-running       # restart"
echo ""
echo "Run MLflow UI (on EC2, forward port 5000 via SSH):"
echo "  docker exec argus-running mlflow ui --host 0.0.0.0 --port 5000 &"
echo "  # SSH: ssh -L 5000:localhost:5000 -i your-key.pem ec2-user@<EC2-IP>"
