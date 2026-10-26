#!/bin/bash
set -euxo pipefail

dnf update -y
dnf install -y docker git python3-pip
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

pip3 install --upgrade pip || true
pip3 install dvc dvc-s3 || true

REGION="${AWS_REGION:-ap-south-1}"
REPO="${GITHUB_REPO:-https://github.com/shwetgaur/image-classification-dvc.git}"
APP_DIR=/opt/image-classification-dvc

rm -rf "$APP_DIR"
git clone "$REPO" "$APP_DIR"
cd "$APP_DIR"

pip3 install "dvc[s3]" pyyaml numpy pandas pillow scikit-learn matplotlib joblib flask gunicorn tensorflow || true

if [ -f .dvc/config ] && grep -q 's3://' .dvc/config; then
  dvc pull -f || true
fi

if [ ! -f models/training_history.json ]; then
  python3 test.py || true
fi

docker build -t image-classification-dvc:latest .
docker rm -f ic-app || true
docker run -d --name ic-app -p 80:8000 --restart unless-stopped image-classification-dvc:latest
