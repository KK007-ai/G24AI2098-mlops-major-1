# Olivetti Faces - MLOps Major Assignment

**Student:** Kaushal Kushwaha (G24AI2098)  
**Branches required:** main, dev, docker_cicd

## Public links (put these on the first page of your PDF)
- GitHub repo: https://github.com/KK007-ai/G24AI2098-mlops-major.git
- Docker Hub image: https://hub.docker.com/r/kaushal299/olivetti-face

> **Important:** Replace the `<your-username>` and `<repo-name>` placeholders above with your actual GitHub and Docker Hub links. These exact links must appear on the **first page** of the submitted PDF.

## Overview
This project implements an end-to-end MLOps pipeline for face classification using the Olivetti faces dataset. It demonstrates:
- model development (scikit-learn `DecisionTreeClassifier`),
- automated CI via GitHub Actions to run training & testing,
- containerization using Docker,
- a minimal Flask app for inference,
- deployment to Kubernetes with 3 replicas.

## Files
- `train.py` - trains `DecisionTreeClassifier` and saves `savedmodel.pth`
- `test.py` - loads `savedmodel.pth` and prints accuracy
- `app.py` - Flask application for inference (image upload & prediction)
- `Dockerfile` - builds container image serving the Flask app
- `.github/workflows/ci.yml` - CI workflow that runs `train.py` and `test.py` on push
- `k8s/` - Kubernetes manifests (`deployment.yaml`, `service.yaml`, optional `hpa.yaml`)
- `requirements.txt` - Python dependencies
- `docs/screenshots/` - screenshots to include in the submission PDF

## How to run locally
See the project instructions or the assignment guide; typical steps:
1. Create virtualenv and install requirements
2. Run `python train.py` then `python test.py`
3. Build Docker and run locally
4. Deploy to Kubernetes

## Contact
Kaushal Kushwaha (G24AI2098)
