# 🚀 AI Model Platform

A modular, scalable platform for **serving, analyzing, and automating AI/ML workflows**.
It uses **FastAPI** microservices, **YAML-based configuration**, and ships with **Docker Compose** for local runs and **Helm** for Kubernetes.

---

## ✨ Features

- **Microservices**: `serving` (gateway, predictor, transformer).
- **Config-driven**: models/transformers and app settings via YAML under `config/`.
- **Shared core**: common base classes for models/transformers + utilities in `src/`.
- **Deploy-ready**: local Compose stacks and Helm charts under `deploy/`.
- **Tests**: unit/integration tests under `tests/`.

---

## 📂 Project Layout

> Only key folders (and a few representative files) are shown.
```bash
.
├── apps/ # Microservices
│ ├── serving/ # Online serving stack
│ │ ├── docker-compose.yaml
│ │ ├── gateway/ # API gateway
│ │ │ ├── Dockerfile
│ │ │ └── app.py
│ │ ├── predictor/ # Model serving
│ │ │ ├── Dockerfile
│ │ │ └── app.py
│ │ └── transformer/ # Pre/post-processing
│ │ ├── Dockerfile
│ │ └── app.py
│ ├── monitoring/ # Prometheus + Grafana
│ └── ...
├── config/ # YAML configs (versioned)
│ ├── models.yaml # model configuration
│ └── transformers.yaml # pre/post-processors configuration
├── data/ # Datasets & model artifacts (mounted at runtime)
│ ├── datasets/
│ ├── models/ # model weights
│ └── transformers/ # pre/post-processors files
├── deploy/ # Deployment layer
│ ├── charts/ # Helm charts for Kubernetes
│ │ └── serving/
│ │ │ ├── templates/ # deployment, service, ingress, hpa, pvc, ...
│ │ │ ├── Chart.yaml # chart definition and version
│ │ │ └── values.yaml # default values
│ │ ├── monitoring/
│ │ ├── ...
│ └── scripts/ # helper scripts (pipeline, loader, etc.)
├── src/ # Shared library (internal packages)
│ ├── core/ # models/transformers + loaders
│ └── shared/ # logging, settings, http helpers
└── tests/ # unit & integration tests
```
---

## 🔧 Prerequisites

- **Python 3.11+**
- **Docker** & **Docker Compose**
- **Helm 3** (+ **kubectl** and optional **Minikube** for local K8s)

---

## ⚙️ Installation (local, optional)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r apps/serving/gateway/requirements.txt
pip install -r apps/serving/transformer/requirements.txt
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r apps/serving/predictor/requirements.txt
```
## 🐍 Run with Python (without Docker)

You can run each service directly with Python. Make sure to install the requirements first:

```bash
python -m apps.serving.gateway.app
python -m apps.serving.predictor.app
python -m apps.serving.transformer.app
```

## ▶️ Run with Docker Compose

### Run single stacks

```bash
# Serving (gateway + predictor + transformer)
docker compose -f apps/serving/docker-compose.yaml up -d --build
```

### Run with Makefile

```bash
make up         # start all stacks
make down       # stop all stacks
# Cleanup dangling images/volumes/cache:
make clean
```
>Note on paths: Compose resolves relative paths from the compose file’s location.
>The provided compose files already use the correct ../../ mounts to reach config/ and data/ at the repo root.

## 🧪 Testing
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```