#!/bin/bash
# Sentinance Kubernetes Deployment Script

set -e

NAMESPACE="sentinance"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Deploying Sentinance to Kubernetes..."

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster."
    exit 1
fi

echo "✅ Connected to cluster"

# Create namespace if not exists
echo "📦 Creating namespace..."
kubectl apply -f "$SCRIPT_DIR/namespace.yaml"

# Apply config and secrets
echo "🔧 Applying configuration..."
kubectl apply -f "$SCRIPT_DIR/config.yaml"

# Deploy databases
echo "🗄️ Deploying databases..."
kubectl apply -f "$SCRIPT_DIR/database.yaml"

# Wait for databases to be ready
echo "⏳ Waiting for databases..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=120s || true
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=60s || true

# Deploy API
echo "🔌 Deploying API..."
kubectl apply -f "$SCRIPT_DIR/api.yaml"

# Deploy Web
echo "🌐 Deploying Web..."
kubectl apply -f "$SCRIPT_DIR/web.yaml"

# Apply Ingress
echo "🔀 Configuring Ingress..."
kubectl apply -f "$SCRIPT_DIR/ingress.yaml"

# Apply autoscaling
echo "📈 Configuring autoscaling..."
kubectl apply -f "$SCRIPT_DIR/autoscaling.yaml"

# Wait for deployments
echo "⏳ Waiting for deployments to be ready..."
kubectl rollout status deployment/api -n $NAMESPACE --timeout=180s
kubectl rollout status deployment/web -n $NAMESPACE --timeout=180s

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Status:"
kubectl get pods -n $NAMESPACE
echo ""
echo "🔗 Services:"
kubectl get svc -n $NAMESPACE
echo ""
echo "🌐 Ingress:"
kubectl get ingress -n $NAMESPACE
