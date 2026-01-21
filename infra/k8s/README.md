# Kubernetes Deployment

This directory contains Kubernetes manifests for deploying Sentinance.

## Files

| File | Description |
|------|-------------|
| `namespace.yaml` | Sentinance namespace |
| `config.yaml` | ConfigMap and Secrets |
| `api.yaml` | API deployment + service |
| `web.yaml` | Web deployment + service |
| `database.yaml` | PostgreSQL + Redis |
| `ingress.yaml` | Ingress with TLS |
| `autoscaling.yaml` | HPA + PDB |
| `kustomization.yaml` | Kustomize config |
| `deploy.sh` | Deployment script |

## Quick Deploy

```bash
# Using kubectl
kubectl apply -k ./

# Using script
chmod +x deploy.sh
./deploy.sh
```

## Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or local)
- `kubectl` configured
- Ingress controller (nginx-ingress)
- Container registry with images

## Building Images

```bash
# From project root
docker build -t sentinance/api:latest -f apps/api/Dockerfile .
docker build -t sentinance/web:latest -f apps/web/Dockerfile .

# Push to registry
docker push sentinance/api:latest
docker push sentinance/web:latest
```

## Configuration

1. Update `config.yaml` with your values
2. Replace secrets in `config.yaml` with proper secrets
3. Update `ingress.yaml` with your domain
4. Create TLS secret for HTTPS:
   ```bash
   kubectl create secret tls sentinance-tls \
     --cert=path/to/tls.crt \
     --key=path/to/tls.key \
     -n sentinance
   ```

## Monitoring

Access Prometheus metrics:
```bash
kubectl port-forward svc/api 8000:8000 -n sentinance
curl http://localhost:8000/metrics
```

## Scaling

Autoscaling is configured. Manual scaling:
```bash
kubectl scale deployment/api --replicas=5 -n sentinance
```
