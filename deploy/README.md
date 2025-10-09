## ☸️ Deploy (Helm)

### local Minikube example

#### 1) Start Cluster
```bash
minikube start
```

#### 2) Copy Files needed
```bash
minikube ssh "sudo mkdir -p /data && sudo chmod 777 /data" # hostPath
# create job-loader
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: loader
  namespace: ai
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 600
  ttlSecondsAfterFinished: 60
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: loader
          image: busybox
          command: ["sleep", "3600000"]
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: data
EOF
POD=$(kubectl -n ai get pod -l job-name=loader -o jsonpath='{.items[0].metadata.name}')
kubectl -n ai exec "$POD" -- sh -lc 'find /data -mindepth 1 -maxdepth 1 -exec rm -rf {} +' # delete [OPTIONAL]
tar -C ./data -cf - . | kubectl -n ai exec -i "$POD" -- sh -lc 'tar -C /data --overwrite -xf -' # load
kubectl -n ai exec "$POD" -- sh -lc "chown -R 1000:1000 /data && find /data -type d -exec chmod 775 {} \; && find /data -type f -exec chmod 664 {} \;" # permissions
```

#### 3) Build images 
```bash
eval $(minikube docker-env)
# serving
docker build -t gateway:latest -f apps/serving/gateway/Dockerfile .
docker build -t predictor:latest -f apps/serving/predictor/Dockerfile .
docker build -t transformer:latest -f apps/serving/transformer/Dockerfile .
```

#### 4) Install/upgrade
```bash
helm upgrade --install ai deploy/charts/serving --namespace ai --create-namespace
```

#### 5) Enable ingress
```bash
minikube addons enable ingress
# if and only in WSL
minikube tunnel
```

test the app http://localhost/docs

#### 6) Other services
a) create new helm chart under deploy/charts/
b) update dependency in umbrella chart /deploy/Chart.yaml
c) override deploy/values.yaml in needed
d) helm update -> install/upgrade
```bash
helm dependency update ./deploy
helm upgrade --install ai ./deploy -n ai --create-namespace -f ./deploy/values.yaml
```