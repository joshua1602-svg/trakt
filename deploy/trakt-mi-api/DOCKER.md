# trakt-mi-api — container deploy (ACR + App Service `trakt-dashboard`)

Run the FastAPI MI Agent API as a **container** on Azure App Service, built and
served from ACR **`traktregistry1602`**, repointing the existing App Service
**`trakt-dashboard`** at the new image.

- Image: **`trakt-mi-api`** → `traktregistry1602.azurecr.io/trakt-mi-api:<tag>`
- Start command (baked into the image `CMD`):
  `gunicorn --bind=0.0.0.0:$PORT -k uvicorn.workers.UvicornWorker mi_agent_api.app:app`
- Dockerfile: `deploy/trakt-mi-api/Dockerfile` (build context = repo root)

No blob-trigger / orchestrator / onboarding logic is changed.

## 0. Variables

```bash
ACR=traktregistry1602
IMAGE=trakt-mi-api
TAG=v1                       # or a git sha; avoid relying on :latest in prod
APP=trakt-dashboard
RG=<resource-group-of-trakt-dashboard>     # az webapp list -o table
IMAGE_REF=$ACR.azurecr.io/$IMAGE:$TAG
```

## 1. Build + push to ACR

Build **in ACR** (no local Docker needed), from the repo root:

```bash
az acr build \
  --registry $ACR \
  --image $IMAGE:$TAG \
  --file deploy/trakt-mi-api/Dockerfile \
  .
```

<details><summary>Alternative: local Docker build + push</summary>

```bash
az acr login --name $ACR
docker build -f deploy/trakt-mi-api/Dockerfile -t $IMAGE_REF .
docker push $IMAGE_REF
```
</details>

## 2. Point `trakt-dashboard` at the image

**Preferred — managed identity pull (no stored creds):**

```bash
# enable the app's system-assigned identity and grant AcrPull on the registry
az webapp identity assign -g $RG -n $APP
PRINCIPAL=$(az webapp identity show -g $RG -n $APP --query principalId -o tsv)
ACR_ID=$(az acr show -n $ACR --query id -o tsv)
az role assignment create --assignee "$PRINCIPAL" --role AcrPull --scope "$ACR_ID"

az webapp config container set -g $RG -n $APP \
  --container-image-name "$IMAGE_REF" \
  --container-registry-url "https://$ACR.azurecr.io"
az resource update -g $RG -n $APP --resource-type "Microsoft.Web/sites" \
  --set properties.siteConfig.acrUseManagedIdentityCreds=true -o none
```

<details><summary>Alternative: ACR admin credentials</summary>

```bash
az acr update -n $ACR --admin-enabled true
ACR_USER=$(az acr credential show -n $ACR --query username -o tsv)
ACR_PASS=$(az acr credential show -n $ACR --query 'passwords[0].value' -o tsv)

az webapp config container set -g $RG -n $APP \
  --container-image-name "$IMAGE_REF" \
  --container-registry-url "https://$ACR.azurecr.io" \
  --container-registry-user "$ACR_USER" \
  --container-registry-password "$ACR_PASS"
```
</details>

## 3. App settings (port + storage + platform canonical)

```bash
az webapp config appsettings set -g $RG -n $APP --settings \
  WEBSITES_PORT=8000 \
  TRAKT_STORAGE_BACKEND=blob \
  TRAKT_BLOB_CONNECTION="<storage-connection-string>" \
  TRAKT_PROCESSED_CONTAINER=processed-v2 \
  MI_AGENT_PLATFORM_URI="blob://processed-v2/platform/ERE/latest/platform_canonical_typed.csv" \
  MI_AGENT_SCRATCH=/tmp/trakt/mi_platform \
  MI_AGENT_CORS_ORIGINS="https://<your-react-host>"
```

The image `CMD` already sets the start command, so no App Service startup
override is needed. (To set it explicitly anyway:
`az webapp config set -g $RG -n $APP --startup-file "gunicorn --bind=0.0.0.0:8000 -k uvicorn.workers.UvicornWorker mi_agent_api.app:app"`.)

## 4. Restart + verify

```bash
az webapp restart -g $RG -n $APP
az webapp log tail -g $RG -n $APP        # watch container pull + gunicorn boot

curl -s https://$APP.azurewebsites.net/health | jq
# expect: {"ok": true, "dataSourceKind": "platform_canonical", "dataAvailable": true, ...}
```

`dataSourceKind: "platform_canonical"` confirms the container loaded the persisted
platform canonical from Blob (the `processed-v2/platform/ERE/latest/…` blob must
exist — run/approve a funded pack first).

## Redeploy a new build

```bash
az acr build --registry $ACR --image $IMAGE:v2 --file deploy/trakt-mi-api/Dockerfile .
az webapp config container set -g $RG -n $APP \
  --container-image-name "$ACR.azurecr.io/$IMAGE:v2" \
  --container-registry-url "https://$ACR.azurecr.io"
az webapp restart -g $RG -n $APP
```

## Notes

- App Service routes to the container port in `WEBSITES_PORT` (8000); the image
  binds gunicorn to `$PORT` (App Service injects it; defaults to 8000).
- `uvicorn.workers.UvicornWorker` is used exactly as required; it emits a
  deprecation notice on newer uvicorn but works. `uvicorn-worker` is also bundled
  if you later switch to `uvicorn_worker.UvicornWorker`.
- The platform canonical is downloaded to `MI_AGENT_SCRATCH` (`/tmp`, writable in
  the container) per resolution; mount + `MI_AGENT_PLATFORM_DIR` for very large tapes.
