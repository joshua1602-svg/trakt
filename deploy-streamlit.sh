#!/usr/bin/env bash
# Deterministic Streamlit deployment to Azure App Service (Linux container).
# - Builds immutable SHA-tagged image in ACR
# - Updates web app to that exact tag
# - Restarts app and verifies configured image

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-trakt}"
ACR_NAME="${ACR_NAME:-}"
APP_NAME="${APP_NAME:-trakt-dashboard}"
APP_SERVICE_PLAN="${APP_SERVICE_PLAN:-trakt-dashboard-plan}"
LOCATION="${LOCATION:-uksouth}"
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-traktstorage}"
IMAGE_NAME="${IMAGE_NAME:-trakt-streamlit}"
if [[ -z "${ACR_NAME}" ]]; then
  # Auto-discover ACR from current Web App container config when available.
  CURRENT_LINUX_FX="$(az webapp config show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query linuxFxVersion -o tsv 2>/dev/null || true)"
  REGISTRY_HOST="$(echo "$CURRENT_LINUX_FX" | sed -E 's#^DOCKER\|([^/]+)/.*#\1#')"
  if [[ -n "$REGISTRY_HOST" && "$REGISTRY_HOST" != "$CURRENT_LINUX_FX" ]]; then
    ACR_NAME="${REGISTRY_HOST%%.*}"
  else
    echo "ERROR: ACR_NAME not provided and auto-discovery failed from linuxFxVersion='$CURRENT_LINUX_FX'."
    echo "Set ACR_NAME=<your-registry-name> and rerun."
    exit 1
  fi
fi

# Use full SHA for immutability; fallback to timestamp outside git worktrees.
IMAGE_VERSION="${IMAGE_VERSION:-$(git rev-parse HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE_TAG="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_VERSION}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing required command '$1'"; exit 1; }
}

require_cmd az

# Make CLI failures visible in logs.
az config set core.only_show_errors=true >/dev/null

echo "==> Deploy target"
echo "Resource Group : ${RESOURCE_GROUP}"
echo "ACR            : ${ACR_NAME}"
echo "Web App        : ${APP_NAME}"
echo "Image          : ${IMAGE_TAG}"

# 1) Ensure ACR exists.
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "==> Creating ACR '${ACR_NAME}'"
  az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic
fi
ACR_LOGIN_SERVER="$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv)"

# 2) Build immutable image tag only (no :latest).
echo "==> Building/pushing ${IMAGE_NAME}:${IMAGE_VERSION}"
az acr build \
  --registry "$ACR_NAME" \
  --image "${IMAGE_NAME}:${IMAGE_VERSION}" \
  --file Dockerfile.streamlit \
  .

# 3) Ensure App Service plan exists.
if ! az appservice plan show --name "$APP_SERVICE_PLAN" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "==> Creating App Service plan '${APP_SERVICE_PLAN}'"
  az appservice plan create \
    --name "$APP_SERVICE_PLAN" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --is-linux \
    --sku B1
fi

# 4) Ensure Web App exists (create if needed).
if ! az webapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "==> Creating Web App '${APP_NAME}'"
  az webapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --plan "$APP_SERVICE_PLAN" \
    --container-image-name "$IMAGE_TAG"
fi

# 5) App settings (idempotent).
echo "==> Configuring app settings"
STORAGE_CONN="$(az storage account show-connection-string \
  --resource-group "$RESOURCE_GROUP" \
  --name "$STORAGE_ACCOUNT" \
  --query connectionString -o tsv)"

az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
    DATA_STORAGE_CONNECTION="$STORAGE_CONN" \
    WEBSITES_PORT=8501 \
    TRAKT_DASHBOARD_BUILD_SHA="$IMAGE_VERSION" >/dev/null

# 6) Managed identity + AcrPull role assignment (production-safe pull auth).
echo "==> Configuring managed identity pull from ACR"
PRINCIPAL_ID="$(az webapp identity assign --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query principalId -o tsv)"
ACR_ID="$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query id -o tsv)"

if ! az role assignment list \
  --assignee-object-id "$PRINCIPAL_ID" \
  --scope "$ACR_ID" \
  --role "AcrPull" \
  --query "[0].id" -o tsv | grep -q .; then
  az role assignment create \
    --assignee-object-id "$PRINCIPAL_ID" \
    --assignee-principal-type ServicePrincipal \
    --scope "$ACR_ID" \
    --role "AcrPull" >/dev/null
fi

# Tell App Service to use MI creds for ACR pulls.
az webapp config set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --generic-configurations '{"acrUseManagedIdentityCreds": true}' >/dev/null

# 7) Update container image to immutable SHA tag.
echo "==> Updating Web App container image"
az webapp config container set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --container-image-name "$IMAGE_TAG" \
  --container-registry-url "https://${ACR_LOGIN_SERVER}" >/dev/null

# 8) Force refresh and verify configured image.
echo "==> Restarting Web App"
az webapp restart --name "$APP_NAME" --resource-group "$RESOURCE_GROUP"

EXPECTED_LINUX_FX="DOCKER|${IMAGE_TAG}"
ACTUAL_LINUX_FX="$(az webapp config show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query linuxFxVersion -o tsv)"
if [[ "$ACTUAL_LINUX_FX" != "$EXPECTED_LINUX_FX" ]]; then
  echo "ERROR: Web App linuxFxVersion mismatch"
  echo "Expected: ${EXPECTED_LINUX_FX}"
  echo "Actual  : ${ACTUAL_LINUX_FX}"
  exit 1
fi

echo ""
echo "Deployment successful"
echo "Image deployed: ${IMAGE_TAG}"
echo "Dashboard URL : https://${APP_NAME}.azurewebsites.net"
