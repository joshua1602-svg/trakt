#!/usr/bin/env bash
#
# Deploy the Streamlit dashboard to Azure App Service (Linux container).
#
# Prerequisites:
#   - Azure CLI (az) logged in
#   - Docker installed
#   - An Azure Container Registry (ACR) or use the existing traktstorage account
#
# Usage:
#   chmod +x deploy-streamlit.sh
#   ./deploy-streamlit.sh
#
# Environment variables (override defaults):
#   RESOURCE_GROUP    - Azure resource group        (default: trakt)
#   ACR_NAME          - Container registry name     (default: traktregistry)
#   APP_NAME          - App Service name            (default: trakt-dashboard)
#   APP_SERVICE_PLAN  - App Service plan name       (default: trakt-dashboard-plan)
#   LOCATION          - Azure region                (default: uksouth)

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-trakt}"
ACR_NAME="${ACR_NAME:-traktregistry}"
APP_NAME="${APP_NAME:-trakt-dashboard}"
APP_SERVICE_PLAN="${APP_SERVICE_PLAN:-trakt-dashboard-plan}"
LOCATION="${LOCATION:-uksouth}"
IMAGE_TAG="${ACR_NAME}.azurecr.io/trakt-streamlit:latest"

echo "=== Step 1: Create Azure Container Registry (if needed) ==="
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true \
  2>/dev/null || echo "ACR already exists"

echo "=== Step 2: Build and push Docker image ==="
az acr build \
  --registry "$ACR_NAME" \
  --image trakt-streamlit:latest \
  --file Dockerfile.streamlit \
  .

echo "=== Step 3: Create App Service Plan (Linux, B1 tier) ==="
az appservice plan create \
  --name "$APP_SERVICE_PLAN" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --is-linux \
  --sku B1 \
  2>/dev/null || echo "Plan already exists"

echo "=== Step 4: Create Web App from container image ==="
az webapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_SERVICE_PLAN" \
  --container-image-name "$IMAGE_TAG" \
  --container-registry-url "https://${ACR_NAME}.azurecr.io" \
  2>/dev/null || echo "Web app already exists"

echo "=== Step 5: Configure app settings ==="
# Fetch the storage connection string from traktstorage
STORAGE_CONN=$(az storage account show-connection-string \
  --resource-group "$RESOURCE_GROUP" \
  --name traktstorage \
  --query connectionString -o tsv)

az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
    DATA_STORAGE_CONNECTION="$STORAGE_CONN" \
    WEBSITES_PORT=8501

echo "=== Step 6: Enable system-assigned Managed Identity ==="
az webapp identity assign \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP"

echo "=== Step 7: Configure ACR pull credentials ==="
ACR_USER=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASS=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

az webapp config container set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --container-image-name "$IMAGE_TAG" \
  --container-registry-url "https://${ACR_NAME}.azurecr.io" \
  --container-registry-user "$ACR_USER" \
  --container-registry-password "$ACR_PASS"

echo ""
echo "=== Deployment complete ==="
echo "Dashboard URL: https://${APP_NAME}.azurewebsites.net"
echo ""
echo "--- Client access options ---"
echo "1. Azure AD Authentication (recommended):"
echo "   az webapp auth update --name $APP_NAME --resource-group $RESOURCE_GROUP \\"
echo "     --enabled true --action LoginWithAzureActiveDirectory"
echo ""
echo "2. Restrict by IP (quick):"
echo "   az webapp config access-restriction add --name $APP_NAME \\"
echo "     --resource-group $RESOURCE_GROUP --priority 100 \\"
echo "     --rule-name 'ClientOffice' --action Allow --ip-address <CLIENT_IP>/32"
