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
#   STORAGE_ACCOUNT   - Blob storage account name   (default: traktstorage)
#   LOCATION          - Azure region                (default: uksouth)

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-trakt}"
ACR_NAME="${ACR_NAME:-traktregistry}"
APP_NAME="${APP_NAME:-trakt-dashboard}"
APP_SERVICE_PLAN="${APP_SERVICE_PLAN:-trakt-dashboard-plan}"
LOCATION="${LOCATION:-uksouth}"
IMAGE_TAG="${ACR_NAME}.azurecr.io/trakt-streamlit:latest"

echo "=== Step 1: Create Azure Container Registry (if needed) ==="
if az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  echo "ACR '$ACR_NAME' already exists"
else
  echo "Creating ACR '$ACR_NAME'..."
  az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true
fi

# Verify ACR is accessible before proceeding
echo "Verifying ACR '$ACR_NAME'..."
az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv || {
  echo "ERROR: ACR '$ACR_NAME' not found in resource group '$RESOURCE_GROUP'."
  echo "Either create it manually or set ACR_NAME=<your-registry> before running this script."
  echo "  az acr create --resource-group $RESOURCE_GROUP --name <unique-name> --sku Basic --admin-enabled true"
  exit 1
}

echo "=== Step 2: Build and push Docker image ==="
az acr build \
  --registry "$ACR_NAME" \
  --image trakt-streamlit:latest \
  --file Dockerfile.streamlit \
  .

echo "=== Step 3: Create App Service Plan (Linux, B1 tier) ==="
if az appservice plan show --name "$APP_SERVICE_PLAN" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  echo "App Service Plan '$APP_SERVICE_PLAN' already exists"
else
  az appservice plan create \
    --name "$APP_SERVICE_PLAN" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --is-linux \
    --sku B1
fi

echo "=== Step 4: Create Web App from container image ==="
if az webapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  echo "Web app '$APP_NAME' already exists — updating container image"
  az webapp config container set \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --container-image-name "$IMAGE_TAG" \
    --container-registry-url "https://${ACR_NAME}.azurecr.io"
else
  az webapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --plan "$APP_SERVICE_PLAN" \
    --container-image-name "$IMAGE_TAG" \
    --container-registry-url "https://${ACR_NAME}.azurecr.io"
fi

STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-traktstorage}"

echo "=== Step 5: Configure app settings ==="
# Fetch the storage connection string
echo "Looking up connection string for storage account '$STORAGE_ACCOUNT'..."
STORAGE_CONN=$(az storage account show-connection-string \
  --resource-group "$RESOURCE_GROUP" \
  --name "$STORAGE_ACCOUNT" \
  --query connectionString -o tsv) || {
  echo "ERROR: Storage account '$STORAGE_ACCOUNT' not found in resource group '$RESOURCE_GROUP'."
  echo "Set STORAGE_ACCOUNT=<your-account> or create it:"
  echo "  az storage account create --name <name> --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS"
  exit 1
}

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

echo "=== Step 8: Restart App Service to pull fresh image ==="
az webapp restart \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP"
echo "App Service restarted — new container image will be pulled."

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
