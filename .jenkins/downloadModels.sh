#!/bin/bash

# Cargar variables del .env
set -a
source .env
set +a

# Separar la variable MODELOS por comas y recorrerla
IFS=',' read -ra MODELOS_ARRAY <<< "$LLMMODELS"
for modelo in "${MODELOS_ARRAY[@]}"; do
  "$SCRIPTS_FOLDER/printLog.sh" "INFO" "downloadModels" "Ejecutando pull para: $modelo"
  docker exec --tty=false ollama-gpu ollama pull "$modelo"
done
