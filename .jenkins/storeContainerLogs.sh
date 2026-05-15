#!/bin/bash
"$SCRIPTS_FOLDER/printLog.sh" "ERROR" "SaveContainerLogs" "Starting to store container logs!"
 # Store docker logs
    DIRECTORY_PATH="$WORKSPACE/logs/"

    if [ ! -d "$DIRECTORY_PATH" ]; then
      "$SCRIPTS_FOLDER/printLog.sh" "ERROR" "SaveContainerLogs" "Directory for storing logs doesnt exist creating..."
      mkdir -p $DIRECTORY_PATH
    fi

    docker logs ollama-gpu &>"$DIRECTORY_PATH/Ollama.log"
echo "Storing of logs finished"