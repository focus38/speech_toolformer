#!/bin/bash

# Define the voice name and directory
EN_VOICE_NAME="en_lessac"
RU_VOICE_NAME="ru_irina"

VOICE_DIR="./data/voices"

EN_MODEL_FILE="${VOICE_DIR}/${EN_VOICE_NAME}.onnx"
EN_CONFIG_FILE="${EN_MODEL_FILE}.json"

RU_MODEL_FILE="${VOICE_DIR}/${RU_VOICE_NAME}.onnx"
RU_CONFIG_FILE="${RU_MODEL_FILE}.json"

# Create the voices directory if it doesn't exist
mkdir -p "$VOICE_DIR"

# Base URL for the official Hugging Face repository
BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Check if the voice model exists
if [ ! -f "$EN_MODEL_FILE" ]; then
    echo "Voice '${EN_VOICE_NAME}' not found locally. Downloading..."

    # Download .onnx model and .json configuration
    curl -L "${BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -o "$EN_MODEL_FILE"
    curl -L "${BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -o "$EN_CONFIG_FILE"

    curl -L "${BASE_URL}/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx" -o "$MODEL_FILE"
    curl -L "${BASE_URL}/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json" -o "$CONFIG_FILE"

    echo "Downloaded '${EN_VOICE_NAME}'."
else
    echo "Voice '${EN_VOICE_NAME}' found locally. Skipping download."
fi

if [ ! -f "$RU_MODEL_FILE" ]; then
    echo "Voice '${RU_VOICE_NAME}' not found locally. Downloading..."

    # Download .onnx model and .json configuration
    curl -L "${BASE_URL}/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx" -o "$RU_MODEL_FILE"
    curl -L "${BASE_URL}/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json" -o "$RU_CONFIG_FILE"

    echo "Downloaded '${RU_VOICE_NAME}'."
else
    echo "Voice '${RU_VOICE_NAME}' found locally. Skipping download."
fi

# Run Piper using the model
echo "Testing Piper TTS..."
echo "Welcome to local text to speech." | piper --model "$EN_MODEL_FILE" --output_file en_output.wav
echo "Карл у Клары украл кораллы, а Клара у Карла украла кларнет." | piper --model "$RU_MODEL_FILE" --output_file ru_output.wav