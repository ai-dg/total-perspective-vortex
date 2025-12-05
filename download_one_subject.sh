#!/bin/bash

BASE_URL="https://physionet.org/files/eegmmidb/1.0.0"

if [ -z "$1" ]; then
    read -p "Which subject do you want to download (ex: S004) ? : " SUBJECT
else
    SUBJECT="$1"
fi

if [[ ! "$SUBJECT" =~ ^S[0-9]{3}$ ]]; then
    echo "❌ Invalid format. Use: S001, S023, S109..."
    exit 1
fi

if [ -z "$2" ]; then
    read -p "In which folder do you want to save the files ? : " DEST
else
    DEST="$2"
fi

mkdir -p "$DEST"

LOCAL_PATH="$DEST/$SUBJECT"

if [ -d "$LOCAL_PATH" ]; then
    echo "✔ The folder '$LOCAL_PATH' already exists. No download needed."
    exit 0
fi

CHECK_URL="$BASE_URL/$SUBJECT/${SUBJECT}R01.edf"
echo "🔎 Checking subject on Physionet..."

if ! curl --output /dev/null --silent --head --fail "$CHECK_URL"; then
    echo "❌ Subject $SUBJECT does not exist on Physionet."
    exit 1
fi

echo "✔ Subject found. Downloading selected runs (imagery only)..."

RUNS=("03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14")

mkdir -p "$LOCAL_PATH"

for RUN in "${RUNS[@]}"; do
    FILE_BASE="${SUBJECT}R${RUN}"
    echo "➡ Downloading ${FILE_BASE}.edf and .edf.event ..."
    wget -c -P "$LOCAL_PATH" "$BASE_URL/$SUBJECT/${FILE_BASE}.edf"
    wget -c -P "$LOCAL_PATH" "$BASE_URL/$SUBJECT/${FILE_BASE}.edf.event"
done

echo "✔ Download complete."
echo "📁 Saved in: $LOCAL_PATH"
echo "🧠 Runs downloaded: R04, R06, R08, R10, R12, R14 (imagery tasks)"
