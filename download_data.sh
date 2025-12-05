#!/bin/bash

BASE_URL="https://physionet.org/files/eegmmidb/1.0.0"

if [ -z "$1" ]; then
    read -p "In which folder do you want to save all subjects ? : " DEST
else
    DEST="$1"
fi

mkdir -p "$DEST"

for i in $(seq -w 1 109); do
    SUBJECT="S${i}"

    LOCAL_PATH="$DEST/$SUBJECT"
    if [ -d "$LOCAL_PATH" ]; then
        echo "✔ The folder '$LOCAL_PATH' already exists for $SUBJECT. Skipping."
        continue
    fi

    # Check if the subject exists
    CHECK_URL="$BASE_URL/$SUBJECT/${SUBJECT}R01.edf"
    echo "🔎 Checking $SUBJECT on Physionet..."

    if ! curl --output /dev/null --silent --head --fail "$CHECK_URL"; then
        echo "❌ Subject $SUBJECT does not exist on Physionet. Skipping."
        continue
    fi

    echo "✔ $SUBJECT found. Downloading selected runs (imagery only)..."

    RUNS=("03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14")
    mkdir -p "$LOCAL_PATH"

    for RUN in "${RUNS[@]}"; do
        FILE_BASE="${SUBJECT}R${RUN}"
        echo "➡ Downloading ${FILE_BASE}.edf and .edf.event ..."
        wget -c -P "$LOCAL_PATH" "$BASE_URL/$SUBJECT/${FILE_BASE}.edf"
        wget -c -P "$LOCAL_PATH" "$BASE_URL/$SUBJECT/${FILE_BASE}.edf.event"
    done

    echo "✔ Download complete for $SUBJECT."
    echo "📁 Saved in: $LOCAL_PATH"
    echo "🧠 Runs downloaded: R04, R06, R08, R10, R12, R14 (imagery tasks)"
    echo "---------------------------------------------"
done

echo "✅ All downloads attempted for subjects S001 to S109."
