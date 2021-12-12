#! /bin/bash

export PROJECT_DIR='/d/DALILA/ProgFormation/M2-IMAGINE/ImageSecu/Projet-Musee/MuseeSecuriseVirtuel'
export BUILD_DIR="$PROJECT_DIR/code/build/Debug"

export PREFIX_BIN='MuseeSecuriseVirtuel'
export POSTFIX_BIN='.out.exe' 

export PHOTOS_DIR="$PROJECT_DIR/code/photos"
export DATA_DIR="$PROJECT_DIR/code/data"

export EXPORTED_DATA="$PROJECT_DIR/code/photos/exportedChain"
export imgprefix='ImageTest'

export SECRET_KEY='27111991'

mkdir -p "$EXPORTED_DATA" 

# Step 0 : create Oeuvre ... 

# Step 1 : Detection feuille 
"$BUILD_DIR/$PREFIX_BIN-DetectImage""$POSTFIX_BIN" "$PHOTOS_DIR/IMG_20211209_124313.jpg" "$EXPORTED_DATA/$imgprefix"

export imageProcess="$EXPORTED_DATA/$imgprefix"'_perspective.png'

# Step 2 :  Detection oeuvre 
"$BUILD_DIR/$PREFIX_BIN-Detection""$POSTFIX_BIN" "$imageProcess" "$EXPORTED_DATA/$imgprefix"