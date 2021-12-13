#! /bin/bash

export PROJECT_DIR='/d/DALILA/ProgFormation/M2-IMAGINE/ImageSecu/Projet-Musee/MuseeSecuriseVirtuel'
export BUILD_DIR="$PROJECT_DIR/code/build/Debug"

export PREFIX_BIN='MuseeSecuriseVirtuel'
export POSTFIX_BIN='.out.exe' 

export PHOTOS_DIR="$PROJECT_DIR/code/photos"
export DATA_DIR="$PROJECT_DIR/code/data"

export EXPORTED_DATA="$PROJECT_DIR/code/data/exportedChain"
export imgprefix='Chaine_'

export SECRET_KEY='27111991'
export OEUVRE_PATH="$EXPORTED_DATA/$imgprefix"'EncryptedOeuvre.png'
export PATTERN="$EXPORTED_DATA/$imgprefix"'patternexported.png'


mkdir -p "$EXPORTED_DATA" 

# Step 0 : create Oeuvre ... 
"$BUILD_DIR/$PREFIX_BIN-MethodeNaive""$POSTFIX_BIN" "$DATA_DIR/15_Girafes.pgm"  "$SECRET_KEY" "$OEUVRE_PATH" "$EXPORTED_DATA"

# Step 1 : Create Feuille oeuvre 
"$BUILD_DIR/$PREFIX_BIN-ExportImage""$POSTFIX_BIN" "$OEUVRE_PATH" "$SECRET_KEY" "$EXPORTED_DATA/$imgprefix""Oeuvre0.png" "$EXPORTED_DATA/$imgprefix"'Oeuvre1.png' "$EXPORTED_DATA/$imgprefix"'Oeuvre2.png' "$PATTERN"


# Step 2 : Detection feuille 
#"$BUILD_DIR/$PREFIX_BIN-DetectImage""$POSTFIX_BIN" "$PHOTOS_DIR/IMG_20211209_124313.jpg" "$EXPORTED_DATA/$imgprefix"

#export imageProcess="$EXPORTED_DATA/$imgprefix"'_perspective.png'

# Step 3 :  Detection oeuvre 
#"$BUILD_DIR/$PREFIX_BIN-Detection""$POSTFIX_BIN" "$imageProcess" "$EXPORTED_DATA/$imgprefix"

# Step 4 :  Fin 
#"$BUILD_DIR/$PREFIX_BIN-Detection""$POSTFIX_BIN" "$imageProcess" "$EXPORTED_DATA/$imgprefix"