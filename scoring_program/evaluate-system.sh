#!/bin/bash

#This script runs the PARSEME shared task evaluation script for subtask 2 (MWE paraphrasing) for a system submission.
#Parameters:
# $1 = gold data directory path
#	It is supposed to contain one folder per language, with a file called test.cupt in it.
# $2 = directory with system results
#	It is supposed to contain one folder per language, with a file called test.system.cupt in it.
# $3 = JSON schema for a prediction file
# $4 = scores directory path
#       The scores in .json and .html format will go there
#
# As a result, a results.txt file, containing the ouput of the evaluation script, is added to every language directory of the system in $1,
# and scores.json and scores.html files are added to $3.
#
# Sample call:
#    ./evaluate-system.sh ../../../codabench/bundle_parseme_2.0_subtask_2/reference_data ../../../system-results/subtask2/test test.system.schema.json .
#
#    ./evaluate-system.sh /app/input/ref /app/input/res test.system.schema.json /app/output/  #In a Codabench docker container
#####################################################################################################################################################

#After results submission for a system, the submission is already unzipped in /app/input/res/ (with no tool name)
#The gold data are in /app/input/ref
#The scores should go to /app/output/, including scores.json, scores.html

#Check the number of parameters
if [ $# -lt 4 ]; then
    echo "Usage: $0 <gold-data-dir> <system-results-dir> <pred-json-schema> <scores-dir>" >&2
    echo "       A results.txt file is added to every language directory in <system-results-dir>."
    exit 1
fi 

#Get the directory paths
REF_DIR=$1
REF_FILE=test.json  #Every gold file for a language should have this name
PRED_DIR=$2
PRED_JSON_SCHEMA=$3
PRED_FILE=test.system.json  #Every prediction for a language should have this name
SCORES_DIR=$4
SCORES_FILE=results.txt  #The scores for a language should be stored in 
SCORES_JSON=$SCORES_DIR/scores.json
SCORES_HTML=$SCORES_DIR/scores.html
LOG_FILE=log.txt

#Get the list of languages
LANGS=($(ls -d $REF_DIR/*))  #Put all the language code
NB_LANGS=${#LANGS[@]}
echo $NB_LANGS languages in total

#Validate and evaluate the submission for each language
for LANG in `ls $PRED_DIR | grep -v metadata`; do

   #Check that the language name exists in the reference
   if [[ ${LANGS[@]} =~ $LANG ]]
   then echo; echo "Evaluating the prediction for $LANG" 
   else 
      echo "$0: $LANG is NOT a valid language name. The predictions will be neglected." >&2
      continue
   fi

   #Validate the format of the prediction
   json validate --schema-file $PRED_JSON_SCHEMA --document-file $PRED_DIR/$LANG/$PRED_FILE > /dev/null 2> $LOG_FILE
   if [ "$?" != "0" ]
   then 
      echo "$0: JSON schema validation failed for $PRED_DIR/$LANG/$PRED_FILE" >&2
      cat $LOG_FILE >&2
      continue
   fi
   
   #Score the prediction for the current language
   if [ "$LANG" == "JA" ]; then
      python3 ./evaluate.py --is_JA  $REF_DIR/$LANG/$REF_FILE $PRED_DIR/$LANG/$PRED_FILE > $SCORES_FILE #2>/dev/null
   else
      python3 ./evaluate.py $REF_DIR/$LANG/$REF_FILE $PRED_DIR/$LANG/$PRED_FILE > $SCORES_FILE #2>/dev/null
   fi
   if [ "$?" != "0" ]; then
      echo "$0: Evaluation script failed for $PRED_DIR/$LANG/$PRED_FILE. Check that the file is aligned with the blind file." >&2
      continue
   else
      mv $SCORES_FILE $PRED_DIR/$LANG/$SCORES_FILE
      echo "$SCORES_FILE file added to $PRED_DIR/$LANG/"
   fi

done

#Generate the average scores for the current language
python3 ./average_of_paraphrase_evaluations.py $NB_LANGS $PRED_DIR $SCORES_JSON $SCORES_HTML

