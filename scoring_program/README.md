# System evaluation tools for PARSEME 2.0 shared task - subtask 2 (MWE paraphrasing)
This folder contains scripts used to evaluate systems running in PARSEME 2.0 shared task, in subtask 2 on MWE paraphrasing.

  * `evaluate.py` - evaluation of paraphrases for a single language
  * `average_of_paraphrase_evaluations.py` - calculating average scores for the predictions, and generating scores in `scores.json` and `scores.html`
  * `test.schema.json` - JSON schema describing the correct format of a gold paraphrase file (see also [here](https://json-schema.org/))
  * `test.system.schema.json` - JSON schema describing the correct format of a system prediction file
  
## Installation
We recommend using the tools in a dedicated environment. Below we provide instructions for one possible way, with `miniconda`.

Install and activate the `miniconda` environment management system:
```
mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```
Create and activate an environment for parseme subtask 2.  Note that afer the 2nd command your prompt should be preceded with `(parseme_subtask2)`. Other new shell windows  :
will now show `(base)` meaning that there you work in your normal environment):
```
conda create -n parseme_subtask2
```
Activate the environment.  Note that afer this command your prompt should be preceded with `(parseme_subtask2)`. The prompt in other new shell windows
will now be preceded by `(base)` meaning that there you work in your normal environment:
```
conda activate parseme_substask2

```
Install the necessary packages. Note that the bert-score package is large. the whole environment might take up to 15 Gigabytes:
```
conda install python==3.11  #Newer Python versions than 3.11 are not sure to be compatible with Numpy
conda install pip
pip install numpy #Python library for multi-dimensional arrays and matrices
pip install bert-score  #Library for calculating the BERTScore, to estimate the semantic closeness of two sentences
pip install spacy  #Spacy linguistic tools (needed for langauge-specific tokenizers)
python3 -m spacy download ja_core_news_sm  #Japanese-specific tokenizers
pip install git+https://github.com/estevelouis/WG4  #Library for caluclating in-text diversity scores (by Louis Estève)
sudo pip install json-spec  #Validator of JSON schemas
```
Check the list of packages installed in your environment and the size of the environment:
```
conda list
cd ~/miniconda3/envs/parseme_subtask2/
du -h .
```
Clone this repository and go to the subtask 2 folder: 
```
git clone git@gitlab.com:parseme/sharedtask-data-dev.git
cd sharedtask-data-dev/2.0/preliminary-sharedtask-data/subtask2/tools
```
To deactivate the environment:
```
conda deactivate
```
To destroy the environment if you no longer need it:
```
conda remove --name parseme_subtask2 --all
```

## MWE paraphrase evaluation for one language
To evaluate the paraphrases for one language, activate your `parseme_substask2` environment (see above) and run e.g.:
```
python3 evaluate.py FR/test.json FR/test.system.json 
```
This should produce an output like:
```
-------------------------------------- FR/test.system.json --------------------------------------


PERFORMANCE :
Average f-bertscore for the current system: 77.55
Number of evaluated elements: 80
Number of elements with a 0 score (MWE not deleted): 15


DIVERSITY :
Entropy, variety, balance for the current system:	 5.325, 326, 0.920
Entropy, variety, balance for the minimal reference:	 4.854, 168, 0.947
Entropy, variety, balance for the creative reference:	 5.212, 256, 0.940

```
## Counting the average of MWE paraphrase evaluations
To count and display the macro-averages of the paraphrase evaluations for several languages:
```
./average_of_paraphrase_evaluations.py [nb-langs] [pred-dir] [ave.json] [ave.html] 
```
where `[nb-langs]` is the number of all langauges in the competition, `[pred-dir]` is the folder with the predictions of a given system,
 and `[scores.json]` and `[ave.html]` are the `json` and `html` files where the average scores should be stored. Example:
 ```
./average_of_paraphrase_evaluations.py 14 ../../../system-results/subtask2/baseline-gpt-oss-120b  scores.json scores.html 
```



## File format validation
To validate gold and prediction files in command line:
```
json validate --schema-file test.schema.json --document-file test.json
json validate --schema-file test.system.schema.json --document-file test.system.json
```
