import ast
from os import listdir
from os.path import isfile, join

directory_path = "./paraphrases/"

filenames = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
#print(files)


for filename in filenames:
    file_to_open = directory_path + filename
    with open(file_to_open, encoding = "utf-8") as f:
        data = ast.literal_eval(f.read())
    
    for i, phrase in enumerate(data):
        if phrase["text"] == phrase["paraphrase"]:
            print("Same phrases detected at: ", (i + 1))