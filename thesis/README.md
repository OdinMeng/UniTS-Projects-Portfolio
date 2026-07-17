# Hybridizing Deep Learning and Classical Mathematical Modeling

This repository contains the code developed for the thesis **"Hybridizing Deep Learning and Classical Mathematical Modeling: Theory and New Applications"**, as well as the PDF files of the final work and slides for the defense.

## Folder Structure

Two main sub-folders:
- `./pdfs/`: Contains the PDF files
- `./code/`: Contains the notebooks used to run the experiments of the thesis, as well as the saved parameters of final UDE models.
    - I have (tried to) organize them neatly into chapters, so there is more or less a corrispondence between the reported case studies and the experiments ran
    - Some extra files worthy of mention:
        - `99_misc.ipynb`: A notebook to make some auxiliary plots for the figures of the thesis
        - `chaotic_recurrence.ipynb`: Under the suggestion of the supervisor, I have experimented with a modification of Chua model by adding a recurrent linear forcing term. There are some interesting results which can imply some things about the PEM-UDE architecture
        - `further_anns.ipynb`: I have experimented with applying different neural network architectures for UDEs, in particular the recursive feed-forward architecture which attempts to keep track of previous states. Unfortunately there are no worthy results here
        - `sr3_modified.py`: Modified algorithm of SR3 algorithm, so it prints the losses for each iteration

## Requirements

Listed in `requirements.txt` of the main folder

