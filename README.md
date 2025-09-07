# ThesisCode

### Folder Structure
- Data : contains code for splitting the data into training and evaluation data and a script for creating datasets with lag features
- XGB: code for XGBoost, both lag and no lag variants, with a notebook for event detection each.
- RF: code for Random Forest, both lag and no lag variants. Event detection plots are created through a second script.
- LSTM: code for supervised LSTM. Event detection plots are created through a second script.
- IF: code for Isolation Forest, both lag and no lag variants, with a notebook for event detection each.
- MBKM: code for MiniBatch K-Means, both lag and no lag variants, with a notebook for event detection each.
- LSTMAE: code for LSTM Autoencoder, event detection plots are in the same script.

### Notes
- The scripts/notebooks are the final versions, there were many edits/ scripts and other code tests before that are not responsible for final results and therefore not included here.
- final results(output of each script) are not included here, but these were presented in the bachelor's thesis, only the notebooks retain their output.
- due to the long process and working on different models when trying to group the relevant code for each results, I was sometimes confused on what code produced which result exactly. I tried to find the exact relevant scripts and hopefully I found the right ones but it is possible that some of the code included here is an older edit or a newer edit(test) than what was used for the final results.
- Each model has a main script, the script that does the hyperparameter search with optuna, but for some models there are two scripts. The second is a continuation of the main script and uses the hyperparameters produced by it. This is done to create relevant plots or because the main script crashed at the end and didn't train the final model so a new script was made to continue.
- Main scripts mostly follow the same style/pipeline, changes are mostly due to memory limits.
- Not all plots that are produced using this code were used in the final results.
- All scripts should be ready to run once the paths are matched correctly.
