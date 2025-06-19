# Notebook setup

🔪 Note 🔪: To have access to accelerators on Kaggle, you need to verify your phone number.

## Notebook setup:

On the first cell, run:
```python
import os

# --- Configuration ---
BRANCH_NAME = "main" # or "my-feature-branch"
# ---------------------

print(f"Setting up environment from branch: {BRANCH_NAME}")
!curl -sSL https://raw.githubusercontent.com/tabtab-labs/caduceus-distill/{BRANCH_NAME}/bin/notebook_bootstrap | bash -s {BRANCH_NAME}

# Change directory into the repo
BASE_DIR = '/kaggle/working' if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else '/content'
%cd {BASE_DIR}/caduceus-distill
print(f"Changed directory to: {os.getcwd()}")
```

Hint: To edit a file in Kaggle, create a new cell in the notebook and run:
```
%%writefile /kaggle/working/caduceus-distill/src/caduceus_inf.py

<CONTENTS OF YOUR PYTHON SCRIPT>
```

## Running the code:

Then to run the *inference* code:
```
!uv run src/caduceus_inf.py data/hg38/hg38.ml.fa inf_output
```


Then to run the *distillation* code:
```
!uv run src/caduceus_distillation.py inf_output
```
