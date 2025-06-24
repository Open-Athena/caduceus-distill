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
!curl -sSL https://raw.githubusercontent.com/Open-Athena/caduceus-distill/{BRANCH_NAME}/bin/notebook_bootstrap | bash -s {BRANCH_NAME}

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

## GCloud SDK Auth in Kaggle Notebook

Start by setting up: "Add-ons/Google Cloud SDK" (1 time operations).

As the first cell:
```
from kaggle_secrets import UserSecretsClient

# Note: Requires a prior setup of "Add-ons/Google Cloud SDK"
UserSecretsClient().set_gcloud_credentials(project="caduceus-distill")
```

To test in the notebook:
```
%%sh
# Test
gcloud storage ls gs://cadu-distill/
# Note: `gsutil ls gs://cadu-distill/` will yield the following error:
# Anonymous caller does not have storage.objects.list access to the Google Cloud Storage bucket. Permission 'storage.objects.list' denied on resource (or it may not exist)
```

## Running the code:

Then to run the *inference* code:
```
!uv run src/caduceus_inf.py data/hg38/hg38.ml.fa inf_output --max-batches=10 --chunk-size=65536
```


Then to run the *distillation* code:
```
!uv run src/caduceus_distillation.py inf_output --no_wandb
```

To run the **eval** code:
```
!uv run src/caduceus_nt_eval.py
```
