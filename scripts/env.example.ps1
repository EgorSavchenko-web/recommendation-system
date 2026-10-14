# Copy to scripts/env.ps1 (git-ignored), fill in, then: . .\scripts\env.ps1
$env:YAMBDA_CLEARML_PROJECT = "TLab-YambdaRL"
$env:YAMBDA_DATASET_ID = "<clearml dataset id>"
$env:YAMBDA_CLEARML_QUEUE = "<default queue>"
# Reach the ClearML server directly even when a system proxy is configured.
$env:NO_PROXY = ".innopolis.university"
$env:no_proxy = ".innopolis.university"
