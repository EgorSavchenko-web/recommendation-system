# Copy to scripts/env.ps1 (git-ignored), fill in, then: . .\scripts\env.ps1
$env:YAMBDA_CLEARML_PROJECT = "TLab-YambdaRL"
$env:YAMBDA_DATASET_ID = "<clearml dataset id>"
$env:YAMBDA_CLEARML_QUEUE = "<default queue>"
# Optional: base image for the agent (default: nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04)
# $env:YAMBDA_CLEARML_DOCKER = "nvidia/cuda:13.4.2-tensorrt-runtime-ubuntu24.04"
# Reach the ClearML server directly even when a system proxy is configured.
$env:NO_PROXY = ".innopolis.university"
$env:no_proxy = ".innopolis.university"
