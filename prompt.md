# Experiment Prompt: ROCm Bug Diagnosis on Strix Halo (gfx1151)

## Primary Goal
Read program.md and kick off a new experiment run, but first start a new branch mar19-dsstrix2, update all uv pip packages possible in a stable manner, read https://github.com/ROCm/ROCm/issues/6034 and optimize the experiment so it confirms (or contradicts) and extends the reported problems, and try to find other issues around ROCm and document them properly in this repo. Commit and push your findings after every experiment and I will send PR to them with the results. Do not stop, continuously experiment to diagnose and properly localize any issues found, repeat if necessary, etc.

## Additional Goals
- When talking about ROCm issues, look around and find similar matching issues and fixes and workarounds at https://github.com/ROCm/ROCm/issues/ then link those issues to the bugreports in the findings.md too.
- Our goal is to contribute testing and compute to issue #6034 primarily, but with maximum technical clarity.
- Trying to figure out causes and root causes and references to other issues or reproduction parameters is all within scope.
- Try to be as useful as possible: identify exact reproduction parameters, narrow down failure boundaries, cross-reference with related ROCm issues, suggest potential fixes or workarounds.
