# Glossary

**Source model**: The upstream checkpoint as published by its authors (e.g. on Hugging Face), before any change for a particular machine.

**Device profile**: What a target machine offers to run a model: its memory, how much of that memory the accelerator may use, free disk, and storage and compute speed.

**Fit plan**: The decision of how a Source model will run on a Device profile: precision per part of the model, which parts stay in memory and which are read from disk, and which runtime executes it.

**Fit**: The runnable result of carrying out a Fit plan: its weights, converted by sarto or taken from a Community build, plus everything needed to start the model, together with its measured speed and quality loss.

**Candidate**: A set of weights a Fit plan measures on a Device profile, either a Community build or one converted by sarto. A Fit's weights are the candidate its plan run chose.

**Plan run**: One execution of a Fit plan on a machine, from listing its candidates to a Fit or a refusal.

**Community build**: A quantized build of a Source model published by someone other than its authors (e.g. unsloth, ggml-org). It is never a Source model: its Quality loss is measured against the Source's Reference pack, like any other Fit's.

**Usable**: A Fit that generates at least 5 tokens per second with at least 8K context while the machine stays responsive.

**Responsive**: The machine stays responsive while a Fit runs: the desktop and the app in front keep answering without hangs, and memory pressure is only rarely critical. Background apps may be swapped out.

**Stall**: A hang during a Fit's benchmark while memory pressure never reaches critical. Unlike a memory failure, it is no evidence about other candidates.

**Slow mode**: An explicit opt-in that accepts a Fit below the Usable speed, trading speed for running at all. It never accepts a Fit that leaves the machine unresponsive.

**Source-precision Fit**: A Fit that runs the Source model's own weights, unchanged, and quantizes nothing at run time either. Its Quality loss is still measured, and what remains is runtime noise, never zero. Asking for one constrains the Fit plan; when the result is below Usable speed, asking for it also opts into Slow mode.

**Batch throughput**: Tokens generated per hour across several sequences decoded together. This is the speed that matters for unattended work such as overnight batches, where single-sequence tokens per second does not.

**Quality loss**: The measured difference in output quality between a Fit and its Source model, taken against the Source's Reference pack on the Source's own responses, never on the prompts it was given. Every Fit reports it; a Fit whose Quality loss cannot be measured is not produced.

**Reference pack**: The Source model's top-k next-token distributions over its own responses to the Prompt set, with the prompts as context, together with how much each of its weights matters on the Calibration set, computed off the device once per Source model revision, Prompt set version and Calibration set version. For a Source model published already quantized, the reference is its published weights, dequantized and computed at full precision; quantizing activations is a runtime choice and counts as Quality loss.

**Prompt set**: The fixed, versioned prompts shared by every Source model, across chat, code, multilingual and tool calls, plus one 8K-context prompt. Each Reference pack adds the Source's own responses to it.

**Calibration set**: The fixed, versioned text a Source model is run on to measure which of its weights matter most before sarto quantizes it. It never overlaps the Prompt set, so Quality loss is never scored on text the quantization was tuned to.

**Quality band**: The grade a Fit's Quality loss earns: Near-lossless, Lossy, or Rejected.

**Degraded mode**: An explicit opt-in that accepts a Rejected Fit, trading quality for running at all. A Rejected Fit is still a Fit.
