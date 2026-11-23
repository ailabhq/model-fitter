# Glossary

**Source model**: The upstream checkpoint as published by its authors (e.g. on Hugging Face), before any change for a particular machine.

**Device profile**: What a target machine offers to run a model: its memory, how much of that memory the accelerator may use, free disk, and storage and compute speed.

**Fit plan**: The decision of how a Source model will run on a Device profile: precision per part of the model, which parts stay in memory and which are read from disk, and which runtime executes it.

**Fit**: The runnable result of carrying out a Fit plan: the converted weights plus everything needed to start the model, together with its measured speed and quality loss.

**Usable**: A Fit that generates at least 5 tokens per second with at least 8K context while the machine stays responsive.

**Responsive**: The machine stays responsive while a Fit runs: the desktop and the app in front keep answering without hangs, and memory pressure is only rarely critical. Background apps may be swapped out.

**Slow mode**: An explicit opt-in that accepts a Fit below the Usable speed, trading speed for running at all. It never accepts a Fit that leaves the machine unresponsive.

**Quality loss**: The measured difference in output quality between a Fit and its Source model, taken against the Source's Reference pack. Every Fit reports it; a Fit whose Quality loss cannot be measured is not produced.

**Reference pack**: The Source model's top-k next-token distributions over a fixed corpus, computed off the device once per Source model revision. For a Source model published already quantized, the published checkpoint is the reference.

**Quality band**: The grade a Fit's Quality loss earns: Near-lossless, Lossy, or Rejected.

**Degraded mode**: An explicit opt-in that accepts a Rejected Fit, trading quality for running at all. A Rejected Fit is still a Fit.
