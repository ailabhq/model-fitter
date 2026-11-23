# Glossary

**Source model**: The upstream checkpoint as published by its authors (e.g. on Hugging Face), before any change for a particular machine.

**Device profile**: What a target machine offers to run a model: its memory, how much of that memory the accelerator may use, free disk, and storage and compute speed.

**Fit plan**: The decision of how a Source model will run on a Device profile: precision per part of the model, which parts stay in memory and which are read from disk, and which runtime executes it.

**Fit**: The runnable result of carrying out a Fit plan: the converted weights plus everything needed to start the model, together with its measured speed and quality loss.

**Usable**: A Fit that generates at least 5 tokens per second with at least 8K context while the machine stays responsive.

**Slow mode**: An explicit opt-in that accepts a Fit below the Usable bar, trading speed for running at all.

**Quality loss**: The measured difference in output quality between a Fit and its Source model. Every Fit reports it.
