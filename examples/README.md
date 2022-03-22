# WeaSEL examples 
This library assumes familiarity with (multi-source) weak supervision, if that's not the case you may want to first
learn its basics in e.g. [this overview slides from Stanford](http://cs229.stanford.edu/notes2019fall/weak_supervision_slides.pdf) or [this Snorkel tutorial](https://www.snorkel.org/use-cases/01-spam-tutorial).

That being said, these examples and notebooks will show you how to use Weasel for your
own dataset, LF set, or end-model. E.g.:

- [***A high-level starter tutorial***](./1_bias_bios.ipynb), with few code, many explanations
 and including Snorkel as a baseline (so that if you are familiar with Snorkel you can see the similarities and differences to Weasel).

- [***See how the whole WeaSEL pipeline works with all details, necessary steps and definitions for a new dataset & custom end-model***](./0_full_pipeline.ipynb). 
    This notebook will probably make you learn the most about WeaSEL and *how to apply it to your own problem*.
    
- [***A realistic ML experiment script***](./1_bias_bios_full.py) with all that's part of a ML pipeline, including logging to Weight&Biases,
arbitrary callbacks, and eventually retrieving your fully trained end-model.

#### Without Hydra dependency:
Check [this notebook](1_bias_bios_no_hydra.ipynb) and/or [this script](0_full_pipeline_no_hydra.py) out.
:
