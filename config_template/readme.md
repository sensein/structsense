## Overview

This folder contains configuration files used to run different tasks. It includes a base template that can be adapted as needed, along with task-specific configuration files.

### Contents
- `config.yaml`: a configuration template
- `ner-config.yaml`: a configuration file that can be used for the NER task
- `resource-extraction-config.yaml`: a configuration file that can be used for the resource extraction task
- `pdf2_reproschema.yaml`: Converts a survey questionnaire PDF into JSON-LD. See [Pdf2ReproSchema](../example/pdf2_reproschema) for details. Generated files are written to the [outputs](outputs) directory. The input PDF used for this workflow is located in the `Pdf2ReproSchema` directory.
