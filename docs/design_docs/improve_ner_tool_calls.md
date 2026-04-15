# Improve NER Extraction by Using Tool Calls Instead of Solely Relying on LLM Prompts

## Overview

The current Named Entity Recognition (NER) pipeline relies primarily on large language model (LLM) prompts to extract entities from text.
While flexible, this approach can lead to inconsistent results and limited domain precision. This design proposes enhancing NER extraction by incorporating tool calls to domain-specific models and specialized NER tools by the `extractor agent`.

## Goals

- Improve accuracy and consistency of entity extraction, especially for domain-specific entities
- Reduce over-reliance on prompt engineering
- Enable parallel execution of multiple NER models
- Produce a unified, reconciled entity output
- Incorporate the provenance
## Proposed Approach
Figure below shows the specialized tool for NER
![](ner_tool.png)

### 1. Tool-Based NER Execution

For each chunk, the system invokes multiple domain-specific NER tools or models in parallel via tool calls. Examples include:

- General-purpose NER models
- Domain-trained models (e.g., biobert_genetic_ner and NCBI-disease-WLT-256-SciBERT-13INS)


### 2. Entity Merging and Global Reconciliation

A post-processing layer merges outputs from all tools by:

- Normalizing entity formats and labels
- Deduplicating overlapping entities
- Resolving conflicts using the weighted majority voting
- Reconciling entities globally across chunks to ensure consistency

### 3. LLM as Orchestrator

The `extractor agent` agent coordinates the workflow by:

- Detects tasks dynamically
- Selecting which tools to execute

## Expected Benefits

- Higher precision and recall for NER, especially in specialized domains
- More robust and explainable extraction pipeline with provenance information


## Risks and Mitigations

- **Conflicting model outputs**: Mitigated via weighted majority voting where domain specific models will have higher say.

## Input/Output
### Input: raw text
These findings highlight that effective amyloid removal depends on the engagement of microglia through the Fc fragment, providing critical insights for optimizing anti-amyloid therapies in Alzheimers disease.', 'Lecanemab, an antibody engineered to target soluble amyloid \u03b2-amyloid (A\u03b2) protofibrils 1 , effectively removes amyloid plaques from the brains of Alzheimers disease (AD) patients, slowing cognitive decline by 27 2.

### Output
```json
{
      "text": "amyloid plaques",
      "label": "disease",
      "start": 1283,
      "end": 1298,
      "weighted_score": 1.0,
      "model_count": 1,
      "occurrences": [
        {
          "start": 324,
          "end": 339,
          "global_start": 1283,
          "global_end": 1298,
          "sentence": "These findings highlight that effective amyloid removal depends on the engagement of microglia through the Fc fragment, providing critical insights for optimizing anti-amyloid therapies in Alzheimers disease.', 'Lecanemab, an antibody engineered to target soluble amyloid \u03b2-amyloid (A\u03b2) protofibrils 1 , effectively removes amyloid plaques from the brains of Alzheimers disease (AD) patients, slowing cognitive decline by 27 2 ."
        }
      ],
      "provenance": [
        {
          "label": "disease",
          "vote_weight": 1.0,
          "sources": [
            {
              "source_model": "NCBI-disease",
              "weight": 1.0,
              "text": "amyloid plaques"
            }
          ]
        }
      ]
    }
```
