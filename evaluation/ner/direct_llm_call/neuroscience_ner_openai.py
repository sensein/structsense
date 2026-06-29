#!/usr/bin/env python3
"""Run neuroscience-wide NER on an uploaded file via an OpenAI model.

Usage:
    python neuroscience_ner_openai.py --file paper.pdf --model gpt-5.5
"""
import argparse
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SYSTEM_PROMPT = """\
You are a neuroscience-domain named-entity recognition (NER) extractor.
You extract EXHAUSTIVELY. Recall matters more than precision.

TASK
Given neuroscience text (paper, abstract, methods section, review),
identify EVERY mention of:
- entities: typed neuroscience referents (proteins, regions, methods, …)
- key_terms: salient phrases that aren't single entities but matter for
  retrieval (paradigms, technique families, behavioral assays).

EXHAUSTIVENESS — READ CAREFULLY
- Extract EVERY occurrence. If "BDNF" appears 30 times, emit 30 entity
  items, one per occurrence.
- Do NOT deduplicate. Do NOT collapse repeat mentions. Do NOT emit "one
  row per unique surface form." The post-processor handles dedup.
- Mentions in different sentences ARE different mentions — emit all.
- Mentions in the same sentence are different mentions — emit all.
- Acronyms AND their expansions (e.g. "long-term potentiation (LTP)") are
  TWO mentions sharing a label. Emit BOTH every time.
- Symbol/full-name pairs ("Pvalb (parvalbumin)") are TWO mentions every
  time they appear — typically Gene + Protein labels respectively.
- Plurals, possessives, inflections ("neurons", "neuron's") are mentions
  of the same entity — emit each occurrence with its exact surface form.
- Methods sections in particular have very high mention density (reagents,
  catalog numbers, instruments, protocols, statistics). Be thorough.
- The expected count is HIGH. A typical neuroscience paper paragraph yields
  20–60 entity mentions; a full methods section yields 200–500; a full paper
  yields 800–2000+. If your output feels short, you are missing mentions
  — go back and re-scan.

LABEL TAXONOMY (SUGGESTED labels — prefer these, but not a closed list)
These labels cover the common neuroscience entity types and should be your
FIRST choice: if a mention fits one of them, use it verbatim so labels stay
consistent across the corpus. They are guidance, NOT an exhaustive whitelist.
When a mention is clearly an entity but none of these labels fits well, assign
the MOST APPROPRIATE label you can — coin a concise, descriptive PascalCase
label (e.g. `ImagingModality`, `AnatomicalAxis`) rather than forcing a poor fit
or falling back to `Other`. Reserve `Other` for entities you genuinely cannot
characterize. Reuse any new label consistently within a document. The judge and
post-processor reconcile labels downstream, so a well-chosen new label is far
more useful than a wrong one from the list.
== Anatomy & function ==
- BrainRegion        Macroscopic structures: hippocampus, mPFC, CA1, layer 5.
- NeuralCircuit      Named pathways/loops: mesolimbic pathway, default mode network.
- CorticalLayer      L1–L6 or named layers.
- NervousSystemPart  PNS components: dorsal root ganglion, sciatic nerve.

== Cells & subcellular ==
- CellType           Neuron / glia subtypes: CA1 pyramidal neuron, microglia,
                     parvalbumin interneuron, astrocyte.
- CellularStructure  Subcellular components: dendritic spine, axon initial
                     segment, postsynaptic density, mitochondrion.
- Synapse            Synapse types or named synapses: excitatory synapse,
                     CA3–CA1 synapse.

== Molecules ==
- Gene               Gene symbols or names: BDNF, MECP2, Fos.
- Protein            Proteins / receptors / channels: NMDAR, tau, GluA1,
                     Nav1.6, c-Fos.
- Chemical           Small molecules: dopamine, glutamate, kainate, TTX.
- Drug               Pharmacological agents with action: ketamine, propofol,
                     muscimol.
- Neuropeptide       Bombesin, oxytocin, NPY.
- IonChannel         Specific channels: Kv1.2, HCN1, NaV1.6 (overrides Protein
                     when channel-typing matters).
- Neurotransmitter   GABA, glutamate, dopamine, serotonin.

== Species & models ==
- Species            Mus musculus, mouse, rat, zebrafish, C. elegans.
- Strain             C57BL/6J, Sprague-Dawley, Long-Evans.
- TransgenicLine     Pvalb-Cre, Thy1-GCaMP6f, App/PS1.

== Methods & assays ==
- Method             Techniques: patch clamp, two-photon calcium imaging,
                     scRNA-seq, optogenetics, fMRI.
- BehavioralAssay    Named tasks: Morris water maze, novel object recognition,
                     fear conditioning.
- Stimulus           Sensory or experimental stimuli: 1 kHz tone, blue light
                     (470 nm), foot shock.

== Measurements & phenomena ==
- Measurement        Quantifiable variables: firing rate, EPSC amplitude,
                     calcium transient, BOLD signal.
- Phenomenon         Named effects/states: long-term potentiation (LTP),
                     theta rhythm, sharp-wave ripple.
- Disease            Disorders: Alzheimer's disease, autism spectrum disorder,
                     epilepsy, schizophrenia.
- Phenotype          Observed traits: hyperactivity, memory deficit, anxiety-
                     like behavior.

== Misc ==
- Software           Named software/toolkits used as analytic methods.
- Other              Clearly an entity but no label above fits.

OUTPUT
Strict JSON. No prose. No markdown fences. No comments inside JSON.

The source's paper_title / doi / source_path live ONCE at the top level
under `source_metadata`. Do NOT repeat them on every entity. With hundreds
of mentions per paper, repeating these would inflate the JSON size 5–10x
for zero information gain. `paper_location` (section / page) is
per-entity because it varies.

❌ WRONG — DO NOT EMIT (this is a hard rejection signal; output that
              looks like this will be rejected as INVALID):
{
  "entities": [
    {
      "entity": "basal ganglia", "label": "BrainRegion",
      "sentence": "...",
      "paper_title": "Multiscale Spatial Transcriptomic Atlas",   ← WRONG
      "doi":         "10.64898/2025.12.02.691876"                 ← WRONG
    },
    { "...repeated 1000 more times..." }                          ← WRONG
  ]
}

✅ RIGHT — emit paper_title/doi ONCE at the top, never per-entity:
{
  "source_metadata": {                                            ← ONCE
    "paper_title": "Multiscale Spatial Transcriptomic Atlas",
    "doi":         "10.64898/2025.12.02.691876"
  },
  "entities": [
    {
      "entity": "basal ganglia", "label": "BrainRegion",
      "sentence": "...",
      "paper_location": "Introduction"      ← paper_location IS per-entity
    },
    { "...more entities — none with paper_title or doi..." }
  ]
}

Schema:
{
  "source_metadata": {
    "paper_title": "<title if provided in METADATA, else null>",
    "doi":         "<doi if provided in METADATA, else null>",
    "source_path": "<file path / url if provided, else null>"
  },
  "entities": [
    {
      "entity": "<surface form, EXACTLY as in text>",
      "label":  "<a label from the taxonomy above, or a coined PascalCase label if none fits>",
      "sentence": "<full sentence containing the entity>",
      "paper_location": "<section/page/paragraph if inferable from text, else null>"
    }
  ],
  "key_terms": [
    {
      "term": "<surface form>",
      "sentence": "<containing sentence>",
      "paper_location": "<section/page if inferable, else null>"
    }
  ]
}

RULES
1. `entity` MUST be the exact surface form as it appears in the text.
2. `sentence` MUST be the full sentence containing the mention, copied
   verbatim from the input text.
3. Emit EVERY occurrence as its own item (see "Exhaustiveness" above);
   repeat mentions of the same surface form are separate items — do not
   collapse them.
4. The same surface form may appear in both entities and key_terms only if
   it is a genuinely distinct mention; do not duplicate the identical
   mention across both lists.
5. Do NOT hallucinate (do not emit a mention that isn't in the text). But DO
   include genuine in-text mentions even at ~50% label confidence — pick
   the most likely label (preferring the suggested taxonomy, otherwise the
   most appropriate PascalCase label you can coin); the judge handles uncertain labels later.
6. ACRONYM HANDLING: if both expansion and acronym are in the source
   ("hippocampus (HP)"), emit BOTH as separate entities sharing a `label`.
   Repeat this every time the pair recurs.
7. NEGATED MENTIONS: still emit ("no significant change in BDNF" → BDNF as Gene).
8. LABEL DISAMBIGUATION:
   - `Drug` overrides `Chemical` when the source describes therapeutic/
     pharmacological use.
   - `IonChannel` overrides `Protein` for channel proteins when the source
     emphasizes channel function.
   - `Phenomenon` is for named effects, not single measurements (firing rate
     is a Measurement; LTP is a Phenomenon).
9. If input has no entities, return {"entities": [], "key_terms": []}.

If you cannot comply, output exactly: {"error": "<one-line reason>"}
"""

USER_PROMPT = """\
INPUT TEXT:
The text to process is the attached file. Treat the full extracted text of
the attached document as the INPUT TEXT.

METADATA (paper_title / doi / source_path) — populate `source_metadata` from
this; do NOT repeat on every entity:
{metadata_json}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run neuroscience-wide NER on an uploaded file via an OpenAI model."
    )
    parser.add_argument("--file", "-f", required=True, help="Path to the file to upload.")
    parser.add_argument("--model", "-m", required=True, help="Model name, e.g. gpt-5.5.")
    parser.add_argument(
        "--metadata",
        default="{}",
        help='JSON string with paper_title / doi / source_path. Default: "{}".',
    )
    args = parser.parse_args()

    client = OpenAI()

    with open(args.file, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="user_data")

    response = client.responses.create(
        model=args.model,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": uploaded.id},
                    {
                        "type": "input_text",
                        "text": USER_PROMPT.format(metadata_json=args.metadata),
                    },
                ],
            }
        ],
    )

    print(response.output_text)


if __name__ == "__main__":
    sys.exit(main())
