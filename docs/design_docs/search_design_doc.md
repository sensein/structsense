# Search Design Document

## Task
Given a query $q$, the system aims to retrieve the most relevant concept or document $d$ from a candidate set $\mathcal{D}$. In our context, this task necessitates identifying the relevant ontological concepts to a given entity (e.g., in Named Entity Recognition scenarios) or to a broader text input, enabling semantic grounding and structured representation of the extracted information.

## Current Situation

Currently, we use **BioPortal** as our ontology database. BioPortal is a well-established, community-trusted platform that hosts and manages a large number of ontologies. However, these benefits come with several trade-offs:

1. **Dependency on BioPortal** — If BioPortal is unavailable (for example, during upgrades), our use case is directly impacted.  
2. **API rate limits** — Rate limiting can slow down API calls. While this is understandable given BioPortal’s design and shared usage model, it affects performance.  
3. **Implementation dependency** — We rely on BioPortal’s implementations (e.g., search), which may not always be optimal or fully aligned with our specific use case.
