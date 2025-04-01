#!/bin/bash

# Define the command parameters
AGENT_CONFIG="config/agents.yaml"
TASK_CONFIG="config/tasks.yaml"
EMBEDDER_CONFIG="config/embedding.yaml"
SOURCE_TEXT="Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function."

# Loop 10 times
for i in {1..10}
do
  echo "🔁 Run #$i"
  structsense-cli extract \
    --agentconfig "$AGENT_CONFIG" \
    --taskconfig "$TASK_CONFIG" \
    --embedderconfig "$EMBEDDER_CONFIG" \
    --source "$SOURCE_TEXT"
  echo "✅ Completed Run #$i"
  echo "-------------------------------------------"
done

