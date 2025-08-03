class SimpleNode {
  id string
  type string
  properties Properties
}

class Properties {
  description string?
}

class SimpleRelationship {
    source_node_id string
    source_node_type string
    target_node_id string
    target_node_type string
    type string
    properties Properties
}

class DynamicGraph {
    nodes SimpleNode[]
    relationships SimpleRelationship[]
}

// Creating a function to extract the DynamicGraph from a string.
function ExtractGraph(graph: string) -> DynamicGraph {
  client Ollama
  prompt #"
    Extract from this content:
    {{ ctx.output_format }}

    {{ graph }}

  "#
}

test TestGraph {
  functions [ExtractGraph]
  args {
    graph #"
      Here is the extracted information in the correct format:\n\n**Nodes:**\n\n1. **Elon Musk** (person) - Entrepreneur and business magnate\n2. **Open AI** (organization) - Artificial intelligence research organization\n\n**Relationships:**\n\n1. **SUED** (Elon Musk, Open AI) - Elon Musk sued Open AI
    "#
  }
}
