class DeDupeResult {
    merged_results string[]
}

// Creating a function to extract the Result from a string.
function ExtractDeDupe(graph: string) -> DeDupeResult {
  client Ollama
  prompt #"
    Extract from this content:
    {{ graph }}

    {{ ctx.output_format }}
  "#
}

test TestDupe {
  functions [ExtractDeDupe]
  args {
    graph #"
      ['Mid-Size Business', 'Mid-Sized Business']
    "#
  }
}


test TestDupe2 {
  functions [ExtractDeDupe]
  args {
    graph #"
      ['Credit Limit']
    "#
  }
}
