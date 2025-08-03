<!-- omit in toc -->
# Graph RAG using Fuzzy Parsing

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![LangChain](https://img.shields.io/badge/LangChain-Framework-purple)](https://www.langchain.com/) [![Ollama](https://img.shields.io/badge/Ollama-Local--LLMs-green)](https://ollama.com/) [![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20DBms-blueviolet)](https://neo4j.com/) [![Medium](https://img.shields.io/badge/Medium-Blog-black?logo=medium)](https://medium.com/@fareedkhandev/improving-langchain-knowledge-graph-rag-using-fuzzy-parsing-a2413b2a4613)


One of the most common challenges when building knowledge graph-based RAG systems or agents with LangChain is the inability to reliably extract nodes and relationships from unstructured data especially when using smaller, quantized local LLMs. This often results in poor performance of your AI product.

![Comparison between Langchain and baml](https://cdn-images-1.medium.com/max/1500/1*oZBsKnxaqw0dVBkG_V1KgA.png)


A key issue with LangChain extraction capabilities is its reliance on strict JSON parsing, which can fail even when using larger models or highly detailed prompt templates.

> In contrast, **BAML** uses a **fuzzy parsing** approach that can extract data even when the LLM output isn’t perfectly formatted as JSON.

In this blog, we will explore the limitations of LangChain extraction when using smaller quantized models, and will see how BAML can improve extraction success rates from around **25% to over 99%**.

This blog is created on top of [BoundaryML BAML](https://github.com/BoundaryML/baml).

All the code is available in my GitHub Repo:

<!-- omit in toc -->
## Table of Contents
- [Initializing Eval Dataset](#initializing-eval-dataset)
- [Quantized Small LLaMA Model](#quantized-small-llama-model)
- [LLMGraphTransformer based Approach](#llmgraphtransformer-based-approach)
- [Understanding the Issue with LangChain](#understanding-the-issue-with-langchain)
- [Will improving the Prompt work?](#will-improving-the-prompt-work)
- [Initialization and Quick Overview of BaML](#initialization-and-quick-overview-of-baml)
- [Integrating BAML with LangChain](#integrating-baml-with-langchain)
- [Running the BAML Experiment](#running-the-baml-experiment)
- [Building and Analyzing the GraphRAG](#building-and-analyzing-the-graphrag)
- [Finding and Linking Similar Entities](#finding-and-linking-similar-entities)
- [Community Detection with Leiden Algorithm](#community-detection-with-leiden-algorithm)
- [Analyzing the Final Graph Structure](#analyzing-the-final-graph-structure)
- [Conclusion](#conclusion)

## Initializing Eval Dataset
To understand the problem and its solution, we need to have evaluation data on which we can perform several tests to understand how BAML is improving our LangChain knowledge graphs.

We will be using [blog datasets](https://github.com/tomasonjo/blog-datasets) from [Tomasonjo](https://github.com/tomasonjo), available on GitHub, so let’s load that data first.

```python
# Import the pandas library for data manipulation and analysis
import pandas as pd

# Load the news articles dataset from a CSV file hosted on GitHub into a pandas DataFrame
news = pd.read_csv(
    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
)

# Display the first 5 rows of the DataFrame
news.head()
```

| #   | Title               | Date               | Text               |
|-----|---------------------|--------------------|--------------------|
| 0   | Chevron: Best Of... | 2031-04-06T01:36... | JHVEPhoto Like many... |
| 1   | FirstEnergy (NYSE...| 2030-04-29T06:55... | FirstEnergy (NYSE:FE... |
| 2   | Dáil almost sus...  | 2023-06-15T14:32... | The Dáil was...        |
| 3   | Epic’s latest to... | 2023-06-15T14:00... | Today, Epic is...      |
| 4   | EU to Ban...        | 2023-06-15T13:50... | The European Commission... |

Our DataFrame is pretty simple (a title with text, which is the description of the news). We need one more column that will contain the total number of tokens corresponding to the text of the news article.

For that we can use the `tiktoken` library from OpenAI to do this. It's straightforward to calculate tokens for our dataset using a loop, so let's do that.

```python
# Import the tiktoken library to count tokens from text
import tiktoken

# Define a function to calculate the number of tokens in a given string for a specific model
def num_tokens_from_string(string: str, model: str = "gpt-4o") -> int:
    """Returns the number of tokens in a text string."""
    # Get the encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)
    # Encode the string into tokens and count them
    num_tokens = len(encoding.encode(string))
    # Return the total number of tokens
    return num_tokens

# Create a new column 'tokens' in the DataFrame
# It calculates the number of tokens for the combined 'title' and 'text' of each article
news["tokens"] = [
    num_tokens_from_string(f"{row['title']} {row['text']}")
    for i, row in news.iterrows()
]
```
It only takes a few seconds to calculate the tokens for our DataFrame.

Here’s the updated version of the DataFrame.

```python
# Display the first 5 rows of the DataFrame to show the new 'tokens' column
news.head()
```
| #   | Title               | Date               | Text                  | Tokens |
|-----|---------------------|--------------------|------------------------|--------|
| 0   | Chevron: Best Of... | 2031-04-06T01:36... | JHVEPhoto Like many... | 78  |
| 1   | FirstEnergy (NYSE...| 2030-04-29T06:55... | FirstEnergy (NYSE:FE...| 130 |
| 2   | Dáil almost sus...  | 2023-06-15T14:32... | The Dáil was...         | 631 |
| 3   | Epic’s latest to... | 2023-06-15T14:00... | Today, Epic is...       | 528 |
| 4   | EU to Ban...        | 2023-06-15T13:50... | The European Commission... | 281 |

These tokens can later be used in our evaluation and analysis phase, which is why we performed this step.

## Quantized Small LLaMA Model
To convert the data into knowledge graphs using AI models, a low level quantized model will be used to perform a hard evaluation of the test.

> In production, open source LLMs are often deployed in quantized form to reduce cost and latency. For this blog, LLaMA 3.1 is used.

Ollama is the chosen platform here, but LangChain supports many API and local LLM providers, so any suitable option can be selected.

```python
# ChatOllama is an interface to Ollama's language models
from langchain_ollama import ChatOllama

# Define the model name to be used
model = "llama3"

# Initialize the ChatOllama language model
# The 'temperature' parameter controls the randomness of the output.
# A low value (e.g., 0.001) makes the model's responses more deterministic.
llm = ChatOllama(model=model, temperature=0.001)
```
You also need to install Ollama on your system. It is available for macOS, Windows, and Linux.

1.  Go to the official Ollama website: [https://ollama.com/](https://www.google.com/url?sa=E&q=https%3A%2F%2Follama.com%2F)
2.  Download the installer for your operating system and follow the installation instructions.

After installation, Ollama runs as a background service.
- On **macOS and Windows**, the application should start automatically and run in the background (you might see an icon in your menu bar or system tray).
- On Linux, you may need to start it manually with `systemctl start ollama`.

To check if the service is running, open your terminal or command prompt and type:
```bash
# Checking available models
ollama list

#### OUTPUT ####
[ ] <-- No models yet
```
If it’s running but you have no models yet, you’ll see an empty list of models, which is perfectly fine at this stage. If you get a “command not found” error, make sure Ollama was installed correctly. If you get a connection error, the server isn’t running.

You can simply download llama3 using pull command. This will take some time and several gigabytes of disk space, as the models are large.

```bash
# Downloading llama3 model
ollama pull llama3
```
After these commands finish, you can run `ollama list` again, and you should now see model listed.

```bash
# Send a request to the local Ollama API to generate text
curl http://localhost:11434/api/generate \
    # Set the Content-Type header to indicate a JSON payload
    -H "Content-Type: application/json" \
    # Provide the data for the request
    -d '{
        "model": "llama3",
        "prompt": "Why is the sky blue?"
    }'

#### OUTPUT ####
{
  "model": "llama3",
  "created_at": "2025-08-03T12:00:00Z",
  "response": "The sky appears blue be ... blue.",
  "done": true
}
```
If it works, you will see a stream of JSON responses printed to your terminal. This confirms that the server is running and can serve the model.

> Now that the eval data and LLM are ready, the next step is to transform the data to better understand the problem within LangChain.

## LLMGraphTransformer based Approach
A proper way to transform raw or structured data into knowledge graphs using LangChain or LangGraph is by using their provided methods. One of the most common approaches is the `LLMGraphTransformer` found in the `langchain_experimental` library.

This tool is designed as an all-in-one solution: provide it with your text and an LLM, and it takes care of the prompting and parsing to return a graph structure.

Let’s see how it performs with our local `llama3` model.

First, we need to import all the necessary components.

```python
# Import the main graph transformer from Langchain's experimental library
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Import the data structures for graphs and documents
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
```
Now, let’s initialize the transformer. We will use the `llm` object we created earlier (which is our `llama3` model).

We also need to tell the transformer what extra information, or "properties" we want it to extract for our nodes and relationships. For this example, we'll just ask for a `description`.

```python
# Initialize the LLMGraphTransformer with our llama3 model
# We specify that we want a 'description' property for both nodes and relationships
llm_transformer = LLMGraphTransformer(
    llm=llm,
    node_properties=["description"],
    relationship_properties=["description"]
)
```
To make the process repeatable and clean, we’ll create a simple helper function. This function will take a string of text, wrap it in Langchain’s `Document` format, and then pass it to our `llm_transformer` to get the graph structure.

```python
# Import the List type for type hinting
from typing import List

# Define a function to process a single text string and convert it into a graph document
def process_text(text: str) -> List[GraphDocument]:
    # Create a Langchain Document object from the raw text
    doc = Document(page_content=text)
    # Use the transformer to convert the document into a list of GraphDocument objects
    return llm_transformer.convert_to_graph_documents([doc])
```
With everything set up, it’s time to run the experiment. To keep things manageable and highlight the core issue, a small sample of 20 articles from the dataset will be processed.

A `ThreadPoolExecutor` will be used to run the processing in parallel and speed up the workflow.

```python
# Import libraries for concurrent processing and a progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Set the number of parallel workers and the number of articles to process
MAX_WORKERS = 10
NUM_ARTICLES = 20

# This list will store the resulting graph documents
graph_documents = []

# Use a ThreadPoolExecutor to process articles in parallel
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit the processing task for each article in our sample
    futures = [
        executor.submit(process_text, f"{row['title']} {row['text']}")
        for i, row in news.head(NUM_ARTICLES).iterrows()
    ]

    # As each task completes, get the result and add it to our list
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing documents"
    ):
        graph_document = future.result()
        graph_documents.extend(graph_document)
```
After running the code, the progress bar shows that all 20 articles were processed.
```bash
#### OUTPUT ####
Processing documents: 100%|██████████| 20/20 [01:32<00:00,  4.64s/it]
```

## Understanding the Issue with LangChain
So, what did we get? Let’s inspect the `graph_documents` list.

```python
# Display the list of graph documents
print(graph_documents)
```
This is the output we get:
```bash
#### OUTPUT ####
[GraphDocument(nodes=[], relationships=[], source=Document(metadata={}, page_content='XPeng Stock Rises...')),
 GraphDocument(nodes=[], relationships=[], source=Document(metadata={}, page_content='Ryanair sacks chief pilot...')),
 GraphDocument(nodes=[], relationships=[], source=Document(metadata={}, page_content='Dáil almost suspended...')),
 GraphDocument(nodes=[Node(id='Jude Bellingham', type='Person', properties={}), Node(id='Real Madrid', type='Organization', properties={})], relationships=[], source=Document(metadata={}, page_content='Arsenal have Rice bid rejected...')),
 ...
]
```

> Immediately, we can spot a problem. Many of the `GraphDocument` objects have empty `nodes` and `relationships` lists.

This means that for those articles, the LLM either produced an output that Langchain couldn't parse into a valid graph structure, or it failed to extract any entities at all.

This is the core challenge with using smaller, quantized LLMs for structured data extraction. They often struggle to follow the strict JSON formatting that tools like `LLMGraphTransformer` expect. If there's even a tiny mistake a trailing comma, a missing quote the parsing fails, and we get nothing.

Let’s quantify this failure rate. We’ll count how many of our 20 documents resulted in an empty graph.

```python
# Initialize a counter for documents with no nodes
empty_count = 0

# Iterate through the generated graph documents
for doc in graph_documents:
    # If the 'nodes' list is empty, increment the counter
    if not doc.nodes:
        empty_count += 1
```
Now, let’s calculate the percentage of failures.
```python
# Calculate and print the percentage of documents that failed to produce any nodes
print(f"Percentage missing: {empty_count/len(graph_documents)*100}")


#### OUTPUT ####
Percentage missing: 75.0
```
A **75% failure rate**. That’s not good at all. This means that for our sample of 20 articles, only 5 were successfully converted into a knowledge graph.

> A success rate of just 25% is not acceptable for any production system.

This is where the problem lies, and it’s a common one. The standard approach is too rigid for the slightly unpredictable nature of smaller LLMs.

## Will improving the Prompt work?
A 75% failure rate is a huge problem. As developers, our first instinct when an LLM doesn’t perform well is often to tweak the prompt. Better instructions should lead to better results, right? The `LLMGraphTransformer` uses a default prompt internally, but we can't easily modify it.

So, let’s build our own simple chain using Langchain `ChatPromptTemplate`. This gives us full control over the instructions we send to `llama3`. We can be more explicit and try to "guide" the model into generating the correct JSON format every single time.

Let’s start by defining the output structure we want using Pydantic models. This is a common pattern in Langchain for structured output.

```python
# Import Pydantic models for defining data structures
from langchain_core.pydantic_v1 import BaseModel, Field

# Define a simple Node structure
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node.")
    type: str = Field(description="Type of the node (e.g., Person, Organization).")

# Define a simple Relationship structure
class Relationship(BaseModel):
    source: Node = Field(description="The source node of the relationship.")
    target: Node = Field(description="The target node of the relationship.")
    type: str = Field(description="The type of the relationship (e.g., WORKS_FOR).")

# Define the overall graph structure
class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="List of nodes in the graph.")
    relationships: List[Relationship] = Field(description="List of relationships in the graph.")
```
Next, we’ll create a new, more detailed prompt. This prompt will explicitly include the JSON schema generated from our Pydantic models and give very specific instructions to the LLM.

> The idea is to leave no room for error.

```python
# Import the prompt template and the output parser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser

# Create an instance of our desired output structure
parser = JsonOutputParser(pydantic_object=KnowledgeGraph)

# Create a detailed prompt template with explicit instructions
template = """
You are a top-tier algorithm for extracting information in structured formats.
Extract a knowledge graph from the given input text, consisting of nodes and relationships.
Your goal is to be as comprehensive as possible, extracting all relevant entities and their connections.

Format your output as a JSON object with 'nodes' and 'relationships' keys.
Adhere strictly to the following JSON schema:
{schema}

Here is the input text:
--------------------
{text}
--------------------
"""

prompt = ChatPromptTemplate.from_template(
    template,
    partial_variables={"schema": parser.get_format_instructions()},
)

# Create the full extraction chain
chain = prompt | llm | parser
```
This new chain is more explicit than the `LLMGraphTransformer`. We've given the model a detailed schema and clear instructions. Let's run our 20-article sample through it again and see if our success rate improves.

```python
# This list will store the new results
graph_documents_prompt_engineered = []
errors = []

for i, row in tqdm(news.head(NUM_ARTICLES).iterrows(), total=NUM_ARTICLES, desc="Processing with better prompt"):
    text = f"{row['title']} {row['text']}"
    try:
        # Invoke our new, improved chain
        graph_data = chain.invoke({"text": text})
        
        # Manually convert the parsed JSON back to GraphDocument format
        nodes = [Node(id=node['id'], type=node['type']) for node in graph_data.get('nodes', [])]
        relationships = [Relationship(source=Node(id=rel['source']['id'], type=rel['source']['type']),
                                      target=Node(id=rel['target']['id'], type=rel['target']['type']),
                                      type=rel['type']) for rel in graph_data.get('relationships', [])]
        
        doc = Document(page_content=text)
        graph_documents_prompt_engineered.append(GraphDocument(nodes=nodes, relationships=relationships, source=doc))
        
    except Exception as e:
        # If the LLM output is not valid JSON, the parser will fail. We'll catch that error.
        errors.append(str(e))
        doc = Document(page_content=text)
        graph_documents_prompt_engineered.append(GraphDocument(nodes=[], relationships=[], source=doc))
```
Now for the moment of truth. Let’s check our failure rate again.

```python
# Initialize a counter for documents with no nodes
empty_count_prompt_engineered = 0

# Iterate through the new results
for doc in graph_documents_prompt_engineered:
    if not doc.nodes:
        empty_count_prompt_engineered += 1

# Calculate and print the new failure percentage
print(f"Percentage missing with improved prompt: {empty_count_prompt_engineered / len(graph_documents_prompt_engineered) * 100}%")
print(f"Number of JSON parsing errors: {len(errors)}")


#### OUTPUT ####
Percentage missing with improved prompt: 62.0%
Number of JSON parsing errors: 13
```
The result? A failure rate of ~**62%**. While this is a slight improvement from our initial 75%, it’s nowhere near reliable enough. We still failed to extract a graph from 13 out of 20 articles. The `JsonOutputParser` threw an error each time because `llama3`, despite our best efforts with prompting, still produced malformed JSON.

This demonstrates a fundamental limitation:

> **Prompt engineering alone cannot fully solve the problem of inconsistent structured output from smaller LLMs.**

So, if a better prompt isn’t the answer, what is? We need a tool that doesn’t just ask for good output but is also smart enough to handle the imperfect output the LLM gives us. This is the problem that BAML is designed to solve.

In the upcoming sections, we will replace this entire chain with a BAML-powered implementation and see the difference it makes.

## Initialization and Quick Overview of BaML
We have established that even with careful prompt engineering, relying on strict JSON parsing with smaller LLMs is a recipe for failure. The models are powerful but not perfect formatters.

This is where BAML (Basically, A Made-up Language) can be very important. BAML offers two key advantages that directly address our problem:

1.  **Simplified Schemas:** Instead of verbose JSON schemas, BAML uses a clean, TypeScript-like syntax to define the data structure. This is easier for both humans to read and LLMs to understand, reducing token usage and the chance of confusion.
2.  **Robust Parsing:** BAML’s client comes with a “fuzzy” or “schema-aligned” parser. It doesn’t expect perfect JSON. It can handle common LLM mistakes like trailing commas, missing quotes, or extra text and still successfully extract the data.

First, you’ll need to install the BAML client and its VS Code extension.

```bash
# Installing baml client
pip install baml-py
```
Search for `baml` in the VS Code marketplace and install the extension. This extension is fantastic because it gives you an interactive playground to test your prompts and schemas without having to run your Python code every time.

Next, we define our graph extraction logic in a `.baml` file. Think of this as a configuration file for our LLM calls. We'll create a file named `extract_graph.baml`:

```csharp
// Define a node in the graph with an ID, type, and optional properties
class SimpleNode {
  id string                   // Unique identifier for the node
  type string                // Type/category of the node
  properties Properties      // Additional attributes associated with the node
}

// Define the structure for optional properties of nodes or relationships
class Properties {
  description string?        // Optional textual description
}

// Define a relationship between two nodes
class SimpleRelationship {
  source_node_id string      // ID of the source node
  source_node_type string    // Type of the source node
  target_node_id string      // ID of the target node
  target_node_type string    // Type of the target node
  type string                // Relationship type (e.g., "connects_to", "belongs_to")
  properties Properties      // Additional properties for the relationship
}

// Define the overall graph consisting of nodes and relationships
class DynamicGraph {
  nodes SimpleNode[]               // List of all nodes in the graph
  relationships SimpleRelationship[] // List of all relationships between nodes
}

// Function to extract a DynamicGraph from a raw input string
function ExtractGraph(graph: string) -> DynamicGraph {
  client Ollama                   // Use the Ollama client to interpret the input
  prompt #"
    Extract from this content:
    {{ ctx.output_format }}

    {{ graph }}                           // Prompt template to instruct Ollama to extract the graph
}
```
The `class` definitions are simple and readable. The `function ExtractGraph` tells BAML to use the `Ollama` client and provides a Jinja prompt template. The special `{{ ctx.output_format }}` variable is where BAML will automatically inject our simplified schema definition.

## Integrating BAML with LangChain
Now, we integrate this BAML function into our LangChain workflow. We’ll need some helper functions to convert BAML’s output into the `GraphDocument` format that Langchain and Neo4j understand.

```python
# Import necessary libraries
from typing import Any, List
import baml_client as client
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.runnables import chain

# Helper function to format nodes correctly (e.g., proper capitalization)
def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize() if el.type else None,
            properties=el.properties
        )
        for el in nodes
    ]

# Helper to map BAML's relationship output to Langchain's Relationship object
def map_to_base_relationship(rel: Any) -> Relationship:
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    return Relationship(
        source=source, target=target, type=rel.type, properties=rel.properties
    )

# Main helper to format all relationships
def _format_relationships(rels) -> List[Relationship]:
    relationships = [
        map_to_base_relationship(rel)
        for rel in rels
        if rel.type and rel.source_node_id and rel.target_node_id
    ]
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
            properties=el.properties,
        )
        for el in relationships
    ]

# Define a LangChain chainable function to call our BAML function
@chain
async def get_graph(message):
    graph = await client.b.ExtractGraph(graph=message.content)
    return graph
```
Let’s understand the purpose of each helper function:
- `_format_nodes(nodes)`: Standardizes node formatting by capitalizing IDs and types, and returns a list of cleanly formatted `Node` objects.
- `map_to_base_relationship(rel)`: Converts a raw BAML relationship into a basic LangChain `Relationship` by wrapping the source and target as `Node` objects.
- `_format_relationships(rels)`: Filters out invalid relationships, maps them to LangChain `Relationship` objects, and formats node types and relationship types for consistency.
- `get_graph(message)`: An asynchronous chain function that sends the input message to the BAML API, calls `ExtractGraph`, and returns the raw graph output.

With these helpers in place, we can define our new processing chain. We’ll use a custom prompt that is much simpler because BAML handles the complex schema injection for us.

```python
# Import the prompt template
from langchain_core.prompts import ChatPromptTemplate

# A simple, effective system prompt
system_prompt = """
You are a knowledgeable assistant skilled in extracting entities and their relationships from text.
Your goal is to create a knowledge graph.
"""

# The final prompt template
default_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format and do not include any explanations. "
                "Use the given format to extract information from the following input: {input}"
            ),
        ),
    ]
)

# Define the full BAML-powered chain
chain = default_prompt | llm | get_graph
```
This prompt template guides the model to extract entities and relationships for a knowledge graph:
- `system_prompt`: Sets the model’s role as an entity-relationship extractor.
- `default_prompt`: Combines system and human messages with a placeholder for input text.
- `chain`: Runs the prompt through the language model, then passes the output to `get_graph` for graph extraction.

## Running the BAML Experiment
Now it’s time to run the experiment again. We’ll process a much larger batch of articles this time to really test the reliability of this new approach.

> I stopped the execution after **344 articles** due to time constraints, but this is a much more robust sample size than our initial 20.

Before the execution the parallel processiong requires some helper function so let’s code those first.

```ruby
import asyncio

# Asynchronous function to process a single document
async def aprocess_response(document: Document) -> GraphDocument:
    # Invoke our BAML chain
    resp = await chain.ainvoke({"input": document.page_content})
    # Format the response into a GraphDocument
    return GraphDocument(
        nodes=_format_nodes(resp.nodes),
        relationships=_format_relationships(resp.relationships),
        source=document,
    )

# Asynchronous function to process a list of documents
async def aconvert_to_graph_documents(
    documents: List[Document],
) -> List[GraphDocument]:
    tasks = [asyncio.create_task(aprocess_response(document)) for document in documents]
    results = await asyncio.gather(*tasks)
    return results

# Asynchronous function to process raw texts
async def aprocess_text(texts: List[str]) -> List[GraphDocument]:
    docs = [Document(page_content=text) for text in texts]
    graph_docs = await aconvert_to_graph_documents(docs)
    return graph_docs
```
Let’s break down the purpose of each asynchronous function:
- `aprocess_response`: Processes one document and returns a `GraphDocument`.
- `aconvert_to_graph_documents`: Processes multiple documents in parallel and returns graph results.
- `aprocess_text`: Converts raw text to documents and extracts graph data.

Now, we can simply execute the main loop that will process our articles.

```python
# Initialize an empty list to store the resulting graph documents.
graph_documents_baml = []

# Set the total number of articles to be processed.
NUM_ARTICLES_BAML = 344

# Create a smaller DataFrame containing only the articles to be processed.
news_baml = news.head(NUM_ARTICLES_BAML)

# Extract titles and texts from the new DataFrame.
titles = news_baml["title"]
texts = news_baml["text"]

# Define the number of articles to process in each batch (chunk).
chunk_size = 4

# Iterate over the articles in chunks, using tqdm to display a progress bar.
for i in tqdm(range(0, len(titles), chunk_size), desc="Processing Chunks with BAML"):
    # Get the titles for the current chunk.
    title_chunk = titles[i : i + chunk_size]
    # Get the texts for the current chunk.
    text_chunk = texts[i : i + chunk_size]
    
    # Combine the title and text for each article in the chunk into a single string.
    combined_docs = [f"{title} {text}" for title, text in zip(title_chunk, text_chunk)]
    
    try:
        # Asynchronously process the combined documents to extract graph structures.
        docs = await aprocess_text(combined_docs)
        # Add the processed graph documents to the main list.
        graph_documents_baml.extend(docs)
    except Exception as e:
        # Handle any errors that occur during processing and print an error message.
        print(f"Error processing chunk starting at index {i}: {e}")

# After the loop, display the total number of graph documents successfully processed.
len(graph_documents_baml)
```
This is the output we get.
```bash
# Total number of graph documents
344
```

![Chunk size distribution](https://miro.medium.com/v2/resize:fit:875/1*LxqVKEcTxUOekyB2srdfPg.png)

We processed 344 articles. Now, let’s run the same failure analysis we did before.

```python
# Initialize a counter for documents with no nodes
empty_count_baml = 0

# Iterate through the results from the BAML approach
for doc in graph_documents_baml:
    if not doc.nodes:
        empty_count_baml += 1

# Calculate and print the new failure percentage
print(f"Percentage missing with BAML: {empty_count_baml / len(graph_documents_baml) * 100}%")


#### OUTPUT ####
Percentage missing with BAML: 0.5813953488372093%
```
This is an incredible result. Our failure rate dropped from **75%** down to just **0.58%**. This means our success rate is now **99.4%**!

By simply replacing the rigid `LLMGraphTransformer` with a BAML-powered chain, we went from a failing prototype to a robust, production-ready pipeline.

> This demonstrates that the bottleneck wasn't the small LLM's ability to understand the task, but rather the fragility of the system expecting perfect JSON.

## Building and Analyzing the GraphRAG
Simply extracting entities isn’t enough. The real power of GraphRAG comes from structuring this knowledge, finding hidden connections, and summarizing communities of related information.

We will now load our high-quality graph data into Neo4j and use graph data science techniques to enrich it.

First, we set up our connection to the Neo4j database.

```python
import os
from langchain_community.graphs import Neo4jGraph

# Set up Neo4j connection details using environment variables
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your_password" # Change this to your password
os.environ["DATABASE"] = "graphragdemo"

# Initialize the Neo4jGraph object
graph = Neo4jGraph()
```
Now, we can add our `graph_documents_baml` to the database. The `baseEntityLabel=True` argument adds an `__Entity__` label to all nodes, which is useful for querying later.

```python
# Add the graph documents to Neo4j
graph.add_graph_documents(graph_documents_baml, baseEntityLabel=True, include_source=True)
```
With our data loaded, we can run some Cypher queries to understand the structure of our new knowledge graph. Let’s start by looking at the relationship between the length of an article (in tokens) and the number of entities extracted from it.

```python
# Import libraries for plotting and data analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Query Neo4j to get the entity count and token count for each document
entity_dist = graph.query(
    """
    MATCH (d:Document)
    RETURN d.text AS text,
           count {(d)-[:MENTIONS]->()} AS entity_count
    """
)
entity_dist_df = pd.DataFrame.from_records(entity_dist)
entity_dist_df["token_count"] = [
    num_tokens_from_string(str(el)) for el in entity_dist_df["text"]
]

# Create a scatter plot with a regression line
sns.lmplot(
    x="token_count", y="entity_count", data=entity_dist_df, line_kws={"color": "red"}
)
plt.title("Entity Count vs Token Count Distribution")
plt.xlabel("Token Count")
plt.ylabel("Entity Count")
plt.show()
```
![Entity count vs Token count](https://miro.medium.com/v2/resize:fit:1250/1*UkVhy-LHGkAndu6RErus_w.png)

The plot shows a clear positive correlation: as the number of tokens in an article increases, the number of entities we extract also tends to increase. This is exactly what we’d expect and confirms our extraction process is behaving logically.

Next, let’s look at the node degree distribution. This tells us how connected our entities are. A few highly connected nodes (hubs) are common in real-world networks.

```python
import numpy as np

# Query for the degree of each entity node
degree_dist = graph.query(
    """
    MATCH (e:__Entity__)
    RETURN count {(e)-[:!MENTIONS]-()} AS node_degree
    """
)
degree_dist_df = pd.DataFrame.from_records(degree_dist)

# Calculate statistics
mean_degree = np.mean(degree_dist_df["node_degree"])
percentiles = np.percentile(degree_dist_df["node_degree"], [25, 50, 75, 90])

# Plot the histogram with a log scale
plt.figure(figsize=(12, 6))
sns.histplot(degree_dist_df["node_degree"], bins=50, kde=False, color="blue")
plt.yscale("log")
plt.title("Node Degree Distribution")
plt.legend()
plt.show()
```

![Node degree distribution](https://miro.medium.com/v2/resize:fit:1250/1*eAegZXc68oYfJxE1j36-mw.png)
*Node degree distribution (Created by Fareed Khan)*

The histogram shows a “long-tail” distribution, which is typical for knowledge graphs. Most entities have only a few connections (a low degree), while a small number of entities are highly connected hubs.

For example, the 90th percentile is a degree of 4, but the maximum degree is 37. This indicates that entities like “USA” or “Microsoft” are likely acting as central points in our graph.

To find entities that are semantically similar (even if they have different names), we need to create vector embeddings for them. An embedding is a numerical representation of a piece of text. We’ll generate embeddings for each entity’s `id` and `description` and store them in the graph.

We will use the `llama3` model via Ollama for embeddings and Langchain's `Neo4jVector` to handle the process.

```python
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings

# Create embeddings using our local llama3 model
embeddings = OllamaEmbeddings(model="llama3")

# Initialize the Neo4jVector instance to manage embeddings in the graph
vector = Neo4jVector.from_existing_graph(
    embeddings,
    node_label="__Entity__",
    text_node_properties=["id", "description"],
    embedding_node_property="embedding",
    database=os.environ["DATABASE"],
)
```
This command iterates through all `__Entity__` nodes in Neo4j, generates an embedding for their properties, and stores it back in the node under the `embedding` property.

## Finding and Linking Similar Entities
With embeddings in place, we can now use the **k-Nearest Neighbors (kNN)** algorithm to find nodes that are close to each other in the vector space. This is a powerful way to identify potential duplicate or highly related entities (e.g., “Man United” and “Manchester United”).

We’ll use Neo4j’s Graph Data Science (GDS) library for this.

```graphql
# Import the GraphDataScience library
from graphdatascience import GraphDataScience

# --- GDS Client Initialization ---
# Initialize the GraphDataScience client to connect to the Neo4j database.
# It uses connection details (URI, username, password) from environment variables.
gds = GraphDataScience(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
)
# Set the specific database for the GDS operations.
gds.set_database(os.environ["DATABASE"])

# --- In-Memory Graph Projection ---
# Project a graph into memory for efficient processing by GDS algorithms.
# This projection is named 'entities'.
G, result = gds.graph.project(
    "entities",                   # Name of the in-memory graph
    "__Entity__",                 # Node label to project
    "*",                          # Project all relationship types
    nodeProperties=["embedding"]  # Include the 'embedding' property for nodes
)

# --- Similarity Calculation using kNN ---
# Define the similarity threshold for creating relationships.
similarity_threshold = 0.95

# Use the k-Nearest Neighbors (kNN) algorithm to find similar nodes.
# This 'mutates' the in-memory graph by adding new relationships.
gds.knn.mutate(
    G,                                  # The in-memory graph to modify
    nodeProperties=["embedding"],       # Property to use for similarity calculation
    mutateRelationshipType="SIMILAR",   # The type of relationship to create
    mutateProperty="score",             # The property on the new relationship to store the similarity score
    similarityCutoff=similarity_threshold, # Threshold to filter relationships
)
```
We are creating `SIMILAR` relationships between nodes whose embedding similarity score is above 0.95.

The kNN algorithm helped us find *candidates* for duplicates, but text similarity alone isn’t perfect. We can further refine this by looking for entities that are not only semantically similar but also have very similar names (a low “edit distance”).

We’ll query for these candidates and then use an LLM to make the final decision on whether to merge them.

```python
# Query for potential duplicates based on community and name similarity
word_edit_distance = 3
potential_duplicate_candidates = graph.query(
    # ... (full Cypher query from the notebook) ...
    """
    MATCH (e:`__Entity__`)
    WHERE size(e.id) > 4
    WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
    WHERE count > 1
    # ... (rest of the complex query) ...
    RETURN distinct(combinedResult)
    """,
    params={"distance": word_edit_distance},
)

# Let's look at a few candidates
potential_duplicate_candidates[:5]
```
The output of the above code is this.
```
#### OUTPUT ####
[{'combinedResult': ['David Van', 'Davidvan']},
 {'combinedResult': ['Cyb003', 'Cyb004']},
 {'combinedResult': ['Delta Air Lines', 'Delta_Air_Lines']},
 {'combinedResult': ['Elon Musk', 'Elonmusk']},
 {'combinedResult': ['Market', 'Markets']}]
```
These look like clear duplicates. We can now use another BAML function to have the LLM decide which name to keep. After running this resolution process, we merge these nodes in Neo4j.

```python
# (Assuming 'merged_entities' is created by the LLM resolution process)
graph.query(
    """
    UNWIND $data AS candidates
    CALL {
      WITH candidates
      MATCH (e:__Entity__) WHERE e.id IN candidates
      RETURN collect(e) AS nodes
    }
    CALL apoc.refactor.mergeNodes(nodes, {properties: {'`.*`': 'discard'}})
    YIELD node
    RETURN count(*)
    """,
    params={"data": merged_entities},
)
```

## Community Detection with Leiden Algorithm
Now for the core of GraphRAG: grouping related entities into communities.

We will project our full graph (including all original relationships) into memory and run the Leiden algorithm, a state-of-the-art community detection algorithm.

```graphql
# Project the full graph, weighting relationships by their frequency
G, result = gds.graph.project(
    "communities",
    "__Entity__",
    {
        "_ALL_": {
            "type": "*",
            "orientation": "UNDIRECTED",
            "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
        }
    },
)

# Run Leiden community detection and write the results back to the nodes
gds.leiden.write(
    G,
    writeProperty="communities",
    includeIntermediateCommunities=True, # This creates hierarchical communities
    relationshipWeightProperty="weight",
)
```
This adds a `communities` property to each entity node, which is a list of community IDs at different levels of granularity (from small, tight-knit groups to larger, broader topics).

Finally, we materialize this hierarchy in the graph by creating `__Community__` nodes and linking them together. This creates a browsable topic structure.

```python
# Create a uniqueness constraint for community nodes
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;")

# Create community nodes and link entities and communities together
graph.query(
    """
    MATCH (e:`__Entity__`)
    UNWIND range(0, size(e.communities) - 1 , 1) AS index
    // ... (full community creation query from notebook) ...
    RETURN count(*)
    """
)
```
This complex query creates a multi-level community structure, for example: `(Entity)-[:IN_COMMUNITY]->(Level_0_Community)-[:IN_COMMUNITY]->(Level_1_Community)`.

## Analyzing the Final Graph Structure
After all that work, what does our knowledge graph look like? Let’s analyze the community sizes at each level.

```python
# Query for the size of each community at each level
community_size = graph.query(
    """
    MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(e:__Entity__)
    WITH c, count(distinct e) AS entities
    RETURN split(c.id, '-')[0] AS level, entities
    """
)

# Printing the processed dataframe
percentiles_df
```

| Lvl | Comms | P25 | P50 | P75 | P90 | P99  | Max |
|-----|-------|-----|-----|-----|-----|------|-----|
| 0   | 858   | 1.0 | 1.0 | 2.0 | 4.0 | 10.4 | 37  |
| 1   | 749   | 1.0 | 1.0 | 2.0 | 5.0 | 18.5 | 77  |
| 2   | 734   | 1.0 | 1.0 | 2.0 | 5.0 | 27.7 | 77  |
| 3   | 732   | 1.0 | 1.0 | 2.0 | 5.0 | 27.7 | 77  |

This table is important here. It shows us how the Leiden algorithm grouped our **1,875 entities**.
- At **Level 0**, we have 858 small, tightly-focused communities. 90% of these have 4 or fewer members.
- As we move up to **Level 3**, the algorithm has merged these into 732 larger, more general communities. The largest community at this level now contains 77 entities.

This hierarchical structure is exactly what we need for effective GraphRAG. We can now perform retrievals at different levels of abstraction.

## Conclusion
The results are clear. While standard Langchain tools provide a quick way to get started, they can be brittle and unreliable when used with smaller, open-source LLMs.

By introducing BAML, we addressed the core issues of overly complex prompts and strict JSON parsing. The outcome was a dramatic increase in success rate from **25% to over 99%**, transforming a failing experiment into a robust and scalable pipeline for building knowledge graphs.

Here’s a quick recap of the key steps we’ve taken:
- We began by preparing a news article dataset and setting up a local llama3 model with Ollama.
- Our first test using Langchain’s LLMGraphTransformer failed 75% of the time due to strict JSON parsing.
- Attempting to fix this with advanced prompt engineering only slightly improved the failure rate to ~62%.
- We then integrated BAML, leveraging its simplified schemas and robust parser to achieve a **99.4% success rate** in graph extraction.
- The high-quality graph data was loaded into Neo4j for structuring and analysis.
- We enriched the graph by generating vector embeddings for all entities to capture semantic meaning.
- Using the k-Nearest Neighbors (kNN) algorithm, we identified and linked semantically similar nodes.
- We further refined the graph by using an LLM to intelligently find and merge duplicate entities.
- Finally, we applied the Leiden algorithm to organize entities into a multi-level community hierarchy, setting the stage for advanced GraphRAG.