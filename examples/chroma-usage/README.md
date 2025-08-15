# Chroma Vector Database Example

This example demonstrates how to use Claude Context with Chroma as the vector database backend.

## Prerequisites

1. **Node.js**: Make sure you have Node.js installed (>= 20.0.0)
2. **Chroma Server**: You need to have a Chroma server running

### Starting Chroma Server

You can run Chroma server using Docker:

```bash
# Pull and run Chroma server
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```

Alternatively, you can install and run Chroma locally:

```bash
pip install chromadb
chroma run --host localhost --port 8000
```

3. **OpenAI API Key**: Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Installation

```bash
# Install dependencies
pnpm install

# or if you're in the root of the project
pnpm install
```

## Running the Example

```bash
# Development mode with file watching
pnpm dev

# One-time execution
pnpm start
```

## What this Example Does

1. **Initializes Chroma Vector Database**: Connects to a Chroma server running on localhost:8000
2. **Sets up Embedding Provider**: Uses OpenAI's text-embedding-3-small model
3. **Creates a Test Collection**: Demonstrates collection creation with proper configuration
4. **Tests Basic Operations**:
   - Collection creation
   - Collection existence check
   - Listing collections
   - Collection deletion
5. **Ready for Code Indexing**: The setup is ready to index your codebase using semantic search

## Configuration Options

The `ChromaVectorDatabase` supports various configuration options:

```typescript
const vectorDatabase = new ChromaVectorDatabase({
  path: "http://localhost:8000", // Chroma server URL
  auth: {
    // Optional authentication
    provider: "basic",
    credentials: "username:password",
  },
  embeddingFunction: new OpenAIEmbeddingFunction({
    openai_api_key: process.env.OPENAI_API_KEY,
    openai_model: "text-embedding-3-small",
  }),
});
```

## Using with Your Codebase

To index and search your actual codebase, uncomment the relevant sections in `index.ts`:

```typescript
// Index your codebase
const stats = await context.indexCodebase("./your-project", (progress) => {
  console.log(`${progress.phase} - ${progress.percentage}%`);
});

// Perform semantic search
const results = await context.semanticSearch(
  "./your-project",
  "vector database operations",
  5
);
```

## Features Supported

- ✅ **Basic Operations**: Create, drop, list collections
- ✅ **Vector Search**: Semantic similarity search
- ✅ **Metadata Filtering**: Query with metadata filters
- ✅ **Batch Operations**: Insert multiple documents at once
- ✅ **Hybrid Search Simulation**: Combines vector and text search results
- ✅ **Automatic Embedding**: Uses OpenAI embeddings by default

## Chroma vs Milvus

While Milvus provides native hybrid search with separate dense and sparse vectors, Chroma's hybrid search is simulated by:

1. Performing vector similarity search
2. Performing text-based search
3. Combining and reranking results

This provides similar functionality while leveraging Chroma's strengths in simplicity and ease of use.
