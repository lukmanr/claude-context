# Chroma Vector Database Implementation

This document describes the Chroma vector database implementation for Claude Context.

## Overview

The `ChromaVectorDatabase` class implements the `VectorDatabase` interface to provide vector database functionality using ChromaDB. This implementation offers an alternative to Milvus with a simpler setup process and excellent developer experience.

## Features

### ✅ Supported Operations

- **Collection Management**: Create, drop, check existence, and list collections
- **Document Operations**: Insert, delete, and query documents with metadata
- **Vector Search**: Semantic similarity search with configurable parameters
- **Metadata Filtering**: Query documents based on metadata criteria
- **Hybrid Search Simulation**: Combines vector and text search for improved results
- **Automatic Embeddings**: Integration with OpenAI and other embedding providers

### 🔧 Configuration Options

```typescript
interface ChromaConfig {
  path?: string; // Chroma server URL (default: http://localhost:8000)
  auth?: {
    // Optional authentication
    provider: string;
    credentials?: string;
    token?: string;
  };
  embeddingFunction?: IEmbeddingFunction; // Custom embedding function
}
```

## Usage Examples

### Basic Setup

```typescript
import {
  ChromaVectorDatabase,
  OpenAIEmbedding,
} from "@zilliz/claude-context-core";
import { OpenAIEmbeddingFunction } from "chromadb";

// Initialize vector database
const vectorDatabase = new ChromaVectorDatabase({
  path: "http://localhost:8000",
  embeddingFunction: new OpenAIEmbeddingFunction({
    openai_api_key: process.env.OPENAI_API_KEY,
    openai_model: "text-embedding-3-small",
  }),
});

// Use with Context
const context = new Context({
  embedding: new OpenAIEmbedding({
    apiKey: process.env.OPENAI_API_KEY,
    model: "text-embedding-3-small",
  }),
  vectorDatabase,
});
```

### Collection Operations

```typescript
// Create collection
await vectorDatabase.createCollection("my_collection", 1536, "Code collection");

// Check existence
const exists = await vectorDatabase.hasCollection("my_collection");

// List all collections
const collections = await vectorDatabase.listCollections();

// Drop collection
await vectorDatabase.dropCollection("my_collection");
```

### Document Operations

```typescript
// Insert documents
const documents: VectorDocument[] = [
    {
        id: 'doc1',
        vector: [0.1, 0.2, 0.3, ...],
        content: 'function example() { return "hello"; }',
        relativePath: 'src/example.ts',
        startLine: 1,
        endLine: 3,
        fileExtension: '.ts',
        metadata: { author: 'developer' }
    }
];

await vectorDatabase.insert('my_collection', documents);

// Search similar vectors
const results = await vectorDatabase.search('my_collection', queryVector, {
    topK: 10,
    filter: { fileExtension: '.ts' }
});

// Delete documents
await vectorDatabase.delete('my_collection', ['doc1']);
```

## Implementation Details

### Chroma Integration

The implementation uses the official ChromaDB JavaScript client:

- **Client Initialization**: Connects to Chroma server with configurable endpoint and auth
- **Embedding Integration**: Supports OpenAI and custom embedding functions
- **Collection Schema**: Maps VectorDocument structure to Chroma's document format
- **Metadata Handling**: Stores file information (path, lines, extension) as metadata

### Data Mapping

| VectorDocument Field | Chroma Mapping               | Description                         |
| -------------------- | ---------------------------- | ----------------------------------- |
| `id`                 | `ids`                        | Unique document identifier          |
| `vector`             | `embeddings`                 | Dense vector representation         |
| `content`            | `documents`                  | Text content for storage and search |
| `relativePath`       | `metadata.relativePath`      | File path within codebase           |
| `startLine/endLine`  | `metadata.startLine/endLine` | Code chunk boundaries               |
| `fileExtension`      | `metadata.fileExtension`     | File type information               |
| `metadata`           | `metadata.*`                 | Additional custom metadata          |

### Hybrid Search Implementation

Since Chroma doesn't have native hybrid search like Milvus, the implementation provides a simulation:

1. **Vector Search**: Uses dense embeddings for semantic similarity
2. **Text Search**: Uses Chroma's text query capabilities
3. **Result Fusion**: Combines and reranks results using weighted scoring
4. **Deduplication**: Merges duplicate documents from different search types

```typescript
// Hybrid search example
const searchRequests: HybridSearchRequest[] = [
  {
    data: denseVector, // Dense vector query
    anns_field: "vector",
    param: { nprobe: 10 },
    limit: 10,
  },
  {
    data: "search query text", // Text query
    anns_field: "sparse_vector",
    param: { drop_ratio_search: 0.2 },
    limit: 10,
  },
];

const results = await vectorDatabase.hybridSearch(
  "collection",
  searchRequests,
  {
    limit: 5,
    filterExpr: 'fileExtension in [".ts", ".js"]',
  }
);
```

## Comparison with Milvus

| Feature                  | Chroma               | Milvus               |
| ------------------------ | -------------------- | -------------------- |
| **Setup Complexity**     | ⭐⭐⭐⭐⭐ Simple    | ⭐⭐⭐ Moderate      |
| **Hybrid Search**        | ⭐⭐⭐ Simulated     | ⭐⭐⭐⭐⭐ Native    |
| **Scalability**          | ⭐⭐⭐ Good          | ⭐⭐⭐⭐⭐ Excellent |
| **Developer Experience** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good        |
| **Production Features**  | ⭐⭐⭐ Basic         | ⭐⭐⭐⭐⭐ Advanced  |

## Installation and Setup

### 1. Install Dependencies

```bash
cd packages/core
pnpm add chromadb
```

### 2. Start Chroma Server

```bash
# Using Docker (recommended)
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma

# Or using Python
pip install chromadb
chroma run --host localhost --port 8000
```

### 3. Configure Environment

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Error Handling

The implementation includes comprehensive error handling:

- **Connection Errors**: Graceful handling of server connectivity issues
- **Authentication Errors**: Clear error messages for auth failures
- **Collection Errors**: Proper error propagation for collection operations
- **Search Errors**: Detailed error information for search failures

## Performance Considerations

- **Batch Operations**: Efficient handling of bulk document operations
- **Connection Pooling**: Reuses client connections for optimal performance
- **Memory Usage**: Careful handling of large document sets
- **Async Operations**: Full async/await support for non-blocking operations

## Testing

Run the example to test the implementation:

```bash
cd examples/chroma-usage
pnpm install
pnpm dev
```

## Future Enhancements

- **Advanced Filtering**: More sophisticated query language support
- **Streaming Operations**: Support for large-scale streaming operations
- **Performance Optimization**: Caching and connection pooling improvements
- **Advanced Reranking**: More sophisticated hybrid search algorithms
