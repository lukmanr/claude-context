// Re-export types and interfaces
export {
    VectorDocument,
    SearchOptions,
    VectorSearchResult,
    VectorDatabase,
    HybridSearchRequest,
    HybridSearchOptions,
    HybridSearchResult,
    RerankStrategy,
    COLLECTION_LIMIT_MESSAGE
} from './types';

// Implementation class exports
export { MilvusRestfulVectorDatabase, MilvusRestfulConfig } from './milvus-restful-vectordb';
export { MilvusVectorDatabase, MilvusConfig } from './milvus-vectordb';
export { PostgresVectorDatabase, PostgresConfig } from './postgres-vectordb';
export { ChromaVectorDatabase, ChromaConfig } from './chroma-vectordb';
export {
    ClusterManager,
    ZillizConfig,
    Project,
    Cluster,
    CreateFreeClusterRequest,
    CreateFreeClusterResponse,
    CreateFreeClusterWithDetailsResponse,
    DescribeClusterResponse
} from './zilliz-utils';

// BM25 index manager for hybrid search
export {
    BM25IndexManager,
    BM25Config,
    BM25Document,
    BM25SearchResult,
    createBM25IndexManager
} from './bm25-index-manager'; 