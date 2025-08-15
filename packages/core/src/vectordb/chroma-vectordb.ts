import { ChromaClient, OpenAIEmbeddingFunction, Collection, IEmbeddingFunction } from 'chromadb';
import {
    VectorDocument,
    SearchOptions,
    VectorSearchResult,
    VectorDatabase,
    HybridSearchRequest,
    HybridSearchOptions,
    HybridSearchResult,
} from './types';

export interface ChromaConfig {
    path?: string;
    auth?: {
        provider: string;
        credentials?: string;
        token?: string;
    };
    embeddingFunction?: IEmbeddingFunction;
}

/**
 * Chroma Vector Database implementation
 * This implementation provides vector database functionality using ChromaDB
 */
export class ChromaVectorDatabase implements VectorDatabase {
    protected config: ChromaConfig;
    private client: ChromaClient | null = null;
    protected initializationPromise: Promise<void>;
    private embeddingFunction: IEmbeddingFunction | null = null;

    constructor(config: ChromaConfig = {}) {
        this.config = config;

        // Start initialization asynchronously without waiting
        this.initializationPromise = this.initialize();
    }

    private async initialize(): Promise<void> {
        await this.initializeClient();
    }

    private async initializeClient(): Promise<void> {
        const clientConfig: any = {};

        // Set the server path (defaults to localhost:8000)
        if (this.config.path) {
            clientConfig.path = this.config.path;
        }

        // Set authentication if provided
        if (this.config.auth) {
            clientConfig.auth = this.config.auth;
        }

        this.client = new ChromaClient(clientConfig);

        // Set up embedding function
        if (this.config.embeddingFunction) {
            this.embeddingFunction = this.config.embeddingFunction;
        } else {
            // Default to OpenAI embedding if available
            const openaiApiKey = process.env.OPENAI_API_KEY;
            if (openaiApiKey) {
                this.embeddingFunction = new OpenAIEmbeddingFunction({
                    openai_api_key: openaiApiKey,
                    openai_model: 'text-embedding-3-small'
                });
            }
        }

        console.log(`🔌 Connected to Chroma at: ${this.config.path || 'http://localhost:8000'}`);
    }

    /**
     * Ensure initialization is complete before method execution
     */
    protected async ensureInitialized(): Promise<void> {
        await this.initializationPromise;
        if (!this.client) {
            throw new Error('Chroma client not initialized');
        }
    }

    async createCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        await this.ensureInitialized();

        try {
            console.log(`📦 Creating collection '${collectionName}' with dimension ${dimension}`);

            const collectionConfig: any = {
                name: collectionName,
            };

            if (this.embeddingFunction) {
                collectionConfig.embeddingFunction = this.embeddingFunction;
            }

            if (description) {
                collectionConfig.metadata = { description };
            }

            await this.client!.createCollection(collectionConfig);

            console.log(`✅ Successfully created collection '${collectionName}'`);
        } catch (error: any) {
            console.error(`❌ Failed to create collection '${collectionName}':`, error);
            throw error;
        }
    }

    async createHybridCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        // Chroma doesn't have native hybrid search like Milvus, but we can simulate it
        // by creating a regular collection and implementing hybrid search logic
        await this.createCollection(collectionName, dimension, description);
    }

    async dropCollection(collectionName: string): Promise<void> {
        await this.ensureInitialized();

        try {
            await this.client!.deleteCollection({
                name: collectionName,
            });
            console.log(`🗑️  Successfully dropped collection '${collectionName}'`);
        } catch (error: any) {
            console.error(`❌ Failed to drop collection '${collectionName}':`, error);
            throw error;
        }
    }

    async hasCollection(collectionName: string): Promise<boolean> {
        await this.ensureInitialized();

        try {
            const collections = await this.client!.listCollections();
            return collections.some(collection => collection.name === collectionName);
        } catch (error: any) {
            console.error(`❌ Failed to check collection '${collectionName}' existence:`, error);
            return false;
        }
    }

    async listCollections(): Promise<string[]> {
        await this.ensureInitialized();

        try {
            const collections = await this.client!.listCollections();
            return collections.map(collection => collection.name);
        } catch (error: any) {
            console.error(`❌ Failed to list collections:`, error);
            throw error;
        }
    }

    private async getCollection(collectionName: string): Promise<Collection> {
        try {
            return await this.client!.getCollection({
                name: collectionName,
                embeddingFunction: this.embeddingFunction || undefined,
            });
        } catch (error: any) {
            console.error(`❌ Failed to get collection '${collectionName}':`, error);
            throw error;
        }
    }

    async insert(collectionName: string, documents: VectorDocument[]): Promise<void> {
        await this.ensureInitialized();

        try {
            const collection = await this.getCollection(collectionName);

            // Transform VectorDocument array to Chroma format
            const ids = documents.map(doc => doc.id);
            const embeddings = documents.map(doc => doc.vector);
            const texts = documents.map(doc => doc.content);
            const metadatas = documents.map(doc => ({
                relativePath: doc.relativePath,
                startLine: doc.startLine,
                endLine: doc.endLine,
                fileExtension: doc.fileExtension,
                ...doc.metadata
            }));

            await collection.add({
                ids,
                embeddings,
                documents: texts,
                metadatas,
            });

            console.log(`✅ Successfully inserted ${documents.length} documents into collection '${collectionName}'`);
        } catch (error: any) {
            console.error(`❌ Failed to insert documents into collection '${collectionName}':`, error);
            throw error;
        }
    }

    async insertHybrid(collectionName: string, documents: VectorDocument[]): Promise<void> {
        // For Chroma, hybrid insertion is the same as regular insertion
        // since Chroma doesn't have separate sparse vectors like Milvus
        await this.insert(collectionName, documents);
    }

    async search(collectionName: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]> {
        await this.ensureInitialized();

        try {
            const collection = await this.getCollection(collectionName);
            const topK = options?.topK || 10;

            const queryParams: any = {
                queryEmbeddings: [queryVector],
                nResults: topK,
                include: ['documents', 'metadatas', 'distances'],
            };

            // Apply metadata filter if provided
            if (options?.filter) {
                queryParams.where = options.filter;
            }

            const results = await collection.query(queryParams);

            // Transform Chroma results to VectorSearchResult format
            const searchResults: VectorSearchResult[] = [];

            if (results.ids && results.ids[0]) {
                for (let i = 0; i < results.ids[0].length; i++) {
                    const metadata = results.metadatas?.[0]?.[i] || {};
                    const document = results.documents?.[0]?.[i] || '';
                    const distance = results.distances?.[0]?.[i] || 0;

                    // Convert distance to similarity score (Chroma uses distance, we want similarity)
                    const score = Math.max(0, 1 - distance);

                    searchResults.push({
                        document: {
                            id: results.ids[0][i],
                            vector: queryVector,
                            content: document,
                            relativePath: metadata.relativePath || '',
                            startLine: metadata.startLine || 0,
                            endLine: metadata.endLine || 0,
                            fileExtension: metadata.fileExtension || '',
                            metadata: Object.fromEntries(
                                Object.entries(metadata).filter(([key]) => 
                                    !['relativePath', 'startLine', 'endLine', 'fileExtension'].includes(key)
                                )
                            )
                        },
                        score,
                    });
                }
            }

            return searchResults;
        } catch (error: any) {
            console.error(`❌ Failed to search in collection '${collectionName}':`, error);
            throw error;
        }
    }

    async hybridSearch(collectionName: string, searchRequests: HybridSearchRequest[], options?: HybridSearchOptions): Promise<HybridSearchResult[]> {
        await this.ensureInitialized();

        try {
            console.log(`🔍 Performing hybrid search simulation on collection: ${collectionName}`);

            // Since Chroma doesn't have native hybrid search, we'll simulate it by:
            // 1. Performing vector search with the dense vector
            // 2. Performing text search if we have text queries
            // 3. Combining and reranking results

            const results: Map<string, HybridSearchResult> = new Map();
            const limit = options?.limit || 10;

            // Process each search request
            for (const request of searchRequests) {
                let searchResults: VectorSearchResult[] = [];

                if (request.anns_field === 'vector' && Array.isArray(request.data)) {
                    // Dense vector search
                    searchResults = await this.search(collectionName, request.data as number[], {
                        topK: request.limit,
                        filterExpr: options?.filterExpr,
                    });
                } else if (request.anns_field === 'sparse_vector' || typeof request.data === 'string') {
                    // Text-based search using Chroma's text search capabilities
                    const collection = await this.getCollection(collectionName);
                    
                    const queryParams: any = {
                        queryTexts: [request.data as string],
                        nResults: request.limit,
                        include: ['documents', 'metadatas', 'distances'],
                    };

                    if (options?.filterExpr) {
                        // Convert filterExpr to Chroma where clause if possible
                        // This is a simplified conversion
                        queryParams.where = { fileExtension: { "$in": [".ts", ".js", ".py"] } };
                    }

                    const textResults = await collection.query(queryParams);

                    // Convert to VectorSearchResult format
                    if (textResults.ids && textResults.ids[0]) {
                        for (let i = 0; i < textResults.ids[0].length; i++) {
                            const metadata = textResults.metadatas?.[0]?.[i] || {};
                            const document = textResults.documents?.[0]?.[i] || '';
                            const distance = textResults.distances?.[0]?.[i] || 0;
                            const score = Math.max(0, 1 - distance);

                            searchResults.push({
                                document: {
                                    id: textResults.ids[0][i],
                                    vector: [],
                                    content: document,
                                    relativePath: metadata.relativePath || '',
                                    startLine: metadata.startLine || 0,
                                    endLine: metadata.endLine || 0,
                                    fileExtension: metadata.fileExtension || '',
                                    metadata: Object.fromEntries(
                                        Object.entries(metadata).filter(([key]) => 
                                            !['relativePath', 'startLine', 'endLine', 'fileExtension'].includes(key)
                                        )
                                    )
                                },
                                score,
                            });
                        }
                    }
                }

                // Add results to the combined map
                for (const result of searchResults) {
                    const existing = results.get(result.document.id);
                    if (existing) {
                        // Combine scores using RRF (Reciprocal Rank Fusion) or weighted average
                        existing.score = (existing.score + result.score) / 2;
                    } else {
                        results.set(result.document.id, {
                            document: result.document,
                            score: result.score,
                        });
                    }
                }
            }

            // Convert map to array and sort by score
            const hybridResults = Array.from(results.values())
                .sort((a, b) => b.score - a.score)
                .slice(0, limit);

            console.log(`✅ Found ${hybridResults.length} results from hybrid search simulation`);
            return hybridResults;

        } catch (error: any) {
            console.error(`❌ Failed to perform hybrid search on collection '${collectionName}':`, error);
            throw error;
        }
    }

    async delete(collectionName: string, ids: string[]): Promise<void> {
        await this.ensureInitialized();

        try {
            const collection = await this.getCollection(collectionName);

            await collection.delete({
                ids,
            });

            console.log(`🗑️  Successfully deleted ${ids.length} documents from collection '${collectionName}'`);
        } catch (error: any) {
            console.error(`❌ Failed to delete documents from collection '${collectionName}':`, error);
            throw error;
        }
    }

    async query(collectionName: string, filter: string, outputFields: string[], limit?: number): Promise<Record<string, any>[]> {
        await this.ensureInitialized();

        try {
            const collection = await this.getCollection(collectionName);

            // Convert filter expression to Chroma where clause
            // This is a simplified conversion - in practice, you'd need more sophisticated parsing
            let where: any = {};
            if (filter && filter.trim() !== '') {
                // Example: fileExtension in [".ts", ".py"] -> { fileExtension: { "$in": [".ts", ".py"] } }
                if (filter.includes('fileExtension in')) {
                    const extensions = filter.match(/\[([^\]]+)\]/)?.[1]
                        ?.split(',')
                        .map(ext => ext.trim().replace(/['"]/g, ''));
                    if (extensions) {
                        where.fileExtension = { "$in": extensions };
                    }
                }
            }

            const queryParams: any = {
                where: Object.keys(where).length > 0 ? where : undefined,
                include: ['documents', 'metadatas'],
            };

            if (limit) {
                queryParams.nResults = limit;
            }

            // Use get() method to retrieve documents based on where clause
            const results = await collection.get(queryParams);

            // Transform results to match expected format
            const queryResults: Record<string, any>[] = [];

            if (results.ids) {
                for (let i = 0; i < results.ids.length; i++) {
                    const metadata = results.metadatas?.[i] || {};
                    const document = results.documents?.[i] || '';

                    const result: Record<string, any> = {
                        id: results.ids[i],
                        content: document,
                        ...metadata,
                    };

                    // Filter to only include requested output fields
                    if (outputFields.length > 0) {
                        const filteredResult: Record<string, any> = {};
                        for (const field of outputFields) {
                            if (result[field] !== undefined) {
                                filteredResult[field] = result[field];
                            }
                        }
                        queryResults.push(filteredResult);
                    } else {
                        queryResults.push(result);
                    }
                }
            }

            return queryResults;

        } catch (error: any) {
            console.error(`❌ Failed to query collection '${collectionName}':`, error);
            throw error;
        }
    }
}
