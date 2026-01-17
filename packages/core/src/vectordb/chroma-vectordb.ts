import { ChromaClient, Collection, IncludeEnum } from 'chromadb';
import {
    VectorDocument,
    SearchOptions,
    VectorSearchResult,
    VectorDatabase,
    HybridSearchRequest,
    HybridSearchOptions,
    HybridSearchResult,
} from './types';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';
import { spawn, ChildProcess } from 'child_process';

export interface ChromaConfig {
    /** Path to store ChromaDB data (default: ~/.velocity/chroma) */
    path?: string;
    /** Port for ChromaDB server (default: 8000) */
    port?: number;
    /** Host for ChromaDB server (default: localhost) */
    host?: string;
    /** Batch size for inserts (default: 100) */
    batchSize?: number;
    /** Skip spawning server (useful if server is managed externally) */
    skipServerSpawn?: boolean;
    /** HNSW ef_search parameter - higher = more accurate but slower (default: 100) */
    hnswEfSearch?: number;
    /** HNSW batch_size for indexing (default: 1000) */
    hnswBatchSize?: number;
    /** HNSW num_threads for indexing, 0 = auto (default: 0) */
    hnswNumThreads?: number;
}

/**
 * ChromaDB Vector Database implementation using file-based persistent storage
 * This implementation provides vector storage and similarity search using ChromaDB
 * with local filesystem persistence.
 * 
 * The server is automatically spawned and managed by this class.
 */
export class ChromaVectorDatabase implements VectorDatabase {
    protected config: ChromaConfig;
    private client: ChromaClient | null = null;
    protected initializationPromise: Promise<void>;
    private chromaPath: string;
    private chromaPort: number;
    private chromaHost: string;
    private serverProcess: ChildProcess | null = null;
    private static serverStarted: boolean = false;
    private static serverProcess: ChildProcess | null = null;
    // Lock to prevent multiple concurrent server start attempts
    private static serverStartPromise: Promise<void> | null = null;

    constructor(config: ChromaConfig) {
        this.config = config;
        // Default to ~/.velocity/chroma if no path specified
        this.chromaPath = config.path || path.join(os.homedir(), '.velocity', 'chroma');
        this.chromaPort = config.port || 8000;
        this.chromaHost = config.host || 'localhost';
        this.initializationPromise = this.initialize();
    }

    private async initialize(): Promise<void> {
        // Ensure the chroma directory exists
        if (!fs.existsSync(this.chromaPath)) {
            fs.mkdirSync(this.chromaPath, { recursive: true });
            console.log(`✅ Created ChromaDB data directory: ${this.chromaPath}`);
        }

        // Start the ChromaDB server if not already running
        if (!this.config.skipServerSpawn) {
            await this.ensureServerRunning();
        }

        await this.initializeClient();
    }

    /**
     * Ensure ChromaDB server is running, starting it if necessary
     * Uses a lock to prevent multiple concurrent server start attempts
     */
    private async ensureServerRunning(): Promise<void> {
        // If another call is already starting the server, wait for it
        if (ChromaVectorDatabase.serverStartPromise) {
            console.log(`⏳ Waiting for ChromaDB server startup (already in progress)...`);
            await ChromaVectorDatabase.serverStartPromise;
            return;
        }

        // Check if server is already running (either started by us or externally)
        if (await this.isServerRunning()) {
            console.log(`✅ ChromaDB server already running at ${this.chromaHost}:${this.chromaPort}`);
            return;
        }

        // Check if we've already started the server in this process
        if (ChromaVectorDatabase.serverStarted && ChromaVectorDatabase.serverProcess) {
            console.log(`✅ ChromaDB server already started by this process`);
            return;
        }

        console.log(`🚀 Starting ChromaDB server at ${this.chromaHost}:${this.chromaPort} with data path: ${this.chromaPath}`);

        // Create a promise that other callers can wait on
        ChromaVectorDatabase.serverStartPromise = this.startServer();
        
        try {
            await ChromaVectorDatabase.serverStartPromise;
        } finally {
            // Clear the promise after completion (success or failure)
            ChromaVectorDatabase.serverStartPromise = null;
        }
    }

    /**
     * Check if ChromaDB server is running by attempting to connect
     * Uses the v2 API heartbeat endpoint (v1 is deprecated in ChromaDB 3.x)
     */
    private async isServerRunning(): Promise<boolean> {
        try {
            const response = await fetch(`http://${this.chromaHost}:${this.chromaPort}/api/v2/heartbeat`, {
                method: 'GET',
                signal: AbortSignal.timeout(2000)
            });
            return response.ok;
        } catch {
            return false;
        }
    }

    /**
     * Start the ChromaDB server process
     * Uses the chromadb npm package's native bindings (Rust-compiled)
     */
    private async startServer(): Promise<void> {
        return new Promise((resolve, reject) => {
            // Use npx chromadb (native bindings) to run the server
            const chromaProcess = spawn('npx', [
                'chromadb', 'run',
                '--path', this.chromaPath,
                '--port', this.chromaPort.toString(),
                '--host', this.chromaHost
            ], {
                stdio: ['ignore', 'pipe', 'pipe'],
                detached: false,
                env: { ...process.env }
            });

            ChromaVectorDatabase.serverProcess = chromaProcess;
            ChromaVectorDatabase.serverStarted = true;
            this.serverProcess = chromaProcess;

            let serverReady = false;
            let startupOutput = '';

            // Handle stdout
            chromaProcess.stdout?.on('data', (data: Buffer) => {
                const output = data.toString();
                startupOutput += output;
                // Log ChromaDB output to stderr (since MCP uses stdout for protocol)
                process.stderr.write(`[ChromaDB] ${output}`);
                
                // Check for server ready indicators
                if (output.includes('Application startup complete') || 
                    output.includes('Uvicorn running on') ||
                    output.includes('Started server process')) {
                    serverReady = true;
                }
            });

            // Handle stderr
            chromaProcess.stderr?.on('data', (data: Buffer) => {
                const output = data.toString();
                startupOutput += output;
                process.stderr.write(`[ChromaDB] ${output}`);
                
                // Uvicorn logs to stderr
                if (output.includes('Application startup complete') || 
                    output.includes('Uvicorn running on') ||
                    output.includes('Started server process')) {
                    serverReady = true;
                }
            });

            // Handle process errors
            chromaProcess.on('error', (error) => {
                console.error(`❌ Failed to start ChromaDB server:`, error);
                ChromaVectorDatabase.serverStarted = false;
                ChromaVectorDatabase.serverProcess = null;
                reject(new Error(`Failed to start ChromaDB server: ${error.message}`));
            });

            // Handle process exit
            chromaProcess.on('exit', (code, signal) => {
                if (!serverReady) {
                    console.error(`❌ ChromaDB server exited prematurely with code ${code}, signal ${signal}`);
                    console.error(`Startup output: ${startupOutput}`);
                    ChromaVectorDatabase.serverStarted = false;
                    ChromaVectorDatabase.serverProcess = null;
                    reject(new Error(`ChromaDB server exited with code ${code}`));
                } else {
                    console.log(`[ChromaDB] Server process exited with code ${code}`);
                    ChromaVectorDatabase.serverStarted = false;
                    ChromaVectorDatabase.serverProcess = null;
                }
            });

            // Register cleanup handlers
            this.registerCleanupHandlers();

            // Wait for server to be ready with polling
            this.waitForServerReady(30000) // 30 second timeout
                .then(() => {
                    console.log(`✅ ChromaDB server started successfully at ${this.chromaHost}:${this.chromaPort}`);
                    resolve();
                })
                .catch((error) => {
                    console.error(`❌ ChromaDB server failed to start within timeout`);
                    this.stopServer();
                    reject(error);
                });
        });
    }

    /**
     * Wait for server to be ready by polling the heartbeat endpoint
     */
    private async waitForServerReady(timeoutMs: number): Promise<void> {
        const startTime = Date.now();
        const pollInterval = 500; // 500ms between polls

        while (Date.now() - startTime < timeoutMs) {
            if (await this.isServerRunning()) {
                return;
            }
            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }

        throw new Error(`ChromaDB server failed to start within ${timeoutMs}ms`);
    }

    /**
     * Register cleanup handlers to stop the server on process exit
     */
    private registerCleanupHandlers(): void {
        const cleanup = () => {
            this.stopServer();
        };

        // Only register once
        if (!ChromaVectorDatabase.serverStarted) {
            return;
        }

        process.once('exit', cleanup);
        process.once('SIGINT', () => {
            cleanup();
            process.exit(0);
        });
        process.once('SIGTERM', () => {
            cleanup();
            process.exit(0);
        });
        process.once('uncaughtException', (error) => {
            console.error('Uncaught exception:', error);
            cleanup();
            process.exit(1);
        });
    }

    /**
     * Stop the ChromaDB server process
     */
    private stopServer(): void {
        if (ChromaVectorDatabase.serverProcess) {
            console.log('🛑 Stopping ChromaDB server...');
            try {
                // Try graceful shutdown first
                ChromaVectorDatabase.serverProcess.kill('SIGTERM');
                
                // Force kill after timeout if still running
                setTimeout(() => {
                    if (ChromaVectorDatabase.serverProcess && !ChromaVectorDatabase.serverProcess.killed) {
                        ChromaVectorDatabase.serverProcess.kill('SIGKILL');
                    }
                }, 5000);
            } catch (error) {
                console.error('Error stopping ChromaDB server:', error);
            }
            ChromaVectorDatabase.serverProcess = null;
            ChromaVectorDatabase.serverStarted = false;
        }
    }

    private async initializeClient(): Promise<void> {
        console.log(`🔌 Connecting to ChromaDB at: ${this.chromaHost}:${this.chromaPort}`);
        
        // Initialize ChromaDB client using the new host/port configuration (3.x API)
        // Note: 'path' param is deprecated in chromadb 3.x
        this.client = new ChromaClient({
            host: this.chromaHost,
            port: this.chromaPort
        });

        // Verify connection by listing collections
        try {
            await this.client.listCollections();
            console.log('✅ ChromaDB client connected successfully');
        } catch (error) {
            console.error('❌ Failed to connect to ChromaDB:', error);
            throw new Error(`Failed to connect to ChromaDB at ${this.chromaHost}:${this.chromaPort}`);
        }
    }

    protected async ensureInitialized(): Promise<void> {
        await this.initializationPromise;
        if (!this.client) {
            throw new Error('ChromaDB client not initialized');
        }
    }

    /**
     * Sanitize collection name for ChromaDB requirements
     * ChromaDB collection names must:
     * - Be 3-63 characters long
     * - Start and end with alphanumeric
     * - Contain only alphanumeric, underscores, hyphens
     * - Not contain consecutive periods
     */
    private sanitizeCollectionName(name: string): string {
        // Replace slashes and other invalid characters with underscores
        let sanitized = name
            .replace(/[\/\\:*?"<>|.]/g, '_')
            .replace(/_{2,}/g, '_')  // Replace multiple underscores with single
            .replace(/^_+|_+$/g, ''); // Remove leading/trailing underscores

        // Ensure it starts with alphanumeric
        if (!/^[a-zA-Z0-9]/.test(sanitized)) {
            sanitized = 'c_' + sanitized;
        }

        // Ensure it ends with alphanumeric
        if (!/[a-zA-Z0-9]$/.test(sanitized)) {
            sanitized = sanitized + '_c';
        }

        // Truncate to 63 characters if needed
        if (sanitized.length > 63) {
            sanitized = sanitized.substring(0, 63);
            // Ensure it still ends with alphanumeric after truncation
            if (!/[a-zA-Z0-9]$/.test(sanitized)) {
                sanitized = sanitized.substring(0, 62) + 'c';
            }
        }

        // Ensure minimum length of 3
        if (sanitized.length < 3) {
            sanitized = sanitized + '_cc';
        }

        return sanitized.toLowerCase();
    }

    async createCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        await this.ensureInitialized();
        const sanitizedName = this.sanitizeCollectionName(collectionName);

        try {
            console.log(`[ChromaDB] Creating collection '${sanitizedName}' (original: ${collectionName}) with dimension ${dimension}, cosine distance`);
            
            // ChromaDB automatically creates the collection if it doesn't exist
            // Configure with cosine distance metric and optimized HNSW parameters
            await this.client!.getOrCreateCollection({
                name: sanitizedName,
                metadata: {
                    description: description || `Claude Context collection: ${collectionName}`,
                    originalName: collectionName,
                    dimension: dimension.toString(),
                    createdAt: new Date().toISOString(),
                    // Configure HNSW with cosine distance for semantic similarity
                    'hnsw:space': 'cosine'
                },
                // HNSW configuration for better search performance
                // These are tuned for code search workloads
                configuration: {
                    hnsw: {
                        // ef_search: higher = more accurate but slower (default: 10)
                        // For code search, accuracy is important
                        ef_search: this.config.hnswEfSearch ?? 100,
                        // batch_size: documents processed per batch during indexing
                        batch_size: this.config.hnswBatchSize ?? 1000,
                        // num_threads: parallel threads for indexing (0 = auto)
                        num_threads: this.config.hnswNumThreads ?? 0
                    }
                }
            });

            console.log(`✅ ChromaDB collection '${sanitizedName}' created successfully with dimension ${dimension}, cosine distance, HNSW index`);
        } catch (error) {
            console.error(`❌ Failed to create collection '${collectionName}':`, error);
            throw error;
        }
    }

    async createHybridCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        // ChromaDB doesn't have a separate hybrid collection concept
        // It stores the full text content alongside embeddings by default
        // We can use metadata for full-text search capabilities
        console.log(`[ChromaDB] 📝 Creating hybrid collection '${collectionName}' (same as regular collection with metadata)`);
        await this.createCollection(collectionName, dimension, description);
    }

    async dropCollection(collectionName: string): Promise<void> {
        await this.ensureInitialized();
        const sanitizedName = this.sanitizeCollectionName(collectionName);

        try {
            await this.client!.deleteCollection({ name: sanitizedName });
            console.log(`✅ ChromaDB collection '${sanitizedName}' dropped successfully`);
        } catch (error: any) {
            // Collection might not exist
            if (error.message?.includes('does not exist')) {
                console.log(`⚠️  Collection '${sanitizedName}' does not exist, nothing to drop`);
                return;
            }
            console.error(`❌ Failed to drop collection '${collectionName}':`, error);
            throw error;
        }
    }

    async hasCollection(collectionName: string): Promise<boolean> {
        await this.ensureInitialized();
        const sanitizedName = this.sanitizeCollectionName(collectionName);

        try {
            const collections = await this.client!.listCollections();
            return collections.some(c => c.name === sanitizedName);
        } catch (error) {
            console.error(`❌ Failed to check collection existence '${collectionName}':`, error);
            return false;
        }
    }

    async listCollections(): Promise<string[]> {
        await this.ensureInitialized();

        try {
            const collections = await this.client!.listCollections();
            return collections.map(c => c.name);
        } catch (error) {
            console.error('❌ Failed to list collections:', error);
            throw error;
        }
    }

    async insert(collectionName: string, documents: VectorDocument[]): Promise<void> {
        await this.ensureInitialized();
        const sanitizedName = this.sanitizeCollectionName(collectionName);

        if (documents.length === 0) {
            return;
        }

        // Deduplicate documents by id (keep the last occurrence)
        const deduplicatedDocs = Array.from(
            new Map(documents.map(doc => [doc.id, doc])).values()
        );

        const originalCount = documents.length;
        const deduplicatedCount = deduplicatedDocs.length;

        if (originalCount !== deduplicatedCount) {
            console.log(`[ChromaDB] 🔄 Deduplicated ${originalCount - deduplicatedCount} duplicate documents (${originalCount} → ${deduplicatedCount})`);
        }

        const startTime = Date.now();
        console.log(`[ChromaDB] 📝 Starting to insert ${deduplicatedCount} documents into collection '${sanitizedName}'...`);

        try {
            const collection = await this.client!.getCollection({ name: sanitizedName });
            
            // Batch insert
            const batchSize = this.config.batchSize || 100;
            for (let i = 0; i < deduplicatedDocs.length; i += batchSize) {
                const batchStartTime = Date.now();
                const batch = deduplicatedDocs.slice(i, i + batchSize);

                const ids = batch.map(doc => doc.id);
                const embeddings = batch.map(doc => doc.vector);
                const documents_text = batch.map(doc => doc.content);
                const metadatas = batch.map(doc => ({
                    relativePath: doc.relativePath,
                    startLine: doc.startLine,
                    endLine: doc.endLine,
                    fileExtension: doc.fileExtension,
                    ...doc.metadata
                }));

                // Use upsert to handle existing documents
                await collection.upsert({
                    ids,
                    embeddings,
                    documents: documents_text,
                    metadatas
                });

                const batchDuration = Date.now() - batchStartTime;
                console.log(`✅ Batch inserted ${batch.length} documents (${i + 1}-${Math.min(i + batchSize, deduplicatedCount)} of ${deduplicatedCount}) in ${batchDuration}ms`);
            }

            const totalDuration = Date.now() - startTime;
            const docsPerSecond = Math.round((deduplicatedCount / totalDuration) * 1000);
            console.log(`✅ Successfully inserted ${deduplicatedCount} documents into ChromaDB collection '${sanitizedName}' in ${totalDuration}ms (${docsPerSecond} docs/sec)`);
        } catch (error) {
            console.error(`❌ Failed to insert documents into collection '${collectionName}':`, error);
            throw error;
        }
    }

    async insertHybrid(collectionName: string, documents: VectorDocument[]): Promise<void> {
        // ChromaDB stores documents alongside embeddings by default
        // No special handling needed for hybrid insertion
        await this.insert(collectionName, documents);
    }

    async search(collectionName: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]> {
        await this.ensureInitialized();
        const sanitizedName = this.sanitizeCollectionName(collectionName);

        const topK = options?.topK || 10;
        const threshold = options?.threshold || 0.0;

        try {
            const collection = await this.client!.getCollection({ name: sanitizedName });

            // Build where filter if filterExpr is provided
            let whereFilter: Record<string, any> | undefined;
            if (options?.filterExpr) {
                whereFilter = this.parseFilterExpression(options.filterExpr);
            }

            const results = await collection.query({
                queryEmbeddings: [queryVector],
                nResults: topK,
                include: [IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
                where: whereFilter
            });

            if (!results.ids || results.ids.length === 0 || !results.ids[0]) {
                return [];
            }

            const searchResults: VectorSearchResult[] = [];
            const ids = results.ids[0];
            const distances = results.distances?.[0] || [];
            const documents_text = results.documents?.[0] || [];
            const metadatas = results.metadatas?.[0] || [];

            for (let i = 0; i < ids.length; i++) {
                // ChromaDB with cosine distance returns distance = 1 - cosine_similarity
                // So similarity = 1 - distance
                // Clamp to [0, 1] range to handle floating point imprecision
                const distance = distances[i] || 0;
                const score = Math.max(0, Math.min(1, 1 - distance));

                // Apply threshold filter
                if (score < threshold) {
                    continue;
                }

                const metadata = metadatas[i] || {};
                searchResults.push({
                    document: {
                        id: ids[i],
                        vector: queryVector,
                        content: documents_text[i] || '',
                        relativePath: (metadata.relativePath as string) || '',
                        startLine: (metadata.startLine as number) || 0,
                        endLine: (metadata.endLine as number) || 0,
                        fileExtension: (metadata.fileExtension as string) || '',
                        metadata: metadata
                    },
                    score
                });
            }

            return searchResults;
        } catch (error) {
            console.error(`❌ Failed to search collection '${collectionName}':`, error);
            throw error;
        }
    }

    async hybridSearch(collectionName: string, searchRequests: HybridSearchRequest[], options?: HybridSearchOptions): Promise<HybridSearchResult[]> {
        await this.ensureInitialized();
        const sanitizedName = this.sanitizeCollectionName(collectionName);
        const limit = options?.limit || 10;

        console.log(`[ChromaDB] 🔍 Performing hybrid search on collection '${sanitizedName}' with ${searchRequests.length} search requests`);

        try {
            // Find dense vector search request
            const denseRequest = searchRequests.find(req => req.anns_field === 'vector');
            // Find sparse/text search request  
            const sparseRequest = searchRequests.find(req => req.anns_field === 'sparse_vector');

            if (!denseRequest) {
                throw new Error('Dense vector search request is required for hybrid search');
            }

            const collection = await this.client!.getCollection({ name: sanitizedName });

            // Build where filter if filterExpr is provided
            let whereFilter: Record<string, any> | undefined;
            if (options?.filterExpr) {
                whereFilter = this.parseFilterExpression(options.filterExpr);
            }

            // ChromaDB doesn't have native hybrid search with BM25, but we can:
            // 1. Use vector search for semantic matching
            // 2. If text query is provided, filter results by document content
            const queryVector = denseRequest.data as number[];
            
            let whereDocumentFilter: string | undefined;
            if (sparseRequest && typeof sparseRequest.data === 'string') {
                // Use ChromaDB's document filter for text-based filtering
                whereDocumentFilter = sparseRequest.data;
                console.log(`[ChromaDB] 🔍 Hybrid search: vector search with text filter: "${sparseRequest.data.substring(0, 50)}..."`);
            }

            const results = await collection.query({
                queryEmbeddings: [queryVector],
                nResults: limit,
                include: [IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
                where: whereFilter,
                whereDocument: whereDocumentFilter ? { "$contains": whereDocumentFilter } : undefined
            });

            if (!results.ids || results.ids.length === 0 || !results.ids[0]) {
                console.log(`[ChromaDB] ⚠️  No results returned from hybrid search`);
                return [];
            }

            console.log(`[ChromaDB] ✅ Found ${results.ids[0].length} results from hybrid search`);

            const hybridResults: HybridSearchResult[] = [];
            const ids = results.ids[0];
            const distances = results.distances?.[0] || [];
            const documents_text = results.documents?.[0] || [];
            const metadatas = results.metadatas?.[0] || [];

            for (let i = 0; i < ids.length; i++) {
                // ChromaDB with cosine distance returns distance = 1 - cosine_similarity
                // So similarity = 1 - distance
                const distance = distances[i] || 0;
                const score = Math.max(0, Math.min(1, 1 - distance));
                const metadata = metadatas[i] || {};

                hybridResults.push({
                    document: {
                        id: ids[i],
                        vector: queryVector,
                        content: documents_text[i] || '',
                        relativePath: (metadata.relativePath as string) || '',
                        startLine: (metadata.startLine as number) || 0,
                        endLine: (metadata.endLine as number) || 0,
                        fileExtension: (metadata.fileExtension as string) || '',
                        metadata: metadata
                    },
                    score
                });
            }

            return hybridResults;
        } catch (error) {
            console.error(`❌ Failed to perform hybrid search on collection '${collectionName}':`, error);
            throw error;
        }
    }

    async delete(collectionName: string, ids: string[]): Promise<void> {
        await this.ensureInitialized();
        const sanitizedName = this.sanitizeCollectionName(collectionName);

        if (ids.length === 0) {
            return;
        }

        try {
            const collection = await this.client!.getCollection({ name: sanitizedName });
            await collection.delete({ ids });
            console.log(`✅ Deleted ${ids.length} documents from ChromaDB collection '${sanitizedName}'`);
        } catch (error) {
            console.error(`❌ Failed to delete documents from collection '${collectionName}':`, error);
            throw error;
        }
    }

    async query(collectionName: string, filter: string, outputFields: string[], limit?: number): Promise<Record<string, any>[]> {
        await this.ensureInitialized();
        const sanitizedName = this.sanitizeCollectionName(collectionName);
        const queryLimit = limit || 100;

        try {
            const collection = await this.client!.getCollection({ name: sanitizedName });

            // Build where filter
            let whereFilter: Record<string, any> | undefined;
            if (filter && filter.trim()) {
                whereFilter = this.parseFilterExpression(filter);
            }

            // Get all documents matching the filter
            const results = await collection.get({
                where: whereFilter,
                include: [IncludeEnum.documents, IncludeEnum.metadatas],
                limit: queryLimit
            });

            if (!results.ids || results.ids.length === 0) {
                return [];
            }

            const queryResults: Record<string, any>[] = [];
            for (let i = 0; i < results.ids.length; i++) {
                const metadata = results.metadatas?.[i] || {};
                const doc: Record<string, any> = {
                    id: results.ids[i],
                    content: results.documents?.[i] || '',
                    relativePath: metadata.relativePath,
                    startLine: metadata.startLine,
                    endLine: metadata.endLine,
                    fileExtension: metadata.fileExtension,
                    metadata
                };

                // Filter to only include requested fields
                if (outputFields.length > 0) {
                    const filtered: Record<string, any> = {};
                    for (const field of outputFields) {
                        if (field in doc) {
                            filtered[field] = doc[field];
                        }
                    }
                    queryResults.push(filtered);
                } else {
                    queryResults.push(doc);
                }
            }

            return queryResults;
        } catch (error) {
            console.error(`❌ Failed to query collection '${collectionName}':`, error);
            throw error;
        }
    }

    async checkCollectionLimit(): Promise<boolean> {
        await this.ensureInitialized();

        try {
            // ChromaDB doesn't have collection limits like cloud services
            // Just verify we can list collections
            await this.client!.listCollections();
            return true;
        } catch (error) {
            console.error('❌ Failed to check collection limit:', error);
            return false;
        }
    }

    /**
     * Get collection statistics
     */
    async getCollectionStats(collectionName: string): Promise<{ entityCount: number }> {
        await this.ensureInitialized();
        const sanitizedName = this.sanitizeCollectionName(collectionName);

        try {
            const collection = await this.client!.getCollection({ name: sanitizedName });
            const count = await collection.count();
            return { entityCount: count };
        } catch (error) {
            console.error(`❌ Failed to get collection stats for '${collectionName}':`, error);
            throw error;
        }
    }

    /**
     * Parse a filter expression string into ChromaDB where clause format
     * Supports basic filters like: fileExtension in [".ts", ".py"]
     */
    private parseFilterExpression(filterExpr: string): Record<string, any> | undefined {
        if (!filterExpr || !filterExpr.trim()) {
            return undefined;
        }

        // Handle 'field in ["value1", "value2"]' pattern
        const inMatch = filterExpr.match(/(\w+)\s+in\s+\[(.+)\]/i);
        if (inMatch) {
            const field = inMatch[1];
            const valuesStr = inMatch[2];
            const values = valuesStr
                .split(',')
                .map(v => v.trim().replace(/^["']|["']$/g, ''));
            return { [field]: { "$in": values } };
        }

        // Handle 'field = "value"' pattern
        const eqMatch = filterExpr.match(/(\w+)\s*=\s*["'](.+)["']/);
        if (eqMatch) {
            return { [eqMatch[1]]: eqMatch[2] };
        }

        // Handle 'field == "value"' pattern
        const eqMatch2 = filterExpr.match(/(\w+)\s*==\s*["'](.+)["']/);
        if (eqMatch2) {
            return { [eqMatch2[1]]: eqMatch2[2] };
        }

        console.warn(`[ChromaDB] ⚠️  Could not parse filter expression: ${filterExpr}`);
        return undefined;
    }

    /**
     * Clean up resources and stop the server if we started it
     */
    async close(): Promise<void> {
        this.client = null;
        console.log('🔌 ChromaDB client connection closed');
        
        // Stop the server if we started it
        this.stopServer();
    }
}
