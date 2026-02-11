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
import { BM25IndexManager, BM25SearchResult, BM25Config } from './bm25-index-manager';

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
    /** BM25 configuration for hybrid search */
    bm25Config?: BM25Config;
    /** Enable BM25 hybrid search (default: true when HYBRID_MODE is enabled) */
    enableBM25?: boolean;
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
    
    // BM25 index manager for hybrid search
    private bm25IndexManager: BM25IndexManager;
    private enableBM25: boolean;
    // Track which collections are hybrid (have BM25 indexes)
    private hybridCollections: Set<string> = new Set();

    constructor(config: ChromaConfig) {
        this.config = config;
        // Default to ~/.velocity/chroma if no path specified
        this.chromaPath = config.path || path.join(os.homedir(), '.velocity', 'chroma');
        this.chromaPort = config.port || 8000;
        this.chromaHost = config.host || 'localhost';
        
        // Initialize BM25 index manager
        this.enableBM25 = config.enableBM25 !== false; // Default to true
        this.bm25IndexManager = new BM25IndexManager(config.bm25Config);
        
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
     * 
     * If startup fails due to port conflict but a server is already running,
     * this method will succeed by connecting to the existing server.
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
            console.log(`✅ ChromaDB server already running at ${this.chromaHost}:${this.chromaPort} (external or previous session)`);
            // Mark as externally managed so we don't try to stop it on shutdown
            ChromaVectorDatabase.serverStarted = true;
            ChromaVectorDatabase.serverProcess = null; // null process means externally managed
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
        } catch (error) {
            // Server startup failed - check if another server is already running
            // This can happen if the initial isServerRunning() check raced with another process
            console.log(`⚠️  ChromaDB server startup failed, checking if external server is available...`);
            
            if (await this.isServerRunning()) {
                console.log(`✅ External ChromaDB server detected at ${this.chromaHost}:${this.chromaPort} - will use existing server`);
                // Mark as externally managed
                ChromaVectorDatabase.serverStarted = true;
                ChromaVectorDatabase.serverProcess = null;
                return; // Success - we can use the existing server
            }
            
            // No external server available, re-throw the error
            throw error;
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
     * Find the chromadb CLI script in node_modules
     * Returns the path to dist/cli.mjs if found, null otherwise
     */
    private findChromaCliPath(): string | null {
        // Log for debugging
        console.log(`[ChromaDB] Looking for CLI, cwd: ${process.cwd()}`);

        // Try to find chromadb's CLI in various locations
        // Note: In bundled Electron apps, process.cwd() is set to the resources directory
        const possiblePaths = [
            // Electron bundled app: resources/mcp/claude-context/node_modules/chromadb/dist/cli.mjs
            path.join(process.cwd(), 'mcp', 'claude-context', 'node_modules', 'chromadb', 'dist', 'cli.mjs'),
            // Development: relative to project root
            path.join(process.cwd(), 'node_modules', 'chromadb', 'dist', 'cli.mjs'),
            // Development: mcp/claude-context workspace
            path.join(process.cwd(), 'mcp', 'claude-context', 'node_modules', 'chromadb', 'dist', 'cli.mjs'),
        ];

        for (const cliPath of possiblePaths) {
            console.log(`[ChromaDB] Checking path: ${cliPath}`);
            if (fs.existsSync(cliPath)) {
                console.log(`[ChromaDB] Found CLI at: ${cliPath}`);
                return cliPath;
            }
        }

        console.log(`[ChromaDB] CLI not found in any of the expected locations`);
        return null;
    }

    /**
     * Kill any stale process occupying the ChromaDB port.
     * This prevents "address already in use" failures when restarting.
     */
    private killStaleProcessOnPort(): void {
        try {
            if (process.platform === 'win32') {
                const { execSync } = require('child_process');
                const result = execSync(`netstat -ano | findstr :${this.chromaPort}`, { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'ignore'] });
                const lines = result.split('\n').filter((line: string) => line.includes('LISTENING'));
                for (const line of lines) {
                    const parts = line.trim().split(/\s+/);
                    const pid = parts[parts.length - 1];
                    if (pid && !isNaN(parseInt(pid))) {
                        execSync(`taskkill /pid ${pid} /f /t`, { stdio: 'ignore' });
                    }
                }
            } else {
                const { execSync } = require('child_process');
                // Try SIGTERM first for graceful shutdown
                execSync(`lsof -ti:${this.chromaPort} | xargs kill -15 2>/dev/null || true`, { stdio: 'ignore' });
                // Brief pause, then force kill any remaining
                execSync(`sleep 0.5 && lsof -ti:${this.chromaPort} | xargs kill -9 2>/dev/null || true`, { stdio: 'ignore' });
            }
        } catch {
            // No process on port or command failed - that's fine
        }
    }

    /**
     * Start the ChromaDB server process
     * Uses the chromadb npm package's native bindings (Rust-compiled)
     */
    private async startServer(): Promise<void> {
        // Kill any stale process on our port before attempting to start
        this.killStaleProcessOnPort();

        return new Promise((resolve, reject) => {
            const isWindows = process.platform === 'win32';

            // Try to find chromadb CLI directly (more reliable than npx)
            const chromaCliPath = this.findChromaCliPath();

            let chromaProcess: ChildProcess;

            if (chromaCliPath) {
                // Use Node.js directly to run the CLI (works in Electron without npx)
                // process.execPath is the path to Node.js (or Electron acting as Node)
                console.log(`🚀 Starting ChromaDB server using CLI at: ${chromaCliPath}`);
                chromaProcess = spawn(process.execPath, [
                    chromaCliPath,
                    'run',
                    '--path', this.chromaPath,
                    '--port', this.chromaPort.toString(),
                    '--host', this.chromaHost
                ], {
                    stdio: ['ignore', 'pipe', 'pipe'],
                    detached: false,
                    env: { ...process.env }
                });
            } else {
                // Fallback to npx (requires npm to be installed)
                // On Windows, we need shell: true for npx to be found correctly
                console.log(`🚀 Starting ChromaDB server using npx (CLI not found directly)`);
                chromaProcess = spawn('npx', [
                    'chromadb', 'run',
                    '--path', this.chromaPath,
                    '--port', this.chromaPort.toString(),
                    '--host', this.chromaHost
                ], {
                    stdio: ['ignore', 'pipe', 'pipe'],
                    detached: false,
                    shell: isWindows, // Required on Windows to find npx.cmd
                    env: { ...process.env }
                });
            }

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

    // Track if cleanup handlers have been registered
    private static cleanupHandlersRegistered: boolean = false;

    /**
     * Register cleanup handlers to stop the server on process exit
     * Only registers handlers if we started the server ourselves (serverProcess is not null)
     */
    private registerCleanupHandlers(): void {
        // Only register handlers once, and only if we own the server process
        if (ChromaVectorDatabase.cleanupHandlersRegistered) {
            return;
        }
        
        // Don't register cleanup if using externally managed server
        if (!ChromaVectorDatabase.serverProcess) {
            console.log('ℹ️  Skipping cleanup handler registration for externally managed server');
            return;
        }

        ChromaVectorDatabase.cleanupHandlersRegistered = true;
        console.log('📝 Registering ChromaDB cleanup handlers');

        const cleanup = () => {
            this.stopServer();
        };

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
     * Only stops the server if we started it (serverProcess is not null)
     * If serverProcess is null but serverStarted is true, the server is externally managed
     */
    private stopServer(): void {
        if (ChromaVectorDatabase.serverProcess) {
            console.log('🛑 Stopping ChromaDB server (started by this process)...');
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
        } else if (ChromaVectorDatabase.serverStarted) {
            // Server is externally managed - don't stop it, just clear our state
            console.log('ℹ️  ChromaDB server is externally managed - not stopping');
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
        // Create the ChromaDB collection
        console.log(`[ChromaDB] 📝 Creating hybrid collection '${collectionName}' with BM25 support`);
        await this.createCollection(collectionName, dimension, description);
        
        const sanitizedName = this.sanitizeCollectionName(collectionName);
        
        // Initialize or load the BM25 index for this collection
        if (this.enableBM25) {
            // Try to load existing BM25 index
            const loaded = await this.bm25IndexManager.loadIndex(sanitizedName);
            if (!loaded) {
                console.log(`[ChromaDB] 📝 No existing BM25 index found for '${collectionName}', will create on first insert`);
            }
            this.hybridCollections.add(sanitizedName);
            console.log(`[ChromaDB] ✅ Hybrid collection '${collectionName}' ready with BM25 indexing enabled`);
        } else {
            console.log(`[ChromaDB] ⚠️ BM25 disabled - hybrid collection will use vector-only search`);
        }
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
            } else {
                console.error(`❌ Failed to drop collection '${collectionName}':`, error);
                throw error;
            }
        }
        
        // Also drop the BM25 index if it exists
        if (this.hybridCollections.has(sanitizedName)) {
            await this.bm25IndexManager.dropIndex(sanitizedName);
            this.hybridCollections.delete(sanitizedName);
            console.log(`✅ BM25 index for '${sanitizedName}' dropped successfully`);
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
        const sanitizedName = this.sanitizeCollectionName(collectionName);
        
        // Insert into ChromaDB (vector store)
        await this.insert(collectionName, documents);
        
        // Insert into BM25 index for hybrid search
        if (this.enableBM25 && documents.length > 0) {
            console.log(`[ChromaDB] 📝 Adding ${documents.length} documents to BM25 index for '${collectionName}'`);
            
            // Prepare documents for BM25 indexing
            const bm25Docs = documents.map(doc => ({
                id: doc.id,
                content: doc.content,
            }));
            
            await this.bm25IndexManager.addDocuments(sanitizedName, bm25Docs);
            
            // Mark as hybrid collection
            this.hybridCollections.add(sanitizedName);
            
            // Save BM25 index to disk (persist after each batch for durability)
            await this.bm25IndexManager.saveIndex(sanitizedName);
            
            const stats = this.bm25IndexManager.getIndexStats(sanitizedName);
            if (stats) {
                console.log(`[ChromaDB] ✅ BM25 index updated: ${stats.documentCount} docs, ${stats.termCount} terms`);
            }
        }
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
        const rrfK = options?.rerank?.params?.k || 60; // RRF k parameter

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

            const queryVector = denseRequest.data as number[];
            const queryText = sparseRequest?.data as string || '';
            
            // Check if BM25 hybrid search is enabled and available for this collection
            const useBM25 = this.enableBM25 && 
                this.hybridCollections.has(sanitizedName) && 
                queryText.length > 0;

            if (useBM25) {
                console.log(`[ChromaDB] 🔍 TRUE HYBRID SEARCH (vector + BM25) for: "${queryText.substring(0, 50)}${queryText.length > 50 ? '...' : ''}"`);
                
                // Ensure BM25 index is loaded
                if (!this.bm25IndexManager.hasIndex(sanitizedName)) {
                    await this.bm25IndexManager.loadIndex(sanitizedName);
                }
                
                // 1. Vector search via ChromaDB
                const vectorResults = await collection.query({
                    queryEmbeddings: [queryVector],
                    nResults: Math.max(limit * 2, 20), // Get more results for better fusion
                    include: [IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
                    where: whereFilter
                });

                // 2. BM25 search
                const bm25Results = await this.bm25IndexManager.search(
                    sanitizedName,
                    queryText,
                    Math.max(limit * 2, 20)
                );

                // 3. Merge results using RRF (Reciprocal Rank Fusion)
                return this.mergeWithRRF(
                    vectorResults,
                    bm25Results,
                    queryVector,
                    rrfK,
                    limit
                );
            } else {
                // Fallback to vector-only search
                if (queryText.length > 0) {
                    console.log(`[ChromaDB] 🔍 VECTOR-ONLY search (BM25 ${this.enableBM25 ? 'not available for this collection' : 'disabled'})`);
                }

                const results = await collection.query({
                    queryEmbeddings: [queryVector],
                    nResults: limit,
                    include: [IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
                    where: whereFilter
                });

                if (!results.ids || results.ids.length === 0 || !results.ids[0]) {
                    console.log(`[ChromaDB] ⚠️  No results returned from search`);
                    return [];
                }

                console.log(`[ChromaDB] ✅ Found ${results.ids[0].length} results from vector search`);

                const hybridResults: HybridSearchResult[] = [];
                const ids = results.ids[0];
                const distances = results.distances?.[0] || [];
                const documents_text = results.documents?.[0] || [];
                const metadatas = results.metadatas?.[0] || [];

                for (let i = 0; i < ids.length; i++) {
                    // ChromaDB with cosine distance returns distance = 1 - cosine_similarity
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
            }
        } catch (error) {
            console.error(`❌ Failed to perform hybrid search on collection '${collectionName}':`, error);
            throw error;
        }
    }

    /**
     * Merge vector search and BM25 search results using Reciprocal Rank Fusion (RRF)
     * 
     * RRF Score = sum(1 / (k + rank)) for each ranking where the document appears
     * This method effectively combines semantic (vector) and lexical (BM25) signals.
     */
    private async mergeWithRRF(
        vectorResults: {
            ids: string[][] | null;
            distances?: (number | null)[][] | null;
            documents?: (string | null)[][] | null;
            metadatas?: (Record<string, unknown> | null)[][] | null;
        },
        bm25Results: BM25SearchResult[],
        queryVector: number[],
        k: number,
        limit: number
    ): Promise<HybridSearchResult[]> {
        // Build a map of document scores and metadata
        const documentScores = new Map<string, {
            vectorRank: number | null;
            bm25Rank: number | null;
            vectorScore: number;
            bm25Score: number;
            content: string;
            metadata: Record<string, unknown>;
        }>();

        // Process vector results
        const vectorIds = vectorResults.ids?.[0] || [];
        const vectorDistances = vectorResults.distances?.[0] || [];
        const vectorDocuments = vectorResults.documents?.[0] || [];
        const vectorMetadatas = vectorResults.metadatas?.[0] || [];

        for (let i = 0; i < vectorIds.length; i++) {
            const id = vectorIds[i];
            const distance = vectorDistances[i] || 0;
            const vectorScore = Math.max(0, Math.min(1, 1 - distance));
            
            documentScores.set(id, {
                vectorRank: i + 1,
                bm25Rank: null,
                vectorScore,
                bm25Score: 0,
                content: vectorDocuments[i] || '',
                metadata: (vectorMetadatas[i] as Record<string, unknown>) || {},
            });
        }

        // Process BM25 results
        for (let i = 0; i < bm25Results.length; i++) {
            const result = bm25Results[i];
            const existing = documentScores.get(result.id);
            
            if (existing) {
                existing.bm25Rank = i + 1;
                existing.bm25Score = result.score;
            } else {
                // Document found only by BM25 - we need to look it up
                // For now, we skip these since we don't have their content/metadata
                // In a production system, you'd fetch the document from ChromaDB
                console.log(`[ChromaDB] 🔍 BM25 found document not in vector results: ${result.id}`);
            }
        }

        // Calculate RRF scores
        const rrfResults: Array<{
            id: string;
            rrfScore: number;
            vectorScore: number;
            bm25Score: number;
            content: string;
            metadata: Record<string, unknown>;
        }> = [];

        for (const [id, data] of documentScores) {
            let rrfScore = 0;
            
            // Add vector rank contribution
            if (data.vectorRank !== null) {
                rrfScore += 1 / (k + data.vectorRank);
            }
            
            // Add BM25 rank contribution
            if (data.bm25Rank !== null) {
                rrfScore += 1 / (k + data.bm25Rank);
            }
            
            rrfResults.push({
                id,
                rrfScore,
                vectorScore: data.vectorScore,
                bm25Score: data.bm25Score,
                content: data.content,
                metadata: data.metadata,
            });
        }

        // Sort by RRF score descending
        rrfResults.sort((a, b) => b.rrfScore - a.rrfScore);

        // Take top results
        const topResults = rrfResults.slice(0, limit);

        // Log fusion statistics
        const bothSignals = topResults.filter(r => 
            documentScores.get(r.id)?.vectorRank !== null && 
            documentScores.get(r.id)?.bm25Rank !== null
        ).length;
        const vectorOnly = topResults.filter(r => 
            documentScores.get(r.id)?.vectorRank !== null && 
            documentScores.get(r.id)?.bm25Rank === null
        ).length;
        const bm25Only = topResults.filter(r => 
            documentScores.get(r.id)?.vectorRank === null && 
            documentScores.get(r.id)?.bm25Rank !== null
        ).length;

        console.log(`[ChromaDB] ✅ RRF merged ${topResults.length} results (both: ${bothSignals}, vector-only: ${vectorOnly}, BM25-only: ${bm25Only})`);

        // Convert to HybridSearchResult format
        return topResults.map(result => ({
            document: {
                id: result.id,
                vector: queryVector,
                content: result.content,
                relativePath: (result.metadata.relativePath as string) || '',
                startLine: (result.metadata.startLine as number) || 0,
                endLine: (result.metadata.endLine as number) || 0,
                fileExtension: (result.metadata.fileExtension as string) || '',
                metadata: result.metadata,
            },
            score: result.rrfScore, // Use RRF score as the final score
        }));
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
        
        // Also remove from BM25 index if this is a hybrid collection
        if (this.hybridCollections.has(sanitizedName)) {
            await this.bm25IndexManager.removeDocuments(sanitizedName, ids);
            await this.bm25IndexManager.saveIndex(sanitizedName);
            console.log(`✅ Removed ${ids.length} documents from BM25 index`);
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
        // Save any dirty BM25 indexes before closing
        await this.bm25IndexManager.saveAllDirtyIndexes();
        this.bm25IndexManager.clearAllIndexes();
        this.hybridCollections.clear();
        
        this.client = null;
        console.log('🔌 ChromaDB client connection closed');
        
        // Stop the server if we started it
        this.stopServer();
    }
    
    /**
     * Get BM25 index statistics for a collection
     * Useful for debugging and testing
     */
    getBM25Stats(collectionName: string): { documentCount: number; termCount: number; avgDocLength: number } | null {
        const sanitizedName = this.sanitizeCollectionName(collectionName);
        return this.bm25IndexManager.getIndexStats(sanitizedName);
    }
    
    /**
     * Check if BM25 hybrid search is enabled
     */
    isBM25Enabled(): boolean {
        return this.enableBM25;
    }
    
    /**
     * Check if a collection has BM25 indexing
     */
    isHybridCollection(collectionName: string): boolean {
        const sanitizedName = this.sanitizeCollectionName(collectionName);
        return this.hybridCollections.has(sanitizedName);
    }

    /**
     * Add documents to BM25 index only (without vector embeddings)
     * This is useful for fast pre-filtering before more expensive vector search.
     * 
     * @param collectionName Collection name
     * @param documents Documents with id and content
     */
    async addBM25Documents(collectionName: string, documents: Array<{ id: string; content: string }>): Promise<void> {
        if (!this.enableBM25) {
            console.warn('[ChromaDB] BM25 is not enabled, skipping addBM25Documents');
            return;
        }

        const sanitizedName = this.sanitizeCollectionName(collectionName);
        console.log(`[ChromaDB] 📝 Adding ${documents.length} documents to BM25-only index for '${collectionName}'`);
        
        await this.bm25IndexManager.addDocuments(sanitizedName, documents);
        
        // Save the index to disk
        await this.bm25IndexManager.saveIndex(sanitizedName);
        
        const stats = this.bm25IndexManager.getIndexStats(sanitizedName);
        if (stats) {
            console.log(`[ChromaDB] ✅ BM25-only index updated: ${stats.documentCount} docs, ${stats.termCount} terms`);
        }
    }

    /**
     * Search using BM25 only (keyword-based search without embeddings)
     * This is useful for fast pre-filtering to identify relevant documents.
     * 
     * @param collectionName Collection name
     * @param query Search query text
     * @param limit Maximum number of results
     * @returns Array of search results with document IDs and BM25 scores
     */
    async searchBM25(collectionName: string, query: string, limit: number = 10): Promise<Array<{ id: string; score: number }>> {
        if (!this.enableBM25) {
            console.warn('[ChromaDB] BM25 is not enabled, returning empty results');
            return [];
        }

        const sanitizedName = this.sanitizeCollectionName(collectionName);
        
        // Ensure BM25 index is loaded
        if (!this.bm25IndexManager.hasIndex(sanitizedName)) {
            const loaded = await this.bm25IndexManager.loadIndex(sanitizedName);
            if (!loaded) {
                console.log(`[ChromaDB] No BM25 index found for '${collectionName}'`);
                return [];
            }
        }
        
        console.log(`[ChromaDB] 🔍 BM25-only search in '${collectionName}' for: "${query.substring(0, 50)}${query.length > 50 ? '...' : ''}"`);
        
        const results = await this.bm25IndexManager.search(sanitizedName, query, limit);
        
        console.log(`[ChromaDB] ✅ BM25 search found ${results.length} results`);
        
        return results;
    }

    /**
     * Check if a BM25 index exists for a collection
     * 
     * @param collectionName Collection name
     */
    async hasBM25Index(collectionName: string): Promise<boolean> {
        if (!this.enableBM25) {
            return false;
        }
        const sanitizedName = this.sanitizeCollectionName(collectionName);
        return this.bm25IndexManager.hasIndex(sanitizedName);
    }

    /**
     * Load a BM25 index from disk if it exists
     * 
     * @param collectionName Collection name
     */
    async loadBM25Index(collectionName: string): Promise<boolean> {
        if (!this.enableBM25) {
            return false;
        }
        const sanitizedName = this.sanitizeCollectionName(collectionName);
        return await this.bm25IndexManager.loadIndex(sanitizedName);
    }

    /**
     * Save a BM25 index to disk
     * 
     * @param collectionName Collection name
     */
    async saveBM25Index(collectionName: string): Promise<void> {
        if (!this.enableBM25) {
            return;
        }
        const sanitizedName = this.sanitizeCollectionName(collectionName);
        await this.bm25IndexManager.saveIndex(sanitizedName);
    }
}
