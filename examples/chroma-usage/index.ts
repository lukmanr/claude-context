import { Context, ChromaVectorDatabase, OpenAIEmbedding } from '@zilliz/claude-context-core';

async function main() {
    try {
        console.log('🚀 Starting Chroma Vector Database Example');

        // Initialize embedding provider
        const embedding = new OpenAIEmbedding({
            apiKey: process.env.OPENAI_API_KEY || 'your-openai-api-key',
            model: 'text-embedding-3-small'
        });

        // Initialize Chroma vector database
        // The ChromaDB server is automatically started and managed
        const vectorDatabase = new ChromaVectorDatabase({
            path: process.env.CHROMA_PATH || '~/.velocity/chroma', // Data storage path
            port: 8000,  // ChromaDB server port
            host: 'localhost'  // ChromaDB server host
        });

        // Create context instance
        const context = new Context({
            embedding,
            vectorDatabase
        });

        console.log('📚 Testing basic Chroma operations...');

        // Test collection operations
        const collectionName = `test_collection_${Date.now()}`;
        
        // Create collection
        await vectorDatabase.createCollection(collectionName, 1536, 'Test collection for Chroma');
        console.log(`✅ Created collection: ${collectionName}`);

        // Check if collection exists
        const exists = await vectorDatabase.hasCollection(collectionName);
        console.log(`📦 Collection exists: ${exists}`);

        // List collections
        const collections = await vectorDatabase.listCollections();
        console.log(`📋 Available collections: ${collections.join(', ')}`);

        // Example: Index a small codebase (optional)
        // const stats = await context.indexCodebase('./your-project', (progress) => {
        //     console.log(`${progress.phase} - ${progress.percentage}%`);
        // });
        // console.log(`Indexed ${stats.indexedFiles} files, ${stats.totalChunks} chunks`);

        // Example: Perform semantic search (optional)
        // const results = await context.semanticSearch('./your-project', 'vector database operations', 5);
        // results.forEach(result => {
        //     console.log(`File: ${result.relativePath}:${result.startLine}-${result.endLine}`);
        //     console.log(`Score: ${(result.score * 100).toFixed(2)}%`);
        //     console.log(`Content: ${result.content.substring(0, 100)}...`);
        // });

        // Clean up - drop the test collection
        await vectorDatabase.dropCollection(collectionName);
        console.log(`🗑️  Dropped collection: ${collectionName}`);

        console.log('✅ Chroma Vector Database Example completed successfully!');

    } catch (error) {
        console.error('❌ Error in Chroma example:', error);
        process.exit(1);
    }
}

// Run the example
if (require.main === module) {
    main();
}

export { main };
