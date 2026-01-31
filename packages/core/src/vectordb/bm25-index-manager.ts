/**
 * BM25 Index Manager for ChromaDB Hybrid Search
 * 
 * Provides file-based BM25 indexing to enable true hybrid search (dense + sparse)
 * with ChromaDB, which doesn't natively support BM25/sparse vectors.
 * 
 * Features:
 * - File-based persistence (survives restarts)
 * - Per-collection indexes
 * - Incremental updates (add/remove documents)
 * - Configurable BM25 parameters (k1, b)
 * - Code-aware tokenization (camelCase, snake_case splitting)
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

/**
 * BM25 configuration options
 */
export interface BM25Config {
    /** Term frequency saturation parameter (default: 1.5) */
    k1?: number;
    /** Document length normalization parameter (default: 0.75) */
    b?: number;
    /** Path to store BM25 indexes */
    indexPath?: string;
}

/**
 * A document in the BM25 index
 */
export interface BM25Document {
    /** Unique document ID */
    id: string;
    /** Tokenized terms with frequencies */
    termFrequencies: Map<string, number>;
    /** Total number of terms in document */
    termCount: number;
}

/**
 * BM25 search result
 */
export interface BM25SearchResult {
    /** Document ID */
    id: string;
    /** BM25 score */
    score: number;
}

/**
 * Serializable BM25 index data for file persistence
 */
interface BM25IndexData {
    /** Index version for compatibility checks */
    version: string;
    /** Collection name */
    collectionName: string;
    /** BM25 k1 parameter */
    k1: number;
    /** BM25 b parameter */
    b: number;
    /** Average document length */
    avgDocLength: number;
    /** Total number of documents */
    documentCount: number;
    /** Document frequency for each term (number of docs containing term) */
    documentFrequencies: Record<string, number>;
    /** Documents with term frequencies */
    documents: Array<{
        id: string;
        termFrequencies: Record<string, number>;
        termCount: number;
    }>;
    /** Timestamp of last update */
    lastUpdated: string;
}

/**
 * BM25 Index Manager
 * 
 * Manages file-based BM25 indexes for hybrid search with ChromaDB.
 */
export class BM25IndexManager {
    private static readonly INDEX_VERSION = '1.0';
    private static readonly DEFAULT_K1 = 1.5;
    private static readonly DEFAULT_B = 0.75;
    
    private k1: number;
    private b: number;
    private indexPath: string;
    
    // Per-collection index data
    private indexes: Map<string, {
        documents: Map<string, BM25Document>;
        documentFrequencies: Map<string, number>;
        avgDocLength: number;
        dirty: boolean; // Track if index needs to be saved
    }> = new Map();
    
    constructor(config: BM25Config = {}) {
        this.k1 = config.k1 ?? BM25IndexManager.DEFAULT_K1;
        this.b = config.b ?? BM25IndexManager.DEFAULT_B;
        this.indexPath = config.indexPath ?? path.join(
            process.env.CONTEXT_DATA_PATH || path.join(require('os').homedir(), '.velocity', 'context'),
            'bm25'
        );
        
        // Ensure index directory exists
        if (!fs.existsSync(this.indexPath)) {
            fs.mkdirSync(this.indexPath, { recursive: true });
            console.log(`[BM25] Created index directory: ${this.indexPath}`);
        }
    }
    
    /**
     * Get the file path for a collection's BM25 index
     */
    private getIndexFilePath(collectionName: string): string {
        // Sanitize collection name for filename
        const sanitized = collectionName
            .replace(/[^a-zA-Z0-9_-]/g, '_')
            .toLowerCase();
        return path.join(this.indexPath, `${sanitized}.bm25.json`);
    }
    
    /**
     * Load a collection's BM25 index from disk
     */
    async loadIndex(collectionName: string): Promise<boolean> {
        const indexPath = this.getIndexFilePath(collectionName);
        
        if (!fs.existsSync(indexPath)) {
            console.log(`[BM25] No existing index found for collection '${collectionName}'`);
            return false;
        }
        
        try {
            const data = JSON.parse(fs.readFileSync(indexPath, 'utf-8')) as BM25IndexData;
            
            // Version check
            if (data.version !== BM25IndexManager.INDEX_VERSION) {
                console.warn(`[BM25] Index version mismatch for '${collectionName}' (got ${data.version}, expected ${BM25IndexManager.INDEX_VERSION}). Rebuilding...`);
                return false;
            }
            
            // Reconstruct in-memory index
            const documents = new Map<string, BM25Document>();
            const documentFrequencies = new Map<string, number>();
            
            for (const doc of data.documents) {
                documents.set(doc.id, {
                    id: doc.id,
                    termFrequencies: new Map(Object.entries(doc.termFrequencies)),
                    termCount: doc.termCount,
                });
            }
            
            for (const [term, freq] of Object.entries(data.documentFrequencies)) {
                documentFrequencies.set(term, freq);
            }
            
            this.indexes.set(collectionName, {
                documents,
                documentFrequencies,
                avgDocLength: data.avgDocLength,
                dirty: false,
            });
            
            console.log(`[BM25] Loaded index for '${collectionName}': ${documents.size} documents, ${documentFrequencies.size} terms`);
            return true;
        } catch (error) {
            console.error(`[BM25] Failed to load index for '${collectionName}':`, error);
            return false;
        }
    }
    
    /**
     * Save a collection's BM25 index to disk
     */
    async saveIndex(collectionName: string): Promise<boolean> {
        const index = this.indexes.get(collectionName);
        if (!index) {
            console.warn(`[BM25] No index to save for collection '${collectionName}'`);
            return false;
        }
        
        const indexPath = this.getIndexFilePath(collectionName);
        
        try {
            const data: BM25IndexData = {
                version: BM25IndexManager.INDEX_VERSION,
                collectionName,
                k1: this.k1,
                b: this.b,
                avgDocLength: index.avgDocLength,
                documentCount: index.documents.size,
                documentFrequencies: Object.fromEntries(index.documentFrequencies),
                documents: Array.from(index.documents.values()).map(doc => ({
                    id: doc.id,
                    termFrequencies: Object.fromEntries(doc.termFrequencies),
                    termCount: doc.termCount,
                })),
                lastUpdated: new Date().toISOString(),
            };
            
            // Write atomically using temp file
            const tempPath = indexPath + '.tmp';
            fs.writeFileSync(tempPath, JSON.stringify(data, null, 2), 'utf-8');
            fs.renameSync(tempPath, indexPath);
            
            index.dirty = false;
            console.log(`[BM25] Saved index for '${collectionName}': ${index.documents.size} documents`);
            return true;
        } catch (error) {
            console.error(`[BM25] Failed to save index for '${collectionName}':`, error);
            return false;
        }
    }
    
    /**
     * Initialize or get an index for a collection
     */
    private ensureIndex(collectionName: string): {
        documents: Map<string, BM25Document>;
        documentFrequencies: Map<string, number>;
        avgDocLength: number;
        dirty: boolean;
    } {
        let index = this.indexes.get(collectionName);
        if (!index) {
            index = {
                documents: new Map(),
                documentFrequencies: new Map(),
                avgDocLength: 0,
                dirty: false,
            };
            this.indexes.set(collectionName, index);
        }
        return index;
    }
    
    /**
     * Tokenize text for BM25 indexing
     * 
     * Handles code-specific patterns:
     * - camelCase splitting
     * - snake_case splitting
     * - Removes common stop words
     * - Lowercases all tokens
     */
    private tokenize(text: string): string[] {
        if (!text) return [];
        
        // Split camelCase and PascalCase
        let processed = text.replace(/([a-z])([A-Z])/g, '$1 $2');
        
        // Split snake_case and kebab-case
        processed = processed.replace(/[_-]/g, ' ');
        
        // Remove special characters but keep alphanumerics and spaces
        processed = processed.replace(/[^\w\s]/g, ' ');
        
        // Lowercase and split
        const tokens = processed.toLowerCase().split(/\s+/).filter(t => t.length > 0);
        
        // Remove common stop words (minimal set to preserve code semantics)
        const stopWords = new Set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'it', 'its',
        ]);
        
        return tokens.filter(t => t.length > 1 && !stopWords.has(t));
    }
    
    /**
     * Calculate term frequencies for a document
     */
    private calculateTermFrequencies(tokens: string[]): Map<string, number> {
        const frequencies = new Map<string, number>();
        for (const token of tokens) {
            frequencies.set(token, (frequencies.get(token) || 0) + 1);
        }
        return frequencies;
    }
    
    /**
     * Add documents to the BM25 index
     * 
     * @param collectionName Collection to add documents to
     * @param documents Array of {id, content} objects
     */
    async addDocuments(
        collectionName: string,
        documents: Array<{ id: string; content: string }>
    ): Promise<void> {
        const index = this.ensureIndex(collectionName);
        
        let totalTermsAdded = 0;
        let docsAdded = 0;
        
        for (const doc of documents) {
            // Skip if document already exists (use upsert logic)
            if (index.documents.has(doc.id)) {
                // Remove old document first
                await this.removeDocuments(collectionName, [doc.id]);
            }
            
            const tokens = this.tokenize(doc.content);
            const termFrequencies = this.calculateTermFrequencies(tokens);
            
            const bm25Doc: BM25Document = {
                id: doc.id,
                termFrequencies,
                termCount: tokens.length,
            };
            
            index.documents.set(doc.id, bm25Doc);
            
            // Update document frequencies
            for (const term of termFrequencies.keys()) {
                index.documentFrequencies.set(
                    term,
                    (index.documentFrequencies.get(term) || 0) + 1
                );
            }
            
            totalTermsAdded += tokens.length;
            docsAdded++;
        }
        
        // Update average document length
        if (index.documents.size > 0) {
            let totalTerms = 0;
            for (const doc of index.documents.values()) {
                totalTerms += doc.termCount;
            }
            index.avgDocLength = totalTerms / index.documents.size;
        }
        
        index.dirty = true;
        
        console.log(`[BM25] Added ${docsAdded} documents to '${collectionName}' (${totalTermsAdded} terms)`);
    }
    
    /**
     * Remove documents from the BM25 index
     * 
     * @param collectionName Collection to remove documents from
     * @param documentIds Array of document IDs to remove
     */
    async removeDocuments(collectionName: string, documentIds: string[]): Promise<void> {
        const index = this.indexes.get(collectionName);
        if (!index) return;
        
        let docsRemoved = 0;
        
        for (const docId of documentIds) {
            const doc = index.documents.get(docId);
            if (!doc) continue;
            
            // Update document frequencies
            for (const term of doc.termFrequencies.keys()) {
                const freq = index.documentFrequencies.get(term) || 0;
                if (freq <= 1) {
                    index.documentFrequencies.delete(term);
                } else {
                    index.documentFrequencies.set(term, freq - 1);
                }
            }
            
            index.documents.delete(docId);
            docsRemoved++;
        }
        
        // Update average document length
        if (index.documents.size > 0) {
            let totalTerms = 0;
            for (const doc of index.documents.values()) {
                totalTerms += doc.termCount;
            }
            index.avgDocLength = totalTerms / index.documents.size;
        } else {
            index.avgDocLength = 0;
        }
        
        if (docsRemoved > 0) {
            index.dirty = true;
            console.log(`[BM25] Removed ${docsRemoved} documents from '${collectionName}'`);
        }
    }
    
    /**
     * Calculate BM25 score for a document given a query
     */
    private calculateBM25Score(
        queryTokens: string[],
        document: BM25Document,
        documentCount: number,
        documentFrequencies: Map<string, number>,
        avgDocLength: number
    ): number {
        let score = 0;
        
        for (const term of queryTokens) {
            const tf = document.termFrequencies.get(term) || 0;
            if (tf === 0) continue;
            
            const df = documentFrequencies.get(term) || 0;
            if (df === 0) continue;
            
            // IDF calculation with smoothing
            const idf = Math.log((documentCount - df + 0.5) / (df + 0.5) + 1);
            
            // TF normalization with BM25 formula
            const tfNorm = (tf * (this.k1 + 1)) / 
                (tf + this.k1 * (1 - this.b + this.b * (document.termCount / avgDocLength)));
            
            score += idf * tfNorm;
        }
        
        return score;
    }
    
    /**
     * Search the BM25 index
     * 
     * @param collectionName Collection to search
     * @param query Search query text
     * @param limit Maximum number of results
     * @returns Array of search results sorted by BM25 score (descending)
     */
    async search(
        collectionName: string,
        query: string,
        limit: number = 10
    ): Promise<BM25SearchResult[]> {
        const index = this.indexes.get(collectionName);
        if (!index || index.documents.size === 0) {
            console.log(`[BM25] No index or empty index for '${collectionName}'`);
            return [];
        }
        
        const queryTokens = this.tokenize(query);
        if (queryTokens.length === 0) {
            console.log(`[BM25] Query produced no tokens after tokenization`);
            return [];
        }
        
        console.log(`[BM25] Searching '${collectionName}' with ${queryTokens.length} terms: ${queryTokens.slice(0, 5).join(', ')}${queryTokens.length > 5 ? '...' : ''}`);
        
        const results: BM25SearchResult[] = [];
        
        for (const doc of index.documents.values()) {
            const score = this.calculateBM25Score(
                queryTokens,
                doc,
                index.documents.size,
                index.documentFrequencies,
                index.avgDocLength
            );
            
            if (score > 0) {
                results.push({ id: doc.id, score });
            }
        }
        
        // Sort by score descending and limit
        results.sort((a, b) => b.score - a.score);
        const topResults = results.slice(0, limit);
        
        console.log(`[BM25] Found ${results.length} matches, returning top ${topResults.length}`);
        
        return topResults;
    }
    
    /**
     * Check if an index exists for a collection
     */
    hasIndex(collectionName: string): boolean {
        return this.indexes.has(collectionName) || 
            fs.existsSync(this.getIndexFilePath(collectionName));
    }
    
    /**
     * Get index statistics for a collection
     */
    getIndexStats(collectionName: string): {
        documentCount: number;
        termCount: number;
        avgDocLength: number;
    } | null {
        const index = this.indexes.get(collectionName);
        if (!index) return null;
        
        return {
            documentCount: index.documents.size,
            termCount: index.documentFrequencies.size,
            avgDocLength: index.avgDocLength,
        };
    }
    
    /**
     * Drop an index for a collection
     */
    async dropIndex(collectionName: string): Promise<void> {
        this.indexes.delete(collectionName);
        
        const indexPath = this.getIndexFilePath(collectionName);
        if (fs.existsSync(indexPath)) {
            fs.unlinkSync(indexPath);
            console.log(`[BM25] Deleted index file for '${collectionName}'`);
        }
    }
    
    /**
     * Save all dirty indexes to disk
     */
    async saveAllDirtyIndexes(): Promise<void> {
        for (const [collectionName, index] of this.indexes) {
            if (index.dirty) {
                await this.saveIndex(collectionName);
            }
        }
    }
    
    /**
     * Clear all in-memory indexes (for cleanup)
     */
    clearAllIndexes(): void {
        this.indexes.clear();
    }
}

// Export a default instance creator
export function createBM25IndexManager(config?: BM25Config): BM25IndexManager {
    return new BM25IndexManager(config);
}
