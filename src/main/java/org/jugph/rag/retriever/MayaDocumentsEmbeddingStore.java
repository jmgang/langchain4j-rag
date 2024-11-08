package org.jugph.rag.retriever;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;

import static org.jugph.rag.config.ConfigLoader.getProperty;

public class MayaDocumentsEmbeddingStore {

    public static final int DEFAULT_DIMENSION = 384;

    private final String database;
    private final int dimension;
    private final EmbeddingStore<TextSegment> embeddingStore;

    public MayaDocumentsEmbeddingStore(String database, int dimension) {
        this.database = database;
        this.dimension = dimension;
        this.embeddingStore = createStore(database, dimension);
    }

    private EmbeddingStore<TextSegment> createStore(String database, int dimension) {
        return PgVectorEmbeddingStore.builder()
                .host(getProperty("vector_store.pg_vector.host"))
                .port(5432)
                .database(database)
                .user(getProperty("vector_store.pg_vector.username"))
                .password(getProperty("vector_store.pg_vector.password"))
                .table(getProperty("vector_store.pg_vector.table"))
                .dimension(dimension)
                .build();
    }

    public EmbeddingStore<TextSegment> getEmbeddingStore() {
        return embeddingStore;
    }

    public String getDatabase() {
        return database;
    }

    public int getDimension() {
        return dimension;
    }
}
