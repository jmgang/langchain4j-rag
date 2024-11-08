package org.jugph.rag.retriever;

import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

public class DocumentVectorRetriever {


//    public static ContentRetriever get() {
//        return EmbeddingStoreContentRetriever.builder()
//                .embeddingStore()
//                .embeddingModel(new BgeSmallEnV15QuantizedEmbeddingModel())
//                .maxResults(3)
//                .minScore(0.8)
//                .build();
//    }
}
