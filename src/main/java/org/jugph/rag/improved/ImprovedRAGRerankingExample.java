package org.jugph.rag.improved;

import dev.langchain4j.model.bedrock.BedrockTitanEmbeddingModel;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.cohere.CohereScoringModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import org.jugph.rag.assistants.Assistant;
import org.jugph.rag.retriever.MayaDocumentsEmbeddingStore;
import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;

import java.util.HashMap;
import java.util.Map;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O_MINI;
import static org.jugph.rag.ApiKeys.OPENAI_API_KEY;
import static org.jugph.rag.Utils.startConversationWith;
import static org.jugph.rag.config.ConfigLoader.getProperty;

public class ImprovedRAGRerankingExample {
    public static void main(String[] args) {
        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName(GPT_4_O_MINI)
                .build();

        EmbeddingModel titanEmbeddingModel = BedrockTitanEmbeddingModel.builder()
                .model("amazon.titan-embed-text-v2:0")
                .credentialsProvider(ProfileCredentialsProvider.create(getProperty("aws.profile")))
                .build();


        var mayaBankDocumentsEmbeddingStore = new MayaDocumentsEmbeddingStore("maya_bank_documents_demo_db",
                titanEmbeddingModel.dimension());

        ContentRetriever mayaBankDocumentsRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(mayaBankDocumentsEmbeddingStore.getEmbeddingStore())
                .embeddingModel(titanEmbeddingModel)
                .maxResults(10)
                .minScore(0.6)
                .build();

        ScoringModel scoringModel = CohereScoringModel.builder()
                .apiKey(getProperty("reranker.cohere.api_key"))
                .modelName("rerank-multilingual-v3.0")
                .build();

        ContentAggregator contentAggregator = ReRankingContentAggregator.builder()
                .scoringModel(scoringModel)
                .minScore(0.8)
                .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(mayaBankDocumentsRetriever)
                .contentAggregator(contentAggregator)
                .build();

        var assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        startConversationWith(assistant);
    }
}
