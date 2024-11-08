package org.jugph.rag.improved.fusion;

import dev.langchain4j.model.bedrock.BedrockTitanEmbeddingModel;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.transformer.ExpandingQueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import org.jugph.rag.aggregator.RAGFusionAggregator;
import org.jugph.rag.assistants.Assistant;
import org.jugph.rag.retriever.MayaDocumentsEmbeddingStore;
import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;

import java.util.HashMap;
import java.util.Map;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O_MINI;
import static org.jugph.rag.ApiKeys.OPENAI_API_KEY;
import static org.jugph.rag.Utils.startConversationWith;
import static org.jugph.rag.config.ConfigLoader.getProperty;

public class RAGFusionExample {
    public static void main(String[] args) {
        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName(GPT_4_O_MINI)
                .build();

        EmbeddingModel titanEmbeddingModel = BedrockTitanEmbeddingModel.builder()
                .model("amazon.titan-embed-text-v2:0")
                .credentialsProvider(ProfileCredentialsProvider.create(getProperty("aws.profile")))
                .build();

        var mayaWalletDocumentsEmbeddingStore = new MayaDocumentsEmbeddingStore("maya_wallet_documents_demo_db",
                titanEmbeddingModel.dimension());

        var mayaBankDocumentsEmbeddingStore = new MayaDocumentsEmbeddingStore("maya_bank_documents_demo_db",
                titanEmbeddingModel.dimension());

        int numberOfAlternativeQueries = 3;
        var expandingQueryTransformer = new ExpandingQueryTransformer(chatLanguageModel, numberOfAlternativeQueries);

        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(getProperty("tavily.api_key"))
                .build();

        ContentRetriever webSearchContentRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(1)
                .build();

        ContentRetriever mayaBankDocumentsRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(mayaBankDocumentsEmbeddingStore.getEmbeddingStore())
                .embeddingModel(titanEmbeddingModel)
                .maxResults(3)
                .minScore(0.7)
                .build();

        QueryRouter queryRouter = new DefaultQueryRouter(mayaBankDocumentsRetriever, webSearchContentRetriever);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(expandingQueryTransformer)
                .queryRouter(queryRouter)
                .contentAggregator(new RAGFusionAggregator())
                .build();

        var assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        startConversationWith(assistant);
    }
}
