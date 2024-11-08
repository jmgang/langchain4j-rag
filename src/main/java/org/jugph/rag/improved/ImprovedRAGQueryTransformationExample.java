package org.jugph.rag.improved;

import dev.langchain4j.model.bedrock.BedrockTitanEmbeddingModel;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.ExpandingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.service.AiServices;
import org.jugph.rag.assistants.Assistant;
import org.jugph.rag.retriever.MayaDocumentsEmbeddingStore;
import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O_MINI;
import static org.jugph.rag.ApiKeys.OPENAI_API_KEY;
import static org.jugph.rag.Utils.startConversationWith;
import static org.jugph.rag.config.ConfigLoader.getProperty;

public class ImprovedRAGQueryTransformationExample {
    public static void main(String[] args) {
        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName(GPT_4_O_MINI)
                .build();

        QueryTransformer queryTransformer = new ExpandingQueryTransformer(chatLanguageModel);
        EmbeddingModel titanEmbeddingModel = BedrockTitanEmbeddingModel.builder()
                .model("amazon.titan-embed-text-v2:0")
                .credentialsProvider(ProfileCredentialsProvider.create(getProperty("aws.profile")))
                .build();

        var mayaWalletDocumentsEmbeddingStore = new MayaDocumentsEmbeddingStore("maya_wallet_documents_demo_db",
                titanEmbeddingModel.dimension());

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(mayaWalletDocumentsEmbeddingStore.getEmbeddingStore())
                .embeddingModel(titanEmbeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(queryTransformer)
                .contentRetriever(contentRetriever)
                .build();

        var assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        startConversationWith(assistant);
    }
}
