package org.jugph.rag.retriever;

import dev.langchain4j.data.document.loader.amazon.s3.AmazonS3DocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.model.bedrock.BedrockTitanEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;

import static org.jugph.rag.config.ConfigLoader.getProperty;

public class MayaDocumentsIngestor {

    private final EmbeddingModel embeddingModel;

    private final String database;

    private final String prefix;

    public MayaDocumentsIngestor(EmbeddingModel embeddingModel, String database, String prefix) {
        this.embeddingModel = embeddingModel;
        this.database = database;
        this.prefix = prefix;
    }

    public void ingest() {
        var s3Loader = AmazonS3DocumentLoader.builder()
                .profile(getProperty("aws.profile"))
                .build();

        var mayaWalletDocumentsEmbeddingStore = new MayaDocumentsEmbeddingStore(database,
                embeddingModel.dimension());

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(450, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(mayaWalletDocumentsEmbeddingStore.getEmbeddingStore())
                .build();

        var documents = s3Loader.loadDocuments("langchain4j-rag-maya-demo", prefix,
                new TextDocumentParser());

        for(var document : documents) {
            var s3Path = document.metadata().getString("source");
            System.out.println("Ingesting " + s3Path);

            String fileNameWithExtension = s3Path.substring(s3Path.lastIndexOf('/') + 1);
            int dotIndex = fileNameWithExtension.lastIndexOf('.');
            var feature = (dotIndex > 0) ? fileNameWithExtension.substring(0, dotIndex) : fileNameWithExtension;

            document.metadata().put("feature", feature);

            ingestor.ingest(document);
        }
    }

    public static void main(String[] args) {
        EmbeddingModel titanEmbeddingModel = BedrockTitanEmbeddingModel.builder()
                .model("amazon.titan-embed-text-v2:0")
                .credentialsProvider(ProfileCredentialsProvider.create(getProperty("aws.profile")))
                .build();

//        var mayaDocumentsIngestor = new MayaDocumentsIngestor(titanEmbeddingModel, "maya_wallet_documents_demo_db", "maya-wallet/");
        var mayaDocumentsIngestor = new MayaDocumentsIngestor(titanEmbeddingModel, "maya_bank_documents_demo_db", "maya-bank/");
        mayaDocumentsIngestor.ingest();
    }
}
