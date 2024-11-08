package org.jugph.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.document.transformer.jsoup.HtmlToTextDocumentTransformer;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.opensearch.OpenSearchEmbeddingStore;
import org.opensearch.client.transport.aws.AwsSdk2Transport;
import org.opensearch.client.transport.aws.AwsSdk2TransportOptions;
import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;
import software.amazon.awssdk.http.SdkHttpClient;
import software.amazon.awssdk.http.apache.ApacheHttpClient;
import software.amazon.awssdk.regions.Region;
import org.opensearch.client.opensearch.OpenSearchClient;

import java.util.List;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static org.jugph.rag.Utils.toPath;

public class OpenSearchRetrieverExample {

    public static void main(String[] args) throws InterruptedException {
        EmbeddingStore<TextSegment> embeddingStore = OpenSearchEmbeddingStore.builder()
                .serverUrl("7cckun0nws3z04mk7w87.us-east-1.aoss.amazonaws.com")
                .region("us-east-1")
                .serviceName("aoss")
                .options(AwsSdk2TransportOptions.builder()
                        .setCredentials(ProfileCredentialsProvider.create("admin-general"))
                        .build())
                .build();

        Document document = UrlDocumentLoader.load("https://www.mayabank.ph/credit/", new TextDocumentParser());

        var transformedDocument = new HtmlToTextDocumentTransformer().transform(document);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(transformedDocument);

        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        embeddingStore.addAll(embeddings, segments);

//        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
//
//        TextSegment segment1 = TextSegment.from("I like football.");
//        Embedding embedding1 = embeddingModel.embed(segment1).content();
//        embeddingStore.add(embedding1, segment1);
//
//        TextSegment segment2 = TextSegment.from("The weather is good today.");
//        Embedding embedding2 = embeddingModel.embed(segment2).content();
//        embeddingStore.add(embedding2, segment2);
//
//        Thread.sleep(1000); // to be sure that embeddings were persisted
//
//        Embedding queryEmbedding = embeddingModel.embed("What is your favourite sport?").content();
//        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(queryEmbedding, 1);
//        EmbeddingMatch<TextSegment> embeddingMatch = relevant.get(0);
//
//        System.out.println(embeddingMatch.score()); // 0.8144289
//        System.out.println(embeddingMatch.embedded().text()); // I like football.
    }
}
