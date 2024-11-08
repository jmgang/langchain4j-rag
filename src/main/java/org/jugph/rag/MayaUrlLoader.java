package org.jugph.rag;

import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.transformer.jsoup.HtmlToTextDocumentTransformer;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;

import java.io.IOException;
import java.nio.file.*;
import java.util.List;
import java.util.Map;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O_MINI;
import static org.jugph.rag.ApiKeys.OPENAI_API_KEY;
import static org.jugph.rag.Utils.toPath;

public class MayaUrlLoader {

    private static final String OUTPUT_DIRECTORY = "/Users/jansen.ang/Documents/langchain4j-rag-demo/maya-bank-documents/";
    private static final String URL_FILE_PATH = "documents/mayabank_urls.txt";

    public static void main(String[] args) {
        try {
            List<String> urls = readUrlsFromFile(URL_FILE_PATH);
            for (String url : urls) {
                processUrl(url, OUTPUT_DIRECTORY);
            }
            System.out.println("All URLs processed successfully!");
        } catch (IOException e) {
            System.err.println("Error reading URLs or processing files: " + e.getMessage());
        }
    }

    private static void processUrl(String url, String outputDirectory) {
        try {
            var document = UrlDocumentLoader.load(url, new TextDocumentParser());
            HtmlToTextDocumentTransformer textExtractor = new HtmlToTextDocumentTransformer();
            var transformedDocument = textExtractor.transform(document);

            String fileName = extractFileNameFromUrl(url);

            ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                    .apiKey(OPENAI_API_KEY)
                    .modelName(GPT_4_O_MINI)
                    .build();

            var cleanedDocument = chatLanguageModel.generate(PromptTemplate.from(
                    """
                    Below is a text retrieved from the maya website. Clean and remove any unwanted text such as headers, footers, and other irrelevant information.
                    This irrelevant information includes Tell me more links or hotlines as well.
                    Your response should only be the cleaned main content of the page. Do not modify or add text in any way. Nothing more.
                    TEXT:
                    {{text}}
                    
                    CLEANED TEXT:
                    """
            ).apply(Map.of("text", transformedDocument.text())).text());

            saveFile(cleanedDocument, outputDirectory + fileName);
        } catch (Exception e) {
            System.err.println("Failed to process URL: " + url + ". Error: " + e.getMessage());
        }
    }

    private static List<String> readUrlsFromFile(String filePath) throws IOException {
        return Files.readAllLines(toPath(filePath));
    }

    private static void saveFile(String content, String filePath) {
        try {
            Files.createDirectories(Paths.get(filePath).getParent());
            Files.write(Paths.get(filePath), content.getBytes(), StandardOpenOption.CREATE);
            System.out.println("File saved successfully to: " + filePath);
        } catch (IOException e) {
            System.err.println("Failed to save file: " + e.getMessage());
        }
    }

    public static String extractFileNameFromUrl(String url) {
        if (url == null || url.isEmpty()) {
            throw new IllegalArgumentException("URL cannot be null or empty");
        }
        String[] parts = url.split("/");
        String fileName = parts[parts.length - 1];
        return fileName.isEmpty() ? "default.txt" : fileName + ".txt";
    }
}
