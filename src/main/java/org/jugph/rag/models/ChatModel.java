package org.jugph.rag.models;

import dev.langchain4j.model.bedrock.BedrockLlamaChatModel;
import dev.langchain4j.model.input.PromptTemplate;
import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;

import java.util.Map;

import static org.jugph.rag.config.ConfigLoader.getProperty;

public class ChatModel {

    public static BedrockLlamaChatModel chatModel() {
        return BedrockLlamaChatModel.builder()
                .credentialsProvider(ProfileCredentialsProvider.create(getProperty("aws.profile")))
                .model("meta.llama3-70b-instruct-v1:0")
                .temperature(0.8)
                .build();
    }

    public static BedrockLlamaChatModel chatModel(String prompt) {
        return BedrockLlamaChatModel.builder()
                .credentialsProvider(ProfileCredentialsProvider.create(getProperty("aws.profile")))
                .model("meta.llama3-70b-instruct-v1:0")
                .temperature(0.7)
                .assistantPrompt(PromptTemplate.from("Based on the information here: {{context}}\n Answer the following question: ")
                        .apply(Map.of("context", prompt)).text())
                .build();
    }
}
