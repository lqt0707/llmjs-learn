import { OllamaEmbeddings } from "@langchain/ollama";

export const embeddings = new OllamaEmbeddings({
  model: "deepseek-r1",
  baseUrl: "http://localhost:11434/",
});
