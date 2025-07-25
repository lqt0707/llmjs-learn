import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";

interface Model {
  model: string;
  baseUrl: string;
}

export const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
  baseUrl: "http://localhost:11434/",
});

export const model = new Ollama({
  model: "deepseek-r1",
  baseUrl: "http://localhost:11434/",
});

export const MoonshotModel = new ChatOpenAI({
  configuration: {
    baseURL: "https://api.moonshot.cn/v1",
    apiKey: process.env.OPENAI_API_KEY,
  },
  modelName: "kimi-k2-0711-preview",
});
