import { Ollama, OllamaEmbeddings } from "@langchain/ollama";

interface Model {
  model: string;
  baseUrl: string;
}

const OllamaCongfig: Model = {
  model: "deepseek-r1",
  baseUrl: "http://localhost:11434/",
};

export const embeddings = new OllamaEmbeddings(OllamaCongfig);

export const model = new Ollama(OllamaCongfig);
