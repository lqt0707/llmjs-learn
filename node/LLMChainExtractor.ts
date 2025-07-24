import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { embeddings, model } from "./utils";
import "dotenv/config";
import { LLMChainExtractor } from "langchain/retrievers/document_compressors/chain_extract";
import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";

process.env.LANGCHAIN_VERBOSE = "true";

async function run() {
  const directory = "../db/kongyiji";
  const vectorstore = await FaissStore.load(directory, embeddings);

  const compressor = LLMChainExtractor.fromLLM(model);

  const retriever = new ContextualCompressionRetriever({
    baseCompressor: compressor,
    baseRetriever: vectorstore.asRetriever(2),
  });
  const res = await retriever.invoke("茴香豆是做什么用的");

  console.log(res);
}

run();
