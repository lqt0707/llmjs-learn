import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { embeddings, model } from "./utils";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";

async function run() {
  const directory = "../db/kongyiji";
  const vectorstore = await FaissStore.load(directory, embeddings);

  const retriever = MultiQueryRetriever.fromLLM({
    llm: model,
    retriever: vectorstore.asRetriever(2),
    queryCount: 3,
    verbose: true,
  });
  const res = await retriever.invoke("茴香豆是做什么用的");

  console.log(res);
}

run();
