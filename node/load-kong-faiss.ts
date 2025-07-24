import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { embeddings } from "./utils";
async function run() {
  const directory = "../db/kongyiji";
  const vectorstore = await FaissStore.load(directory, embeddings);

  const retriever = vectorstore.asRetriever(2);
  const res = await retriever.invoke("茴香豆是做什么用的");

  console.log(res);
}

run();
