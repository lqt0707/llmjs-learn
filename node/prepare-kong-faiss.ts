import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { embeddings } from "./utils";

const run = async () => {
  const loader = new TextLoader("../data/kong.txt");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });
  const solitDocs = await splitter.splitDocuments(docs);

  const vectorStore = await FaissStore.fromDocuments(solitDocs, embeddings);

  const directory = "../db/kongyiji";
  await vectorStore.save(directory);
};

run();
