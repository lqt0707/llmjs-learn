import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import "dotenv/config";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import path from "path";
import { embeddings } from "../utils";

const run = async () => {
  const loader = new TextLoader(path.join(__dirname, "../../data/qiu.txt"));
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });
  const splitDocs = await splitter.splitDocuments(docs);

  const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);

  const directory = path.join(__dirname, "../../db/qiu");
  await vectorStore.save(directory);
};

run();
