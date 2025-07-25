import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAIEmbeddings } from "@langchain/openai";
import { OllamaEmbeddings } from "@langchain/ollama";
import "dotenv/config";
import path from "path";
import { JSONChatHistory } from "../JSONChatHistory/index";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
  RunnableWithMessageHistory,
  Runnable,
} from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";

async function loadVectorStore(): Promise<FaissStore> {
  console.log("📁 步骤 2.1: 正在加载向量存储...");
  const directory = path.join(__dirname, "../../db/qiu");

  // 检查向量存储文件是否存在
  const fs = require("fs");
  if (!fs.existsSync(directory)) {
    throw new Error(`向量存储目录不存在: ${directory}`);
  }

  const requiredFiles = ["faiss.index", "docstore.json"];
  for (const file of requiredFiles) {
    if (!fs.existsSync(path.join(directory, file))) {
      throw new Error(`向量存储文件缺失: ${file}`);
    }
  }
  console.log("✓ 向量存储文件检查通过");

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434/",
  });

  try {
    console.log("📂 正在读取向量存储数据...");
    // 为向量存储加载设置超时
    const vectorStore = (await Promise.race([
      FaissStore.load(directory, embeddings),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("向量存储加载超时")), 30000)
      ),
    ])) as FaissStore;
    console.log("✓ 向量存储加载完成");

    // 添加调试信息：检查向量存储中的文档数量
    console.log("🔍 步骤 2.2: 检查向量存储中的文档数量...");
    try {
      // 先尝试一个非常简单的搜索
      console.log("🔍 执行简单测试搜索...");
      const testDocs = await vectorStore.similaritySearch("球状闪电", 1);
      console.log(`📊 测试搜索返回 ${testDocs.length} 个文档片段`);

      if (testDocs.length > 0) {
        console.log("📋 测试文档内容预览:");
        testDocs.slice(0, 1).forEach((doc, index) => {
          console.log(
            `  文档 ${index + 1}: ${doc.pageContent.substring(0, 100)}${
              doc.pageContent.length > 100 ? "..." : ""
            }`
          );
        });
      }

      // 为相似性搜索添加超时控制
      const similaritySearchPromise = vectorStore.similaritySearch(
        "球状闪电",
        2
      );
      const timeoutPromise = new Promise<Document[]>((_, reject) =>
        setTimeout(() => reject(new Error("相似性搜索超时")), 10000)
      );

      const docs = await Promise.race([
        similaritySearchPromise,
        timeoutPromise,
      ]);
      console.log(`📊 向量存储中共有 ${docs.length} 个文档片段`);

      if (docs.length > 0) {
        console.log("📋 存储的文档内容预览:");
        docs.slice(0, 2).forEach((doc, index) => {
          console.log(
            `  文档 ${index + 1}: ${doc.pageContent.substring(0, 100)}${
              doc.pageContent.length > 100 ? "..." : ""
            }`
          );
        });
      } else {
        console.log("⚠️ 警告：向量存储中没有找到任何文档");
        console.log("💡 请运行: yarn prepare-qiu-faiss 来准备数据");
      }
      console.log("✓ 文档数量检查完成");
    } catch (error) {
      console.log("⚠️ 无法获取文档数量信息，可能存储为空或格式问题");
      console.log("错误详情:", error.message);
    }

    return vectorStore;
  } catch (error) {
    console.error("❌ 向量存储加载时发生错误:", error.message);
    throw error;
  }
}

async function getRephraseChain() {
  console.log("🔄 正在构建问题重述链...");

  console.log("💬 步骤 2.7.1: 配置问题重述提示模板...");
  const rephraseChainPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "给定以下对话和一个后续问题，请将后续问题重述为一个独立的问题。请注意，重述的问题应该包含足够的信息，使得没有看过对话历史的人也能理解。",
    ],
    new MessagesPlaceholder("history"),
    ["human", "将以下问题重述为一个独立的问题：\n{question}"],
  ]);
  console.log("✓ 问题重述提示模板配置完成");

  console.log("🤖 步骤 2.7.2: 配置语言模型...");
  const rephraseChain = RunnableSequence.from([
    rephraseChainPrompt,
    new ChatOpenAI({
      configuration: {
        baseURL: "https://api.moonshot.cn/v1",
        apiKey: process.env.OPENAI_API_KEY,
      },
      modelName: "kimi-k2-0711-preview",
      temperature: 0.4,
    }),
    new StringOutputParser(),
  ]);
  console.log("✓ 语言模型配置完成");

  console.log("✓ 问题重述链构建完成");
  return rephraseChain;
}

async function testRephraseChain() {
  const historyMessages = [
    new HumanMessage("你好，我叫小明"),
    new AIMessage("你好小明"),
  ];
  const rephraseChain = await getRephraseChain();

  const question = "你觉得我的名字怎么样？";
  const standaloneQuestion = await rephraseChain.invoke({
    history: historyMessages,
    question,
  });

  console.log(standaloneQuestion);
}

export async function getRagChain(): Promise<Runnable> {
  console.log("🔗 步骤 2.3: 创建检索器...");
  const vectorStore = await loadVectorStore();
  const retriever = vectorStore.asRetriever(2);
  console.log("✓ 检索器创建完成");

  console.log("📄 步骤 2.4: 创建文档转换器...");
  const convertDocsToString = (documents: Document[]): string => {
    return documents.map((document) => document.pageContent).join("\n");
  };
  console.log("✓ 文档转换器创建完成");

  console.log("⛓️ 步骤 2.5: 构建上下文检索链...");
  const contextRetrieverChain = RunnableSequence.from([
    (input) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);
  console.log("✓ 上下文检索链构建完成");

  console.log("💬 步骤 2.6: 配置提示模板...");
  const SYSTEM_TEMPLATE = `
    你是一个熟读刘慈欣的《球状闪电》的终极原着党，精通根据作品原文详细解释和回答问题，你在回答时会引用作品原文。
    并且回答时仅根据原文，尽可能回答用户问题，如果原文中没有相关内容，你可以回答“原文中没有相关内容”，

    以下是原文中跟用户回答相关的内容：
    {context}
  `;

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    ["human", "现在，你需要基于原文，回答以下问题：\n{standalone_question}`"],
  ]);
  console.log("✓ 提示模板配置完成");

  console.log("🔄 步骤 2.7: 构建问题重述链...");
  const rephraseChain = await getRephraseChain();
  console.log("✓ 问题重述链构建完成");

  console.log("⚙️ 步骤 2.8: 组装RAG链...");
  const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseChain,
    }),
    RunnablePassthrough.assign({
      context: contextRetrieverChain,
    }),
    prompt,
    new ChatOpenAI({
      configuration: {
        baseURL: "https://api.moonshot.cn/v1",
        apiKey: process.env.OPENAI_API_KEY,
      },
      modelName: "kimi-k2-0711-preview",
    }),
    new StringOutputParser(),
  ]);
  console.log("✓ RAG链组装完成");

  console.log("💾 步骤 2.9: 配置聊天历史...");
  const chatHistoryDir = path.join(__dirname, "../../chat_data");

  const ragChainWithHistory = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory: (sessionId) =>
      new JSONChatHistory({ sessionId, dir: chatHistoryDir }),
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });
  console.log("✓ 聊天历史配置完成");

  return ragChainWithHistory;
}

async function run() {
  console.log("开始运行 RAG 应用...");
  console.log("步骤 1: 检查环境配置...");

  // 提前检查API密钥
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY 环境变量未设置");
  }
  console.log("✓ API密钥已配置");

  try {
    console.log("步骤 2: 构建RAG链...");
    const ragChain = await getRagChain();
    console.log("✓ RAG链构建完成");

    console.log("步骤 3: 执行查询...");
    console.log("查询问题: 什么是球状闪电？");

    // 添加超时控制的查询执行
    const queryPromise = ragChain.invoke(
      {
        question: "什么是球状闪电？",
      },
      {
        configurable: { sessionId: "test-history" },
      }
    );

    // 设置总超时时间为5分钟
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("查询超时，超过5分钟")), 300000)
    );

    const res = await Promise.race([queryPromise, timeoutPromise]);

    console.log("✓ 查询执行完成");
    console.log("=".repeat(50));
    console.log("最终回答:");
    console.log("=".repeat(50));
    console.log(res);
    console.log("=".repeat(50));
  } catch (error) {
    console.error("❌ 运行 RAG 应用时发生错误:");
    console.error("错误类型:", error.constructor.name);
    console.error("错误消息:", error.message);

    if (error.message.includes("timeout")) {
      console.error("\n💡 超时问题排查:");
      console.error("1. 检查网络连接速度");
      console.error("2. 尝试减少检索文档数量");
      console.error("3. 考虑使用更小的文本块");
    } else {
      console.error("\n💡 通用排查建议:");
      console.error("1. 检查网络连接是否正常");
      console.error("2. 检查API密钥是否正确配置");
      console.error("3. 检查向量存储文件是否存在");
      console.error("4. 检查模型服务是否可用");
    }

    console.error("\n🔍 详细错误堆栈:");
    console.error(error.stack);
  }
}

run();

// testRephraseChain();
