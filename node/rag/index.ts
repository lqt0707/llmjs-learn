// 导入 Faiss 向量存储库，用于存储和检索向量数据
import { FaissStore } from "@langchain/community/vectorstores/faiss";
// 加载环境变量配置
import "dotenv/config";
// 导入 Node.js 的 path 模块，用于处理文件路径
import path from "path";
// 导入自定义的 JSON 聊天历史记录模块
import { JSONChatHistory } from "../JSONChatHistory/index";
// 从 langchain 核心库导入聊天提示模板和消息占位符
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
// 从 langchain 核心库导入可运行序列、可传递对象、带消息历史的可运行对象和可运行对象
import {
  RunnableSequence,
  RunnablePassthrough,
  RunnableWithMessageHistory,
  Runnable,
} from "@langchain/core/runnables";
// 导入 OpenAI 聊天模型
import { ChatOpenAI } from "@langchain/openai";
// 导入字符串输出解析器
import { StringOutputParser } from "@langchain/core/output_parsers";
// 导入人类消息和 AI 消息类
import { HumanMessage, AIMessage } from "@langchain/core/messages";
// 导入文档类
import { Document } from "@langchain/core/documents";
// 导入自定义的嵌入函数
import { embeddings } from "../utils";

/**
 * 加载向量存储库
 * @returns 加载好的 Faiss 向量存储实例
 */
async function loadVectorStore() {
  // 拼接向量存储库的目录路径
  const directory = path.join(__dirname, "../../db/qiu");
  // 从指定目录加载 Faiss 向量存储
  const vectorStore = await FaissStore.load(directory, embeddings);

  return vectorStore;
}

/**
 * 获取问题重述链
 * @returns 问题重述链实例
 */
async function getRephraseChain() {
  // 定义问题重述链的提示模板
  const rephraseChainPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "给定以下对话和一个后续问题，请将后续问题重述为一个独立的问题。请注意，重述的问题应该包含足够的信息，使得没有看过对话历史的人也能理解。",
    ],
    new MessagesPlaceholder("history"),
    ["human", "将以下问题重述为一个独立的问题：\n{question}"],
  ]);

  // 构建问题重述链
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

  return rephraseChain;
}

/**
 * 测试问题重述链
 */
async function testRephraseChain() {
  // 定义聊天历史消息
  const historyMessages = [
    new HumanMessage("你好，我叫小明"),
    new AIMessage("你好小明"),
  ];
  // 获取问题重述链实例
  const rephraseChain = await getRephraseChain();

  // 定义待重述的问题
  const question = "你觉得我的名字怎么样？";
  // 调用问题重述链获取独立问题
  const standaloneQuestion = await rephraseChain.invoke({
    history: historyMessages,
    question,
  });

  // 打印重述后的独立问题
  console.log(standaloneQuestion);
}

/**
 * 获取 RAG（检索增强生成）链
 * @returns 带消息历史的 RAG 链实例
 */
export async function getRagChain(): Promise<Runnable> {
  // 加载向量存储库
  const vectorStore = await loadVectorStore();
  // 获取向量存储的检索器，每次检索返回 2 个文档
  const retriever = vectorStore.asRetriever(2);

  /**
   * 将文档数组转换为字符串
   * @param documents 文档数组
   * @returns 拼接后的文档内容字符串
   */
  const convertDocsToString = (documents: Document[]): string => {
    return documents.map((document) => document.pageContent).join("\n");
  };
  // 构建上下文检索链
  const contextRetrieverChain = RunnableSequence.from([
    (input) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);

  // 定义系统提示模板
  const SYSTEM_TEMPLATE = `
    你是一个熟读刘慈欣的《球状闪电》的终极原着党，精通根据作品原文详细解释和回答问题，你在回答时会引用作品原文。
    并且回答时仅根据原文，尽可能回答用户问题，如果原文中没有相关内容，你可以回答“原文中没有相关内容”，

    以下是原文中跟用户回答相关的内容：
    {context}
  `;

  // 定义聊天提示模板
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    ["human", "现在，你需要基于原文，回答以下问题：\n{standalone_question}`"],
  ]);
  // 获取问题重述链实例
  const rephraseChain = await getRephraseChain();

  // 构建 RAG 链
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

  // 拼接聊天历史记录的存储目录路径
  const chatHistoryDir = path.join(__dirname, "../../chat_data");

  // 构建带消息历史的 RAG 链
  const ragChainWithHistory = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory: (sessionId) =>
      new JSONChatHistory({ sessionId, dir: chatHistoryDir }),
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });

  return ragChainWithHistory;
}

/**
 * 运行测试函数，调用 RAG 链处理问题并打印结果
 */
async function run() {
  // 获取带消息历史的 RAG 链实例
  const ragChain = await getRagChain();

  // 调用 RAG 链处理问题
  const res = await ragChain.invoke(
    {
      //   question: "什么是球状闪电？",
      question: "这个现象在文中有什么故事",
    },
    {
      configurable: { sessionId: "test-history" },
    }
  );

  // 打印处理结果
  console.log(res);
}

// 执行测试函数
run();

// 注释掉的测试函数，可用于测试问题重述链
// testRephraseChain();
