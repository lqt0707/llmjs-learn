import { BaseListChatMessageHistory } from "@langchain/core/chat_history";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  StoredMessage,
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
} from "@langchain/core/messages";
import fs from "node:fs";
import path from "node:path";

/**
 * JSONChatHistory 输入参数接口
 * 定义创建 JSONChatHistory 实例所需的字段
 */
export interface JSONChatHistoryInput {
  sessionId: string;
  dir: string;
}

/**
 * 基于 JSON 文件的聊天消息历史记录类
 * 实现了从 JSON 文件中读取、写入和管理聊天消息的功能
 */
export class JSONChatHistory extends BaseListChatMessageHistory {
  sessionId: string;
  dir: string;

  /**
   * 构造函数，初始化 JSONChatHistory 实例
   * @param fields - 包含会话 ID 和存储目录的输入参数
   */
  constructor(fields: JSONChatHistoryInput) {
    super(fields);
    this.sessionId = fields.sessionId;
    this.dir = fields.dir;
  }

  /**
   * 从 JSON 文件中获取聊天消息
   * 如果文件不存在，则创建一个空文件并返回空数组
   * @returns 聊天消息数组
   */
  async getMessages(): Promise<BaseMessage[]> {
    const filePath = path.join(this.dir, `${this.sessionId}.json`);
    try {
      if (!fs.existsSync(filePath)) {
        this.saveMessagesToFile([]);
        return [];
      }
      const data = fs.readFileSync(filePath, { encoding: "utf-8" });
      const storedMessages = JSON.parse(data) as StoredMessage[];
      return mapStoredMessagesToChatMessages(storedMessages);
    } catch (error) {
      console.error(`Failed to read chat history from ${filePath}`, error);
      return [];
    }
  }

  /**
   * 向聊天历史中添加一条消息
   * @param message - 要添加的聊天消息
   */
  async addMessage(message: BaseMessage): Promise<void> {
    const messages = await this.getMessages();
    messages.push(message);
    await this.saveMessagesToFile(messages);
  }

  /**
   * 向聊天历史中添加多条消息
   * @param messages - 要添加的聊天消息数组
   */
  async addMessages(messages: BaseMessage[]): Promise<void> {
    const existingMessages = await this.getMessages();
    const allMessages = existingMessages.concat(messages);
    await this.saveMessagesToFile(allMessages);
  }

  /**
   * 清除聊天历史记录，删除对应的 JSON 文件
   */
  async clear(): Promise<void> {
    const filePath = path.join(this.dir, `${this.sessionId}.json`);
    try {
      fs.unlinkSync(filePath);
    } catch (error) {
      console.error(`Failed to clear chat history from ${filePath}`, error);
    }
  }

  /**
   * 将聊天消息保存到 JSON 文件中
   * 如果存储目录不存在，则创建该目录
   * @param messages - 要保存的聊天消息数组
   */
  private async saveMessagesToFile(messages: BaseMessage[]): Promise<void> {
    const filePath = path.join(this.dir, `${this.sessionId}.json`);

    const serializedMessages = mapChatMessagesToStoredMessages(messages);
    try {
      if (!fs.existsSync(this.dir)) {
        fs.mkdirSync(this.dir);
      }
      fs.writeFileSync(filePath, JSON.stringify(serializedMessages, null, 2), {
        encoding: "utf-8",
      });
    } catch (error) {
      console.error(`Failed to save chat history to ${filePath}`, error);
    }
  }
  /**
   * 获取所有消息的唯一ID列表
   */
  async getMessageIds(): Promise<string[]> {
    const messages = await this.getMessages();
    // 假设每个 message 都有 id 字段
    return messages.map((msg: any) => msg.id).filter(Boolean);
  }

  /**
   * 根据消息ID删除指定消息
   * @param messageId - 要删除的消息的 ID
   */
  async deleteMessage(messageId: string): Promise<void> {
    const messages = await this.getMessages();
    const filtered = messages.filter((msg: any) => msg.id !== messageId);
    await this.saveMessagesToFile(filtered);
  }

  /**
   * 获取最后一条消息
   * @returns 最后一条消息，如果没有消息则返回 undefined
   */
  async getLastMessage(): Promise<BaseMessage | undefined> {
    const messages = await this.getMessages();
    return messages.length > 0 ? messages[messages.length - 1] : undefined;
  }

  /**
   * 获取消息总数
   * @returns 聊天消息的总数
   */
  async count(): Promise<number> {
    const messages = await this.getMessages();
    return messages.length;
  }

  /**
   * 添加用户消息的便捷方法
   * @param message - 用户消息内容
   */
  async addUserMessage(message: string): Promise<void> {
    await this.addMessage(new HumanMessage(message));
  }

  /**
   * 添加AI消息的便捷方法
   * @param message - AI消息内容
   */
  async addAIChatMessage(message: string): Promise<void> {
    await this.addMessage(new AIMessage(message));
  }

  /**
   * 添加AI消息的别名方法
   * @param message - AI消息内容
   */
  async addAIMessage(message: string): Promise<void> {
    await this.addAIChatMessage(message);
  }

  /**
   * LangChain 序列化标识
   */
  lc_serializable = true;

  /**
   * LangChain 命名空间
   */
  get lc_namespace(): string[] {
    return ["langchain", "stores", "message", "json"];
  }

  /**
   * LangChain 密钥
   */
  get lc_secrets(): { [key: string]: string } | undefined {
    return undefined;
  }

  /**
   * LangChain 属性
   */
  get lc_attributes(): { [key: string]: any } | undefined {
    return {
      sessionId: this.sessionId,
      dir: this.dir,
    };
  }

  /**
   * LangChain 别名
   */
  get lc_aliases(): { [key: string]: string } | undefined {
    return undefined;
  }
}
