#!/usr/bin/env ts-node

/**
 * 启动 RAG 客户端和服务端的脚本
 * 该脚本会检查并安装 concurrently 包（如果尚未安装），
 * 然后并行启动服务端和客户端。
 */

import { spawn } from 'child_process';
import { existsSync } from 'fs';
import { join } from 'path';

// 检查是否已安装 concurrently
function isConcurrentlyInstalled(): boolean {
  try {
    const packageJsonPath = join(__dirname, '..', 'package.json');
    if (existsSync(packageJsonPath)) {
      const packageJson = require(packageJsonPath);
      return !!(packageJson.devDependencies?.concurrently || packageJson.dependencies?.concurrently);
    }
    return false;
  } catch (error) {
    return false;
  }
}

// 安装 concurrently
function installConcurrently(): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log('正在安装 concurrently...');
    const installProcess = spawn('yarn', ['add', '-D', 'concurrently'], {
      cwd: join(__dirname, '..'),
      stdio: 'inherit'
    });

    installProcess.on('close', (code) => {
      if (code === 0) {
        console.log('concurrently 安装成功');
        resolve();
      } else {
        reject(new Error(`安装 concurrently 失败，退出码: ${code}`));
      }
    });
  });
}

// 启动服务端和客户端
async function startServices(): Promise<void> {
  // 检查是否已安装 concurrently，如果没有则安装
  if (!isConcurrentlyInstalled()) {
    await installConcurrently();
  }

  // 使用 concurrently 启动服务端和客户端
  const startProcess = spawn('npx', ['concurrently', 
    '"ts-node ./rag/server.ts"', 
    '"sleep 3 && ts-node ./rag/client.ts"' // 等待3秒让服务端启动
  ], {
    cwd: join(__dirname, '..'),
    stdio: 'inherit'
  });

  startProcess.on('close', (code) => {
    if (code === 0) {
      console.log('服务已正常退出');
    } else {
      console.error(`服务退出，退出码: ${code}`);
    }
  });
}

// 启动服务
startServices().catch(error => {
  console.error('启动服务时出错:', error);
  process.exit(1);
});