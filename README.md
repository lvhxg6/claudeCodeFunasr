# claudeCodeFunasr

FunASR 语音识别服务 - 支持流式识别（边说边出文字）

## 特性

- **流式识别**：实时返回识别结果，延迟 200-500ms
- **本地部署**：完全免费，无需联网
- **中英混合**：支持中英文混合识别
- **双模式**：支持 HTTP API（非流式）和 WebSocket（流式）
- **OpenAI 兼容**：HTTP API 兼容 OpenAI Whisper 接口

## 流式 vs 非流式对比

### Whisper（非流式）
```
用户说话 → 等待 1.5s 静音 → 完整录音 → HTTP API → 返回全部文本 → 显示
延迟：1.5s（静音等待）+ 1-2s（处理）= 2.5-3.5s
```

### FunASR（流式）
```
用户说话 → 每 200ms 发送音频块 → WebSocket → 实时返回部分文本 → 实时显示
延迟：200-500ms（几乎实时）
```

## 安装

### 系统要求

- Python 3.10+
- macOS / Linux
- 约 500MB 磁盘空间（用于模型）

### 安装步骤

```bash
cd /Users/liubu/hx/AI-SystemService/claudeCodeFunasr
./scripts/install.sh
```

安装脚本会自动：
1. 检查 Python 版本
2. 创建虚拟环境
3. 安装依赖
4. 下载 FunASR 模型（首次约 200-500MB）
5. 创建命令行工具符号链接

## 使用

### 启动服务

```bash
funasr-service start
```

### 查看状态

```bash
funasr-service status
```

### 查看日志

```bash
funasr-service logs
```

### 停止服务

```bash
funasr-service stop
```

### 重启服务

```bash
funasr-service restart
```

## API 接口

### 1. 健康检查

```bash
curl http://localhost:10095/health
```

响应：
```json
{
  "status": "ok",
  "models": {
    "asr": true,
    "vad": true,
    "punc": true
  }
}
```

### 2. HTTP API（非流式）

OpenAI Whisper 兼容接口：

```bash
curl -X POST http://localhost:10095/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=json"
```

响应：
```json
{
  "text": "识别的文本内容"
}
```

### 3. WebSocket API（流式）

连接：`ws://localhost:10095/ws/transcribe`

#### 客户端发送

```json
{
  "type": "audio",
  "data": "<base64编码的音频数据>",
  "is_final": false
}
```

#### 服务端返回

```json
{
  "type": "partial",
  "text": "识别的增量文本",
  "is_final": false
}
```

- `type`: `partial`（中间结果）或 `final`（最终结果）
- `text`: 增量文本（只包含新识别的部分）
- `is_final`: 是否为最后一块

## 命令行工具

### 转录音频文件

```bash
funasr-transcribe audio.wav
```

支持的音频格式：WAV, MP3, FLAC, OGG 等

## 配置

编辑 `config.yaml` 修改配置：

```yaml
model:
  name: "paraformer-zh-streaming"  # 流式模型
  vad_model: "fsmn-vad"            # VAD 模型
  punc_model: "ct-punc"            # 标点模型

server:
  host: "127.0.0.1"
  port: 10095

streaming:
  chunk_ms: 200      # 音频块大小（毫秒）
  sample_rate: 16000 # 采样率

logging:
  level: "INFO"
  file: "logs/service.log"
```

## 与 claudeCodeTrigger 集成

在 `claudeCodeTrigger/config.yaml` 中配置：

```yaml
stt:
  engine: "funasr"
  streaming: true  # 启用流式模式

funasr:
  api_url: "http://localhost:10095/v1/audio/transcriptions"
  ws_url: "ws://localhost:10095/ws/transcribe"
```

## 端口

- HTTP API: `http://localhost:10095`
- WebSocket: `ws://localhost:10095/ws/transcribe`

## 日志

日志文件位置：`logs/service.log`

## 故障排除

### 服务无法启动

1. 检查端口是否被占用：
```bash
lsof -i :10095
```

2. 查看日志：
```bash
funasr-service logs
```

### 模型下载失败

首次启动会自动下载模型，如果下载失败：
1. 检查网络连接
2. 重新运行安装脚本：`./scripts/install.sh`

### 识别效果不佳

1. 确保音频采样率为 16000Hz
2. 确保音频为单声道
3. 检查音频质量（噪音、音量等）

## 项目结构

```
claudeCodeFunasr/
├── README.md              # 项目文档
├── server.py              # FastAPI + WebSocket 服务
├── config.yaml            # 配置文件
├── requirements.txt       # Python 依赖
├── scripts/
│   ├── install.sh         # 安装脚本
│   ├── funasr-service.sh  # 服务管理脚本
│   └── funasr-transcribe.sh # CLI 工具
├── logs/                  # 日志目录
│   └── service.log
└── temp/                  # 临时文件目录
```

## 技术栈

- **FunASR**: 阿里达摩院开源的语音识别框架
- **FastAPI**: 现代 Web 框架
- **WebSocket**: 实时双向通信
- **ModelScope**: 模型下载和管理

## 许可证

MIT License

## 相关项目

- [claudeCodeTrigger](../claudeCodeTrigger) - 语音输入触发器
- [claudeCodeWhisper](../claudeCodeWhisper) - Whisper 语音识别服务
