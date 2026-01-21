#!/usr/bin/env python3
"""
FunASR 语音识别服务
支持 HTTP API（非流式）和 WebSocket（流式）
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import yaml
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from funasr import AutoModel

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 加载配置
with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# 配置日志
log_file = PROJECT_ROOT / CONFIG["logging"]["file"]
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=CONFIG["logging"]["level"],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(title="FunASR Service", version="1.0.0")

# 全局模型实例
asr_model: Optional[AutoModel] = None
vad_model: Optional[AutoModel] = None
punc_model: Optional[AutoModel] = None


def load_models():
    """加载 FunASR 模型"""
    global asr_model, vad_model, punc_model

    logger.info("开始加载 FunASR 模型...")

    try:
        # 加载 ASR 模型（流式）
        model_name = CONFIG["model"]["name"]
        logger.info(f"加载 ASR 模型: {model_name}")
        asr_model = AutoModel(
            model=model_name,
            device="cpu",  # 可以改为 "cuda" 如果有 GPU
        )
        logger.info("ASR 模型加载成功")

        # 加载 VAD 模型
        vad_model_name = CONFIG["model"].get("vad_model")
        if vad_model_name:
            logger.info(f"加载 VAD 模型: {vad_model_name}")
            vad_model = AutoModel(model=vad_model_name, device="cpu")
            logger.info("VAD 模型加载成功")

        # 加载标点恢复模型
        punc_model_name = CONFIG["model"].get("punc_model")
        if punc_model_name:
            logger.info(f"加载标点模型: {punc_model_name}")
            punc_model = AutoModel(model=punc_model_name, device="cpu")
            logger.info("标点模型加载成功")

        logger.info("所有模型加载完成")

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_models()


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "models": {
            "asr": asr_model is not None,
            "vad": vad_model is not None,
            "punc": punc_model is not None
        }
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    response_format: str = Form("json")
):
    """
    OpenAI 兼容的音频转录接口（非流式）

    Args:
        file: 音频文件
        language: 语言（可选，FunASR 自动检测）
        response_format: 响应格式（json 或 text）

    Returns:
        转录结果
    """
    try:
        # 读取音频文件
        audio_bytes = await file.read()

        # 解码音频
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # 如果是立体声，转换为单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # 确保采样率为 16000
        if sample_rate != 16000:
            logger.warning(f"音频采样率为 {sample_rate}，需要重采样到 16000")
            # 简单的重采样（生产环境建议使用 librosa）
            from scipy import signal
            audio_data = signal.resample(
                audio_data,
                int(len(audio_data) * 16000 / sample_rate)
            )

        # 调用 FunASR 进行识别
        logger.info(f"开始识别音频，长度: {len(audio_data)/16000:.2f}秒")

        result = asr_model.generate(
            input=audio_data,
            batch_size_s=300,
        )

        # 提取文本
        text = ""
        if result and len(result) > 0:
            text = result[0].get("text", "")

        # 标点恢复
        if punc_model and text:
            punc_result = punc_model.generate(input=text)
            if punc_result and len(punc_result) > 0:
                text = punc_result[0].get("text", text)

        logger.info(f"识别结果: {text}")

        # 返回结果
        if response_format == "text":
            return text
        else:
            return {"text": text}

    except Exception as e:
        logger.error(f"转录失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# 全局连接管理
active_connections: set = set()

# 从配置文件读取流式配置
streaming_config = CONFIG.get("streaming", {})
MAX_CONNECTIONS = streaming_config.get("max_connections", 10)
INFERENCE_BATCH_SIZE = streaming_config.get("inference_batch_size", 10)  # 每 N 块推理一次
MAX_BUFFER_SIZE = streaming_config.get("max_buffer_size", 100)  # 最大缓冲区大小（约 3 秒音频）


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket 流式识别接口

    协议：
    客户端发送：
    {
        "type": "audio",
        "data": "<base64编码的原始PCM数据>",
        "is_final": false,
        "format": {
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2
        }
    }

    服务端返回：
    {
        "type": "partial",  # partial=中间结果, final=最终结果
        "text": "识别的文本",
        "is_final": false
    }
    """
    # 检查连接数限制
    if len(active_connections) >= MAX_CONNECTIONS:
        await websocket.close(code=1008, reason="服务器连接数已满")
        logger.warning("拒绝新连接：已达到最大连接数")
        return

    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"WebSocket 连接已建立，当前连接数: {len(active_connections)}")

    # 音频缓冲区
    audio_buffer = []
    accumulated_text = ""
    chunks_since_inference = 0

    try:
        while True:
            # 接收客户端消息
            message = await websocket.receive_text()
            data = json.loads(message)

            msg_type = data.get("type")

            if msg_type == "audio":
                # 解码音频数据
                audio_base64 = data.get("data", "")
                is_final = data.get("is_final", False)
                audio_format = data.get("format", {})

                if audio_base64:
                    # Base64 解码
                    audio_bytes = base64.b64decode(audio_base64)

                    # 从原始 PCM 数据转换为 numpy 数组
                    sample_rate = audio_format.get("sample_rate", 16000)
                    channels = audio_format.get("channels", 1)
                    sample_width = audio_format.get("sample_width", 2)

                    # 将字节转换为 int16 数组
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

                    # 归一化到 [-1, 1]
                    audio_data = audio_data.astype(np.float32) / 32768.0

                    # 如果是立体声，转换为单声道
                    if channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                    # 添加到缓冲区
                    audio_buffer.append(audio_data)
                    chunks_since_inference += 1

                    # 限制缓冲区大小，防止内存溢出
                    if len(audio_buffer) > MAX_BUFFER_SIZE:
                        # 移除最旧的块
                        audio_buffer.pop(0)
                        logger.warning("缓冲区已满，移除最旧的音频块")

                    # 优化：每 INFERENCE_BATCH_SIZE 块或最终块才推理
                    should_infer = is_final or chunks_since_inference >= INFERENCE_BATCH_SIZE

                    if should_infer and audio_buffer:
                        # 合并音频
                        full_audio = np.concatenate(audio_buffer)
                        chunks_since_inference = 0

                        # 流式识别
                        try:
                            result = asr_model.generate(
                                input=full_audio,
                                batch_size_s=300,
                            )

                            # 提取文本
                            new_text = ""
                            if result and len(result) > 0:
                                new_text = result[0].get("text", "")

                            # 计算增量文本
                            if new_text.startswith(accumulated_text):
                                increment = new_text[len(accumulated_text):]
                            else:
                                # 如果不是前缀，说明有修正，返回完整文本
                                increment = new_text
                                accumulated_text = ""

                            # 更新累积文本
                            accumulated_text = new_text

                            # 发送结果
                            if increment:
                                await websocket.send_json({
                                    "type": "partial" if not is_final else "final",
                                    "text": increment,
                                    "is_final": is_final
                                })
                                logger.debug(f"发送增量文本: {increment}")

                        except Exception as e:
                            logger.error(f"识别失败: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "error": str(e)
                            })

                # 如果是最后一块，进行标点恢复
                if is_final and accumulated_text and punc_model:
                    try:
                        punc_result = punc_model.generate(input=accumulated_text)
                        if punc_result and len(punc_result) > 0:
                            final_text = punc_result[0].get("text", accumulated_text)

                            # 发送最终结果
                            await websocket.send_json({
                                "type": "final",
                                "text": final_text,
                                "is_final": True
                            })
                            logger.info(f"最终识别结果: {final_text}")
                    except Exception as e:
                        logger.error(f"标点恢复失败: {e}")

                    # 重置状态
                    audio_buffer = []
                    accumulated_text = ""
                    chunks_since_inference = 0

            elif msg_type == "reset":
                # 重置状态
                audio_buffer = []
                accumulated_text = ""
                chunks_since_inference = 0
                await websocket.send_json({
                    "type": "reset",
                    "status": "ok"
                })
                logger.info("状态已重置")

    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
    except asyncio.CancelledError:
        logger.warning("WebSocket 任务被取消")
        raise  # 重新抛出以正确清理
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
    finally:
        # 确保资源清理
        audio_buffer.clear()
        accumulated_text = ""
        active_connections.discard(websocket)
        logger.info(f"WebSocket 资源已清理，当前连接数: {len(active_connections)}")


if __name__ == "__main__":
    import uvicorn

    host = CONFIG["server"]["host"]
    port = CONFIG["server"]["port"]

    logger.info(f"启动 FunASR 服务: {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=CONFIG["logging"]["level"].lower()
    )
