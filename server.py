#!/usr/bin/env python3
"""
FunASR 语音识别服务
支持 HTTP API（非流式）和 WebSocket（流式）
使用 FunASR cache 机制实现真正的流式识别
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf
import torch
import yaml
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from funasr import AutoModel


def get_device() -> str:
    """检测并返回最佳计算设备"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def calculate_audio_energy(audio_data: np.ndarray) -> float:
    """计算音频能量（RMS）"""
    return float(np.sqrt(np.mean(audio_data ** 2)))


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

# 线程池执行器（用于异步执行同步的推理操作，避免阻塞事件循环）
inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr_inference")


def load_models():
    """加载 FunASR 模型"""
    global asr_model, vad_model, punc_model

    logger.info("开始加载 FunASR 模型...")

    # 检测设备
    device = get_device()
    logger.info(f"使用计算设备: {device}")

    try:
        # 加载 ASR 模型（流式）
        model_name = CONFIG["model"]["name"]
        logger.info(f"加载 ASR 模型: {model_name}")
        asr_model = AutoModel(
            model=model_name,
            device=device,
        )
        logger.info("ASR 模型加载成功")

        # 加载 VAD 模型
        vad_model_name = CONFIG["model"].get("vad_model")
        if vad_model_name:
            logger.info(f"加载 VAD 模型: {vad_model_name}")
            vad_model = AutoModel(model=vad_model_name, device=device)
            logger.info("VAD 模型加载成功")

        # 加载标点恢复模型
        punc_model_name = CONFIG["model"].get("punc_model")
        if punc_model_name:
            logger.info(f"加载标点模型: {punc_model_name}")
            punc_model = AutoModel(model=punc_model_name, device=device)
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
ENERGY_THRESHOLD = streaming_config.get("energy_threshold", 0.02)

# FunASR 流式参数
CHUNK_SIZE = streaming_config.get("chunk_size", [0, 10, 5])
CHUNK_STRIDE = streaming_config.get("chunk_stride", CHUNK_SIZE[1] * 960)  # 9600 采样点 = 600ms
ENCODER_CHUNK_LOOK_BACK = streaming_config.get("encoder_chunk_look_back", 4)
DECODER_CHUNK_LOOK_BACK = streaming_config.get("decoder_chunk_look_back", 1)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket 流式识别接口
    使用 FunASR cache 机制实现真正的流式识别

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

    # FunASR 流式识别状态 - 每个连接独立的 cache
    cache: Dict[str, Any] = {}
    audio_buffer: list = []           # 累积音频块
    accumulated_text = ""              # 累积识别文本

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
                    try:
                        # Base64 解码
                        audio_bytes = base64.b64decode(audio_base64)

                        # 从原始 PCM 数据转换为 numpy 数组
                        sample_rate = audio_format.get("sample_rate", 16000)
                        channels = audio_format.get("channels", 1)

                        # 将字节转换为 int16 数组
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

                        # 归一化到 [-1, 1]
                        audio_data = audio_data.astype(np.float32) / 32768.0

                        # 如果是立体声，转换为单声道
                        if channels == 2:
                            audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                        # 将音频块加入缓冲区
                        audio_buffer.append(audio_data)
                        total_samples = sum(len(chunk) for chunk in audio_buffer)

                        # 达到 CHUNK_STRIDE (600ms) 或 is_final 时才推理
                        if total_samples >= CHUNK_STRIDE or is_final:
                            # 合并所有音频块
                            full_audio = np.concatenate(audio_buffer)

                            # 处理完整的 600ms 块
                            while len(full_audio) >= CHUNK_STRIDE or (is_final and len(full_audio) > 0):
                                # 取出要处理的音频块
                                if len(full_audio) >= CHUNK_STRIDE:
                                    chunk_to_process = full_audio[:CHUNK_STRIDE]
                                    full_audio = full_audio[CHUNK_STRIDE:]
                                else:
                                    # is_final 且剩余不足一个 chunk
                                    chunk_to_process = full_audio
                                    full_audio = np.array([], dtype=np.float32)

                                # 音量过滤：检查音频能量是否超过阈值
                                audio_energy = calculate_audio_energy(chunk_to_process)
                                if audio_energy < ENERGY_THRESHOLD and not is_final:
                                    logger.debug(f"音频能量 {audio_energy:.4f} 低于阈值 {ENERGY_THRESHOLD}，跳过识别")
                                    continue

                                # 使用 generate() 进行流式识别
                                def run_streaming_inference(audio, is_final_chunk):
                                    """执行流式推理，使用 cache 维护状态"""
                                    return asr_model.generate(
                                        input=audio,
                                        cache=cache,  # 传入 cache，模型会自动更新
                                        is_final=is_final_chunk,
                                        chunk_size=CHUNK_SIZE,
                                        encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                                        decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
                                    )

                                # 在线程池中执行推理
                                loop = asyncio.get_event_loop()
                                # 只有最后一个块且 is_final 才传 True
                                is_final_chunk = is_final and len(full_audio) == 0
                                result = await loop.run_in_executor(
                                    inference_executor,
                                    run_streaming_inference,
                                    chunk_to_process,
                                    is_final_chunk
                                )

                                # 提取增量文本并累积
                                if result and len(result) > 0:
                                    increment = result[0].get("text", "")
                                    if increment:
                                        accumulated_text += increment  # 累积增量文本
                                        # 发送累积文本给客户端
                                        await websocket.send_json({
                                            "type": "partial" if not is_final_chunk else "final",
                                            "text": accumulated_text,
                                            "is_final": is_final_chunk
                                        })
                                        logger.debug(f"发送累积文本: {accumulated_text} (增量: {increment})")

                            # 保留剩余的音频到下次处理
                            audio_buffer = [full_audio] if len(full_audio) > 0 else []

                    except Exception as e:
                        logger.error(f"音频处理失败: {e}", exc_info=True)
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

                    # 重置状态 - is_final=True 时重置 cache
                    cache = {}
                    audio_buffer = []
                    accumulated_text = ""

            elif msg_type == "reset":
                # 重置状态
                cache = {}
                audio_buffer = []
                accumulated_text = ""
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
        cache = {}
        audio_buffer = []
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
