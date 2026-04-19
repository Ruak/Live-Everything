import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
} from '@xenova/transformers';
import type { DetectionResult, DetectionService } from '../types/detection';

/**
 * 对应本地目录 `models/Xenova/gelan-c_all`（Vite 映射到 `/models/`）。
 *
 * 选择 **gelan-c**（YOLOv9 的精简 backbone，COCO 80 类）而非 `yolov9-c_all`，
 * 以贴齐 HF Space `Xenova/video-object-detection` 的运行时行为与流畅度：
 *  - q8 权重 ~26MB（yolov9-c 约 35MB），fp16 ~51MB（yolov9-c 约 68MB）
 *  - 前向算子更少，单帧推理延迟更低
 *
 * 注意：`model_type=yolov9` 不能走 `pipeline('object-detection')`（仅支持 DETR/YOLOS 等），
 * 必须按官方 README 使用 `AutoModel` + `AutoProcessor`。
 * @see https://huggingface.co/Xenova/gelan-c_all
 * @see https://huggingface.co/spaces/Xenova/video-object-detection
 */
const LOCAL_MODEL_ID = 'Xenova/gelan-c_all';

function configureTransformersEnv() {
  env.allowLocalModels = true;
  env.localModelPath = '/models/';
  env.allowRemoteModels = false;
  /**
   * ★ 关键：把 ONNX Runtime 的 WASM backend 放到 Web Worker 里跑
   * （与 HF Space `Xenova/video-object-detection` 行为一致），
   * 避免每次推理阻塞主线程导致 RAF 掉帧、video 卡顿。
   * @see https://huggingface.co/spaces/Xenova/video-object-detection/blob/main/assets/index-C0Q5FIv3.js
   */
  const onnx = (env.backends as unknown as {
    onnx?: { wasm?: { proxy?: boolean; numThreads?: number } };
  }).onnx;
  if (onnx?.wasm) {
    onnx.wasm.proxy = true;
  }
}

/**
 * 权重变体。注意：传给 Transformers.js 的 `model_file_name` 必须是不带后缀的基名（`model`），
 * 库会按 `dtype` 拼成 `model_quantized.onnx` / `model_fp16.onnx`。
 * 若误传 `model_quantized.onnx`，会变成 `model_quantized.onnx_quantized.onnx` 等无效 URL，
 * 下载到 HTML/空响应后 ORT 会报 protobuf 解析失败。
 */
export type YoloOnnxVariant = 'quantized' | 'fp16';

type LoadedModel = Awaited<ReturnType<typeof AutoModel.from_pretrained>>;
type LoadedProcessor = Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>;
type ProcessorWithFeatureExtractor = LoadedProcessor & {
  feature_extractor?: {
    size?: {
      shortest_edge?: number;
    };
  };
};

/** `[xmin, ymin, xmax, ymax, score, classId]`（坐标在 reshaped 空间内） */
export type YoloPrediction = [number, number, number, number, number, number];

export interface RawDetectionFrame {
  predictions: YoloPrediction[];
  /** 原站 `reshaped_input_sizes[0].reverse()`：[width, height] */
  reshapedSize: [number, number];
  /** 原始图像 [width, height]（推理输入 canvas 的尺寸） */
  originalSize: [number, number];
  id2label: Record<string, string>;
  /** 已按 threshold 过滤过的高层结果；若只需原始 predictions 请用 `predictions`。 */
  results: DetectionResult[];
  threshold: number;
}

function createRawImageFromCanvas(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('[yolo] 无法从 canvas 获取 2D 上下文');
  }
  const { width, height } = canvas;
  const imageData = ctx.getImageData(0, 0, width, height);
  return new RawImage(imageData.data, width, height, 4);
}

function iterPredictionRows(raw: unknown): [number, number, number, number, number, number][] {
  if (!Array.isArray(raw) || raw.length === 0) return [];
  const out: [number, number, number, number, number, number][] = [];
  const first = raw[0];
  if (Array.isArray(first)) {
    for (const row of raw as unknown[][]) {
      if (row.length >= 6) {
        out.push([
          Number(row[0]),
          Number(row[1]),
          Number(row[2]),
          Number(row[3]),
          Number(row[4]),
          Number(row[5]),
        ]);
      }
    }
  } else {
    const flat = raw as number[];
    for (let i = 0; i + 5 < flat.length; i += 6) {
      out.push([
        flat[i],
        flat[i + 1],
        flat[i + 2],
        flat[i + 3],
        flat[i + 4],
        flat[i + 5],
      ]);
    }
  }
  return out;
}

export class YoloDetectionService implements DetectionService {
  private model: LoadedModel | null = null;
  private processor: LoadedProcessor | null = null;
  private initPromise: Promise<void> | null = null;
  private threshold: number;
  private readonly onnxVariant: YoloOnnxVariant;

  constructor(opts?: { threshold?: number; onnxVariant?: YoloOnnxVariant }) {
    this.threshold = opts?.threshold ?? 0.25;
    this.onnxVariant = opts?.onnxVariant ?? 'quantized';
  }

  setThreshold(value: number) {
    this.threshold = value;
  }

  setProcessorShortestEdge(shortestEdge: number) {
    const fe = (this.processor as ProcessorWithFeatureExtractor | null)?.feature_extractor;
    if (fe?.size) fe.size.shortest_edge = shortestEdge;
  }

  async initialize(): Promise<void> {
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      configureTransformersEnv();
      /**
       * 不主动指定 `device`，让 Transformers.js 走 wasm backend（已通过
       * `env.backends.onnx.wasm.proxy = true` 放到 Worker）。对 q8/fp16 权重
       * 这通常比 webgpu 更稳定、更快，且不会阻塞主线程渲染 —— 与原站一致。
       */
      // @huggingface/transformers 用 `dtype` 指定权重精度：'q8' → model_quantized.onnx, 'fp16' → model_fp16.onnx
      const dtype = this.onnxVariant === 'fp16' ? 'fp16' : 'q8';
      // Xenova 运行时需要 dtype 选择 q8/fp16 ONNX；类型定义未收录该字段
      this.model = await AutoModel.from_pretrained(LOCAL_MODEL_ID, {
        local_files_only: true,
        model_file_name: 'model',
        dtype,
      } as Parameters<typeof AutoModel.from_pretrained>[1]);
      this.processor = await AutoProcessor.from_pretrained(LOCAL_MODEL_ID, {
        local_files_only: true,
      });
    })();

    return this.initPromise;
  }

  async detect(frame: ImageData): Promise<DetectionResult[]> {
    const img = new RawImage(frame.data, frame.width, frame.height, 4);
    const { results } = await this.runRaw(img);
    return results;
  }

  /** 避免 ImageData 拷贝，供实时页面使用 */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<DetectionResult[]> {
    const img = createRawImageFromCanvas(canvas);
    const { results } = await this.runRaw(img);
    return results;
  }

  /**
   * 与 Hugging Face Space 的 `updateCanvas` 对齐：直接返回原始 predictions
   * 与 `reshaped_input_sizes`（[width, height]），供调用方按原站百分比公式绘制。
   */
  async detectRaw(canvas: HTMLCanvasElement): Promise<RawDetectionFrame> {
    const img = createRawImageFromCanvas(canvas);
    return this.runRaw(img);
  }

  /** 模型 id2label 字典，用于展示名称。需在 `initialize` 之后访问。 */
  get id2label(): Record<string, string> {
    return (
      (this.model?.config as { id2label?: Record<string, string> } | undefined)
        ?.id2label ?? {}
    );
  }

  private async runRaw(
    img: InstanceType<typeof RawImage>
  ): Promise<RawDetectionFrame> {
    if (!this.model || !this.processor) await this.initialize();

    const inputs = await this.processor!(img);
    const orig = inputs.original_sizes?.[0] as [number, number] | undefined;
    const reshaped = inputs.reshaped_input_sizes?.[0] as [number, number] | undefined;
    if (!orig || !reshaped || reshaped[0] <= 0 || reshaped[1] <= 0) {
      throw new Error('[yolo] 预处理未返回 original_sizes / reshaped_input_sizes');
    }
    /** image_processors_utils 约定：[height, width] */
    const [origH, origW] = orig;
    const [resH, resW] = reshaped;
    const sx = origW / resW;
    const sy = origH / resH;

    const forward = await this.model!(inputs);
    const tensor = forward.outputs;
    if (!tensor?.tolist) {
      throw new Error('[yolo] 模型输出缺少 outputs 或 tolist()');
    }

    const predictions = iterPredictionRows(tensor.tolist());
    const id2label =
      (this.model!.config as { id2label?: Record<string, string> }).id2label ??
      {};

    const results: DetectionResult[] = [];
    for (const [xmin, ymin, xmax, ymax, score, id] of predictions) {
      if (score < this.threshold) continue;
      const label = id2label[String(id)] ?? String(id);
      const x = xmin * sx;
      const y = ymin * sy;
      const w = (xmax - xmin) * sx;
      const h = (ymax - ymin) * sy;
      results.push({
        classId: label,
        className: label,
        classIdNum: id,
        confidence: score,
        bbox: { x, y, width: w, height: h },
      });
    }
    return {
      predictions,
      /** 与原站一致：取 reshaped 的 [width, height] 方向 */
      reshapedSize: [resW, resH],
      originalSize: [origW, origH],
      id2label,
      results,
      threshold: this.threshold,
    };
  }

  dispose(): void {
    void this.model?.dispose?.();
    this.model = null;
    this.processor = null;
    this.initPromise = null;
  }
}
