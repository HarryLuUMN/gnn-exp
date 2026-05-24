import {
  buildStaticVisualizationGeometry,
  type StaticVisualizationScene,
} from "./staticScene";
import type { StaticRendererResult } from "./webglStaticRenderer";

const STATIC_SHADER = `
struct Uniforms {
  viewport: vec2f,
  pad: vec2f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec2f,
  @location(1) color: vec4f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
}

@vertex
fn vsMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let clip = (input.position / uniforms.viewport) * vec2f(2.0, 2.0) - vec2f(1.0, 1.0);
  output.position = vec4f(clip.x, -clip.y, 0.0, 1.0);
  output.color = input.color;
  return output;
}

@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
`;

type WebGpuRuntime = {
  bufferUsage: {
    COPY_DST: number;
    UNIFORM: number;
    VERTEX: number;
  };
  shaderStage: {
    VERTEX: number;
  };
};

const MAX_GPU_BACKING_STORE_DIMENSION = 4096;

function getWebGpuRuntime(): WebGpuRuntime | { error: string } {
  const bufferUsage = (globalThis as {
    GPUBufferUsage?: WebGpuRuntime["bufferUsage"];
  }).GPUBufferUsage;
  const shaderStage = (globalThis as {
    GPUShaderStage?: WebGpuRuntime["shaderStage"];
  }).GPUShaderStage;

  if (!bufferUsage || !shaderStage) {
    return { error: "WebGPU runtime constants unavailable." };
  }

  return { bufferUsage, shaderStage };
}

function resolvePixelRatio(
  displayWidth: number,
  displayHeight: number,
  maxDimension: number
) {
  const targetPixelRatio = window.devicePixelRatio || 1;
  const largestDisplayDimension = Math.max(displayWidth, displayHeight, 1);
  const maxPixelRatio = maxDimension / largestDisplayDimension;
  return Math.min(targetPixelRatio, maxPixelRatio);
}

function setCanvasSize(
  canvas: HTMLCanvasElement,
  scene: StaticVisualizationScene,
  maxDimension: number
) {
  const displayWidth = Math.max(1, Math.ceil(scene.width));
  const displayHeight = Math.max(1, Math.ceil(scene.height));
  const pixelRatio = resolvePixelRatio(displayWidth, displayHeight, maxDimension);

  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;
  canvas.width = Math.max(1, Math.floor(displayWidth * pixelRatio));
  canvas.height = Math.max(1, Math.floor(displayHeight * pixelRatio));
}

export async function renderWebgpuStaticVisualization(
  canvas: HTMLCanvasElement,
  scene: StaticVisualizationScene
): Promise<StaticRendererResult> {
  const gpu = (
    navigator as Navigator & {
      gpu?: {
        requestAdapter: () => Promise<{
          limits?: { maxTextureDimension2D?: number };
          requestDevice: () => Promise<any>;
        } | null>;
        getPreferredCanvasFormat: () => string;
      };
    }
  ).gpu;

  if (!gpu) {
    return { error: "WebGPU is unavailable in this browser environment." };
  }

  const runtime = getWebGpuRuntime();
  if ("error" in runtime) {
    return runtime;
  }

  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    return { error: "WebGPU adapter unavailable." };
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu") as any;
  if (!context) {
    device.destroy();
    return { error: "WebGPU canvas context unavailable." };
  }

  const format = gpu.getPreferredCanvasFormat();
  const maxDimension = Math.max(
    1,
    Math.min(
      Number(adapter.limits?.maxTextureDimension2D) || MAX_GPU_BACKING_STORE_DIMENSION,
      MAX_GPU_BACKING_STORE_DIMENSION
    )
  );
  setCanvasSize(canvas, scene, maxDimension);
  context.configure({
    device,
    format,
    alphaMode: "premultiplied",
  });

  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: runtime.bufferUsage.UNIFORM | runtime.bufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([scene.width, scene.height, 0, 0])
  );

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: runtime.shaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const module = device.createShaderModule({ code: STATIC_SHADER });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: "vsMain",
      buffers: [
        {
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            { shaderLocation: 1, offset: 8, format: "float32x4" },
          ],
        },
      ],
    },
    fragment: {
      module,
      entryPoint: "fsMain",
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const geometry = buildStaticVisualizationGeometry(scene);
  const vertexBuffer = device.createBuffer({
    size: Math.max(geometry.byteLength, 4),
    usage: runtime.bufferUsage.VERTEX | runtime.bufferUsage.COPY_DST,
  });
  if (geometry.byteLength > 0) {
    device.queue.writeBuffer(vertexBuffer, 0, geometry);
  }

  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      },
    ],
  });

  if (geometry.length > 0) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(geometry.length / 6);
  }

  pass.end();
  device.queue.submit([commandEncoder.finish()]);

  return {
    destroy() {
      vertexBuffer.destroy();
      uniformBuffer.destroy();
      device.destroy();
    },
  };
}
