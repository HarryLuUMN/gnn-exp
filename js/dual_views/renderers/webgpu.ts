import {
  buildGraphGeometry,
  buildMatrixGeometry,
  type EngineResult,
  type GraphCanvasDrawArgs,
  type GraphCanvasEngine,
  type MatrixCanvasDrawArgs,
  type MatrixCanvasEngine,
} from "./shared";

const COLOR_SHADER = `
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

const NODE_SHADER = `
struct Uniforms {
  viewport: vec2f,
  pad: vec2f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec2f,
  @location(1) local: vec2f,
  @location(2) fill: vec4f,
  @location(3) stroke: vec4f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) fill: vec4f,
  @location(2) stroke: vec4f,
}

@vertex
fn vsMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let clip = (input.position / uniforms.viewport) * vec2f(2.0, 2.0) - vec2f(1.0, 1.0);
  output.position = vec4f(clip.x, -clip.y, 0.0, 1.0);
  output.local = input.local;
  output.fill = input.fill;
  output.stroke = input.stroke;
  return output;
}

@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4f {
  let dist = dot(input.local, input.local);
  if (dist > 1.0) {
    discard;
  }

  if (dist > 0.72) {
    return input.stroke;
  }

  return input.fill;
}
`;

type BufferState = {
  buffer: { destroy: () => void } | null;
  capacity: number;
};

type WebGpuSharedState = {
  device: any;
  context: any;
  format: string;
  uniformBuffer: { destroy: () => void };
  bindGroupLayout: any;
  bindGroup: any;
  bufferUsage: {
    COPY_DST: number;
    UNIFORM: number;
    VERTEX: number;
  };
  shaderStage: {
    VERTEX: number;
  };
};

function setCanvasSize(canvas: HTMLCanvasElement) {
  const pixelRatio = window.devicePixelRatio || 1;
  const displayWidth = Math.max(1, Math.floor(canvas.clientWidth * pixelRatio));
  const displayHeight = Math.max(1, Math.floor(canvas.clientHeight * pixelRatio));

  if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
    canvas.width = displayWidth;
    canvas.height = displayHeight;
  }
}

function ensureBuffer(
  device: any,
  state: BufferState,
  data: Float32Array,
  usage: number
) {
  if (data.byteLength === 0) {
    return state;
  }

  if (!state.buffer || state.capacity < data.byteLength) {
    state.buffer?.destroy();
    const capacity = Math.max(data.byteLength, state.capacity * 2, 256);
    state.buffer = device.createBuffer({
      size: capacity,
      usage,
    });
    state.capacity = capacity;
  }

  device.queue.writeBuffer(state.buffer, 0, data);
  return state;
}

function createColorPipeline(
  device: any,
  format: string,
  bindGroupLayout: any,
  topology: string
) {
  const module = device.createShaderModule({ code: COLOR_SHADER });

  return device.createRenderPipeline({
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
      targets: [{ format, blend: undefined }],
    },
    primitive: {
      topology,
    },
  });
}

function createNodePipeline(
  device: any,
  format: string,
  bindGroupLayout: any
) {
  const module = device.createShaderModule({ code: NODE_SHADER });

  return device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: "vsMain",
      buffers: [
        {
          arrayStride: 48,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            { shaderLocation: 1, offset: 8, format: "float32x2" },
            { shaderLocation: 2, offset: 16, format: "float32x4" },
            { shaderLocation: 3, offset: 32, format: "float32x4" },
          ],
        },
      ],
    },
    fragment: {
      module,
      entryPoint: "fsMain",
      targets: [{ format, blend: undefined }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });
}

async function createSharedState(
  canvas: HTMLCanvasElement
): Promise<WebGpuSharedState | { error: string }> {
  const gpu = (
    navigator as Navigator & {
      gpu?: {
        requestAdapter: () => Promise<{ requestDevice: () => Promise<any> } | null>;
        getPreferredCanvasFormat: () => string;
      };
    }
  ).gpu;
  if (!gpu) {
    return { error: "WebGPU is unavailable in this browser environment." } as const;
  }

  const bufferUsage = (globalThis as {
    GPUBufferUsage?: { COPY_DST: number; UNIFORM: number; VERTEX: number };
  }).GPUBufferUsage;
  const shaderStage = (globalThis as {
    GPUShaderStage?: { VERTEX: number };
  }).GPUShaderStage;

  if (!bufferUsage || !shaderStage) {
    return { error: "WebGPU runtime constants unavailable." };
  }

  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    return { error: "WebGPU adapter unavailable." } as const;
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu") as any;

  if (!context) {
    device.destroy();
    return { error: "WebGPU canvas context unavailable." } as const;
  }

  const format = gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "premultiplied",
  });

  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: shaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  return {
    device,
    context,
    format,
    uniformBuffer,
    bindGroupLayout,
    bindGroup,
    bufferUsage,
    shaderStage,
  };
}

export async function createWebgpuGraphEngine(
  canvas: HTMLCanvasElement
): Promise<EngineResult<GraphCanvasEngine>> {
  const shared = await createSharedState(canvas);
  if ("error" in shared) {
    return shared;
  }

  const edgePipeline = createColorPipeline(
    shared.device,
    shared.format,
    shared.bindGroupLayout,
    "line-list"
  );
  const nodePipeline = createNodePipeline(
    shared.device,
    shared.format,
    shared.bindGroupLayout
  );

  const edgeState: BufferState = { buffer: null, capacity: 0 };
  const nodeState: BufferState = { buffer: null, capacity: 0 };

  return {
    draw(args: GraphCanvasDrawArgs) {
      setCanvasSize(canvas);
      shared.device.queue.writeBuffer(
        shared.uniformBuffer,
        0,
        new Float32Array([args.width, args.height, 0, 0])
      );

      const nodesById = new Map(args.nodes.map((node) => [node.id, node]));
      const geometry = buildGraphGeometry(
        args.nodes,
        args.links,
        nodesById,
        args.hover,
        args.transform
      );

      ensureBuffer(
        shared.device,
        edgeState,
        geometry.edgeVertices,
        shared.bufferUsage.VERTEX | shared.bufferUsage.COPY_DST
      );
      ensureBuffer(
        shared.device,
        nodeState,
        geometry.nodeVertices,
        shared.bufferUsage.VERTEX | shared.bufferUsage.COPY_DST
      );

      const commandEncoder = shared.device.createCommandEncoder();
      const pass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: shared.context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          },
        ],
      });

      if (geometry.edgeVertices.length > 0 && edgeState.buffer) {
        pass.setPipeline(edgePipeline);
        pass.setBindGroup(0, shared.bindGroup);
        pass.setVertexBuffer(0, edgeState.buffer);
        pass.draw(geometry.edgeVertices.length / 6);
      }

      if (geometry.nodeVertices.length > 0 && nodeState.buffer) {
        pass.setPipeline(nodePipeline);
        pass.setBindGroup(0, shared.bindGroup);
        pass.setVertexBuffer(0, nodeState.buffer);
        pass.draw(geometry.nodeVertices.length / 12);
      }

      pass.end();
      shared.device.queue.submit([commandEncoder.finish()]);
    },
    destroy() {
      edgeState.buffer?.destroy();
      nodeState.buffer?.destroy();
      shared.uniformBuffer.destroy();
      shared.device.destroy();
    },
  };
}

export async function createWebgpuMatrixEngine(
  canvas: HTMLCanvasElement
): Promise<EngineResult<MatrixCanvasEngine>> {
  const shared = await createSharedState(canvas);
  if ("error" in shared) {
    return shared;
  }

  const pipeline = createColorPipeline(
    shared.device,
    shared.format,
    shared.bindGroupLayout,
    "triangle-list"
  );
  const state: BufferState = { buffer: null, capacity: 0 };

  return {
    draw(args: MatrixCanvasDrawArgs) {
      setCanvasSize(canvas);
      shared.device.queue.writeBuffer(
        shared.uniformBuffer,
        0,
        new Float32Array([args.width, args.height, 0, 0])
      );

      const geometry = buildMatrixGeometry(
        args.nodes,
        args.matrix,
        args.hover,
        args.layout
      );

      ensureBuffer(
        shared.device,
        state,
        geometry,
        shared.bufferUsage.VERTEX | shared.bufferUsage.COPY_DST
      );

      const commandEncoder = shared.device.createCommandEncoder();
      const pass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: shared.context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          },
        ],
      });

      if (geometry.length > 0 && state.buffer) {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, shared.bindGroup);
        pass.setVertexBuffer(0, state.buffer);
        pass.draw(geometry.length / 6);
      }

      pass.end();
      shared.device.queue.submit([commandEncoder.finish()]);
    },
    destroy() {
      state.buffer?.destroy();
      shared.uniformBuffer.destroy();
      shared.device.destroy();
    },
  };
}
