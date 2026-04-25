import {
  buildGraphGeometry,
  buildMatrixGeometry,
  type EngineResult,
  type GraphCanvasDrawArgs,
  type GraphCanvasEngine,
  type MatrixCanvasDrawArgs,
  type MatrixCanvasEngine,
} from "./shared";

type PositionColorProgram = {
  program: WebGLProgram;
  position: number;
  color: number;
  viewport: WebGLUniformLocation;
};

type GraphNodeProgram = {
  program: WebGLProgram;
  position: number;
  local: number;
  fill: number;
  stroke: number;
  viewport: WebGLUniformLocation;
};

function setCanvasSize(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext) {
  const pixelRatio = window.devicePixelRatio || 1;
  const displayWidth = Math.max(1, Math.floor(canvas.clientWidth * pixelRatio));
  const displayHeight = Math.max(1, Math.floor(canvas.clientHeight * pixelRatio));

  if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
    canvas.width = displayWidth;
    canvas.height = displayHeight;
  }

  gl.viewport(0, 0, canvas.width, canvas.height);
}

function createShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string
): WebGLShader | { error: string } {
  const shader = gl.createShader(type);
  if (!shader) {
    return { error: "Unable to allocate a shader." } as const;
  }

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) ?? "Shader compilation failed.";
    gl.deleteShader(shader);
    return { error: message } as const;
  }

  return shader;
}

function createProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string
): WebGLProgram | { error: string } {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  if ("error" in vertexShader) {
    return vertexShader;
  }

  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  if ("error" in fragmentShader) {
    gl.deleteShader(vertexShader);
    return fragmentShader;
  }

  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    return { error: "Unable to allocate a shader program." } as const;
  }

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) ?? "Program linking failed.";
    gl.deleteProgram(program);
    return { error: message } as const;
  }

  return program;
}

function createPositionColorProgram(
  gl: WebGL2RenderingContext
): EngineResult<PositionColorProgram> {
  const program = createProgram(
    gl,
    `#version 300 es
    in vec2 a_position;
    in vec4 a_color;
    uniform vec2 u_viewport;
    out vec4 v_color;

    void main() {
      vec2 clip = (a_position / u_viewport) * 2.0 - 1.0;
      gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
      v_color = a_color;
    }`,
    `#version 300 es
    precision mediump float;
    in vec4 v_color;
    out vec4 outColor;

    void main() {
      outColor = v_color;
    }`
  );

  if ("error" in program) {
    return program;
  }

  const viewport = gl.getUniformLocation(program, "u_viewport");
  if (!viewport) {
    gl.deleteProgram(program);
    return { error: "Unable to bind the viewport uniform." };
  }

  return {
    program,
    position: gl.getAttribLocation(program, "a_position"),
    color: gl.getAttribLocation(program, "a_color"),
    viewport,
  };
}

function createGraphNodeProgram(
  gl: WebGL2RenderingContext
): EngineResult<GraphNodeProgram> {
  const program = createProgram(
    gl,
    `#version 300 es
    in vec2 a_position;
    in vec2 a_local;
    in vec4 a_fill;
    in vec4 a_stroke;
    uniform vec2 u_viewport;
    out vec2 v_local;
    out vec4 v_fill;
    out vec4 v_stroke;

    void main() {
      vec2 clip = (a_position / u_viewport) * 2.0 - 1.0;
      gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
      v_local = a_local;
      v_fill = a_fill;
      v_stroke = a_stroke;
    }`,
    `#version 300 es
    precision mediump float;
    in vec2 v_local;
    in vec4 v_fill;
    in vec4 v_stroke;
    out vec4 outColor;

    void main() {
      float dist = dot(v_local, v_local);
      if (dist > 1.0) {
        discard;
      }

      if (dist > 0.72) {
        outColor = v_stroke;
        return;
      }

      outColor = v_fill;
    }`
  );

  if ("error" in program) {
    return program;
  }

  const viewport = gl.getUniformLocation(program, "u_viewport");
  if (!viewport) {
    gl.deleteProgram(program);
    return { error: "Unable to bind the viewport uniform." };
  }

  return {
    program,
    position: gl.getAttribLocation(program, "a_position"),
    local: gl.getAttribLocation(program, "a_local"),
    fill: gl.getAttribLocation(program, "a_fill"),
    stroke: gl.getAttribLocation(program, "a_stroke"),
    viewport,
  };
}

function deleteBuffer(gl: WebGL2RenderingContext, buffer: WebGLBuffer | null) {
  if (buffer) {
    gl.deleteBuffer(buffer);
  }
}

function bindFloatAttribute(
  gl: WebGL2RenderingContext,
  location: number,
  size: number,
  stride: number,
  offset: number
) {
  if (location < 0) {
    return;
  }

  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offset);
}

function disableAttribute(gl: WebGL2RenderingContext, location: number) {
  if (location >= 0) {
    gl.disableVertexAttribArray(location);
  }
}

export function createWebglGraphEngine(
  canvas: HTMLCanvasElement
): EngineResult<GraphCanvasEngine> {
  const gl = canvas.getContext("webgl2", {
    alpha: true,
    antialias: true,
    premultipliedAlpha: true,
  });

  if (!gl) {
    return { error: "WebGL2 context unavailable." };
  }

  const edgeProgram = createPositionColorProgram(gl);
  if ("error" in edgeProgram) {
    return edgeProgram;
  }

  const nodeProgram = createGraphNodeProgram(gl);
  if ("error" in nodeProgram) {
    gl.deleteProgram(edgeProgram.program);
    return nodeProgram;
  }

  const edgeBuffer = gl.createBuffer();
  const nodeBuffer = gl.createBuffer();

  if (!edgeBuffer || !nodeBuffer) {
    gl.deleteProgram(edgeProgram.program);
    gl.deleteProgram(nodeProgram.program);
    deleteBuffer(gl, edgeBuffer);
    deleteBuffer(gl, nodeBuffer);
    return { error: "Unable to allocate WebGL buffers." };
  }

  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);

  return {
    draw(args: GraphCanvasDrawArgs) {
      setCanvasSize(canvas, gl);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      const nodesById = new Map(args.nodes.map((node) => [node.id, node]));
      const geometry = buildGraphGeometry(
        args.nodes,
        args.links,
        nodesById,
        args.hover,
        args.transform
      );

      if (geometry.edgeVertices.length > 0) {
        gl.useProgram(edgeProgram.program);
        gl.uniform2f(edgeProgram.viewport, args.width, args.height);
        gl.bindBuffer(gl.ARRAY_BUFFER, edgeBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, geometry.edgeVertices, gl.DYNAMIC_DRAW);
        bindFloatAttribute(gl, edgeProgram.position, 2, 24, 0);
        bindFloatAttribute(gl, edgeProgram.color, 4, 24, 8);
        gl.drawArrays(gl.LINES, 0, geometry.edgeVertices.length / 6);
        disableAttribute(gl, edgeProgram.position);
        disableAttribute(gl, edgeProgram.color);
      }

      if (geometry.nodeVertices.length > 0) {
        gl.useProgram(nodeProgram.program);
        gl.uniform2f(nodeProgram.viewport, args.width, args.height);
        gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, geometry.nodeVertices, gl.DYNAMIC_DRAW);
        bindFloatAttribute(gl, nodeProgram.position, 2, 48, 0);
        bindFloatAttribute(gl, nodeProgram.local, 2, 48, 8);
        bindFloatAttribute(gl, nodeProgram.fill, 4, 48, 16);
        bindFloatAttribute(gl, nodeProgram.stroke, 4, 48, 32);
        gl.drawArrays(gl.TRIANGLES, 0, geometry.nodeVertices.length / 12);
        disableAttribute(gl, nodeProgram.position);
        disableAttribute(gl, nodeProgram.local);
        disableAttribute(gl, nodeProgram.fill);
        disableAttribute(gl, nodeProgram.stroke);
      }
    },
    destroy() {
      gl.deleteProgram(edgeProgram.program);
      gl.deleteProgram(nodeProgram.program);
      gl.deleteBuffer(edgeBuffer);
      gl.deleteBuffer(nodeBuffer);
    },
  };
}

export function createWebglMatrixEngine(
  canvas: HTMLCanvasElement
): EngineResult<MatrixCanvasEngine> {
  const gl = canvas.getContext("webgl2", {
    alpha: true,
    antialias: true,
    premultipliedAlpha: true,
  });

  if (!gl) {
    return { error: "WebGL2 context unavailable." };
  }

  const program = createPositionColorProgram(gl);
  if ("error" in program) {
    return program;
  }

  const buffer = gl.createBuffer();
  if (!buffer) {
    gl.deleteProgram(program.program);
    return { error: "Unable to allocate a WebGL buffer." };
  }

  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);

  return {
    draw(args: MatrixCanvasDrawArgs) {
      setCanvasSize(canvas, gl);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      const geometry = buildMatrixGeometry(
        args.nodes,
        args.matrix,
        args.hover,
        args.layout
      );

      if (geometry.length === 0) {
        return;
      }

      gl.useProgram(program.program);
      gl.uniform2f(program.viewport, args.width, args.height);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, geometry, gl.DYNAMIC_DRAW);
      bindFloatAttribute(gl, program.position, 2, 24, 0);
      bindFloatAttribute(gl, program.color, 4, 24, 8);
      gl.drawArrays(gl.TRIANGLES, 0, geometry.length / 6);
      disableAttribute(gl, program.position);
      disableAttribute(gl, program.color);
    },
    destroy() {
      gl.deleteProgram(program.program);
      gl.deleteBuffer(buffer);
    },
  };
}
