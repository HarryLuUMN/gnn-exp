import {
  buildStaticVisualizationGeometry,
  type StaticVisualizationScene,
} from "./staticScene";

export type StaticRendererResult =
  | {
      destroy: () => void;
    }
  | {
      error: string;
    };

type StaticProgram = {
  program: WebGLProgram;
  position: number;
  color: number;
  viewport: WebGLUniformLocation;
};

function createShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string
): WebGLShader | { error: string } {
  const shader = gl.createShader(type);
  if (!shader) {
    return { error: "Unable to allocate a shader." };
  }

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) ?? "Shader compilation failed.";
    gl.deleteShader(shader);
    return { error: message };
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
    return { error: "Unable to allocate a shader program." };
  }

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) ?? "Program linking failed.";
    gl.deleteProgram(program);
    return { error: message };
  }

  return program;
}

function createStaticProgram(
  gl: WebGL2RenderingContext
): StaticProgram | { error: string } {
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

function setCanvasSize(canvas: HTMLCanvasElement, scene: StaticVisualizationScene) {
  const pixelRatio = window.devicePixelRatio || 1;
  const displayWidth = Math.max(1, Math.ceil(scene.width));
  const displayHeight = Math.max(1, Math.ceil(scene.height));

  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;
  canvas.width = Math.max(1, Math.floor(displayWidth * pixelRatio));
  canvas.height = Math.max(1, Math.floor(displayHeight * pixelRatio));
}

function bindAttribute(
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

export function renderWebglStaticVisualization(
  canvas: HTMLCanvasElement,
  scene: StaticVisualizationScene
): StaticRendererResult {
  const gl = canvas.getContext("webgl2", {
    alpha: true,
    antialias: true,
    premultipliedAlpha: true,
  });

  if (!gl) {
    return { error: "WebGL2 context unavailable." };
  }

  const program = createStaticProgram(gl);
  if ("error" in program) {
    return program;
  }

  const buffer = gl.createBuffer();
  if (!buffer) {
    gl.deleteProgram(program.program);
    return { error: "Unable to allocate a vertex buffer." };
  }

  const geometry = buildStaticVisualizationGeometry(scene);
  setCanvasSize(canvas, scene);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  gl.useProgram(program.program);
  gl.uniform2f(program.viewport, scene.width, scene.height);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, geometry, gl.STATIC_DRAW);

  const stride = 6 * Float32Array.BYTES_PER_ELEMENT;
  bindAttribute(gl, program.position, 2, stride, 0);
  bindAttribute(gl, program.color, 4, stride, 2 * Float32Array.BYTES_PER_ELEMENT);

  if (geometry.length > 0) {
    gl.drawArrays(gl.TRIANGLES, 0, geometry.length / 6);
  }

  return {
    destroy() {
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program.program);
    },
  };
}
