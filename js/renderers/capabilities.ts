export type RendererMode = "auto" | "svg" | "webgl" | "webgpu";
export type ResolvedRenderer = "svg" | "webgl" | "webgpu";

export interface RenderCapabilities {
  webgl: boolean;
  webgpu: boolean;
}

export interface RendererResolution {
  effectiveRenderer: ResolvedRenderer;
  reason: string | null;
}

type FailureMap = Partial<Record<ResolvedRenderer, string>>;
type WebGpuNavigator = Navigator & {
  gpu?: {
    requestAdapter?: () => Promise<unknown>;
  };
};

function joinFallbackReason(prefix: string, target: ResolvedRenderer) {
  return `${prefix} Using ${target.toUpperCase()} instead.`;
}

function getWebGpuRuntime() {
  if (typeof navigator === "undefined") {
    return undefined;
  }

  return (navigator as WebGpuNavigator).gpu;
}

export function detectCapabilities(): RenderCapabilities {
  if (typeof document === "undefined") {
    return { webgl: false, webgpu: false };
  }

  const canvas = document.createElement("canvas");
  const gpu = getWebGpuRuntime();

  return {
    webgl: canvas.getContext("webgl2") !== null,
    webgpu: typeof gpu?.requestAdapter === "function",
  };
}

export async function detectWebgpuCapability(): Promise<boolean> {
  const gpu = getWebGpuRuntime();

  if (typeof gpu?.requestAdapter !== "function") {
    return false;
  }

  try {
    return (await gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

export function resolveRenderer(
  mode: RendererMode,
  capabilities: RenderCapabilities,
  failures: FailureMap = {}
): RendererResolution {
  const webglReady = capabilities.webgl && !failures.webgl;
  const webgpuReady = capabilities.webgpu && !failures.webgpu;

  if (mode === "svg") {
    return { effectiveRenderer: "svg", reason: null };
  }

  if (mode === "webgl") {
    if (webglReady) {
      return { effectiveRenderer: "webgl", reason: null };
    }

    const reason =
      failures.webgl ?? "WebGL is unavailable in this browser environment.";
    return {
      effectiveRenderer: "svg",
      reason: joinFallbackReason(reason, "svg"),
    };
  }

  if (mode === "webgpu") {
    if (webgpuReady) {
      return { effectiveRenderer: "webgpu", reason: null };
    }

    const reason =
      failures.webgpu ?? "WebGPU is unavailable in this browser environment.";
    return {
      effectiveRenderer: "svg",
      reason: joinFallbackReason(reason, "svg"),
    };
  }

  if (webgpuReady) {
    return { effectiveRenderer: "webgpu", reason: null };
  }

  if (webglReady) {
    const reason =
      failures.webgpu ?? "WebGPU is unavailable in this browser environment.";
    return {
      effectiveRenderer: "webgl",
      reason: joinFallbackReason(reason, "webgl"),
    };
  }

  const fallbackReason =
    failures.webgl ??
    failures.webgpu ??
    "Neither WebGPU nor WebGL is available in this browser environment.";

  return {
    effectiveRenderer: "svg",
    reason: joinFallbackReason(fallbackReason, "svg"),
  };
}
