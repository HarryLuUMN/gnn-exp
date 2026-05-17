import { expect, test } from "@playwright/test";

const cases = [
  { model: "gat", aggregation: "attention", expectsAttention: true },
  { model: "graphsage", aggregation: "mean", expectsAttention: false },
  { model: "gin", aggregation: "sum", expectsAttention: false },
] as const;

for (const { model, aggregation, expectsAttention } of cases) {
  test(`${model} renders and expands`, async ({ page }) => {
    const failures: string[] = [];
    page.on("pageerror", (error) => failures.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") {
        failures.push(message.text());
      }
    });

    await page.goto(`/tests/playwright/fixtures/gnn-layer-harness.html?model=${model}`);
    await page.waitForFunction(() => window.__GNN_LAYER_READY === true);

    await expect(page.locator("#matrix-svg")).toBeVisible();
    await expect(page.locator("#feature-layer-frame-1-node-1")).toBeVisible();
    await expect(page.locator("#fc-feature-layer-frame-node-1")).toBeVisible();

    const content = page.locator(".gnn-model-content");
    const initialTransform = await content.evaluate((element) => getComputedStyle(element).transform);
    await page.locator(".gnn-model-viewport").hover();
    await page.mouse.wheel(120, 0);
    await expect.poll(
      () => content.evaluate((element) => getComputedStyle(element).transform)
    ).not.toBe(initialTransform);

    await page.locator("#feature-layer-1-node-1-dim-0").click();

    await expect(page.getByText(`agg: ${aggregation}`)).toBeVisible();
    await expect(page.locator(".weight-matrix-frame")).toBeVisible();
    await expect(page.locator(".weight-matrix-cell").first()).toHaveCSS("stroke", "rgb(216, 222, 219)");
    await expect(page.locator(".weight-matrix-cell").first()).toHaveCSS("stroke-width", "0.35px");
    if (expectsAttention) {
      const fixture = await page.evaluate(async () => {
        const model = new URLSearchParams(window.location.search).get("model");
        const response = await fetch(`../.cache/fixtures/${model}.json`);
        return response.json();
      });
      expect(fixture.modelInfo.conv1.attention.edges.length).toBeGreaterThan(0);
      await expect(page.locator(".attention-coefficient-text").first()).toBeVisible();
      const coefficients = await page.locator(".attention-coefficient-text").allTextContents();
      expect(coefficients.some((value) => /^\d+\.\d{2}$/.test(value.trim()))).toBe(true);

      const sourceNodes = fixture.modelInfo.conv1.attention.edges
        .filter((edge: { target: number }) => edge.target === 1)
        .map((edge: { source: number }) => edge.source);
      const pathAnchors = await page.evaluate((sources) => {
        const parsePathStart = (path: Element | null) => {
          const d = path?.getAttribute("d") ?? "";
          const match = d.match(/^M\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/);
          return match ? { x: Number(match[1]), y: Number(match[2]) } : null;
        };

        return sources.map((source, index) => {
          const frame = document.querySelector(`#feature-layer-frame-0-node-${source}`);
          const path = document.querySelector(`#link-path-aggregated-1-1-to-${index}`);
          const label = document.querySelector(`#degree-multiplier-text-1-1-to-${index}`);
          const frameX = Number(frame?.getAttribute("x"));
          const frameY = Number(frame?.getAttribute("y"));
          const frameWidth = Number(frame?.getAttribute("width"));
          const frameHeight = Number(frame?.getAttribute("height"));
          return {
            source,
            pathStart: parsePathStart(path),
            expectedX: frameX + frameWidth,
            expectedY: frameY + frameHeight / 2,
            labelX: Number(label?.getAttribute("x")),
          };
        });
      }, sourceNodes);

      for (const anchor of pathAnchors) {
        expect(anchor.pathStart, `missing aggregation path for source node ${anchor.source}`).not.toBeNull();
        expect(anchor.pathStart?.x).toBeCloseTo(anchor.expectedX, 3);
        expect(anchor.pathStart?.y).toBeCloseTo(anchor.expectedY, 3);
        expect(anchor.labelX).toBeCloseTo(anchor.expectedX + 3, 3);
      }
    }
    await expect(page.locator(".bias-frame")).toBeVisible();
    expect(failures).toEqual([]);
  });
}
