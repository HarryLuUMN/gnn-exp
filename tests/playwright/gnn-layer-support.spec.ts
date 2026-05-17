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
    }
    await expect(page.locator(".bias-frame")).toBeVisible();
    expect(failures).toEqual([]);
  });
}
