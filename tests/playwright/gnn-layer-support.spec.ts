import { expect, test } from "@playwright/test";

const cases = [
  { model: "gat", aggregation: "sum" },
  { model: "graphsage", aggregation: "mean" },
  { model: "gin", aggregation: "sum" },
] as const;

for (const { model, aggregation } of cases) {
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

    await page.locator("#feature-layer-1-node-1-dim-0").click();

    await expect(page.getByText(`agg: ${aggregation}`)).toBeVisible();
    await expect(page.locator(".weight-matrix-frame")).toBeVisible();
    await expect(page.locator(".weight-matrix-cell").first()).toHaveCSS("stroke", "rgb(216, 222, 219)");
    await expect(page.locator(".weight-matrix-cell").first()).toHaveCSS("stroke-width", "0.35px");
    await expect(page.locator(".bias-frame")).toBeVisible();
    expect(failures).toEqual([]);
  });
}
