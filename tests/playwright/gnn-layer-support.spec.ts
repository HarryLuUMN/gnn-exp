import { expect, test } from "@playwright/test";

const cases = [
  { model: "gcn_logits", aggregation: "GCN norm", expectsAttention: false },
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
    if (model === "graphsage") {
      await expect(page.locator(".sampling-icon")).toHaveCount(1);
      await expect(page.locator(".sampled-out-link")).toHaveCount(1);
      await expect(page.locator(".sampled-out-link").first()).toHaveAttribute("stroke-dasharray", "3,2");
      const samplingIconLayout = await page.evaluate(() => {
        const toBox = (element: Element) => {
          const rect = element.getBoundingClientRect();
          return {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            right: rect.right,
            bottom: rect.bottom,
          };
        };
        const intersects = (
          a: ReturnType<typeof toBox>,
          b: ReturnType<typeof toBox>
        ) =>
          a.x < b.right &&
          a.right > b.x &&
          a.y < b.bottom &&
          a.bottom > b.y;
        const icon = document.querySelector(".sampling-icon");
        const iconBox = icon ? toBox(icon) : null;
        const overlappingFrames = iconBox
          ? Array.from(document.querySelectorAll(".feature-layer-frame"))
            .map((frame) => ({ id: frame.id, box: toBox(frame) }))
            .filter(({ box }) => intersects(iconBox, box))
            .map(({ id }) => id)
          : [];
        return { iconBox, overlappingFrames };
      });
      expect(samplingIconLayout.iconBox).not.toBeNull();
      expect(samplingIconLayout.overlappingFrames).toEqual([]);
    } else {
      await expect(page.locator(".sampling-icon")).toHaveCount(0);
      await expect(page.locator(".sampled-out-link")).toHaveCount(0);
    }
    await expect(page.locator(".bias-frame")).toBeVisible();
    expect(failures).toEqual([]);
  });
}

test("graph pooling readout fades and shifts during layer expansion", async ({ page }) => {
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      failures.push(message.text());
    }
  });

  await page.goto(`/tests/playwright/fixtures/gnn-layer-harness.html?model=graph_gat`);
  await page.waitForFunction(() => window.__GNN_LAYER_READY === true);

  await expect(page.locator("#matrix-svg")).toBeVisible();
  await expect(page.locator("#agg-feature-layer-node-graph")).toBeVisible();
  await expect(page.locator("#graph-aggregation-label")).toBeVisible();

  const before = await page.locator("#agg-feature-layer-node-graph").evaluate((element) => ({
    opacity: getComputedStyle(element).opacity,
    transform: element.getAttribute("transform"),
  }));

  await page.locator("#feature-layer-1-node-1-dim-0").click();

  await expect(page.locator(".weight-matrix-frame")).toBeVisible();
  await expect.poll(
    () => page.locator("#agg-feature-layer-node-graph").evaluate((element) => element.getAttribute("transform"))
  ).not.toBe(before.transform);

  const after = await page.locator("#agg-feature-layer-node-graph").evaluate((element) => ({
    opacity: getComputedStyle(element).opacity,
    transform: element.getAttribute("transform"),
  }));
  const aggLinks = page.locator(".agg-link-path-fc");
  expect(await aggLinks.count()).toBeGreaterThan(0);
  const aggLinkOpacity = await aggLinks.first().evaluate((element) => getComputedStyle(element).opacity);

  expect(after.opacity).toBe("0.1");
  expect(after.transform).toMatch(/^translate\(/);
  expect(aggLinkOpacity).toBe("0");
  expect(failures).toEqual([]);
});
