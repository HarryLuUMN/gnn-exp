import * as d3 from "d3";

const INLINE_ICONS: Record<string, string> = {
    matmul: `<svg viewBox="0 0 10 10" width="10" height="10" xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="5" r="4.5" fill="#fff" stroke="#000" stroke-width=".35"/><g fill="#fff" stroke="#000" stroke-width=".18"><rect x="1.7" y="2.2" width="1" height="1"/><rect x="1.7" y="3.4" width="1" height="1"/><rect x="1.7" y="4.6" width="1" height="1"/><rect x="2.9" y="2.2" width="1" height="1"/><rect x="2.9" y="3.4" width="1" height="1"/><rect x="2.9" y="4.6" width="1" height="1"/></g><text x="6" y="6.2" font-size="2.6" font-family="serif" text-anchor="middle" fill="#000">×</text></svg>`,
    relu: `<svg viewBox="0 0 10 10" width="10" height="10" xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="5" r="4.5" fill="#fff" stroke="#000" stroke-width=".35"/><path d="M1.5 7h7" stroke="#000" stroke-width=".18" opacity=".25"/><path d="M5 1.4v7" stroke="#000" stroke-width=".18" opacity=".35"/><path d="M1.7 6.8h3.1L8.1 3" fill="none" stroke="#c6501d" stroke-width=".45" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    sampling: `<svg viewBox="0 0 10 10" width="10" height="10" xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="5" r="4.35" fill="#fff" stroke="#000" stroke-width=".35"/><text x="5" y="6.6" font-size="5.6" font-family="sans-serif" text-anchor="middle" fill="#000" stroke="#000" stroke-width=".18">X</text></svg>`,
};
const ICON_SCALE = 2;
const ICON_VIEWBOX_SIZE = 10;

function fallbackIconForPath(path: string) {
    const normalized = path.toLowerCase();
    if (normalized.includes("matmul")) return INLINE_ICONS.matmul;
    if (normalized.includes("relu") || normalized.includes("activation")) return INLINE_ICONS.relu;
    if (normalized.includes("sampling")) return INLINE_ICONS.sampling;
    return null;
}

function appendSVGText(g: any, x: number, y: number, svgText: string, svgClass: string, iconScale: number) {
    const parser = new DOMParser();
    const svgDoc = parser.parseFromString(svgText, "image/svg+xml");
    const parserError = svgDoc.querySelector("parsererror");
    if (parserError) {
        throw new Error("SVG parsing error: " + parserError.textContent);
    }

    const svgElement = svgDoc.documentElement;
    if (!svgElement || svgElement.nodeName !== "svg") {
        throw new Error("Failed to parse SVG element");
    }

    const offset = (ICON_VIEWBOX_SIZE * iconScale) / 2;
    const group = d3.select(g.node()).append("g")
        .attr("class", svgClass)
        .attr("transform", `translate(${x - offset}, ${y - offset}) scale(${iconScale})`);

    const groupNode = group.node();
    if (!groupNode) {
        throw new Error("Failed to create group element");
    }

    const svgNode = groupNode.appendChild(svgElement.cloneNode(true) as SVGElement);
    d3.select(svgNode)
        .attr("x", null)
        .attr("y", null)
        .style("display", "block")
        .style("overflow", "visible");
    return svgNode;
}

export function injectSVG(g:any, x: number, y: number, SVGPath:string, svgClass: string, iconScale: number = ICON_SCALE){
    if (!g || !g.node()) {
        console.error("Invalid d3 selection passed to injectSVG");
        return;
    }

    const inlineIcon = fallbackIconForPath(SVGPath);
    if (inlineIcon) {
        return Promise.resolve(appendSVGText(g, x, y, inlineIcon, svgClass, iconScale));
    }
    
    // Try multiple path resolution strategies
    const tryPaths = [
        SVGPath, // Original path
        SVGPath.startsWith("./") ? SVGPath.substring(2) : SVGPath, // Remove leading ./
        `/gnn_visualizer/${SVGPath.startsWith("./") ? SVGPath.substring(2) : SVGPath}`, // Absolute from root
        `./gnn_visualizer/${SVGPath.startsWith("./") ? SVGPath.substring(2) : SVGPath}`, // Relative with module name
    ];
    
    let pathIndex = 0;
    
    function tryLoadPath(path: string): Promise<any> {
        return fetch(path)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const contentType = response.headers.get("content-type");
                if (contentType && !contentType.includes("svg") && !contentType.includes("xml")) {
                    throw new Error(`Unexpected content type: ${contentType}`);
                }
                return response.text();
            })
            .then(svgText => {
                if (svgText.includes("<html") || svgText.includes("parsererror") || svgText.includes("This page contains the following errors")) {
                    throw new Error("Received HTML error page instead of SVG");
                }
                return appendSVGText(g, x, y, svgText, svgClass, iconScale);
            });
    }
    
    function attemptLoad(): Promise<any> {
        if (pathIndex >= tryPaths.length) {
            return Promise.reject(new Error("All path attempts failed"));
        }
        
        const currentPath = tryPaths[pathIndex];
        console.log(`Attempting to load SVG (attempt ${pathIndex + 1}/${tryPaths.length}):`, currentPath);
        
        return tryLoadPath(currentPath).catch(function(error) {
            console.warn(`Failed to load SVG from path ${currentPath}:`, error);
            pathIndex++;
            return attemptLoad();
        });
    }
    
    return attemptLoad().catch(function(error) {
        console.error("Error loading SVG after all attempts:", SVGPath, error);
    });
}








