import * as d3 from "d3";

export function injectSVG(g:any, x: number, y: number, SVGPath:string, svgClass: string){
    // Check if g is valid
    if (!g || !g.node()) {
        console.error("Invalid d3 selection passed to injectSVG");
        return;
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
        // Use fetch instead of d3.xml to better handle errors
        return fetch(path)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                // Check if response is actually SVG, not HTML error page
                const contentType = response.headers.get("content-type");
                if (contentType && !contentType.includes("svg") && !contentType.includes("xml")) {
                    throw new Error(`Unexpected content type: ${contentType}`);
                }
                return response.text();
            })
            .then(svgText => {
                // Check if we got HTML error page instead of SVG
                if (svgText.includes("<html") || svgText.includes("parsererror") || svgText.includes("This page contains the following errors")) {
                    throw new Error("Received HTML error page instead of SVG");
                }
                
                // Parse SVG text into DOM element
                const parser = new DOMParser();
                const svgDoc = parser.parseFromString(svgText, "image/svg+xml");
                
                // Check for parsing errors
                const parserError = svgDoc.querySelector("parsererror");
                if (parserError) {
                    throw new Error("SVG parsing error: " + parserError.textContent);
                }
                
                const svgElement = svgDoc.documentElement;
                if (!svgElement || svgElement.nodeName !== "svg") {
                    throw new Error("Failed to parse SVG element");
                }
                
                // Clone the SVG element
                const clonedSvg = svgElement.cloneNode(true) as SVGElement;
                
                // Create a group element to contain the SVG and position it
                // Scale by 2x and translate to position
                const group = d3.select(g.node()).append("g")
                    .attr("class", svgClass)
                    .attr("transform", `translate(${x}, ${y}) scale(2)`);
                
                const groupNode = group.node();
                if (!groupNode) {
                    throw new Error("Failed to create group element");
                }
                
                // Append the cloned SVG to the group
                const svgNode = groupNode.appendChild(clonedSvg);
                
                // Clean up attributes that might interfere with positioning
                const svgSelection = d3.select(svgNode);
                svgSelection
                    .attr("x", null)
                    .attr("y", null)
                    .style("display", "block")
                    .style("overflow", "visible");
                
                console.log("SVG successfully injected:", svgClass, "from path:", path);
                return svgNode;
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
        if (error instanceof Error) {
            console.error("Error message:", error.message);
        }
    });
}













