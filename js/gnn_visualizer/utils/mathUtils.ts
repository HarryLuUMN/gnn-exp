export function divideVector(v: number[], n: number): number[] {
    if (n === 0) {
        throw new Error("Division by zero");
    }
    return v.map(x => x / n);
}

export function addVector(a: number[], b: number[]): number[] {
    if (a.length !== b.length) {
        throw new Error("Vector length mismatch");
    }
    return a.map((v, i) => v + b[i]);
}

export function scaleVector(k: number, v: number[]): number[] {
    return v.map(x => k * x);
}

export const countOnes = (arr: number[]) =>
    arr.reduce((sum, x) => sum + (x === 1 ? 1 : 0), 0);

export function vecMatMul(v: number[], W: number[][]): number[] {
    const n = v.length;
    const m = W[0].length;

    if (W.length !== n) {
        throw new Error("Dimension mismatch");
    }

    const result = Array(m).fill(0);

    for (let j = 0; j < m; j++) {
        for (let i = 0; i < n; i++) {
            result[j] += v[i] * W[i][j];
        }
    }

    return result;
}

export function randomMatrix(m: number, n: number): number[][] {
    return Array.from({ length: m }, () =>
        Array.from({ length: n }, () => Math.random())
    );
}

export function randomVector(n: number): number[] {
    return Array.from({ length: n }, () => Math.random());
}

export function matrixTranspose<T>(matrix: T[][]): T[][] {
  return matrix[0].map((_, colIndex) =>
    matrix.map(row => row[colIndex])
  )
}

