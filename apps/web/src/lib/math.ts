export function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

export function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function ema(prev: number, next: number, alpha: number) {
  return prev + alpha * (next - prev);
}
