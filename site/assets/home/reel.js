// The robots reel plays only while it is on screen. Under reduced motion it stays on its poster.
import { whenVisible } from './code.js';

export function init(video) {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  whenVisible(video, (visible) => {
    if (visible) video.play().catch(() => {});
    else video.pause();
  }, { once: false, margin: '0px' });
}
