// The robots reel plays only while it is on screen, and never under reduced motion.
import { whenVisible } from './code.js';

export function init(video) {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
    video.controls = true;
    return;
  }
  whenVisible(video, (visible) => {
    if (visible) video.play().catch(() => { video.controls = true; });
    else video.pause();
  }, { once: false, margin: '0px' });
}
