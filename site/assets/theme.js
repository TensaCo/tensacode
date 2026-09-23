// Light/dark switch shared by every page (marketing and docs).
//
// An inline snippet in each page's <head> has already written `data-theme` on <html>
// from the stored choice or the device setting, before first paint. This file only
// wires the button: it unhides it, remembers a click, and keeps following the device
// for as long as the visitor has not chosen.
(() => {
  const KEY = 'tc-theme';
  const root = document.documentElement;
  const device = window.matchMedia('(prefers-color-scheme: dark)');
  const buttons = document.querySelectorAll('[data-theme-toggle]');

  const stored = () => {
    try {
      const v = localStorage.getItem(KEY);
      return v === 'light' || v === 'dark' ? v : null;
    } catch { return null; }
  };

  const apply = (theme) => {
    root.dataset.theme = theme;
    const to = theme === 'dark' ? 'light' : 'dark';
    for (const b of buttons) {
      b.hidden = false;
      b.setAttribute('aria-label', `Switch to ${to} theme`);
      b.title = `Switch to ${to} theme`;
    }
  };

  apply(stored() ?? (device.matches ? 'dark' : 'light'));

  for (const b of buttons) {
    b.addEventListener('click', () => {
      const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(KEY, next); } catch { /* private window: this visit only */ }
      apply(next);
    });
  }

  device.addEventListener('change', (e) => { if (!stored()) apply(e.matches ? 'dark' : 'light'); });
})();
