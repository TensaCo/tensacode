// Marketing-page behaviour: copy-prompt buttons on image placeholders, and a gentle
// reveal as sections scroll into view. Everything here is progressive enhancement:
// without JavaScript every section is visible and every prompt is selectable text.
(() => {
  // ---- Copy prompt ------------------------------------------------------------
  async function copyText(text) {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch { /* fall through to the legacy path */ }
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    area.style.pointerEvents = 'none';
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    area.remove();
    return ok;
  }

  function selectContents(el) {
    const range = document.createRange();
    range.selectNodeContents(el);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  for (const button of document.querySelectorAll('[data-copy-prompt]')) {
    const target = document.getElementById(button.getAttribute('data-copy-prompt'));
    if (!target) continue;
    const label = button.querySelector('span') || button;
    const idle = label.textContent;
    let timer;
    button.addEventListener('click', async () => {
      const ok = await copyText(target.textContent.trim());
      if (!ok) selectContents(target);
      label.textContent = ok ? 'Copied' : 'Selected: press Ctrl+C';
      button.classList.toggle('is-done', ok);
      clearTimeout(timer);
      timer = setTimeout(() => { label.textContent = idle; button.classList.remove('is-done'); }, 2200);
    });
  }

  // ---- Header hairline once the page has scrolled --------------------------------
  const top = document.querySelector('.site-top');
  if (top) {
    const onScroll = () => {
      const scrolled = window.scrollY > 8;
      top.toggleAttribute('data-scrolled', scrolled);
      top.toggleAttribute('data-top', !scrolled);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  // ---- Reveal on scroll -------------------------------------------------------
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const items = document.querySelectorAll('[data-reveal]');
  if (reduce || !('IntersectionObserver' in window) || !items.length) return;
  document.documentElement.classList.add('can-reveal');
  const io = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-in');
        io.unobserve(entry.target);
      }
    }
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
  for (const el of items) io.observe(el);
})();
