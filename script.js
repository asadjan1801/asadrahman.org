// ---------- Theme toggle (light / dark) ----------
(function () {
  const root = document.documentElement;
  const btn = document.getElementById('theme-toggle');
  const stored = localStorage.getItem('theme');
  if (stored === 'dark' || stored === 'light') {
    root.setAttribute('data-theme', stored);
  }
  btn?.addEventListener('click', () => {
    const current = root.getAttribute('data-theme')
      || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  });
})();

// ---------- Current year ----------
document.getElementById('year').textContent = new Date().getFullYear();

// ---------- TOC scroll-spy ----------
(function () {
  const links = document.querySelectorAll('.toc a[href^="#"]');
  if (!links.length) return;
  const map = new Map();
  links.forEach((a) => {
    const id = a.getAttribute('href').slice(1);
    const section = document.getElementById(id);
    if (section) map.set(section, a);
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const link = map.get(entry.target);
        if (!link) return;
        if (entry.isIntersecting) {
          links.forEach((l) => l.classList.remove('active'));
          link.classList.add('active');
        }
      });
    },
    { rootMargin: '-40% 0px -55% 0px', threshold: 0 }
  );
  map.forEach((_, section) => observer.observe(section));
})();
