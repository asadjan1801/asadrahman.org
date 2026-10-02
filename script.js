(function () {
  const root = document.documentElement;
  const themeButton = document.getElementById('theme-toggle');
  const storedTheme = localStorage.getItem('theme');
  if (storedTheme === 'dark' || storedTheme === 'light') root.setAttribute('data-theme', storedTheme);

  function currentTheme() {
    return root.getAttribute('data-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  function updateThemeLabel() {
    if (!themeButton) return;
    const nextTheme = currentTheme() === 'dark' ? 'light' : 'dark';
    themeButton.setAttribute('aria-label', `Use ${nextTheme} theme`);
    themeButton.title = `Use ${nextTheme} theme`;
  }
  themeButton?.addEventListener('click', function () {
    const nextTheme = currentTheme() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', nextTheme);
    localStorage.setItem('theme', nextTheme);
    updateThemeLabel();
  });
  updateThemeLabel();

  const menuButton = document.getElementById('nav-toggle');
  const navPanel = document.getElementById('nav-panel');
  function closeMenu() {
    if (!menuButton || !navPanel) return;
    navPanel.hidden = true;
    menuButton.setAttribute('aria-expanded', 'false');
  }
  menuButton?.addEventListener('click', function () {
    const willOpen = navPanel.hidden;
    navPanel.hidden = !willOpen;
    menuButton.setAttribute('aria-expanded', String(willOpen));
  });
  navPanel?.querySelectorAll('a').forEach(function (link) {
    link.addEventListener('click', function () {
      if (window.matchMedia('(max-width: 880px)').matches) closeMenu();
    });
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && navPanel && !navPanel.hidden) {
      closeMenu();
      menuButton?.focus();
    }
  });
  window.matchMedia('(min-width: 881px)').addEventListener('change', function (event) {
    if (event.matches && navPanel) {
      navPanel.hidden = false;
      menuButton?.setAttribute('aria-expanded', 'false');
    } else closeMenu();
  });
  if (window.matchMedia('(max-width: 880px)').matches) closeMenu();
  document.querySelectorAll('[data-year]').forEach(function (element) {
    element.textContent = new Date().getFullYear();
  });
})();
