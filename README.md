# asadrahman.org

Personal website of Asad Ur Rahman — Structural Engineering PhD candidate at the University of Houston.

Static HTML / CSS / JS. No build step. Designed to deploy on Cloudflare Pages.

## Local preview

```sh
python3 -m http.server 8000
# then open http://localhost:8000
```

## Deploy (Cloudflare Pages)

1. Push this repo to GitHub.
2. In Cloudflare dashboard: **Workers & Pages → Create → Pages → Connect to Git**.
3. Pick this repo. Framework preset: **None**. Build command: *(leave empty)*. Output directory: `/`.
4. After first deploy, add the custom domain `asadrahman.org` under **Custom domains**.

## Files

- `index.html` — single-page site
- `styles.css` — typography, layout, light/dark theme
- `script.js` — theme toggle, TOC scroll‑spy, year stamp
- `Resume_Asad ur rahman.pdf` — linked from the Contact section
