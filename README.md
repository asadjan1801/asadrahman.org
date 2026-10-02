# asadrahman.org

Academic website of Asad Ur Rahman, PhD — Assistant Professor at the NUST Institute of Civil Engineering, School of Civil and Environmental Engineering.

The site is built with static HTML, CSS, and JavaScript. It has no build step and is designed for deployment on Cloudflare Pages.

## Local preview

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Site structure

- `/` — academic homepage
- `/research/` — research areas and featured projects
- `/publications/` — full publication record
- `/teaching/` — teaching overview
- `/teaching/structural-dynamics/` — CE-809 course page
- `/teaching/structural-dynamics/interactive-lab/` — Structural Dynamics Virtual Lab
- `/teaching/ai-built-environment/` — CE-312 course page
- `/interactive-labs/` — interactive learning tools
- `/cv/` — web CV and PDF download
- `/contact/` — contact information and academic profiles

Shared presentation and behavior are in `styles.css` and `script.js`.

## Private course material

Files under `Structural_dynamics_Interactive_Labs/lectures/` are private instructor materials. They are ignored by Git and must not be linked, committed, or deployed. The public interactive lab is self-contained and does not require those files.

## Deploy with Cloudflare Pages

1. Push the repository to GitHub.
2. In Cloudflare, open **Workers & Pages → Create → Pages → Connect to Git**.
3. Select this repository.
4. Use framework preset **None**, leave the build command empty, and use `/` as the output directory.
5. Add `asadrahman.org` under **Custom domains**.

Before deploying, verify that the private `Structural_dynamics_Interactive_Labs/lectures/` directory is not included in the commit.
