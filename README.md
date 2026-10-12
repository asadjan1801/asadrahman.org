# asadrahman.org

Academic website of Asad Ur Rahman, PhD — Assistant Professor at the NUST Institute of Civil Engineering, School of Civil and Environmental Engineering.

The site is built with static HTML, CSS, and JavaScript. It has no build step and is deployed with Cloudflare Workers static assets.

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
- `/feedback/` — anonymous student feedback form
- `/contact/` — contact information and academic profiles

Shared presentation and behavior are in `styles.css` and `script.js`.

## Private course material

Files under `Structural_dynamics_Interactive_Labs/lectures/` and `AI_in_built_env_interactive_lab/` are private instructor materials. They are ignored by Git and must not be linked, committed, or deployed. The public interactive labs are self-contained and do not require those files.

## Deploy with Cloudflare Workers

1. Push the repository to GitHub.
2. In Cloudflare, connect this GitHub repository to the `asadrahman-org` Worker.
3. Use `main` as the production branch and enable preview builds for feature branches.
4. Leave the build command empty. Use `npx wrangler deploy` as the production deploy command and `npx wrangler versions upload` as the preview command for feature branches.
5. The checked-in `wrangler.jsonc` publishes the repository's public static files; `.assetsignore` prevents repository-only and private-source files from being uploaded.
6. Add `asadrahman.org` under **Settings → Domains & Routes**.

Before deploying, verify that the private `Structural_dynamics_Interactive_Labs/lectures/` directory is not included in the commit.

## Anonymous feedback setup

The `/feedback/` page submits to the Worker endpoint at `/api/feedback`. The Worker validates the form with Cloudflare Turnstile and sends a plain-text email to `arahman@nice.nust.edu.pk`. It does not write submissions to a database or include request IP addresses, browser details, or reply addresses in the email.

Complete these one-time Cloudflare account steps before deploying the feature:

1. Under **Compute → Email Service**, onboard `asadrahman.org` for email sending.
2. Add `arahman@nice.nust.edu.pk` as a destination address and follow the verification link sent to that inbox.
3. Create a managed Turnstile widget for `asadrahman.org` (and `www.asadrahman.org` if that hostname is used).
4. Add the widget values to the Worker as runtime secrets so they never enter the repository:

   ```sh
   npx wrangler secret put TURNSTILE_SITE_KEY
   npx wrangler secret put TURNSTILE_SECRET_KEY
   ```

The site key is returned to the feedback page by `/api/feedback/config`; the secret key remains server-only. Until both values are configured, the submit button stays disabled and the page reports that the service is being configured.

For full local Worker testing, use `npx wrangler dev` rather than the basic Python static server. Use Cloudflare's published Turnstile test keys only in an ignored `.dev.vars` file, never in production.

Run the feedback endpoint unit tests with:

```sh
node --test tests/feedback-worker.test.mjs
```
