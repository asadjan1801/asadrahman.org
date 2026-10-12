import test from 'node:test';
import assert from 'node:assert/strict';
import worker, { buildFeedbackEmail, validateFeedbackPayload } from '../worker.mjs';

const validPayload = {
  category: 'pace-workload',
  course: 'ce-809',
  message: 'More worked examples before the assignment would be helpful.',
  companyWebsite: '',
  turnstileToken: 'valid-test-token'
};

test('validates only approved feedback fields', function () {
  const valid = validateFeedbackPayload(validPayload);
  assert.equal(valid.ok, true);
  assert.equal(valid.isSpam, false);
  assert.equal(valid.message, validPayload.message);

  assert.equal(validateFeedbackPayload({ ...validPayload, category: 'injected-subject' }).ok, false);
  assert.equal(validateFeedbackPayload({ ...validPayload, message: 'Too short' }).ok, false);
  assert.equal(validateFeedbackPayload({ ...validPayload, turnstileToken: '' }).ok, false);

  const honeypot = validateFeedbackPayload({ ...validPayload, companyWebsite: 'https://spam.example' });
  assert.equal(honeypot.ok, true);
  assert.equal(honeypot.isSpam, true);
});

test('builds a fixed-recipient plain-text email', function () {
  const validated = validateFeedbackPayload(validPayload);
  const email = buildFeedbackEmail(validated);

  assert.equal(email.to, 'arahman@nice.nust.edu.pk');
  assert.deepEqual(email.from, { email: 'feedback@asadrahman.org', name: 'Anonymous Feedback' });
  assert.match(email.subject, /^\[Anonymous feedback\]/);
  assert.match(email.text, /More worked examples/);
  assert.equal('html' in email, false);
  assert.equal('replyTo' in email, false);
});

test('serves public configuration without exposing the Turnstile secret', async function () {
  const response = await worker.fetch(
    new Request('https://asadrahman.org/api/feedback/config'),
    { TURNSTILE_SITE_KEY: 'public-site-key', TURNSTILE_SECRET_KEY: 'private-secret' }
  );
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.deepEqual(body, { siteKey: 'public-site-key' });
  assert.doesNotMatch(JSON.stringify(body), /private-secret/);
});

test('rejects cross-origin submissions', async function () {
  const response = await worker.fetch(
    new Request('https://asadrahman.org/api/feedback', {
      method: 'POST',
      headers: { Origin: 'https://spam.example', 'Content-Type': 'application/json' },
      body: JSON.stringify(validPayload)
    }),
    {}
  );

  assert.equal(response.status, 403);
});

test('verifies Turnstile and sends no connection details in the email', async function () {
  const originalFetch = globalThis.fetch;
  let sentEmail;
  globalThis.fetch = async function (url) {
    assert.equal(url, 'https://challenges.cloudflare.com/turnstile/v0/siteverify');
    return new Response(JSON.stringify({ success: true, action: 'feedback_submit' }), {
      headers: { 'Content-Type': 'application/json' }
    });
  };

  try {
    const request = new Request('https://asadrahman.org/api/feedback', {
      method: 'POST',
      headers: {
        Origin: 'https://asadrahman.org',
        'Content-Type': 'application/json',
        'CF-Connecting-IP': '203.0.113.42',
        'User-Agent': 'Identifying Browser Test'
      },
      body: JSON.stringify(validPayload)
    });
    const response = await worker.fetch(request, {
      TURNSTILE_SECRET_KEY: 'private-secret',
      FEEDBACK_EMAIL: {
        async send(email) {
          sentEmail = email;
          return { messageId: 'test-message-id' };
        }
      }
    });
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.deepEqual(body, { ok: true });
    assert.equal(sentEmail.to, 'arahman@nice.nust.edu.pk');
    assert.doesNotMatch(sentEmail.text, /203\.0\.113\.42/);
    assert.doesNotMatch(sentEmail.text, /Identifying Browser Test/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
