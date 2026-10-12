const FEEDBACK_RECIPIENT = 'arahman@nice.nust.edu.pk';
const FEEDBACK_SENDER = 'feedback@asadrahman.org';
const MAX_REQUEST_BYTES = 12_000;
const MIN_MESSAGE_LENGTH = 20;
const MAX_MESSAGE_LENGTH = 3_000;

const CATEGORY_LABELS = Object.freeze({
  teaching: 'Teaching approach',
  'course-content': 'Course content',
  'pace-workload': 'Pace or workload',
  assessment: 'Assessment or grading',
  'labs-resources': 'Interactive labs or resources',
  supervision: 'Supervision or mentoring',
  other: 'Something else'
});

const COURSE_LABELS = Object.freeze({
  'prefer-not-to-say': 'Prefer not to say',
  'ce-809': 'CE-809 · Structural Dynamics',
  'ce-312': 'CE-312 · AI in the Built Environment',
  'research-supervision': 'Research supervision',
  'website-resources': 'Website or learning resources',
  other: 'Other'
});

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff'
    }
  });
}

function normalizeMessage(value) {
  return value.replace(/\r\n?/g, '\n').trim();
}

export function validateFeedbackPayload(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return { ok: false };
  }

  const category = typeof payload.category === 'string' ? payload.category : '';
  const course = typeof payload.course === 'string' ? payload.course : '';
  const message = typeof payload.message === 'string' ? normalizeMessage(payload.message) : '';
  const turnstileToken = typeof payload.turnstileToken === 'string' ? payload.turnstileToken : '';
  const companyWebsite = typeof payload.companyWebsite === 'string' ? payload.companyWebsite.trim() : '';

  if (companyWebsite) return { ok: true, isSpam: true };
  if (!Object.hasOwn(CATEGORY_LABELS, category) || !Object.hasOwn(COURSE_LABELS, course)) {
    return { ok: false };
  }
  if (message.length < MIN_MESSAGE_LENGTH || message.length > MAX_MESSAGE_LENGTH) {
    return { ok: false };
  }
  if (/\p{Cc}/u.test(message.replace(/\n|\t/g, ''))) {
    return { ok: false };
  }
  if (!turnstileToken || turnstileToken.length > 2_048) {
    return { ok: false };
  }

  return { ok: true, category, course, message, turnstileToken, isSpam: false };
}

export function buildFeedbackEmail(feedback) {
  const categoryLabel = CATEGORY_LABELS[feedback.category];
  const courseLabel = COURSE_LABELS[feedback.course];
  const subjectCourse = feedback.course === 'prefer-not-to-say' ? 'General' : courseLabel;

  return {
    to: FEEDBACK_RECIPIENT,
    from: { email: FEEDBACK_SENDER, name: 'Anonymous Feedback' },
    subject: `[Anonymous feedback] ${subjectCourse} · ${categoryLabel}`,
    text: [
      'Anonymous student feedback',
      '',
      `Feedback about: ${categoryLabel}`,
      `Course or area: ${courseLabel}`,
      '',
      'Message:',
      feedback.message,
      '',
      '---',
      'Submitted through asadrahman.org/feedback/.',
      'The form did not request a name, email address, student ID, login, or reply address.',
      'Connection information handled by Cloudflare is not included in this email.'
    ].join('\n')
  };
}

async function verifyTurnstile(token, secret) {
  if (!secret) return false;

  const body = new FormData();
  body.set('secret', secret);
  body.set('response', token);

  const response = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', {
    method: 'POST',
    body
  });
  if (!response.ok) return false;

  const result = await response.json();
  return result.success === true && result.action === 'feedback_submit';
}

function hasSameOrigin(request) {
  const origin = request.headers.get('Origin');
  return origin === new URL(request.url).origin;
}

async function handleFeedbackSubmission(request, env) {
  if (!hasSameOrigin(request)) {
    return jsonResponse({ ok: false, message: 'This submission could not be verified.' }, 403);
  }

  if (!request.headers.get('Content-Type')?.toLowerCase().startsWith('application/json')) {
    return jsonResponse({ ok: false, message: 'Please submit the feedback form from the website.' }, 415);
  }

  const declaredLength = Number(request.headers.get('Content-Length') || 0);
  if (declaredLength > MAX_REQUEST_BYTES) {
    return jsonResponse({ ok: false, message: 'The feedback is too long.' }, 413);
  }

  let rawBody;
  try {
    rawBody = await request.text();
  } catch {
    return jsonResponse({ ok: false, message: 'The feedback could not be read.' }, 400);
  }
  if (new TextEncoder().encode(rawBody).byteLength > MAX_REQUEST_BYTES) {
    return jsonResponse({ ok: false, message: 'The feedback is too long.' }, 413);
  }

  let payload;
  try {
    payload = JSON.parse(rawBody);
  } catch {
    return jsonResponse({ ok: false, message: 'Please check the form and try again.' }, 400);
  }

  const validation = validateFeedbackPayload(payload);
  if (!validation.ok) {
    return jsonResponse({ ok: false, message: 'Please check the form and try again.' }, 400);
  }
  if (validation.isSpam) {
    return jsonResponse({ ok: true });
  }

  let turnstileIsValid = false;
  try {
    turnstileIsValid = await verifyTurnstile(validation.turnstileToken, env.TURNSTILE_SECRET_KEY);
  } catch {
    return jsonResponse({ ok: false, message: 'Verification is temporarily unavailable. Please try again.' }, 503);
  }
  if (!turnstileIsValid) {
    return jsonResponse({ ok: false, message: 'Verification failed. Please refresh and try again.' }, 403);
  }

  if (!env.FEEDBACK_EMAIL || typeof env.FEEDBACK_EMAIL.send !== 'function') {
    return jsonResponse({ ok: false, message: 'The feedback service is temporarily unavailable.' }, 503);
  }

  try {
    await env.FEEDBACK_EMAIL.send(buildFeedbackEmail(validation));
  } catch {
    return jsonResponse({ ok: false, message: 'Your feedback could not be sent right now. Please try again later.' }, 503);
  }

  return jsonResponse({ ok: true });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/api/feedback/config' && request.method === 'GET') {
      return jsonResponse({ siteKey: env.TURNSTILE_SITE_KEY || null });
    }

    if (url.pathname === '/api/feedback' && request.method === 'POST') {
      return handleFeedbackSubmission(request, env);
    }

    if (url.pathname.startsWith('/api/feedback')) {
      return jsonResponse({ ok: false, message: 'Not found.' }, 404);
    }

    return new Response('Not found', {
      status: 404,
      headers: { 'Content-Type': 'text/plain; charset=utf-8' }
    });
  }
};
