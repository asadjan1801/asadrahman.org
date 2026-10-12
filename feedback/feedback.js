(function () {
  const form = document.getElementById('feedback-form');
  if (!form) return;

  const submitButton = document.getElementById('feedback-submit');
  const messageField = document.getElementById('feedback-message');
  const count = document.getElementById('feedback-count');
  const status = document.getElementById('feedback-status');
  const verificationHelp = document.getElementById('verification-help');
  const turnstileContainer = document.getElementById('feedback-turnstile');

  let turnstileToken = '';
  let turnstileWidgetId = null;
  let isSubmitting = false;

  function updateCount() {
    count.textContent = `${messageField.value.length.toLocaleString()} / 3,000`;
  }

  function showStatus(type, message, focus) {
    status.className = `form-status is-visible form-status-${type}`;
    status.textContent = message;
    if (focus) status.focus();
  }

  function clearStatus() {
    status.className = 'form-status';
    status.textContent = '';
  }

  function updateSubmitState() {
    submitButton.disabled = isSubmitting || !turnstileToken;
  }

  function loadTurnstile() {
    return new Promise(function (resolve, reject) {
      if (window.turnstile) {
        resolve(window.turnstile);
        return;
      }

      const callbackName = `turnstileLoaded${Date.now()}`;
      const script = document.createElement('script');
      window[callbackName] = function () {
        delete window[callbackName];
        resolve(window.turnstile);
      };
      script.src = `https://challenges.cloudflare.com/turnstile/v0/api.js?onload=${callbackName}&render=explicit`;
      script.async = true;
      script.defer = true;
      script.onerror = function () {
        delete window[callbackName];
        reject(new Error('Turnstile could not be loaded.'));
      };
      document.head.appendChild(script);
    });
  }

  async function initializeVerification() {
    try {
      const configResponse = await fetch('/api/feedback/config', {
        headers: { Accept: 'application/json' },
        cache: 'no-store'
      });
      if (!configResponse.ok) throw new Error('Feedback service is not configured.');

      const config = await configResponse.json();
      if (!config.siteKey) throw new Error('Feedback service is not configured.');

      const turnstile = await loadTurnstile();
      turnstileWidgetId = turnstile.render(turnstileContainer, {
        sitekey: config.siteKey,
        action: 'feedback_submit',
        theme: 'auto',
        callback: function (token) {
          turnstileToken = token;
          verificationHelp.textContent = 'Verification complete. You can submit your feedback.';
          updateSubmitState();
        },
        'expired-callback': function () {
          turnstileToken = '';
          verificationHelp.textContent = 'Verification expired. Please complete it again.';
          updateSubmitState();
        },
        'error-callback': function () {
          turnstileToken = '';
          verificationHelp.textContent = 'Verification could not be completed. Please refresh the page and try again.';
          updateSubmitState();
        }
      });
    } catch (error) {
      verificationHelp.textContent = 'Anonymous feedback is temporarily unavailable while the service is being configured.';
      showStatus('error', 'The feedback service is not ready yet. Please try again later.');
    }
  }

  messageField.addEventListener('input', updateCount);
  updateCount();

  form.addEventListener('submit', async function (event) {
    event.preventDefault();
    clearStatus();

    if (!form.reportValidity()) return;
    if (!turnstileToken) {
      showStatus('error', 'Please complete the anti-spam verification before submitting.', true);
      return;
    }

    isSubmitting = true;
    submitButton.textContent = 'Sending…';
    updateSubmitState();

    const formData = new FormData(form);
    const payload = {
      category: formData.get('category'),
      course: formData.get('course'),
      message: formData.get('message'),
      companyWebsite: formData.get('companyWebsite'),
      turnstileToken: turnstileToken
    };

    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      const result = await response.json().catch(function () { return {}; });

      if (!response.ok || !result.ok) {
        throw new Error(result.message || 'Your feedback could not be sent. Please try again.');
      }

      form.reset();
      updateCount();
      showStatus('success', 'Thank you. Your feedback was emailed anonymously. Because no contact information was collected, I will not be able to reply directly.', true);
    } catch (error) {
      showStatus('error', error.message || 'Your feedback could not be sent. Please try again later.', true);
    } finally {
      isSubmitting = false;
      submitButton.textContent = 'Submit anonymously';
      turnstileToken = '';
      if (window.turnstile && turnstileWidgetId !== null) window.turnstile.reset(turnstileWidgetId);
      updateSubmitState();
    }
  });

  initializeVerification();
})();
