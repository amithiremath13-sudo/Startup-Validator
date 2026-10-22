/**
 * api.js
 * -------
 * Thin wrapper around POST /api/validate.
 *
 * VITE_API_BASE_URL is set at build time from:
 *   .env.development  → http://localhost:8000
 *   .env.production   → https://YOUR_EC2_URL   (must be set before npm run build)
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

/**
 * Call the validation endpoint.
 *
 * @param {string} idea    - startup idea description
 * @param {string} country - country name or ISO-3 code
 * @returns {Promise<object>} parsed API response
 * @throws {Error} with a user-friendly message on failure
 */
export async function validateIdea(idea, country) {
  const url = `${BASE_URL}/api/validate`

  let response
  try {
    response = await fetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ idea: idea.trim(), country: country.trim() }),
    })
  } catch (networkError) {
    throw new Error(
      'Could not reach the server. ' +
      'Make sure the backend is running and VITE_API_BASE_URL is correct.'
    )
  }

  if (!response.ok) {
    // Try to extract a detail message from the API error body
    let detail = `Server error (${response.status})`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch { /* ignore parse errors */ }
    throw new Error(detail)
  }

  return response.json()
}
