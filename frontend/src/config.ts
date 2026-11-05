const DEFAULT_MAX_UPLOAD_MB = 50;
const BYTES_PER_MB = 1024 * 1024;

/**
 * Reads the upload limit in MB from a build setting.
 *
 * The backend enforces its own limit, so this check only spares the user a
 * doomed upload. An unset or invalid value falls back to the default instead
 * of breaking the page.
 */
export function parseMaxUploadMb(raw: string | undefined): number {
  if (raw === undefined || !/^[1-9]\d*$/.test(raw)) return DEFAULT_MAX_UPLOAD_MB;
  const mb = Number(raw);
  return Number.isSafeInteger(mb * BYTES_PER_MB) ? mb : DEFAULT_MAX_UPLOAD_MB;
}

export const MAX_UPLOAD_MB = parseMaxUploadMb(import.meta.env.VITE_MAX_UPLOAD_MB);
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * BYTES_PER_MB;
