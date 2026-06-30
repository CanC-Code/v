// model-cache.js
// Downloads ONNX model files and persists them in IndexedDB so the app
// works fully offline after the first successful download.

const DB_NAME = "motionforge-models";
const DB_VERSION = 1;
const STORE_NAME = "models";

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet(key) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).get(key);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

async function idbSet(key, value) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function idbDelete(key) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function clearAllModels() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/**
 * Fetch a model from `url`, caching the raw bytes in IndexedDB keyed by URL.
 * Returns an ArrayBuffer. Calls onProgress(loadedBytes, totalBytes|null) while downloading.
 */
export async function fetchModelBytes(url, onProgress) {
  const cached = await idbGet(url);
  if (cached) {
    onProgress && onProgress(cached.byteLength, cached.byteLength);
    return cached;
  }

  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to download model: ${resp.status} ${resp.statusText} (${url})`);
  }

  const total = Number(resp.headers.get("content-length")) || null;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress && onProgress(received, total);
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  await idbSet(url, buffer.buffer);
  return buffer.buffer;
}

export async function deleteCachedModel(url) {
  await idbDelete(url);
}
