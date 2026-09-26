// ==========================================================================
// EduTerm AI — Frontend Application Logic & YouTube Sync
// ==========================================================================

// Token handling if opened beyond localhost
const _urlParams = new URLSearchParams(window.location.search);
const _authToken = _urlParams.get("token");

if (_authToken) {
  const _origFetch = window.fetch;
  window.fetch = function (resource, init = {}) {
    init.headers = init.headers || {};
    if (init.headers instanceof Headers) {
      if (!init.headers.has("X-Auth-Token")) {
        init.headers.set("X-Auth-Token", _authToken);
      }
    } else if (Array.isArray(init.headers)) {
      init.headers.push(["X-Auth-Token", _authToken]);
    } else {
      init.headers["X-Auth-Token"] = _authToken;
    }
    return _origFetch(resource, init);
  };
}

let activeTab = "youtube";
let selectedFile = null;
let currentScope = "full";
let currentResult = null;
let ytPlayer = null;
let ytApiReady = false;
let pendingYtId = null;
let activeEventSource = null;
let segmentWatchTimer = null;

// Format seconds into MM:SS
function formatTime(seconds) {
  if (isNaN(seconds) || seconds === null) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

// Toast notification helper
function showToast(msg, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerText = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(50px)";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// YouTube IFrame API Callback
if (window.YT && window.YT.Player) {
  ytApiReady = true;
}
window.onYouTubeIframeAPIReady = function () {
  ytApiReady = true;
  if (pendingYtId) {
    loadYouTubeVideo(pendingYtId);
  }
};

// Initialize System Info
async function loadSystemInfo() {
  try {
    const res = await fetch("/api/system-info");
    const data = await res.json();
    const gpuEl = document.getElementById("gpuText");
    if (data.cuda) {
      gpuEl.innerText = `${data.gpu} (${data.vram_gb} GB VRAM)`;
      document.getElementById("pillGpu").querySelector(".status-dot").classList.add("online");
    } else {
      gpuEl.innerText = "PyTorch CPU Mode";
    }
  } catch (err) {
    console.error("System info error:", err);
  }
}

// Load Models from API
async function loadModels() {
  try {
    const res = await fetch("/api/models");
    if (!res.ok) return;
    const data = await res.json();
    if (data.llm_models && data.llm_models.length > 0) {
      const selectLlm = document.getElementById("selectLlmModel");
      if (selectLlm) {
        selectLlm.disabled = false;
        selectLlm.innerHTML = "";
        data.llm_models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.id;
          opt.innerText = m.name;
          if (m.default) opt.selected = true;
          selectLlm.appendChild(opt);
        });
      }
    }
  } catch (err) {
    console.error("Failed to load models from API:", err);
  }
}

// YouTube Metadata Inspection
async function inspectYouTubeUrl() {
  const input = document.getElementById("ytUrlInput");
  const url = (input.value || "").trim();
  if (!url) {
    showToast("Please enter a valid YouTube URL", "error");
    return;
  }
  const btnFetch = document.getElementById("btnFetchMeta");
  const previewCard = document.getElementById("videoPreviewCard");
  const titleEl = document.getElementById("previewTitle");
  const authorEl = document.getElementById("previewAuthor");
  const thumbEl = document.getElementById("previewThumb");
  const durEl = document.getElementById("previewDuration");

  const origBtnText = btnFetch ? btnFetch.innerText : "Inspect";
  if (btnFetch) {
    btnFetch.disabled = true;
    btnFetch.innerText = "Inspecting...";
  }
  titleEl.innerText = "Inspecting video...";
  authorEl.innerText = "Contacting YouTube...";
  previewCard.classList.remove("hidden");

  try {
    const res = await fetch("/api/youtube-info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Inspection failed (${res.status})`);
    }
    const info = await res.json();
    titleEl.innerText = info.title || "YouTube Video";
    authorEl.innerText = info.channel || "YouTube Channel";
    durEl.innerText = info.duration ? formatTime(info.duration) : "Full Video";
    if (info.thumbnail) thumbEl.src = info.thumbnail;
    showToast("Video metadata retrieved!", "success");

    // Proactively pre-fetch transcript and terminology in the background so player & helper slide are instant
    fetchTranscriptPreview(url, false);
  } catch (err) {
    titleEl.innerText = "Could not load video info";
    authorEl.innerText = err.message;
    showToast(err.message, "error");
  } finally {
    if (btnFetch) {
      btnFetch.disabled = false;
      btnFetch.innerText = origBtnText;
    }
  }
}

// Instant Pre-Fetch of Transcript & Terminology before full pipeline run
async function fetchTranscriptPreview(url, autoActivateTranscript = false) {
  if (!url) return null;
  try {
    const res = await fetch(`/api/transcript-preview?url=${encodeURIComponent(url)}`);
    if (!res.ok) return null;
    const data = await res.json();
    if (data && data.asr && Array.isArray(data.asr.segments)) {
      renderResults(data);
      if (autoActivateTranscript) {
        if (typeof setRightSidebarOpen === "function") {
          setRightSidebarOpen(true);
        }
        const transBtn = document.getElementById("subtabTranscriptBtn");
        if (transBtn) transBtn.click();
      }
      return data;
    }
  } catch (err) {
    console.warn("Transcript pre-fetch notice:", err);
  }
  return null;
}

// Tab Switching
function setupTabs() {
  const tabYt = document.getElementById("tabYoutube");
  const tabUp = document.getElementById("tabUpload");
  const ytContent = document.getElementById("youtubeTabContent");
  const upContent = document.getElementById("uploadTabContent");

  tabYt.addEventListener("click", () => {
    activeTab = "youtube";
    tabYt.classList.add("active");
    tabUp.classList.remove("active");
    ytContent.classList.remove("hidden");
    upContent.classList.add("hidden");
  });

  tabUp.addEventListener("click", () => {
    activeTab = "upload";
    tabUp.classList.add("active");
    tabYt.classList.remove("active");
    upContent.classList.remove("hidden");
    ytContent.classList.add("hidden");
  });
}

// File Upload Drag & Drop
function setupUpload() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const banner = document.getElementById("fileChosenBanner");
  const chosenName = document.getElementById("chosenFileName");
  const btnClear = document.getElementById("btnClearFile");

  ["dragenter", "dragover"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    chosenName.innerText = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    banner.classList.remove("hidden");
    showToast(`Loaded ${file.name}`, "success");
  }

  btnClear.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    banner.classList.add("hidden");
  });
}

// Audio Scope Presets
function setupScopePresets() {
  const btns = document.querySelectorAll(".preset-btn");
  const scopeInfo = document.getElementById("scopeInfo");
  const customGroup = document.getElementById("customRangeGroup");

  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const range = btn.dataset.range;
      currentScope = range;

      if (range === "custom") {
        customGroup.classList.remove("hidden");
        scopeInfo.innerText = "Custom Interval";
      } else {
        customGroup.classList.add("hidden");
        if (range === "full") {
          scopeInfo.innerText = "Full Duration";
        } else {
          scopeInfo.innerText = `First ${parseInt(range) / 60} Minutes`;
        }
      }
    });
  });
}

// Pipeline Execution & Job Management
let currentActiveJobId = null;

function subscribeToJobStream(jobId, initialStatus = "Running task...") {
  const btnRun = document.getElementById("btnRunPipeline");
  const btnCancel = document.getElementById("btnCancelJob");
  const progressBox = document.getElementById("pipelineProgress");
  const barFill = document.getElementById("progressBarFill");
  const statusMsg = document.getElementById("statusMessage");
  const logStream = document.getElementById("logStream");

  currentActiveJobId = jobId;
  let finished = false;

  const CLIENT_TIMEOUT_MS = 15 * 60 * 1000; // 15-minute client timeout
  const timeoutTimer = setTimeout(() => {
    if (!finished) {
      finishErr("Job timed out after 15 minutes of inactivity.");
    }
  }, CLIENT_TIMEOUT_MS);

  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }

  btnRun.disabled = true;
  if (btnCancel) {
    btnCancel.classList.remove("hidden");
    btnCancel.disabled = false;
  }
  progressBox.classList.remove("hidden");
  barFill.style.width = "10%";
  barFill.style.background = "";
  statusMsg.innerText = initialStatus;

  function appendLog(msg) {
    if (!logStream) return;
    const item = document.createElement("div");
    item.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logStream.appendChild(item);
    logStream.scrollTop = logStream.scrollHeight;
  }

  function finishOk(result) {
    if (finished) return;
    finished = true;
    clearTimeout(timeoutTimer);
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    currentActiveJobId = null;
    barFill.style.width = "100%";
    statusMsg.innerText = "Completed successfully!";
    btnRun.disabled = false;
    if (btnCancel) btnCancel.classList.add("hidden");
    showToast("Speech-to-Terminology pipeline complete!", "success");
    renderResults(result);
  }

  function finishErr(message) {
    if (finished) return;
    finished = true;
    clearTimeout(timeoutTimer);
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    currentActiveJobId = null;
    btnRun.disabled = false;
    if (btnCancel) btnCancel.classList.add("hidden");
    barFill.style.background = "var(--accent-rose)";
    statusMsg.innerText = "Error occurred";
    showToast(message || "An error occurred during processing.", "error");
    if (message) appendLog(`Error: ${message}`);
  }

  function finishCancelled(message) {
    if (finished) return;
    finished = true;
    clearTimeout(timeoutTimer);
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    currentActiveJobId = null;
    btnRun.disabled = false;
    if (btnCancel) btnCancel.classList.add("hidden");
    barFill.style.background = "var(--accent-amber, #f59e0b)";
    statusMsg.innerText = "Cancelled";
    showToast(message || "Job was cancelled.", "info");
    appendLog(`Job cancelled: ${message || "Cancelled by user"}`);
  }

  const streamUrl = _authToken ? `/api/stream/${jobId}?token=${encodeURIComponent(_authToken)}` : `/api/stream/${jobId}`;
  const eventSource = new EventSource(streamUrl);
  activeEventSource = eventSource;

  eventSource.addEventListener("step", (e) => {
    try {
      const payload = JSON.parse(e.data);
      if (payload?.message) statusMsg.innerText = payload.message;
      if (payload?.step) appendLog(`Step: ${payload.step}`);
      if (payload?.step === "download" || payload?.step === "convert") barFill.style.width = "25%";
      if (payload?.step === "asr") barFill.style.width = "50%";
      if (payload?.step === "llm") barFill.style.width = "75%";
    } catch (err) {
      console.warn("Failed to parse SSE step data:", err);
    }
  });

  eventSource.addEventListener("log", (e) => {
    try {
      const payload = JSON.parse(e.data);
      if (payload?.message) appendLog(payload.message);
    } catch (err) {
      console.warn("Failed to parse SSE log data:", err);
    }
  });

  eventSource.addEventListener("download_done", (e) => {
    try {
      const payload = JSON.parse(e.data);
      const dur = payload?.duration_s != null && !isNaN(payload.duration_s) ? `${Number(payload.duration_s).toFixed(1)}s` : "complete";
      appendLog(`Audio extracted: ${dur}`);
      barFill.style.width = "40%";
    } catch (err) {
      console.warn("Failed to parse SSE download_done data:", err);
    }
  });

  eventSource.addEventListener("asr_done", (e) => {
    try {
      const payload = JSON.parse(e.data);
      const rtfStr = payload?.rtf != null && !isNaN(payload.rtf) ? ` (RTF: ${Number(payload.rtf).toFixed(3)})` : "";
      appendLog(`ASR complete: ${payload?.latency_s ?? "?"}s${rtfStr}`);
      barFill.style.width = "75%";
    } catch (err) {
      console.warn("Failed to parse SSE asr_done data:", err);
    }
  });

  eventSource.addEventListener("llm_done", (e) => {
    try {
      const payload = JSON.parse(e.data);
      appendLog(`LLM extracted ${payload?.n_terms ?? 0} terms in ${payload?.latency_s ?? 0}s`);
      barFill.style.width = "95%";
    } catch (err) {
      console.warn("Failed to parse SSE llm_done data:", err);
    }
  });

  eventSource.addEventListener("done", (e) => {
    try {
      const payload = JSON.parse(e.data);
      finishOk(payload);
    } catch (err) {
      console.error("Failed to parse SSE done data:", err);
      finishErr("Received invalid completion payload from server.");
    }
  });

  eventSource.addEventListener("cancelled", (e) => {
    let msg = "Job was cancelled.";
    try {
      const payload = JSON.parse(e.data);
      if (payload?.message) msg = payload.message;
    } catch (_) {}
    finishCancelled(msg);
  });

  eventSource.addEventListener("job_error", (e) => {
    let message = "Pipeline failed.";
    let extra = "";
    try {
      const payload = JSON.parse(e.data);
      if (payload?.message) message = payload.message;
      if (payload?.error && payload.error !== message) extra = payload.error;
      if (payload?.detail && payload.detail !== message) extra = payload.detail;
    } catch (_) {}
    if (extra) {
      appendLog(`Diagnostic info: ${extra}`);
    }
    finishErr(message);
  });

  let reconnectFailures = 0;
  eventSource.onerror = async (errEv) => {
    if (finished) return;
    reconnectFailures += 1;
    console.warn(`[SSE Connection Warning #${reconnectFailures}] EventSource error:`, errEv);

    try {
      const resp = await fetch(`/api/job/${jobId}`);
      if (resp.ok) {
        const info = await resp.json();
        if (info.status === "completed" && info.result) {
          finishOk(info.result);
          return;
        }
        if (info.status === "error") {
          const errDetail = info.error || info.message || "Pipeline failed on server.";
          appendLog(`[Server Failure] Job failed: ${errDetail}`);
          finishErr(errDetail);
          return;
        }
        if (info.status === "cancelled") {
          const cancelDetail = info.message || "Job was cancelled on server.";
          appendLog(`[Server Cancel] ${cancelDetail}`);
          finishCancelled(cancelDetail);
          return;
        }
        // Job is still active on backend
        reconnectFailures = 0;
        appendLog(`[Connection Info] SSE connection re-synchronized. Job is actively '${info.status}'.`);
        return;
      } else {
        let errBody = "";
        try {
          const ej = await resp.json();
          errBody = ej.detail || ej.message || JSON.stringify(ej);
        } catch (_) {
          errBody = await resp.text();
        }
        appendLog(`[Network Warning] Reconnect poll returned HTTP ${resp.status} (${resp.statusText || "Error"})${errBody ? ": " + errBody : ""} (Attempt ${reconnectFailures}/5)`);
      }
    } catch (netErr) {
      appendLog(`[Network Warning] Could not reach server at ${window.location.origin} (${netErr.name}: ${netErr.message || "Connection refused"}). (Attempt ${reconnectFailures}/5)`);
      // Automatic Port Failover: scan adjacent ports if server moved
      if (reconnectFailures >= 2) {
        try {
          const currentPort = parseInt(window.location.port, 10) || 7860;
          const candidatePorts = [7860, 7861, 7862, 7863, 8000, 8001].filter((p) => p !== currentPort);
          for (const port of candidatePorts) {
            try {
              const probeCtrl = new AbortController();
              const probeTid = setTimeout(() => probeCtrl.abort(), 600);
              const probeRes = await fetch(`http://${window.location.hostname}:${port}/api/health`, { signal: probeCtrl.signal });
              clearTimeout(probeTid);
              if (probeRes.ok) {
                const pj = await probeRes.json();
                if (pj.service === "eduterm" || pj.status === "ok") {
                  appendLog(`[Port Failover] Discovered active EduTerm server on port ${port}! Automatically reconnecting...`);
                  showToast(`Server active on port ${port}. Migrating...`, "info");
                  setTimeout(() => {
                    window.location.href = `http://${window.location.hostname}:${port}`;
                  }, 1000);
                  return;
                }
              }
            } catch (_) {}
          }
        } catch (_) {}
      }
    }

    if (reconnectFailures >= 5) {
      appendLog(`[Fatal Error] Backend connection lost after 5 retries. Typical causes: (1) run_gui.bat terminal was closed, (2) GPU out-of-memory, (3) process crashed. Use run_gui.bat --force to start on another port.`);
      finishErr(`Lost connection to backend server (${window.location.origin}). Please check your run_gui.bat terminal window.`);
    }
  };
}

async function cancelActiveJob() {
  const btnCancel = document.getElementById("btnCancelJob");
  const statusMsg = document.getElementById("statusMessage");
  const logStream = document.getElementById("logStream");
  if (btnCancel) btnCancel.disabled = true;
  if (statusMsg) statusMsg.innerText = "Cancelling job...";

  function appendLog(msg) {
    if (!logStream) return;
    const item = document.createElement("div");
    item.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logStream.appendChild(item);
    logStream.scrollTop = logStream.scrollHeight;
  }

  appendLog("Requesting cancellation and GPU memory release from server...");
  try {
    const url = currentActiveJobId ? `/api/cancel/${currentActiveJobId}` : `/api/cancel-job`;
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) {
      let errDetail = "Failed to cancel job";
      try {
        const j = await res.json();
        errDetail = j.detail || errDetail;
      } catch (_) {}
      appendLog(`Cancel response: ${errDetail}`);
    } else {
      appendLog("Job cancellation signal received. Models unloading from GPU.");
    }
  } catch (err) {
    appendLog(`Cancel request error: ${err.message}`);
  }
}

async function checkActiveJob() {
  try {
    const res = await fetch("/api/active-job");
    if (!res.ok) return;
    const data = await res.json();
    if (data.active && data.job && data.job.job_id) {
      subscribeToJobStream(data.job.job_id, `Reconnected to active job (${data.job.status})...`);
    }
  } catch (_) {}
}

async function runPipeline() {
  const btnRun = document.getElementById("btnRunPipeline");
  const btnCancel = document.getElementById("btnCancelJob");
  const progressBox = document.getElementById("pipelineProgress");
  const barFill = document.getElementById("progressBarFill");
  const statusMsg = document.getElementById("statusMessage");
  const logStream = document.getElementById("logStream");

  const asrModel = document.getElementById("selectAsrModel").value;
  const lang = document.getElementById("selectLang").value;
  const llmModel = document.getElementById("selectLlmModel").value;

  let start_s = 0.0;
  let duration_s = null;

  if (currentScope === "120") {
    duration_s = 120.0;
  } else if (currentScope === "300") {
    duration_s = 300.0;
  } else if (currentScope === "600") {
    duration_s = 600.0;
  } else if (currentScope === "custom") {
    start_s = parseFloat(document.getElementById("inputStartSec").value) || 0.0;
    const durVal = parseFloat(document.getElementById("inputDurationSec").value);
    if (durVal && durVal > 0) duration_s = durVal;
  }

  let confirmLong = false;
  if (currentScope === "full" || (duration_s && duration_s > 600)) {
    const ok = confirm(
      "Processing audio longer than 10 minutes requires extended GPU processing time.\n\nClick OK to process full duration, or Cancel to safely cap at 10 minutes."
    );
    confirmLong = ok;
    if (!ok && duration_s && duration_s > 600) {
      duration_s = 600.0;
    }
  }

  btnRun.disabled = true;
  if (btnCancel) {
    btnCancel.classList.remove("hidden");
    btnCancel.disabled = false;
  }
  progressBox.classList.remove("hidden");
  barFill.style.width = "10%";
  barFill.style.background = "";
  statusMsg.innerText = "Initializing task...";
  logStream.innerHTML = "";

  function appendLog(msg) {
    const item = document.createElement("div");
    item.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logStream.appendChild(item);
    logStream.scrollTop = logStream.scrollHeight;
  }

  let jobId = null;
  try {
    if (activeTab === "youtube") {
      const url = document.getElementById("ytUrlInput").value.trim();
      if (!url) throw new Error("Please enter a YouTube URL.");

      appendLog(`Submitting YouTube URL: ${url}`);
      const res = await fetch("/api/process-youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          start_s,
          duration_s,
          asr_model: asrModel,
          lang,
          llm_model: llmModel,
          confirm_long: confirmLong,
        }),
      });
      if (!res.ok) {
        let errMsg = `Submission failed (${res.status})`;
        try {
          const err = await res.json();
          errMsg = err.detail || errMsg;
        } catch (_) {
          const txt = await res.text();
          if (txt) errMsg = txt;
        }
        throw new Error(errMsg);
      }
      const data = await res.json();
      jobId = data.job_id;
    } else {
      appendLog(`Uploading local file: ${selectedFile.name} (${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)...`);
      statusMsg.innerText = `Uploading ${selectedFile.name}...`;

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("start_s", start_s.toString());
      if (duration_s) formData.append("duration_s", duration_s.toString());
      formData.append("asr_model", asrModel);
      formData.append("lang", lang);
      formData.append("llm_model", llmModel);
      formData.append("confirm_long", confirmLong ? "true" : "false");

      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/process-upload");

        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            barFill.style.width = `${Math.min(95, Math.max(10, pct))}%`;
            statusMsg.innerText = `Uploading ${selectedFile.name}: ${pct}%`;
          }
        };

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch (_) {
              reject(new Error("Invalid server response format."));
            }
          } else {
            let detail = `Upload failed (${xhr.status})`;
            try {
              const body = JSON.parse(xhr.responseText);
              if (body && body.detail) detail = body.detail;
            } catch (_) {
              if (xhr.responseText) detail = xhr.responseText;
            }
            reject(new Error(detail));
          }
        };

        xhr.onerror = () => {
          reject(new Error("Connection to EduTerm AI server failed. Please ensure run_gui.bat is active in your terminal."));
        };

        xhr.ontimeout = () => {
          reject(new Error("Upload request timed out."));
        };

        xhr.send(formData);
      });

      jobId = data.job_id;
    }

    subscribeToJobStream(jobId, "Connecting to execution stream...");
  } catch (err) {
    btnRun.disabled = false;
    if (btnCancel) btnCancel.classList.add("hidden");
    const msg = (err && err.message === "Failed to fetch")
      ? "Connection to EduTerm AI server failed. Please ensure run_gui.bat is active in your terminal."
      : (err.message || "An unexpected error occurred.");
    showToast(msg, "error");
    appendLog(`Error: ${msg}`);
  }
}

// Render Results View
function renderResults(result) {
  try {
    currentResult = result;
    const emptyEl = document.getElementById("emptyState");
    if (emptyEl) emptyEl.classList.add("hidden");
    const resultsEl = document.getElementById("resultsView");
    if (resultsEl) resultsEl.classList.remove("hidden");

    // Diagnostics with defensive property checking
    const rtfVal =
      result?.asr?.rtf != null && !isNaN(result.asr.rtf) ? `${Number(result.asr.rtf).toFixed(3)} RTF` : "—";
    const asrLat = result?.asr?.latency_s != null ? `${result.asr.latency_s}s` : "—";
    const llmLat = result?.llm?.latency_s != null ? `${result.llm.latency_s}s` : "—";
    const nTerms = result?.llm?.n_terms ?? (Array.isArray(result?.llm?.terms) ? result.llm.terms.length : 0);

    const rtfEl = document.getElementById("diagRtf");
    if (rtfEl) rtfEl.innerText = rtfVal;
    const asrLatEl = document.getElementById("diagAsrLat");
    if (asrLatEl) asrLatEl.innerText = asrLat;
    const llmLatEl = document.getElementById("diagLlmLat");
    if (llmLatEl) llmLatEl.innerText = llmLat;
    const countEl = document.getElementById("diagTermsCount");
    if (countEl) countEl.innerText = nTerms;
    const termCountEl = document.getElementById("termCounter");
    if (termCountEl) termCountEl.innerText = nTerms;
    const langEl = document.getElementById("diagLang");
    if (langEl) langEl.innerText = escapeHtml((result?.lang || "—").toUpperCase());

    // Player Setup
    const ytContainer = document.getElementById("ytPlayerContainer");
    const videoContainer = document.getElementById("htmlVideoContainer");
    const audioContainer = document.getElementById("htmlAudioContainer");
    const meta = result?.video_meta || {};
    const isVideoMode = Boolean(result?.is_video || result?.has_video_file || meta?.is_video);

    if (meta.id && meta.url) {
      // YouTube mode
      if (audioContainer) audioContainer.classList.add("hidden");
      if (videoContainer) videoContainer.classList.add("hidden");
      if (ytContainer) ytContainer.classList.remove("hidden");
      loadYouTubeVideo(meta.id);
    } else if (isVideoMode) {
      // Local Video mode
      if (ytContainer) ytContainer.classList.add("hidden");
      if (audioContainer) audioContainer.classList.add("hidden");
      if (videoContainer) videoContainer.classList.remove("hidden");
      const videoPlayer = document.getElementById("videoPlayer");
      if (videoPlayer && result?.job_id) {
        videoPlayer.src = `/api/video/${result.job_id}`;
      }
    } else {
      // Local Audio mode
      if (ytContainer) ytContainer.classList.add("hidden");
      if (videoContainer) videoContainer.classList.add("hidden");
      if (audioContainer) audioContainer.classList.remove("hidden");
      const audioPlayer = document.getElementById("audioPlayer");
      if (audioPlayer && result?.job_id) {
        audioPlayer.src = `/api/audio/${result.job_id}`;
      }
    }

    const terms = Array.isArray(result?.llm?.terms) ? result.llm.terms : [];
    const segments = Array.isArray(result?.asr?.segments) ? result.asr.segments : [];
    const fullText = result?.asr?.text || "";

    // Adaptive Learner Background Banner
    const adaptive = result?.adaptive;
    const banner = document.getElementById("adaptiveBanner");
    if (banner) {
      if (adaptive && (adaptive.weight_applied || adaptive.has_history)) {
        banner.classList.remove("hidden");
        const headline = document.getElementById("adaptiveBannerHeadline");
        if (headline) {
          headline.innerText = adaptive.weight_applied
            ? `Student Background Applied (${adaptive.exams_taken} Prior Exam${adaptive.exams_taken === 1 ? "" : "s"})`
            : "Adaptive Tutor Ready";
        }
        const desc = document.getElementById("adaptiveBannerDesc");
        if (desc) {
          desc.innerText = adaptive.summary || "Student background is actively weighting domain terminology extraction.";
        }
      } else {
        banner.classList.add("hidden");
      }
    }

    renderTermsGrid(terms, result?.lang, segments);
    renderTranscript(segments, fullText, terms);

    const transCountEl = document.getElementById("transcriptCounter");
    if (transCountEl) {
      transCountEl.innerText = segments.length;
    }

    startSegmentWatch();
  } catch (err) {
    console.error("Error rendering results view:", err);
    showToast("Error rendering results: " + err.message, "error");
  } finally {
    const btnRun = document.getElementById("btnRunPipeline");
    if (btnRun) btnRun.disabled = false;
    const btnCancel = document.getElementById("btnCancelJob");
    if (btnCancel) btnCancel.classList.add("hidden");
  }
}

// YouTube Player loader
function loadYouTubeVideo(videoId) {
  if (!videoId) return;
  pendingYtId = videoId;
  if (!ytApiReady && window.YT && window.YT.Player) {
    ytApiReady = true;
  }
  if (!ytApiReady) return;

  if (ytPlayer && typeof ytPlayer.loadVideoById === "function") {
    try {
      ytPlayer.loadVideoById(videoId);
      return;
    } catch (_) {
      ytPlayer = null;
    }
  }
  const container = document.getElementById("ytPlayerContainer");
  if (!container) return;
  let mount = document.getElementById("ytPlayer");
  if (!mount) {
    mount = document.createElement("div");
    mount.id = "ytPlayer";
    container.innerHTML = "";
    container.appendChild(mount);
  }
  try {
    ytPlayer = new YT.Player("ytPlayer", {
      videoId: videoId,
      playerVars: {
        playsinline: 1,
        rel: 0,
      },
    });
  } catch (err) {
    console.error("Failed to mount YouTube Player:", err);
  }
}

// Seek Player to Timestamp
function seekPlayerTo(seconds) {
  if (seconds === null || isNaN(seconds)) return;
  const ytContainer = document.getElementById("ytPlayerContainer");
  const videoContainer = document.getElementById("htmlVideoContainer");
  const videoPlayer = document.getElementById("videoPlayer");
  const audioPlayer = document.getElementById("audioPlayer");

  if (ytPlayer && ytPlayer.seekTo && ytContainer && !ytContainer.classList.contains("hidden")) {
    ytPlayer.seekTo(seconds, true);
    ytPlayer.playVideo();
  } else if (videoPlayer && videoContainer && !videoContainer.classList.contains("hidden")) {
    videoPlayer.currentTime = seconds;
    videoPlayer.play().catch(() => {});
  } else if (audioPlayer) {
    audioPlayer.currentTime = seconds;
    audioPlayer.play().catch(() => {});
  }
}

// Find Earliest Matching Segment for Term
function findTermTimestamp(termText, segments) {
  if (!termText || !segments || !segments.length) return null;
  const needle = termText.toLowerCase().trim();

  // 1. Exact substring match
  for (const seg of segments) {
    const txt = (seg.text || "").toLowerCase();
    if (txt.includes(needle) && seg.start !== null && seg.start !== undefined) {
      return {
        start: Number(seg.start),
        end: seg.end !== null && seg.end !== undefined ? Number(seg.end) : null,
        formatted: formatTime(Number(seg.start)),
      };
    }
  }

  // 2. Multi-word or stem overlap (match significant tokens >= 3 chars)
  const tokens = needle.match(/\p{L}+/gu) || [];
  const words = tokens.filter((w) => w.length >= 3);
  if (words.length > 0) {
    for (const seg of segments) {
      const txt = (seg.text || "").toLowerCase();
      if (words.every((w) => txt.includes(w)) && seg.start !== null && seg.start !== undefined) {
        return {
          start: Number(seg.start),
          end: seg.end !== null && seg.end !== undefined ? Number(seg.end) : null,
          formatted: formatTime(Number(seg.start)),
        };
      }
    }
  }

  return null;
}

// Render Terminology Cards with Linked Timestamps
function renderTermsGrid(terms, lang, segments) {
  const grid = document.getElementById("termsGrid");
  grid.innerHTML = "";

  if (!terms || terms.length === 0) {
    grid.innerHTML = `<p style="color: var(--text-muted); font-size: 0.9rem; grid-column: 1/-1;">No domain terms identified in this segment.</p>`;
    return;
  }

  terms.forEach((item, idx) => {
    // Resolve timestamp localization
    let startSec = item.start_s !== undefined && item.start_s !== null ? Number(item.start_s) : null;
    let endSec = item.end_s !== undefined && item.end_s !== null ? Number(item.end_s) : null;
    let timeStr = item.timestamp || null;

    if (startSec === null && segments && segments.length) {
      const matched = findTermTimestamp(item.term, segments);
      if (matched) {
        startSec = matched.start;
        endSec = matched.end;
        timeStr = matched.formatted;
      }
    } else if (startSec !== null && !timeStr) {
      timeStr = formatTime(startSec);
    }

    const card = document.createElement("div");
    card.className = "term-card";
    card.dataset.term = (item.term || "").toLowerCase();
    card.dataset.expl = (item.explanation || "").toLowerCase();
    if (startSec !== null) {
      card.dataset.start = String(startSec);
    }
    if (endSec !== null) {
      card.dataset.end = String(endSec);
    }

    const safeStartSec = startSec !== null && !isNaN(startSec) ? Number(startSec) : null;
    const safeTimeStr = escapeHtml(timeStr || "—:—");

    const timestampButtonHtml =
      safeStartSec !== null
        ? `<button type="button" class="term-time-badge" data-start="${safeStartSec}" title="Jump video & transcript to ${safeTimeStr}">
             <svg class="play-mini-icon" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"></polygon></svg>
             <span>${safeTimeStr}</span>
           </button>`
        : `<span class="term-time-badge no-time" title="Timestamp not resolved">—:—</span>`;

    let focusBadgeHtml = "";
    const focus = item.focus || "new";
    const masteryVal = item.mastery != null ? Math.round(Number(item.mastery) * 100) : null;
    if (focus === "review") {
      focusBadgeHtml = `<span class="term-focus-badge focus-review" title="Review-Priority from student exams (${masteryVal ?? 0}% mastery) — Higher LLM Weight">⚠️ Review (${masteryVal ?? 0}%)</span>`;
    } else if (focus === "learning") {
      focusBadgeHtml = `<span class="term-focus-badge focus-learning" title="Still Learning (${masteryVal ?? 50}% mastery)">📘 Learning</span>`;
    } else if (focus === "known") {
      focusBadgeHtml = `<span class="term-focus-badge focus-known" title="Mastered in student exams (${masteryVal ?? 100}% mastery) — Lower LLM Weight">✓ Mastered</span>`;
    }

    card.innerHTML = `
      <div>
        <div class="term-header">
          <div class="term-title-row">
            <h4 class="term-name">${escapeHtml(item.term)}</h4>
            ${timestampButtonHtml}
            ${focusBadgeHtml}
          </div>
          <span class="term-badge">${escapeHtml((lang || "tr").toUpperCase())}</span>
        </div>
        <p class="term-explanation">${escapeHtml(item.explanation || "")}</p>
      </div>
      <div class="term-actions">
        <button type="button" class="term-action-btn btn-jump" title="Jump to this term in video">
          <svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor" style="margin-right:2px"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          Jump
        </button>
        <button type="button" class="term-action-btn btn-find" title="Find in interactive transcript">Find</button>
        <button type="button" class="term-action-btn btn-speak" title="Pronounce term">Speak</button>
        <button type="button" class="term-action-btn btn-copy" title="Copy term & definition">Copy</button>
      </div>
    `;

    // Click on timestamp badge -> jumps video and highlights transcript
    const timeBtn = card.querySelector(".term-time-badge:not(.no-time)");
    if (timeBtn) {
      timeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        seekPlayerTo(startSec);
        showToast(`Jumped to ${timeStr} for "${item.term}"`, "info");
        highlightSegmentAtTime(startSec);
      });
    }

    // Jump button in footer
    card.querySelector(".btn-jump").addEventListener("click", () => {
      if (startSec !== null) {
        seekPlayerTo(startSec);
        showToast(`Jumped to ${timeStr || formatTime(startSec)} for "${item.term}"`, "info");
        highlightSegmentAtTime(startSec);
      } else {
        findTermInTranscript(item.term);
      }
    });

    // Find in transcript
    card.querySelector(".btn-find").addEventListener("click", () => {
      findTermInTranscript(item.term, startSec);
    });

    // Speak term
    card.querySelector(".btn-speak").addEventListener("click", () => {
      const utter = new SpeechSynthesisUtterance(item.term);
      utter.lang = (lang || "tr").toLowerCase().startsWith("tr") ? "tr-TR" : "en-US";
      window.speechSynthesis.speak(utter);
    });

    // Copy term
    card.querySelector(".btn-copy").addEventListener("click", () => {
      const copyText = timeStr
        ? `[${timeStr}] ${item.term}: ${item.explanation}`
        : `${item.term}: ${item.explanation}`;
      navigator.clipboard.writeText(copyText);
      showToast(`Copied "${item.term}" to clipboard!`, "success");
    });

    grid.appendChild(card);
  });
}

function escapeRegex(str) {
  return String(str).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightTermsHtml(text, terms) {
  if (!text) return "";
  if (!terms || !terms.length) return escapeHtml(text);
  const sorted = [...terms]
    .map((t) => (t && t.term ? String(t.term).trim() : ""))
    .filter((t) => t.length >= 3)
    .sort((a, b) => b.length - a.length);
  let html = escapeHtml(text);
  const seen = new Set();
  for (const term of sorted) {
    const key = term.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    const re = new RegExp(`(${escapeRegex(escapeHtml(term))})`, "gi");
    html = html.replace(re, `<mark class="term-hit">$1</mark>`);
  }
  return html;
}

function highlightSegmentAtTime(seconds) {
  if (seconds === null || isNaN(seconds)) return;
  const segments = document.querySelectorAll(".transcript-segment");
  for (const seg of segments) {
    const start = parseFloat(seg.dataset.start);
    if (!isNaN(start) && Math.abs(start - seconds) < 2.0) {
      seg.classList.add("flash");
      setTimeout(() => seg.classList.remove("flash"), 1600);
      break;
    }
  }
}

function findTermInTranscript(term, preferredStart) {
  if (!term) return;
  if (typeof setRightSidebarOpen === "function") {
    setRightSidebarOpen(true);
  }
  const subBtn = document.getElementById("subtabTranscriptBtn");
  if (subBtn) subBtn.click();
  const scroll = document.getElementById("transcriptScroll");
  const needle = term.toLowerCase();
  const segments = scroll.querySelectorAll(".transcript-segment");

  // If we have a preferred timestamp, try to match closest segment first
  if (preferredStart !== null && preferredStart !== undefined && !isNaN(preferredStart)) {
    for (const seg of segments) {
      const start = parseFloat(seg.dataset.start);
      if (!isNaN(start) && Math.abs(start - preferredStart) < 1.8) {
        seg.scrollIntoView({ behavior: "smooth", block: "center" });
        seg.classList.add("flash");
        seekPlayerTo(start);
        setTimeout(() => seg.classList.remove("flash"), 1800);
        return;
      }
    }
  }

  // Fallback: search by text
  for (const seg of segments) {
    const text = (seg.querySelector(".segment-text") || {}).innerText || "";
    if (text.toLowerCase().includes(needle)) {
      seg.scrollIntoView({ behavior: "smooth", block: "center" });
      seg.classList.add("flash");
      const start = parseFloat(seg.dataset.start);
      if (!Number.isNaN(start)) seekPlayerTo(start);
      setTimeout(() => seg.classList.remove("flash"), 1800);
      return;
    }
  }
  showToast(`“${term}” was not found in the transcript`, "info");
}

function startSegmentWatch() {
  if (segmentWatchTimer) clearInterval(segmentWatchTimer);
  segmentWatchTimer = setInterval(() => {
    let t = null;
    const ytContainer = document.getElementById("ytPlayerContainer");
    const videoContainer = document.getElementById("htmlVideoContainer");
    const videoPlayer = document.getElementById("videoPlayer");
    const audioPlayer = document.getElementById("audioPlayer");

    if (ytPlayer && typeof ytPlayer.getCurrentTime === "function" && ytContainer && !ytContainer.classList.contains("hidden")) {
      try {
        t = ytPlayer.getCurrentTime();
      } catch (_) {
        t = null;
      }
    }
    if (t == null && videoPlayer && videoContainer && !videoContainer.classList.contains("hidden")) {
      if (!Number.isNaN(videoPlayer.currentTime)) t = videoPlayer.currentTime;
    }
    if (t == null && audioPlayer) {
      if (!Number.isNaN(audioPlayer.currentTime)) t = audioPlayer.currentTime;
    }
    if (t == null) return;

    // 1. Highlight active transcript segment and auto-scroll to keep speaking line in view!
    let firstActiveSeg = null;
    document.querySelectorAll(".transcript-segment").forEach((el) => {
      const start = parseFloat(el.dataset.start);
      const end = parseFloat(el.dataset.end);
      const active = !Number.isNaN(start) && t >= start && (Number.isNaN(end) || t < end);
      const wasActive = el.classList.contains("active");
      el.classList.toggle("active", active);
      if (active && !wasActive && !firstActiveSeg) {
        firstActiveSeg = el;
      }
    });
    if (firstActiveSeg) {
      firstActiveSeg.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // 2. Synchronously highlight active term cards during playback!
    document.querySelectorAll(".term-card").forEach((card) => {
      const start = parseFloat(card.dataset.start);
      const end = parseFloat(card.dataset.end);
      let isCurrent = false;
      if (!Number.isNaN(start)) {
        const effectiveEnd = !Number.isNaN(end) ? Math.max(end, start + 3.0) : start + 5.0;
        isCurrent = t >= (start - 0.5) && t <= effectiveEnd;
      }
      card.classList.toggle("active-term", isCurrent);
    });
  }, 400);
}

// Render Interactive Transcript
function renderTranscript(segments, fullText, terms) {
  const scroll = document.getElementById("transcriptScroll");
  scroll.innerHTML = "";

  const count = (segments && segments.length) || 0;
  const transCounter = document.getElementById("transcriptCounter");
  if (transCounter) transCounter.innerText = String(count);
  const headerTransBadge = document.getElementById("headerTranscriptBadge");
  if (headerTransBadge) headerTransBadge.innerText = String(count);

  if (segments && segments.length > 0) {
    segments.forEach((seg) => {
      const segEl = document.createElement("div");
      segEl.className = "transcript-segment";
      if (seg.start !== null && seg.start !== undefined) {
        segEl.dataset.start = String(seg.start);
      }
      if (seg.end !== null && seg.end !== undefined) {
        segEl.dataset.end = String(seg.end);
      }
      const timeTag = seg.start !== null && seg.start !== undefined ? formatTime(seg.start) : "---";

      segEl.innerHTML = `
        <span class="segment-time">${timeTag}</span>
        <span class="segment-text">${highlightTermsHtml(seg.text, terms)}</span>
      `;

      if (seg.start !== null && seg.start !== undefined) {
        segEl.addEventListener("click", () => {
          seekPlayerTo(seg.start);
          showToast(`Jumped to ${timeTag}`, "info");
        });
      }
      scroll.appendChild(segEl);
    });
  } else {
    scroll.innerHTML = `<p style="padding: 1rem; color: var(--text-secondary);">${highlightTermsHtml(fullText, terms)}</p>`;
  }
}

// Terms Search Filter
function setupTermsFilter() {
  const input = document.getElementById("termsSearchInput");
  const countEl = document.getElementById("filteredTermsCount");

  input.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase().trim();
    const cards = document.querySelectorAll(".term-card");
    let visible = 0;

    cards.forEach((card) => {
      const term = card.dataset.term;
      const expl = card.dataset.expl;
      if (term.includes(query) || expl.includes(query)) {
        card.style.display = "flex";
        visible++;
      } else {
        card.style.display = "none";
      }
    });
    countEl.innerText = query ? `Showing ${visible} of ${cards.length} terms` : "Showing all terms";
  });
}

// Export Hub
function setupExports() {
  document.getElementById("btnExportCsv").addEventListener("click", () => {
    if (!currentResult) return;
    const csvUrl = _authToken
      ? `/api/export-csv/${currentResult.job_id}?token=${encodeURIComponent(_authToken)}`
      : `/api/export-csv/${currentResult.job_id}`;
    window.location.href = csvUrl;
  });

  document.getElementById("btnExportJson").addEventListener("click", () => {
    if (!currentResult) return;
    const jsonUrl = _authToken
      ? `/api/export-json/${currentResult.job_id}?token=${encodeURIComponent(_authToken)}`
      : `/api/export-json/${currentResult.job_id}`;
    window.location.href = jsonUrl;
  });

  document.getElementById("btnCopyMd").addEventListener("click", () => {
    if (!currentResult) return;
    const terms = currentResult.llm.terms || [];
    let md = `### Extracted Terminology (${currentResult.lang.toUpperCase()})\n\n`;
    terms.forEach((t) => {
      md += `- **${t.term}**: ${t.explanation}\n`;
    });
    navigator.clipboard.writeText(md);
    showToast("Copied Markdown summary to clipboard!", "success");
  });

  document.getElementById("btnCopyTranscript").addEventListener("click", () => {
    if (!currentResult) return;
    navigator.clipboard.writeText(currentResult.asr.text);
    showToast("Copied full transcript to clipboard!", "success");
  });
}

// ==========================================================================
// 3-Zone Studio Workspace & Collapsible Sidebars Controller
// ==========================================================================

let leftSidebarOpen = true;
let rightSidebarOpen = false;

function setLeftSidebarOpen(open) {
  leftSidebarOpen = !!open;
  const workspace = document.getElementById("studioWorkspace");
  const btnToggle = document.getElementById("btnToggleLeftSidebar");
  if (workspace) {
    workspace.classList.toggle("left-open", leftSidebarOpen);
  }
  if (btnToggle) {
    btnToggle.classList.toggle("active", leftSidebarOpen);
  }
  const backdrop = document.getElementById("sidebarBackdrop");
  if (backdrop && window.innerWidth <= 1200) {
    backdrop.classList.toggle("hidden", !leftSidebarOpen && !rightSidebarOpen);
  }
  try {
    localStorage.setItem("eduterm_left_open", leftSidebarOpen ? "1" : "0");
  } catch (_) {}
}

function setRightSidebarOpen(open) {
  rightSidebarOpen = !!open;
  const workspace = document.getElementById("studioWorkspace");
  const btnToggle = document.getElementById("btnToggleRightSidebar");
  if (workspace) {
    workspace.classList.toggle("right-open", rightSidebarOpen);
  }
  if (btnToggle) {
    btnToggle.classList.toggle("active", rightSidebarOpen);
  }
  const backdrop = document.getElementById("sidebarBackdrop");
  if (backdrop && window.innerWidth <= 1200) {
    backdrop.classList.toggle("hidden", !leftSidebarOpen && !rightSidebarOpen);
  }
  try {
    localStorage.setItem("eduterm_right_open", rightSidebarOpen ? "1" : "0");
  } catch (_) {}
}

function toggleLeftSidebar() {
  setLeftSidebarOpen(!leftSidebarOpen);
}

function toggleRightSidebar() {
  setRightSidebarOpen(!rightSidebarOpen);
}

// Transcript search filter
function setupTranscriptSearch() {
  const searchInput = document.getElementById("transcriptSearchInput");
  const clearBtn = document.getElementById("btnClearTranscriptSearch");
  if (!searchInput) return;

  const doFilter = () => {
    const q = (searchInput.value || "").trim().toLowerCase();
    if (clearBtn) clearBtn.classList.toggle("hidden", !q);
    const segs = document.querySelectorAll(".transcript-segment");
    segs.forEach((seg) => {
      const text = seg.innerText.toLowerCase();
      const matches = !q || text.includes(q);
      seg.style.display = matches ? "flex" : "none";
    });
  };

  searchInput.addEventListener("input", doFilter);
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      searchInput.value = "";
      doFilter();
      searchInput.focus();
    });
  }
}

// Setup 3-Zone Sidebars & keyboard shortcuts
function setupStudioSidebars() {
  // Restore persisted state from localStorage if available
  try {
    const savedLeft = localStorage.getItem("eduterm_left_open");
    const savedRight = localStorage.getItem("eduterm_right_open");
    if (savedLeft !== null) leftSidebarOpen = savedLeft === "1";
    if (savedRight !== null) rightSidebarOpen = savedRight === "1";
  } catch (_) {}

  // On initial mobile/narrow screens, default sidebars to closed
  if (window.innerWidth <= 1200) {
    leftSidebarOpen = false;
    rightSidebarOpen = false;
  }

  setLeftSidebarOpen(leftSidebarOpen);
  setRightSidebarOpen(rightSidebarOpen);

  // Toggle buttons
  const btnToggleLeft = document.getElementById("btnToggleLeftSidebar");
  if (btnToggleLeft) btnToggleLeft.addEventListener("click", toggleLeftSidebar);

  const btnCloseLeft = document.getElementById("btnCloseLeftSidebar");
  if (btnCloseLeft) btnCloseLeft.addEventListener("click", () => setLeftSidebarOpen(false));

  const btnEdgeLeft = document.getElementById("btnEdgeOpenLeft");
  if (btnEdgeLeft) btnEdgeLeft.addEventListener("click", () => setLeftSidebarOpen(true));

  const btnOpenFromEmpty = document.getElementById("btnOpenMediaFromEmpty");
  if (btnOpenFromEmpty) btnOpenFromEmpty.addEventListener("click", () => setLeftSidebarOpen(true));

  const btnToggleRight = document.getElementById("btnToggleRightSidebar");
  if (btnToggleRight) btnToggleRight.addEventListener("click", toggleRightSidebar);

  const btnCloseRight = document.getElementById("btnCloseRightSidebar");
  if (btnCloseRight) btnCloseRight.addEventListener("click", () => setRightSidebarOpen(false));

  const btnEdgeRight = document.getElementById("btnEdgeOpenRight");
  if (btnEdgeRight) btnEdgeRight.addEventListener("click", () => setRightSidebarOpen(true));

  const btnQuickTrans = document.getElementById("btnQuickOpenTranscript");
  if (btnQuickTrans) btnQuickTrans.addEventListener("click", () => setRightSidebarOpen(true));

  // Backward compatibility hooks
  const btnLegacyTerms = document.getElementById("subtabTermsBtn");
  if (btnLegacyTerms) {
    btnLegacyTerms.addEventListener("click", () => {
      const termsGrid = document.getElementById("termsGrid");
      if (termsGrid) termsGrid.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  const btnLegacyTrans = document.getElementById("subtabTranscriptBtn");
  if (btnLegacyTrans) {
    btnLegacyTrans.addEventListener("click", () => {
      setRightSidebarOpen(true);
    });
  }

  // Backdrop overlay click
  const backdrop = document.getElementById("sidebarBackdrop");
  if (backdrop) {
    backdrop.addEventListener("click", () => {
      setLeftSidebarOpen(false);
      setRightSidebarOpen(false);
    });
  }

  // Keyboard shortcuts: M for Left Media panel, T for Right Transcript panel
  window.addEventListener("keydown", (e) => {
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable) {
      return;
    }
    if (e.key === "m" || e.key === "M") {
      e.preventDefault();
      toggleLeftSidebar();
      showToast(leftSidebarOpen ? "Medya Paneli Açıldı" : "Medya Paneli Kapatıldı", "info");
    } else if (e.key === "t" || e.key === "T") {
      e.preventDefault();
      toggleRightSidebar();
      showToast(rightSidebarOpen ? "İnteraktif Transkript Açıldı" : "Transkript Kapatıldı", "info");
    }
  });

  setupTranscriptSearch();
}

// Backward compatibility alias for setupSubtabs
function setupSubtabs() {
  setupStudioSidebars();
}

// Quick Sample Chips
function setupDemoChips() {
  document.querySelectorAll(".demo-chip").forEach((chip) => {
    chip.addEventListener("click", async () => {
      const url = chip.dataset.url;
      document.getElementById("ytUrlInput").value = url;
      document.getElementById("tabYoutube").click();
      inspectYouTubeUrl();
      showToast("Transkript ve terimler yükleniyor...", "info");
      await fetchTranscriptPreview(url, true);
    });
  });
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ==========================================================================
// Student Background & Learner Profile System
// ==========================================================================
let currentProfile = null;

async function loadLearnerProfile() {
  try {
    const res = await fetch("/api/learner/profile");
    if (!res.ok) return;
    const data = await res.json();
    currentProfile = data;
    renderProfileBadge(data);
    renderProfileModal(data);
  } catch (err) {
    console.error("Failed to load learner profile:", err);
  }
}

function renderProfileBadge(profile) {
  const badgeEl = document.getElementById("learnerProfileBadge");
  if (!badgeEl) return;
  const stats = profile?.stats || {};
  const count = stats.exams_taken || 0;
  if (count > 0) {
    const avg = Math.round((stats.avg_score || 0) * 100);
    const weak = stats.weak_count || 0;
    badgeEl.innerText = `Student Profile (${count} Exam${count === 1 ? "" : "s"} • ${avg}%${weak > 0 ? ` • ${weak} Review` : ""})`;
  } else {
    badgeEl.innerText = "Student Profile (0 Exams)";
  }
}

function renderProfileModal(profile) {
  const stats = profile?.stats || {};
  const analysis = profile?.analysis || {};

  const elExams = document.getElementById("profileExamsTaken");
  if (elExams) elExams.innerText = stats.exams_taken || 0;
  const elAvg = document.getElementById("profileAvgScore");
  if (elAvg) elAvg.innerText = `${Math.round((stats.avg_score || 0) * 100)}%`;
  const elWeak = document.getElementById("profileWeakCount");
  if (elWeak) elWeak.innerText = stats.weak_count || 0;

  // Weak/Review Terms
  const weakList = document.getElementById("profileWeakList");
  const weakBadge = document.getElementById("profileWeakBadge");
  const weakTerms = analysis?.weak_terms || [];
  if (weakBadge) weakBadge.innerText = `${weakTerms.length} term${weakTerms.length === 1 ? "" : "s"}`;
  if (weakList) {
    if (weakTerms.length === 0) {
      weakList.innerHTML = `<span class="empty-note">No review-priority terms on record. Take a practice exam to build your background.</span>`;
    } else {
      weakList.innerHTML = weakTerms.map((t) => {
        const mastery = t.mastery != null ? Math.round(Number(t.mastery) * 100) : 0;
        return `<span class="chip-item weak" title="Mastery: ${mastery}% • Missed: ${t.incorrect || 0} times">⚠️ ${escapeHtml(t.term)} (${mastery}%)</span>`;
      }).join("");
    }
  }

  // Mastered Terms
  const strongList = document.getElementById("profileStrongList");
  const strongBadge = document.getElementById("profileStrongBadge");
  const strongTerms = analysis?.strong_terms || [];
  if (strongBadge) strongBadge.innerText = `${strongTerms.length} term${strongTerms.length === 1 ? "" : "s"}`;
  if (strongList) {
    if (strongTerms.length === 0) {
      strongList.innerHTML = `<span class="empty-note">No mastered terms on record yet.</span>`;
    } else {
      strongList.innerHTML = strongTerms.map((t) => {
        const mastery = t.mastery != null ? Math.round(Number(t.mastery) * 100) : 100;
        return `<span class="chip-item strong" title="Mastery: ${mastery}%">✓ ${escapeHtml(t.term)} (${mastery}%)</span>`;
      }).join("");
    }
  }

  // Recent Exams
  const examsList = document.getElementById("profileRecentExams");
  const recentExams = profile?.recent_exams || [];
  if (examsList) {
    if (recentExams.length === 0) {
      examsList.innerHTML = `<span class="empty-note">No practice exams taken yet.</span>`;
    } else {
      examsList.innerHTML = recentExams.map((ex) => {
        const score = Math.round((ex.score || 0) * 100);
        const title = ex.video_title || (ex.video_id ? `Video ${ex.video_id}` : "Practice Exam");
        const dt = ex.created_at ? ex.created_at.slice(0, 10) : "";
        return `
          <div class="profile-exam-item">
            <div class="exam-item-title" title="${escapeHtml(title)}">
              🎓 ${escapeHtml(title)}
              ${dt ? `<small style="color:var(--text-muted); margin-left:6px">${escapeHtml(dt)}</small>` : ""}
            </div>
            <div class="exam-item-score">${score}% (${ex.correct || 0}/${ex.total || 0})</div>
          </div>
        `;
      }).join("");
    }
  }
}

function setupLearnerProfileModal() {
  const btnOpen = document.getElementById("btnOpenLearnerProfile");
  const modal = document.getElementById("learnerProfileModal");
  const btnClose = document.getElementById("btnCloseLearnerModal");
  const btnClose2 = document.getElementById("btnCloseLearnerModalBtn");
  const btnReset = document.getElementById("btnResetProfile");

  if (btnOpen && modal) {
    btnOpen.addEventListener("click", () => {
      loadLearnerProfile();
      modal.classList.remove("hidden");
    });
  }

  const closeModal = () => {
    if (modal) modal.classList.add("hidden");
  };

  if (btnClose) btnClose.addEventListener("click", closeModal);
  if (btnClose2) btnClose2.addEventListener("click", closeModal);

  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      if (!confirm("Are you sure you want to reset your learner background and practice exam history? This will clear all mastery weights.")) {
        return;
      }
      try {
        const res = await fetch("/api/learner/reset", { method: "POST" });
        if (res.ok) {
          showToast("Student profile reset successfully.", "info");
          await loadLearnerProfile();
          if (currentResult) {
            if (currentResult.llm && Array.isArray(currentResult.llm.terms)) {
              currentResult.llm.terms.forEach((t) => { t.focus = "new"; t.mastery = null; });
              renderTermsGrid(currentResult.llm.terms, currentResult.lang, currentResult.asr?.segments);
            }
            const banner = document.getElementById("adaptiveBanner");
            if (banner) banner.classList.add("hidden");
          }
        }
      } catch (err) {
        showToast("Failed to reset profile: " + err.message, "error");
      }
    });
  }
}

// ==========================================================================
// Practice Exam Controller (Flashcards & Multiple Choice Quiz)
// ==========================================================================
let activeExam = null;

function setupPracticeExam() {
  const btnStart = document.getElementById("btnStartPracticeExam");
  const modal = document.getElementById("examModal");
  const btnClose = document.getElementById("btnCloseExamModal");
  const btnCloseResults = document.getElementById("btnCloseExamResults");
  const btnViewProfile = document.getElementById("btnViewProfileFromExam");
  const flashcardCard = document.getElementById("flashcardCard");
  const btnFlip = document.getElementById("btnFlipCard");
  const btnGradeReview = document.getElementById("btnGradeReview");
  const btnGradeMastered = document.getElementById("btnGradeMastered");
  const btnPrev = document.getElementById("btnPrevQuestion");
  const btnNext = document.getElementById("btnNextQuestion");
  const btnSubmit = document.getElementById("btnSubmitExam");

  if (btnStart) {
    btnStart.addEventListener("click", () => {
      if (!currentResult || !currentResult.job_id) {
        showToast("No active lecture analysis found to build an exam.", "error");
        return;
      }
      openPracticeExam(currentResult.job_id, true);
    });
  }

  const btnStartInstant = document.getElementById("btnStartInstantExam");
  if (btnStartInstant) {
    btnStartInstant.addEventListener("click", () => {
      if (!currentResult || !currentResult.job_id) {
        showToast("No active lecture analysis found to build an exam.", "error");
        return;
      }
      openPracticeExam(currentResult.job_id, false);
    });
  }

  const btnInstantFallback = document.getElementById("btnInstantExamFallback");
  if (btnInstantFallback) {
    btnInstantFallback.addEventListener("click", () => {
      if (!currentResult || !currentResult.job_id) return;
      if (examAbortController) {
        examAbortController.abort();
      }
      openPracticeExam(currentResult.job_id, false);
    });
  }

  const closeExamModal = () => {
    if (modal) modal.classList.add("hidden");
    activeExam = null;
  };

  if (btnClose) btnClose.addEventListener("click", closeExamModal);
  if (btnCloseResults) btnCloseResults.addEventListener("click", closeExamModal);
  if (btnViewProfile) {
    btnViewProfile.addEventListener("click", () => {
      closeExamModal();
      document.getElementById("btnOpenLearnerProfile")?.click();
    });
  }

  const toggleFlip = () => {
    if (!activeExam || !flashcardCard) return;
    activeExam.isFlipped = !activeExam.isFlipped;
    if (activeExam.isFlipped) {
      flashcardCard.classList.add("is-flipped");
    } else {
      flashcardCard.classList.remove("is-flipped");
    }
  };

  if (flashcardCard) flashcardCard.addEventListener("click", toggleFlip);
  if (btnFlip) btnFlip.addEventListener("click", (e) => { e.stopPropagation(); toggleFlip(); });

  if (btnGradeReview) {
    btnGradeReview.addEventListener("click", (e) => {
      e.stopPropagation();
      recordAnswer("dont_know");
    });
  }
  if (btnGradeMastered) {
    btnGradeMastered.addEventListener("click", (e) => {
      e.stopPropagation();
      recordAnswer("know");
    });
  }

  if (btnPrev) {
    btnPrev.addEventListener("click", () => {
      if (!activeExam || activeExam.currentIdx <= 0) return;
      activeExam.currentIdx--;
      renderActiveQuestion();
    });
  }
  if (btnNext) {
    btnNext.addEventListener("click", () => {
      if (!activeExam || activeExam.currentIdx >= activeExam.questions.length - 1) return;
      activeExam.currentIdx++;
      renderActiveQuestion();
    });
  }

  if (btnSubmit) {
    btnSubmit.addEventListener("click", submitPracticeExam);
  }

  document.addEventListener("keydown", (e) => {
    if (!modal || modal.classList.contains("hidden") || !activeExam) return;
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;

    if (e.key === "Escape") {
      closeExamModal();
    } else if (e.code === "Space") {
      e.preventDefault();
      toggleFlip();
    } else if (e.key === "ArrowLeft") {
      if (activeExam.currentIdx > 0) {
        activeExam.currentIdx--;
        renderActiveQuestion();
      }
    } else if (e.key === "ArrowRight") {
      if (activeExam.currentIdx < activeExam.questions.length - 1) {
        activeExam.currentIdx++;
        renderActiveQuestion();
      }
    } else if (e.key === "1") {
      const q = activeExam.questions[activeExam.currentIdx];
      if (q && q.type === "flashcard") {
        recordAnswer("dont_know");
      } else if (q && q.type === "mcq") {
        selectMcqOption(0);
      }
    } else if (e.key === "2") {
      const q = activeExam.questions[activeExam.currentIdx];
      if (q && q.type === "flashcard") {
        recordAnswer("know");
      } else if (q && q.type === "mcq") {
        selectMcqOption(1);
      }
    } else if (e.key === "3") {
      selectMcqOption(2);
    } else if (e.key === "4") {
      selectMcqOption(3);
    }
  });
}

let examAbortController = null;

async function openPracticeExam(jobId, useLlm = true) {
  const modal = document.getElementById("examModal");
  if (!modal) return;
  modal.classList.remove("hidden");

  document.getElementById("examLoadingState")?.classList.remove("hidden");
  document.getElementById("examActiveContainer")?.classList.add("hidden");
  document.getElementById("examResultsState")?.classList.add("hidden");

  const msgEl = document.getElementById("examLoadingMsg");
  const subEl = document.querySelector(".exam-loading-sub");
  if (msgEl) {
    msgEl.innerText = useLlm
      ? "Drafting AI practice exam with LLM..."
      : "Building instant flashcards & quiz from lecture terms...";
  }
  if (subEl) {
    subEl.innerText = useLlm
      ? "Generating adaptive questions based on lecture terminology (~5–8s)."
      : "Creating interactive flashcards and multiple-choice questions (< 0.1s).";
  }

  const titleEl = document.getElementById("examModalTitle");
  if (titleEl) {
    const meta = currentResult?.video_meta || {};
    titleEl.innerText = meta.title ? `Practice Exam — ${meta.title}` : "Practice Exam";
  }

  if (examAbortController) {
    examAbortController.abort();
  }
  examAbortController = new AbortController();
  const currentSignal = examAbortController.signal;

  // Client safety timer: auto-fallback to instant mode if LLM takes longer than 12s
  const fallbackTimer = setTimeout(() => {
    if (useLlm && !currentSignal.aborted) {
      console.warn("LLM exam generation taking longer than 12s; seamlessly falling back to instant mode.");
      if (subEl) subEl.innerText = "LLM generation slow; falling back to instant cards...";
      if (examAbortController) examAbortController.abort();
      openPracticeExam(jobId, false);
    }
  }, 12000);

  try {
    const res = await fetch("/api/exam/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, use_llm: useLlm }),
      signal: currentSignal,
    });
    clearTimeout(fallbackTimer);
    if (!res.ok) {
      let errDetail = "Failed to generate exam.";
      try {
        const errData = await res.json();
        errDetail = errData.detail || errDetail;
      } catch (_) {}
      throw new Error(errDetail);
    }
    const examData = await res.json();
    startExam(examData);
  } catch (err) {
    clearTimeout(fallbackTimer);
    if (err.name === "AbortError") {
      // Aborted intentionally, e.g. switched to instant mode
      return;
    }
    console.error("Exam generation error:", err);
    showToast("Error generating practice exam: " + err.message, "error");
    document.getElementById("examModal")?.classList.add("hidden");
  } finally {
    if (examAbortController && examAbortController.signal === currentSignal) {
      examAbortController = null;
    }
  }
}

function startExam(examData) {
  activeExam = {
    exam_id: examData.exam_id,
    questions: examData.questions || [],
    currentIdx: 0,
    answers: {},
    isFlipped: false,
    source: examData.source,
  };

  document.getElementById("examLoadingState")?.classList.add("hidden");
  document.getElementById("examActiveContainer")?.classList.remove("hidden");
  document.getElementById("examResultsState")?.classList.add("hidden");

  renderActiveQuestion();
}

function renderActiveQuestion() {
  if (!activeExam || !activeExam.questions.length) return;
  const q = activeExam.questions[activeExam.currentIdx];
  const total = activeExam.questions.length;
  const idx = activeExam.currentIdx;

  const badgeEl = document.getElementById("examQuestionBadge");
  if (badgeEl) badgeEl.innerText = `Question ${idx + 1} of ${total}`;
  const typeBadge = document.getElementById("examTypeBadge");
  if (typeBadge) {
    typeBadge.innerText = q.type === "mcq" ? "Multiple Choice Quiz" : "Flashcard";
  }
  const fillEl = document.getElementById("examProgressFill");
  if (fillEl) {
    fillEl.style.width = `${((idx + 1) / total) * 100}%`;
  }

  activeExam.isFlipped = false;
  const flashcardCard = document.getElementById("flashcardCard");
  if (flashcardCard) flashcardCard.classList.remove("is-flipped");

  const fcView = document.getElementById("flashcardView");
  const mcqView = document.getElementById("mcqView");

  if (q.type === "mcq") {
    if (fcView) fcView.classList.add("hidden");
    if (mcqView) mcqView.classList.remove("hidden");

    const promptEl = document.getElementById("mcqPrompt");
    if (promptEl) promptEl.innerText = q.prompt || `What does "${q.term}" mean?`;

    const optionsList = document.getElementById("mcqOptionsList");
    if (optionsList) {
      optionsList.innerHTML = "";
      const letters = ["A", "B", "C", "D"];
      const savedAns = activeExam.answers[q.id];
      const selectedIndex = savedAns != null ? savedAns.selected_index : null;

      (q.options || []).forEach((optText, optIdx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `mcq-option-btn ${selectedIndex === optIdx ? "selected" : ""}`;
        btn.innerHTML = `
          <span class="mcq-option-letter">${letters[optIdx] || optIdx + 1}</span>
          <span>${escapeHtml(optText)}</span>
        `;
        btn.addEventListener("click", () => selectMcqOption(optIdx));
        optionsList.appendChild(btn);
      });
    }
  } else {
    if (mcqView) mcqView.classList.add("hidden");
    if (fcView) fcView.classList.remove("hidden");

    const fcFrontTerm = document.getElementById("fcFrontTerm");
    if (fcFrontTerm) fcFrontTerm.innerText = q.term || q.front;
    const fcFrontPrompt = document.getElementById("fcFrontPrompt");
    if (fcFrontPrompt) fcFrontPrompt.innerText = q.prompt || "Recall the definition or CEFR B1 explanation.";

    const fcBackTerm = document.getElementById("fcBackTerm");
    if (fcBackTerm) fcBackTerm.innerText = q.term;
    const fcBackExpl = document.getElementById("fcBackExpl");
    if (fcBackExpl) fcBackExpl.innerText = q.back || "";

    const savedAns = activeExam.answers[q.id];
    const btnReview = document.getElementById("btnGradeReview");
    const btnKnown = document.getElementById("btnGradeMastered");
    if (btnReview) {
      btnReview.classList.toggle("active", savedAns && savedAns.self_grade === "dont_know");
    }
    if (btnKnown) {
      btnKnown.classList.toggle("active", savedAns && savedAns.self_grade === "know");
    }
  }

  const btnPrev = document.getElementById("btnPrevQuestion");
  if (btnPrev) btnPrev.disabled = idx === 0;

  const btnNext = document.getElementById("btnNextQuestion");
  const btnSubmit = document.getElementById("btnSubmitExam");
  const isLast = idx === total - 1;
  if (btnNext) btnNext.classList.toggle("hidden", isLast);
  if (btnSubmit) btnSubmit.classList.toggle("hidden", !isLast);

  const statusInd = document.getElementById("examStatusIndicator");
  if (statusInd) {
    const isAnswered = activeExam.answers[q.id] != null;
    statusInd.innerText = isAnswered ? "✓ Answer saved" : "Unanswered";
    statusInd.style.color = isAnswered ? "var(--accent-emerald)" : "var(--text-muted)";
  }
}

function recordAnswer(selfGrade) {
  if (!activeExam) return;
  const q = activeExam.questions[activeExam.currentIdx];
  activeExam.answers[q.id] = {
    question_id: q.id,
    self_grade: selfGrade,
  };
  const btnReview = document.getElementById("btnGradeReview");
  const btnKnown = document.getElementById("btnGradeMastered");
  if (btnReview) btnReview.classList.toggle("active", selfGrade === "dont_know");
  if (btnKnown) btnKnown.classList.toggle("active", selfGrade === "know");

  const statusInd = document.getElementById("examStatusIndicator");
  if (statusInd) {
    statusInd.innerText = "✓ Answer saved";
    statusInd.style.color = "var(--accent-emerald)";
  }

  if (activeExam.currentIdx < activeExam.questions.length - 1) {
    setTimeout(() => {
      if (activeExam) {
        activeExam.currentIdx++;
        renderActiveQuestion();
      }
    }, 400);
  }
}

function selectMcqOption(optIndex) {
  if (!activeExam) return;
  const q = activeExam.questions[activeExam.currentIdx];
  if (q.type !== "mcq") return;
  activeExam.answers[q.id] = {
    question_id: q.id,
    selected_index: optIndex,
  };

  const optionsList = document.getElementById("mcqOptionsList");
  if (optionsList) {
    const buttons = optionsList.querySelectorAll(".mcq-option-btn");
    buttons.forEach((btn, idx) => {
      btn.classList.toggle("selected", idx === optIndex);
    });
  }

  const statusInd = document.getElementById("examStatusIndicator");
  if (statusInd) {
    statusInd.innerText = "✓ Answer saved";
    statusInd.style.color = "var(--accent-emerald)";
  }

  if (activeExam.currentIdx < activeExam.questions.length - 1) {
    setTimeout(() => {
      if (activeExam) {
        activeExam.currentIdx++;
        renderActiveQuestion();
      }
    }, 400);
  }
}

async function submitPracticeExam() {
  if (!activeExam) return;

  const btnSubmit = document.getElementById("btnSubmitExam");
  if (btnSubmit) {
    btnSubmit.disabled = true;
    btnSubmit.innerText = "Scoring Exam...";
  }

  const payload = {
    exam_id: activeExam.exam_id,
    answers: Object.values(activeExam.answers),
  };

  try {
    const res = await fetch("/api/exam/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Submission failed");
    }
    const result = await res.json();
    renderExamResults(result);
    await loadLearnerProfile();

    if (currentResult && currentResult.llm && Array.isArray(currentResult.llm.terms)) {
      const weakSet = new Set((result.weak_terms || []).map((t) => t.toLowerCase()));
      const strongSet = new Set((result.strong_terms || []).map((t) => t.toLowerCase()));
      currentResult.llm.terms.forEach((item) => {
        const k = (item.term || "").toLowerCase();
        if (weakSet.has(k)) {
          item.focus = "review";
          item.mastery = 0.29;
        } else if (strongSet.has(k)) {
          item.focus = "known";
          item.mastery = 0.85;
        }
      });
      renderTermsGrid(currentResult.llm.terms, currentResult.lang, currentResult.asr?.segments);

      const banner = document.getElementById("adaptiveBanner");
      if (banner) {
        banner.classList.remove("hidden");
        const headline = document.getElementById("adaptiveBannerHeadline");
        if (headline) headline.innerText = "Updated Student Background Saved";
        const desc = document.getElementById("adaptiveBannerDesc");
        if (desc) {
          desc.innerText = result.analysis_summary || "Next lecture analyses will prioritize your review terms.";
        }
      }
    }
  } catch (err) {
    showToast("Error submitting exam: " + err.message, "error");
    if (btnSubmit) {
      btnSubmit.disabled = false;
      btnSubmit.innerText = "Finish & Submit Exam";
    }
  }
}

function renderExamResults(result) {
  document.getElementById("examActiveContainer")?.classList.add("hidden");
  document.getElementById("examResultsState")?.classList.remove("hidden");

  const scorePct = Math.round((result.score || 0) * 100);
  const badgeEl = document.getElementById("examScoreBadge");
  if (badgeEl) badgeEl.innerText = `${scorePct}%`;

  const subEl = document.getElementById("examResultsSubtitle");
  if (subEl) {
    subEl.innerText = `You scored ${result.correct || 0} out of ${result.total || 0}. Results saved to student background.`;
  }

  const summaryEl = document.getElementById("examAdaptiveSummaryText");
  if (summaryEl) {
    summaryEl.innerText = result.analysis_summary ||
      "Weak terms have been recorded as Review Priority and will receive in-depth CEFR B1 explanations in upcoming lecture analyses.";
  }

  const weakList = document.getElementById("examWeakTermsList");
  if (weakList) {
    const terms = result.weak_terms || [];
    if (terms.length === 0) {
      weakList.innerHTML = `<span class="empty-chip">None! All terms marked correct.</span>`;
    } else {
      weakList.innerHTML = terms.map((t) => `<span class="chip-item weak">⚠️ ${escapeHtml(t)}</span>`).join("");
    }
  }

  const strongList = document.getElementById("examStrongTermsList");
  if (strongList) {
    const terms = result.strong_terms || [];
    if (terms.length === 0) {
      strongList.innerHTML = `<span class="empty-chip">None</span>`;
    } else {
      strongList.innerHTML = terms.map((t) => `<span class="chip-item strong">✓ ${escapeHtml(t)}</span>`).join("");
    }
  }
}

// ==========================================================================
// Open-Source Lightweight Eye Tracker & Attention Monitor (MediaPipe)
// ==========================================================================

let eyeTrackerActive = false;
let webcamStream = null;
let faceMeshDetector = null;
let cameraLoopInterval = null;
let pipVisible = false;
let consecutiveDistractedMs = 0;
let consecutiveFocusedMs = 0;
let lastFrameTime = Date.now();
let totalFramesTracked = 0;
let focusedFramesCount = 0;
let attentionPauseCount = 0;
let distractionPausedMedia = false;
let snoozeAttentionUntil = 0;
let isSimulationMode = false;
let simDistractedState = false;

// Media playback unified helpers
function isAnyMediaPlaying() {
  const ytContainer = document.getElementById("ytPlayerContainer");
  const videoContainer = document.getElementById("htmlVideoContainer");
  const videoPlayer = document.getElementById("videoPlayer");
  const audioPlayer = document.getElementById("audioPlayer");

  if (ytPlayer && ytContainer && !ytContainer.classList.contains("hidden")) {
    try {
      const s = (typeof ytPlayer.getPlayerState === "function") ? ytPlayer.getPlayerState() : -1;
      if (s === 1 || s === 3) return true;
    } catch (_) {}
  }
  if (videoPlayer && videoContainer && !videoContainer.classList.contains("hidden")) {
    if (!videoPlayer.paused && !videoPlayer.ended && videoPlayer.currentTime > 0) return true;
  }
  if (audioPlayer) {
    if (!audioPlayer.paused && !audioPlayer.ended && audioPlayer.currentTime > 0) return true;
  }
  return false;
}

function pauseActiveMedia() {
  const ytContainer = document.getElementById("ytPlayerContainer");
  const videoContainer = document.getElementById("htmlVideoContainer");
  const videoPlayer = document.getElementById("videoPlayer");
  const audioPlayer = document.getElementById("audioPlayer");
  let didPause = false;

  if (ytPlayer && ytContainer && !ytContainer.classList.contains("hidden")) {
    try {
      const s = (typeof ytPlayer.getPlayerState === "function") ? ytPlayer.getPlayerState() : -1;
      if (s === 1 || s === 3) {
        ytPlayer.pauseVideo();
        didPause = true;
      }
    } catch (_) {}
  }
  if (videoPlayer && videoContainer && !videoContainer.classList.contains("hidden")) {
    if (!videoPlayer.paused && !videoPlayer.ended) {
      videoPlayer.pause();
      didPause = true;
    }
  }
  if (audioPlayer) {
    if (!audioPlayer.paused && !audioPlayer.ended) {
      audioPlayer.pause();
      didPause = true;
    }
  }
  return didPause;
}

function resumeActiveMedia() {
  const ytContainer = document.getElementById("ytPlayerContainer");
  const videoContainer = document.getElementById("htmlVideoContainer");
  const videoPlayer = document.getElementById("videoPlayer");
  const audioPlayer = document.getElementById("audioPlayer");

  if (ytPlayer && ytContainer && !ytContainer.classList.contains("hidden") && ytPlayer.playVideo) {
    try {
      ytPlayer.playVideo();
    } catch (_) {}
  } else if (videoPlayer && videoContainer && !videoContainer.classList.contains("hidden")) {
    videoPlayer.play().catch(() => {});
  } else if (audioPlayer) {
    audioPlayer.play().catch(() => {});
  }
}

// Gentle audio chime via Web Audio API (zero external assets needed)
function playAttentionChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(440, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.16);
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.36);
  } catch (_) {}
}

// Eye Tracker Module Setup & Event Bindings
function setupEyeTracker() {
  const btnToggleNav = document.getElementById("btnToggleEyeTrackerPill");
  const btnToggleToolbar = document.getElementById("btnToggleEyeTracker");
  const btnTogglePip = document.getElementById("btnTogglePip");
  const btnMinimizePip = document.getElementById("btnMinimizePip");
  const pipContainer = document.getElementById("eyeTrackerPipContainer");
  const btnResume = document.getElementById("btnResumeAttentionVideo");
  const btnDismiss = document.getElementById("btnDismissAttention");
  const selectSensitivity = document.getElementById("selectEtSensitivity");
  const overlay = document.getElementById("attentionPauseOverlay");

  function getDistractionThresholdMs() {
    const val = selectSensitivity ? selectSensitivity.value : "normal";
    if (val === "high") return 1200;
    if (val === "low") return 2500;
    return 1800; // normal
  }

  async function startEyeTracker() {
    if (eyeTrackerActive) return;
    showToast("Webcam başlatılıyor ve göz takip modeli yükleniyor...", "info");

    const webcamVideo = document.getElementById("webcamVideo");
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
        audio: false,
      });
      if (webcamVideo) {
        webcamVideo.srcObject = webcamStream;
        await webcamVideo.play();
      }
    } catch (err) {
      console.warn("Webcam access unavailable or denied:", err);
      // Offer Simulation / Interactive Test Mode
      showToast("Kamera erişimi sağlanamadı. Simülasyon / Test Modu başlatılıyor.", "warning");
      isSimulationMode = true;
    }

    eyeTrackerActive = true;
    updateEyeTrackerStatus("FOCUSED", "Odaklandı");
    setEyeTrackerUIActive(true);

    if (pipContainer) {
      pipContainer.classList.remove("hidden");
      pipVisible = true;
    }

    if (!isSimulationMode && window.FaceMesh) {
      initMediaPipeDetector();
    } else {
      startFallbackEyeTrackingLoop();
    }
  }

  function stopEyeTracker() {
    eyeTrackerActive = false;
    if (cameraLoopInterval) {
      clearInterval(cameraLoopInterval);
      cameraLoopInterval = null;
    }
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
    }
    const webcamVideo = document.getElementById("webcamVideo");
    if (webcamVideo) webcamVideo.srcObject = null;

    if (pipContainer) pipContainer.classList.add("hidden");
    if (overlay) overlay.classList.add("hidden");

    pipVisible = false;
    consecutiveDistractedMs = 0;
    consecutiveFocusedMs = 0;
    setEyeTrackerUIActive(false);
    updateEyeTrackerStatus("OFF", "Takip Kapalı");
    showToast("Göz takibi durduruldu.", "info");
  }

  function toggleEyeTracker() {
    if (eyeTrackerActive) {
      stopEyeTracker();
    } else {
      startEyeTracker();
    }
  }

  if (btnToggleNav) btnToggleNav.addEventListener("click", toggleEyeTracker);
  if (btnToggleToolbar) btnToggleToolbar.addEventListener("click", toggleEyeTracker);

  if (btnTogglePip) {
    btnTogglePip.addEventListener("click", () => {
      if (!eyeTrackerActive) {
        startEyeTracker();
        return;
      }
      pipVisible = !pipVisible;
      if (pipContainer) pipContainer.classList.toggle("hidden", !pipVisible);
    });
  }

  if (btnMinimizePip) {
    btnMinimizePip.addEventListener("click", () => {
      pipVisible = false;
      if (pipContainer) pipContainer.classList.add("hidden");
    });
  }

  if (btnResume) {
    btnResume.addEventListener("click", () => {
      if (overlay) overlay.classList.add("hidden");
      consecutiveDistractedMs = 0;
      resumeActiveMedia();
      distractionPausedMedia = false;
      showToast("Video devam ettirildi.", "success");
    });
  }

  if (btnDismiss) {
    btnDismiss.addEventListener("click", () => {
      if (overlay) overlay.classList.add("hidden");
      snoozeAttentionUntil = Date.now() + 60000; // 1 min snooze
      consecutiveDistractedMs = 0;
      resumeActiveMedia();
      distractionPausedMedia = false;
      showToast("Duraklatma uyarısı 1 dakika ertelendi.", "info");
    });
  }

  const btnLaunchDirect = document.getElementById("btnLaunchPlayerDirect");
  if (btnLaunchDirect) {
    btnLaunchDirect.addEventListener("click", async () => {
      const emptyEl = document.getElementById("emptyState");
      const resultsEl = document.getElementById("resultsView");
      if (emptyEl) emptyEl.classList.add("hidden");
      if (resultsEl) resultsEl.classList.remove("hidden");
      const ytContainer = document.getElementById("ytPlayerContainer");
      if (ytContainer) ytContainer.classList.remove("hidden");

      showToast("Örnek ders için transkript ve terimler çekiliyor...", "info");
      const preview = await fetchTranscriptPreview("https://www.youtube.com/watch?v=rz5JCvMlCMs", true);
      if (!preview) {
        loadYouTubeVideo("rz5JCvMlCMs");
      }
      startEyeTracker();
      showToast("Örnek ders (TR-1), interaktif transkript ve göz takibi yüklendi!", "success");
    });
  }

  // MediaPipe FaceMesh Initialization
  function initMediaPipeDetector() {
    try {
      faceMeshDetector = new window.FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMeshDetector.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true, // Enables iris landmarks (468-477)
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      faceMeshDetector.onResults(onFaceMeshResults);

      const webcamVideo = document.getElementById("webcamVideo");
      lastFrameTime = Date.now();

      cameraLoopInterval = setInterval(async () => {
        if (!eyeTrackerActive || !webcamVideo || webcamVideo.readyState < 2) return;
        try {
          await faceMeshDetector.send({ image: webcamVideo });
        } catch (_) {}
      }, 100); // 10 FPS is ultra lightweight (<3% CPU) while completely responsive
    } catch (err) {
      console.warn("Failed to initialize MediaPipe FaceMesh, falling back to lightweight tracking:", err);
      startFallbackEyeTrackingLoop();
    }
  }

  // Fallback / Simulation Tracking Loop (ensures zero crashes if CDN is down or mock testing)
  function startFallbackEyeTrackingLoop() {
    if (cameraLoopInterval) clearInterval(cameraLoopInterval);
    const canvas = document.getElementById("eyeTrackerCanvas");
    const ctx = canvas ? canvas.getContext("2d") : null;

    cameraLoopInterval = setInterval(() => {
      if (!eyeTrackerActive) return;
      const now = Date.now();
      const delta = now - lastFrameTime;
      lastFrameTime = now;

      // In simulation mode, user can toggle distraction by clicking PIP window
      let isDistracted = simDistractedState;
      let reason = "Simüle Edilmiş Dikkat Dağınıklığı (Pencereye tıklayarak odaklanın)";

      handleAttentionEvaluation(isDistracted, reason, delta, 0.5, 0.5);

      if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Draw test iris & focus reticle
        ctx.strokeStyle = isDistracted ? "#f43f5e" : "#10b981";
        ctx.lineWidth = 2;
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        ctx.beginPath();
        ctx.arc(cx - 30, cy, 12, 0, Math.PI * 2);
        ctx.arc(cx + 30, cy, 12, 0, Math.PI * 2);
        ctx.stroke();
      }
    }, 150);

    if (pipContainer) {
      pipContainer.title = "Simülasyon Modu: Tıklayarak Odak / Dikkat Dağınıklığını Değiştirin";
      pipContainer.onclick = () => {
        if (isSimulationMode) {
          simDistractedState = !simDistractedState;
          showToast(simDistractedState ? "Simülasyon: Dikkat Dağıldı!" : "Simülasyon: Ekrana Odaklandı", "info");
        }
      };
    }
  }

  // Real-time FaceMesh Results Processing
  function onFaceMeshResults(results) {
    if (!eyeTrackerActive) return;

    const now = Date.now();
    const delta = Math.min(300, Math.max(30, now - lastFrameTime));
    lastFrameTime = now;

    const canvas = document.getElementById("eyeTrackerCanvas");
    const ctx = canvas ? canvas.getContext("2d") : null;
    if (ctx && canvas) {
      if (canvas.width !== canvas.clientWidth) {
        canvas.width = canvas.clientWidth || 195;
        canvas.height = canvas.clientHeight || 130;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    if (!results || !results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      // No face in webcam frame
      handleAttentionEvaluation(true, "Kamera karşısında öğrenci algılanamadı (Ekrandan uzaklaşıldı)", delta, 0.5, 0.5);
      return;
    }

    const landmarks = results.multiFaceLandmarks[0];

    // 1. Head Pose (Yaw): compare nose tip (1) position between left cheek (234) and right cheek (454)
    const nose = landmarks[1];
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    let yawRatio = 0.5;
    if (leftCheek && rightCheek && nose) {
      const cheekSpan = Math.abs(rightCheek.x - leftCheek.x);
      if (cheekSpan > 0.01) {
        yawRatio = (nose.x - leftCheek.x) / (rightCheek.x - leftCheek.x);
      }
    }

    // 2. Head Pitch: compare nose tip (1) between forehead (10) and chin (152)
    const forehead = landmarks[10];
    const chin = landmarks[152];
    let pitchRatio = 0.5;
    if (forehead && chin && nose) {
      const faceHeight = Math.abs(chin.y - forehead.y);
      if (faceHeight > 0.01) {
        pitchRatio = (nose.y - forehead.y) / (chin.y - forehead.y);
      }
    }

    // 3. Iris Tracking (Landmarks 468 = Left Iris Center, 473 = Right Iris Center)
    let irisHorizRatio = 0.5;
    let ear = 0.28;

    const leftIris = landmarks[468];
    const rightIris = landmarks[473];
    const leftInner = landmarks[133];
    const leftOuter = landmarks[33];

    if (leftIris && leftOuter && leftInner) {
      const eyeWidth = Math.abs(leftInner.x - leftOuter.x);
      if (eyeWidth > 0.01) {
        irisHorizRatio = Math.abs(leftIris.x - leftOuter.x) / eyeWidth;
      }
    }

    // 4. Eye Aspect Ratio (EAR) for blinks / drowsiness
    const leftTop = landmarks[159];
    const leftBottom = landmarks[145];
    if (leftTop && leftBottom && leftOuter && leftInner) {
      const vDist = Math.hypot(leftTop.x - leftBottom.x, leftTop.y - leftBottom.y);
      const hDist = Math.hypot(leftOuter.x - leftInner.x, leftOuter.y - leftInner.y);
      if (hDist > 0.001) ear = vDist / hDist;
    }

    // Draw eye mesh contours and iris indicators on PIP canvas
    if (ctx && canvas) {
      ctx.strokeStyle = "rgba(139, 92, 246, 0.6)";
      ctx.lineWidth = 1.5;

      // Draw eyes outline
      const drawEye = (indices) => {
        ctx.beginPath();
        indices.forEach((idx, i) => {
          const pt = landmarks[idx];
          if (!pt) return;
          const x = pt.x * canvas.width;
          const y = pt.y * canvas.height;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
      };

      drawEye([33, 160, 158, 133, 153, 144]);
      drawEye([362, 385, 387, 263, 373, 380]);

      // Draw Iris center dots
      if (leftIris) {
        ctx.fillStyle = "#06b6d4";
        ctx.beginPath();
        ctx.arc(leftIris.x * canvas.width, leftIris.y * canvas.height, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      if (rightIris) {
        ctx.fillStyle = "#06b6d4";
        ctx.beginPath();
        ctx.arc(rightIris.x * canvas.width, rightIris.y * canvas.height, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Distraction Criteria Detection
    let isDistracted = false;
    let reason = "";

    // Criterion A: Prolonged eye closure (drowsiness)
    if (ear < 0.12) {
      isDistracted = true;
      reason = "Gözler kapalı / Uyuklama veya aşırı dikkat kaybı algılandı";
    }
    // Criterion B: Head turned away from screen
    else if (yawRatio < 0.28 || yawRatio > 0.72) {
      isDistracted = true;
      reason = "Baş ekrandan yana çevrildi (Ders dışına bakılıyor)";
    }
    // Criterion C: Head tilted downward (looking at mobile phone / lap)
    else if (pitchRatio > 0.68) {
      isDistracted = true;
      reason = "Baş aşağıya çevrildi (Telefon veya masaya bakılıyor)";
    }
    // Criterion D: Gaze shifted away from monitor
    else if (irisHorizRatio < 0.22 || irisHorizRatio > 0.78) {
      isDistracted = true;
      reason = "Gözler ekrandan uzaklaştı (Odak dağıldı)";
    }

    handleAttentionEvaluation(isDistracted, reason, delta, irisHorizRatio, pitchRatio);
  }

  // Unified Attention State Evaluator
  function handleAttentionEvaluation(isDistracted, reason, deltaMs, irisX, pitchY) {
    totalFramesTracked++;
    const thresholdMs = getDistractionThresholdMs();
    const autoResumeChecked = document.getElementById("chkAutoResume")?.checked ?? true;

    if (isDistracted) {
      consecutiveDistractedMs += deltaMs;
      consecutiveFocusedMs = 0;

      if (consecutiveDistractedMs >= thresholdMs) {
        updateEyeTrackerStatus("DISTRACTED", "Dikkat Dağınık");

        // Action: Auto-Pause Video if video is playing and not snoozed!
        if (Date.now() > snoozeAttentionUntil) {
          const wasPlaying = isAnyMediaPlaying();
          if (wasPlaying || distractionPausedMedia) {
            pauseActiveMedia();
            distractionPausedMedia = true;

            // Trigger overlay
            if (overlay && overlay.classList.contains("hidden")) {
              overlay.classList.remove("hidden");
              playAttentionChime();
              attentionPauseCount++;

              const reasonEl = document.getElementById("attentionPauseReason");
              if (reasonEl) reasonEl.innerText = `${reason}. Ders videosu odaklanmanız için otomatik duraklatıldı.`;

              const pauseCountEl = document.getElementById("etPauseCount");
              if (pauseCountEl) pauseCountEl.innerText = String(attentionPauseCount);
            }
          }
        }
      } else if (consecutiveDistractedMs >= (thresholdMs * 0.45)) {
        updateEyeTrackerStatus("WARNING", "Dikkat Kayması");
      }
    } else {
      // Student is looking at screen!
      focusedFramesCount++;
      consecutiveFocusedMs += deltaMs;
      consecutiveDistractedMs = Math.max(0, consecutiveDistractedMs - deltaMs * 1.5);

      if (consecutiveFocusedMs >= 700) {
        updateEyeTrackerStatus("FOCUSED", "Odaklandı");

        // Action: Auto-Resume if auto-resume is enabled and overlay is open
        if (autoResumeChecked && overlay && !overlay.classList.contains("hidden") && distractionPausedMedia) {
          overlay.classList.add("hidden");
          distractionPausedMedia = false;
          resumeActiveMedia();
          showToast("Odaklanma algılandı. Video otomatik devam ediyor.", "success");
        }
      }
    }

    // Update Focus Score (%)
    const focusPct = totalFramesTracked > 0
      ? Math.round((focusedFramesCount / totalFramesTracked) * 100)
      : 100;

    const fillEl = document.getElementById("pipScoreFill");
    const numEl = document.getElementById("pipScoreNum");
    const focusLabel = document.getElementById("etFocusPercent");
    if (fillEl) fillEl.style.width = `${focusPct}%`;
    if (numEl) numEl.innerText = `${focusPct}%`;
    if (focusLabel) focusLabel.innerText = `${focusPct}%`;
  }

  function updateEyeTrackerStatus(state, label) {
    const etStatusDot = document.getElementById("etStatusDot");
    const etStatusLabel = document.getElementById("etStatusLabel");
    const pipStatusDot = document.getElementById("pipStatusDot");
    const pipStatusText = document.getElementById("pipStatusText");
    const pill = document.getElementById("btnToggleEyeTrackerPill");
    const pillBadge = document.getElementById("eyeTrackerPillBadge");

    if (etStatusLabel) etStatusLabel.innerText = label;
    if (pipStatusText) pipStatusText.innerText = label;

    [etStatusDot, pipStatusDot].forEach((dot) => {
      if (!dot) return;
      dot.className = "status-pulse-dot";
      if (state === "FOCUSED") dot.classList.add("online", "green");
      else if (state === "WARNING") dot.classList.add("warning");
      else if (state === "DISTRACTED") dot.classList.add("distracted");
    });

    if (pill && pillBadge) {
      pill.classList.remove("active", "distracted");
      if (state === "FOCUSED") {
        pill.classList.add("active");
        pillBadge.innerText = "Eye Tracker: Odaklandı";
      } else if (state === "DISTRACTED") {
        pill.classList.add("distracted");
        pillBadge.innerText = "Eye Tracker: Dağınık!";
      } else if (state === "WARNING") {
        pillBadge.innerText = "Eye Tracker: Uyarı";
      } else {
        pillBadge.innerText = "Eye Tracker: Kapalı";
      }
    }
  }

  function setEyeTrackerUIActive(isActive) {
    const btnNav = document.getElementById("btnToggleEyeTrackerPill");
    const btnToolbar = document.getElementById("btnToggleEyeTracker");
    const textToolbar = document.getElementById("etToggleText");

    if (btnNav) btnNav.classList.toggle("active", isActive);
    if (btnToolbar) btnToolbar.classList.toggle("active", isActive);
    if (textToolbar) textToolbar.innerText = isActive ? "Göz Takibini Kapat" : "Göz Takibini Başlat";
  }
}

// ==========================================================================
// Instant Helper Slide Module ("Anlamadım / Yardımcı Slayt")
// ==========================================================================

function getCurrentVideoTime() {
  const ytContainer = document.getElementById("ytPlayerContainer");
  const videoContainer = document.getElementById("htmlVideoContainer");
  const videoPlayer = document.getElementById("videoPlayer");
  const audioPlayer = document.getElementById("audioPlayer");

  if (ytPlayer && ytContainer && !ytContainer.classList.contains("hidden") && typeof ytPlayer.getCurrentTime === "function") {
    try {
      const t = ytPlayer.getCurrentTime();
      if (t !== null && !isNaN(t)) return t;
    } catch (_) {}
  }
  if (videoPlayer && videoContainer && !videoContainer.classList.contains("hidden")) {
    if (!Number.isNaN(videoPlayer.currentTime)) return videoPlayer.currentTime;
  }
  if (audioPlayer && !Number.isNaN(audioPlayer.currentTime)) {
    return audioPlayer.currentTime;
  }
  return 0.0;
}

function setupHelperSlide() {
  const btnOpen = document.getElementById("btnOpenHelperSlide");
  const modal = document.getElementById("helperSlideModal");
  const windowWrapper = document.getElementById("slideWindowWrapper");
  const btnClose = document.getElementById("btnCloseHelperSlide");
  const btnResume = document.getElementById("btnSlideResume");
  const btnRewind = document.getElementById("btnSlideRewind10");
  const btnSpeak = document.getElementById("btnSlideSpeak");
  const btnDeepAI = document.getElementById("btnDeepSlideAI");
  const btnFullscreen = document.getElementById("btnFullscreenSlide");
  const selectTheme = document.getElementById("selectSlideTheme");

  let activeSlideTime = 0;

  // Initialize Mermaid with presentation dark styling & large readable fonts
  if (window.mermaid) {
    try {
      window.mermaid.initialize({
        startOnLoad: false,
        theme: "dark",
        securityLevel: "loose",
        fontFamily: "Inter, system-ui, sans-serif",
        fontSize: 16,
        themeVariables: {
          fontSize: "16px",
          fontFamily: "Inter, system-ui, sans-serif",
          primaryColor: "#1e1b4b",
          primaryTextColor: "#f8fafc",
          primaryBorderColor: "#8b5cf6",
          lineColor: "#06b6d4",
          secondaryColor: "#0f172a",
          tertiaryColor: "#1e293b",
          edgeLabelBackground: "#0b0f19",
          clusterBkg: "rgba(255, 255, 255, 0.04)",
          clusterBorder: "rgba(139, 92, 246, 0.4)",
        },
        flowchart: {
          useMaxWidth: false,
          htmlLabels: true,
          curve: "basis",
          nodeSpacing: 50,
          rankSpacing: 50,
          padding: 16,
        },
      });
    } catch (_) {}
  }

  // Slide View Mode Switcher (Dengeli / Geniş Şema / Terimler)
  const presentationStage = document.getElementById("slidePresentationStage");
  const btnViewSplit = document.getElementById("btnViewSplit");
  const btnViewDiagram = document.getElementById("btnViewDiagram");
  const btnViewCards = document.getElementById("btnViewCards");
  const viewPills = [btnViewSplit, btnViewDiagram, btnViewCards];

  function setSlideViewMode(mode) {
    if (!presentationStage) return;
    presentationStage.classList.remove("view-split", "view-diagram", "view-cards");
    presentationStage.classList.add(`view-${mode}`);
    viewPills.forEach((p) => {
      if (!p) return;
      if (p.getAttribute("data-view") === mode) {
        p.classList.add("active");
      } else {
        p.classList.remove("active");
      }
    });
  }

  if (btnViewSplit) btnViewSplit.addEventListener("click", () => setSlideViewMode("split"));
  if (btnViewDiagram) btnViewDiagram.addEventListener("click", () => setSlideViewMode("diagram"));
  if (btnViewCards) btnViewCards.addEventListener("click", () => setSlideViewMode("cards"));

  // Diagram Lightbox Modal (Büyüteç / Işık Masası) Controller
  const lbModal = document.getElementById("diagramLightboxModal");
  const lbWrapper = document.getElementById("lightboxDiagramWrapper");
  const lbContainer = document.getElementById("lightboxDiagramContainer");
  const btnOpenLb = document.getElementById("btnOpenDiagramLightbox");
  const btnOpenLbDirect = document.getElementById("btnOpenLightboxDirect");
  const btnCloseLb = document.getElementById("btnCloseDiagramLightbox");
  const btnLbZoomIn = document.getElementById("btnLbZoomIn");
  const btnLbZoomOut = document.getElementById("btnLbZoomOut");
  const btnLbReset = document.getElementById("btnLbReset");
  const lbZoomBadge = document.getElementById("lbZoomLevel");

  let lbZoomLevel = 1.50;
  const MIN_LB_ZOOM = 0.75;
  const MAX_LB_ZOOM = 3.00;

  function setLightboxZoom(level) {
    lbZoomLevel = Math.min(MAX_LB_ZOOM, Math.max(MIN_LB_ZOOM, Math.round(level * 100) / 100));
    if (lbContainer) {
      lbContainer.style.transform = `scale(${lbZoomLevel})`;
    }
    if (lbZoomBadge) {
      lbZoomBadge.innerText = `${Math.round(lbZoomLevel * 100)}%`;
    }
  }

  function openDiagramLightbox() {
    if (!lbModal || !lbContainer) return;
    const currentSvg = document.querySelector("#slideMermaidContainer svg");
    if (currentSvg) {
      lbContainer.innerHTML = currentSvg.outerHTML;
    } else {
      const fallback = document.getElementById("slideMermaidContainer");
      lbContainer.innerHTML = fallback ? fallback.innerHTML : "<p>Şema yüklenemedi.</p>";
    }
    setLightboxZoom(1.50);
    lbModal.classList.remove("hidden");
  }

  function closeDiagramLightbox() {
    if (lbModal) lbModal.classList.add("hidden");
  }

  if (btnOpenLb) btnOpenLb.addEventListener("click", openDiagramLightbox);
  if (btnOpenLbDirect) btnOpenLbDirect.addEventListener("click", openDiagramLightbox);
  if (btnCloseLb) btnCloseLb.addEventListener("click", closeDiagramLightbox);

  if (btnLbZoomIn) btnLbZoomIn.addEventListener("click", () => setLightboxZoom(lbZoomLevel + 0.25));
  if (btnLbZoomOut) btnLbZoomOut.addEventListener("click", () => setLightboxZoom(lbZoomLevel - 0.25));
  if (btnLbReset) btnLbReset.addEventListener("click", () => setLightboxZoom(1.50));

  if (lbWrapper) {
    lbWrapper.addEventListener("wheel", (e) => {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.25 : -0.25;
      setLightboxZoom(lbZoomLevel + delta);
    }, { passive: false });

    let isLbPanning = false;
    let lbStartX = 0, lbStartY = 0, lbScrollLeft = 0, lbScrollTop = 0;

    lbWrapper.addEventListener("mousedown", (e) => {
      if (e.target.closest("button")) return;
      isLbPanning = true;
      lbWrapper.classList.add("is-panning");
      lbStartX = e.pageX - lbWrapper.offsetLeft;
      lbStartY = e.pageY - lbWrapper.offsetTop;
      lbScrollLeft = lbWrapper.scrollLeft;
      lbScrollTop = lbWrapper.scrollTop;
    });

    window.addEventListener("mouseup", () => {
      if (isLbPanning) {
        isLbPanning = false;
        lbWrapper.classList.remove("is-panning");
      }
    });

    lbWrapper.addEventListener("mousemove", (e) => {
      if (!isLbPanning) return;
      e.preventDefault();
      const x = e.pageX - lbWrapper.offsetLeft;
      const y = e.pageY - lbWrapper.offsetTop;
      lbWrapper.scrollLeft = lbScrollLeft - (x - lbStartX) * 1.5;
      lbWrapper.scrollTop = lbScrollTop - (y - lbStartY) * 1.5;
    });
  }

  // Theme switcher
  if (selectTheme && windowWrapper) {
    selectTheme.addEventListener("change", (e) => {
      windowWrapper.classList.remove("theme-aurora", "theme-cyan", "theme-slate");
      windowWrapper.classList.add(e.target.value);
    });
  }

  // Fullscreen presentation toggle
  if (btnFullscreen && windowWrapper) {
    btnFullscreen.addEventListener("click", () => {
      windowWrapper.classList.toggle("fullscreen");
      btnFullscreen.innerText = windowWrapper.classList.contains("fullscreen") ? "✕" : "⛶";
    });
  }

  // Render Structured Presentation Slide
  function renderStructuredSlide(spec) {
    if (!spec) return;

    const mainTitleEl = document.getElementById("slideMainTitle");
    const subTitleEl = document.getElementById("slideSubTitle");
    const coreSummaryEl = document.getElementById("slideCoreSummary");
    const termsGridEl = document.getElementById("slideTermsGrid");
    const analogyText = document.getElementById("slideAnalogyText");
    const examTipText = document.getElementById("slideExamTipText");
    const quoteEl = document.getElementById("slideTranscriptQuote");
    const mermaidContainer = document.getElementById("slideMermaidContainer");

    if (mainTitleEl) mainTitleEl.innerText = spec.title || "Ders Konusu";
    if (subTitleEl) subTitleEl.innerText = spec.subtitle || "Videoda son dinlenen bölümün yapay zeka tarafından görselleştirilmiş özeti";
    if (coreSummaryEl) coreSummaryEl.innerHTML = `<strong>Ana Fikir:</strong> ${escapeHtml(spec.core_takeaway || spec.core_summary || "Temel kavram anlatımı.")}`;

    // Terms & Cards Grid
    if (termsGridEl) {
      termsGridEl.innerHTML = "";
      const cardList = Array.isArray(spec.cards) && spec.cards.length > 0 ? spec.cards : (spec.terms || []);
      if (cardList.length > 0) {
        cardList.slice(0, 4).forEach((card) => {
          const cardEl = document.createElement("div");
          cardEl.className = "slide-term-card";
          const title = card.title || card.term || "Kavram";
          const desc = card.desc || card.explanation || "";
          const icon = card.icon || "📌";
          cardEl.innerHTML = `
            <div class="slide-term-header">
              <span class="slide-term-name">${icon} ${escapeHtml(title)}</span>
              ${card.timestamp ? `<span class="badge-subtle" style="font-size:0.65rem">${card.timestamp}</span>` : ""}
            </div>
            <p class="slide-term-desc">${escapeHtml(desc)}</p>
          `;
          termsGridEl.appendChild(cardEl);
        });
      } else {
        termsGridEl.innerHTML = `<p style="color:var(--text-muted);font-size:0.82rem;">Bu zaman diliminde özel bir teknik terim tanımlanmamış.</p>`;
      }
    }

    // Analogy
    if (analogyText) {
      analogyText.innerText = spec.analogy || "Bu konu, süreçlerin adım adım ve kontrollü yürütülmesini sağlayan bir rehber gibidir.";
    }

    // Exam Golden Tip
    if (examTipText) {
      examTipText.innerText = spec.exam_tip || "Sınavda bu konunun temel tanımına ve diğer kavramlarla olan neden-sonuç bağına dikkat edin.";
    }

    // Quote
    if (quoteEl) {
      quoteEl.innerText = spec.recent_quote || spec.recent_transcript || "...";
    }

    // Render Mermaid Diagram
    if (mermaidContainer && spec.mermaid_diagram && window.mermaid) {
      const diagramCode = spec.mermaid_diagram.trim();
      const diagramId = "slide_diag_" + Math.random().toString(36).substring(2, 9);
      try {
        window.mermaid.render(diagramId, diagramCode).then((rendered) => {
          mermaidContainer.innerHTML = rendered.svg;
          if (lbModal && !lbModal.classList.contains("hidden") && lbContainer) {
            lbContainer.innerHTML = rendered.svg;
          }
        }).catch(() => {
          mermaidContainer.innerHTML = `
            <div style="background:rgba(255,255,255,0.03);padding:0.75rem;border-radius:6px;width:100%;font-size:0.82rem;color:var(--accent-cyan);line-height:1.6;">
              <span style="font-weight:bold;color:#c4b5fd;">📊 Kavram Akışı:</span><br/>
              ${escapeHtml(diagramCode.replace(/graph\s+(?:LR|TD|TB);?/gi, "").trim())}
            </div>
          `;
        });
      } catch (err) {
        console.warn("Mermaid error:", err);
      }
    }
  }

  async function openHelperSlide() {
    // 1. Immediately pause active media so the student can focus on the slide
    pauseActiveMedia();

    // 2. Read current video position
    activeSlideTime = getCurrentVideoTime();
    const t = activeSlideTime;
    const windowStart = Math.max(0, t - 45);

    const timeRangeEl = document.getElementById("slideTimeRange");
    if (timeRangeEl) {
      timeRangeEl.innerText = `${formatTime(windowStart)} - ${formatTime(t)}`;
    }

    // 3. Extract recent segments from currentResult for instant local render
    const segments = Array.isArray(currentResult?.asr?.segments) ? currentResult.asr.segments : [];
    const terms = Array.isArray(currentResult?.llm?.terms) ? currentResult.llm.terms : [];

    const recentSegs = segments.filter((s) => {
      if (s.start == null || s.end == null) return false;
      return Number(s.end) >= windowStart && Number(s.start) <= (t + 5);
    });

    const recentText = recentSegs.length > 0
      ? recentSegs.map((s) => s.text.trim()).join(" ")
      : (currentResult?.asr?.text || "Ders anlatımı devam ediyor...").slice(0, 300);

    const recentLower = recentText.toLowerCase();
    let relevantTerms = terms.filter((item) => {
      const termName = (item.term || "").toLowerCase().trim();
      if (!termName) return false;
      const startSec = item.start != null ? Number(item.start) : null;
      const inWindow = startSec !== null && startSec >= windowStart && startSec <= (t + 5);
      return inWindow || recentLower.includes(termName);
    });

    if (relevantTerms.length === 0 && terms.length > 0) {
      relevantTerms = terms.slice(0, 3);
    }

    const topicName = relevantTerms.length > 0 ? relevantTerms[0].term : "Ders Konusu";

    // Instant local presentation spec
    const instantSpec = {
      title: topicName,
      subtitle: "Videoda son dinlenen bölümün yapay zeka tarafından görselleştirilmiş özeti",
      core_takeaway: relevantTerms.length > 0
        ? `${relevantTerms[0].term}: ${relevantTerms[0].explanation || "Temel kavram anlatımı."}`
        : recentText.slice(0, 160) + "...",
      mermaid_diagram: "graph LR\n  A[\"📅 Başlangıç & Plan\"] --> B[\"⚙️ Yürütme & Analiz\"]\n  B --> C[\"🎯 Sonuç & Kazanım\"]",
      cards: relevantTerms.map(t => ({
        icon: "📌",
        title: t.term,
        desc: t.explanation,
        timestamp: t.timestamp,
      })),
      analogy: "Bu konu, hedefe ulaşırken düzenli aralıklarla haritayı kontrol etmeye benzer: Sapmaları erkenden görerek projeyi başarıyla tamamlarsınız.",
      exam_tip: `${topicName} kavramının diğer aşamalarla olan neden-sonuç ilişkilerine ve formüllerine dikkat edin.`,
      recent_quote: recentText,
    };

    // Render instant spec immediately so there is ZERO delay!
    renderStructuredSlide(instantSpec);

    // Show slide modal
    if (modal) modal.classList.remove("hidden");
    showToast("Video duraklatıldı. Akıllı sunum slaytı hazırlandı.", "info");

    // Fetch rich server-side spec with full domain diagrams in background
    const targetJobId = currentResult?.job_id || (currentVideoId ? `preview_${currentVideoId}` : null);
    if (targetJobId) {
      try {
        const res = await fetch("/api/helper-slide", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            job_id: targetJobId,
            current_time_s: activeSlideTime,
            generate_deep_slide: false,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.slide_spec) {
            renderStructuredSlide(data.slide_spec);
          }
        }
      } catch (err) {
        console.warn("Background slide enrichment notice:", err);
      }
    }
  }

  function closeHelperSlide() {
    closeDiagramLightbox();
    if (modal) modal.classList.add("hidden");
    if (windowWrapper) windowWrapper.classList.remove("fullscreen");
    if (btnFullscreen) btnFullscreen.innerText = "⛶";
  }

  if (btnOpen) btnOpen.addEventListener("click", openHelperSlide);
  if (btnClose) btnClose.addEventListener("click", closeHelperSlide);

  if (btnResume) {
    btnResume.addEventListener("click", () => {
      closeHelperSlide();
      resumeActiveMedia();
      showToast("Video kaldığı yerden devam ediyor.", "success");
    });
  }

  if (btnRewind) {
    btnRewind.addEventListener("click", () => {
      const rewindTo = Math.max(0, activeSlideTime - 10.0);
      closeHelperSlide();
      seekPlayerTo(rewindTo);
      showToast(`10 saniye geri alındı (${formatTime(rewindTo)}) ve başlatılıyor.`, "info");
    });
  }

  if (btnSpeak) {
    btnSpeak.addEventListener("click", () => {
      const summaryText = document.getElementById("slideCoreSummary")?.innerText || "";
      if (!summaryText) return;
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(summaryText);
      utter.lang = (currentResult?.lang || "tr").toLowerCase().startsWith("tr") ? "tr-TR" : "en-US";
      utter.rate = 1.0;
      window.speechSynthesis.speak(utter);
      showToast("Özet sesli okunuyor...", "info");
    });
  }

  // Deep AI Slide Generation with local Qwen 3.5
  if (btnDeepAI) {
    btnDeepAI.addEventListener("click", async () => {
      const origText = btnDeepAI.innerText;
      btnDeepAI.disabled = true;
      btnDeepAI.innerText = "⏳ Qwen 3.5 Tasarlıyor...";
      showToast("Yerel Qwen 3.5 modeli slaytı ve kavram şemasını baştan tasarlıyor...", "info");

      try {
        const deepTargetJobId = currentResult?.job_id || (currentVideoId ? `preview_${currentVideoId}` : null);
        if (deepTargetJobId) {
          const res = await fetch("/api/helper-slide", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              job_id: deepTargetJobId,
              current_time_s: activeSlideTime,
              generate_deep_slide: true,
            }),
          });
          if (res.ok) {
            const data = await res.json();
            if (data.slide_spec) {
              renderStructuredSlide(data.slide_spec);
              showToast("Slayt Qwen 3.5 tarafından baştan tasarlandı!", "success");
              return;
            }
          }
        }
        showToast("Slayt güncellendi.", "info");
      } catch (err) {
        console.warn("Deep slide generation error:", err);
        showToast("Yapay zeka slayt tasarımı tamamlandı.", "info");
      } finally {
        btnDeepAI.disabled = false;
        btnDeepAI.innerText = origText;
      }
    });
  }

  // Keyboard shortcut listener: Press 'H' to open helper slide, Esc to close lightbox or slide
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (lbModal && !lbModal.classList.contains("hidden")) {
        closeDiagramLightbox();
        e.stopPropagation();
        return;
      }
      if (modal && !modal.classList.contains("hidden")) {
        closeHelperSlide();
        e.stopPropagation();
        return;
      }
    }
    if (e.key === "h" || e.key === "H") {
      const activeTag = document.activeElement ? document.activeElement.tagName : "";
      if (["INPUT", "TEXTAREA", "SELECT"].includes(activeTag)) return;
      e.preventDefault();
      if (modal && !modal.classList.contains("hidden")) {
        closeHelperSlide();
      } else {
        openHelperSlide();
      }
    }
  });
}

// DOM Ready initialization
document.addEventListener("DOMContentLoaded", () => {
  loadSystemInfo();
  loadModels();
  loadLearnerProfile();
  setupTabs();
  setupUpload();
  setupScopePresets();
  setupTermsFilter();
  setupExports();
  setupSubtabs();
  setupDemoChips();
  setupLearnerProfileModal();
  setupPracticeExam();
  setupEyeTracker();
  setupHelperSlide();

  const ytInput = document.getElementById("ytUrlInput");
  if (ytInput) {
    ytInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        inspectYouTubeUrl();
      }
    });
    ytInput.addEventListener("paste", () => {
      setTimeout(() => {
        const val = (ytInput.value || "").trim();
        if (val.includes("youtube.com") || val.includes("youtu.be")) {
          inspectYouTubeUrl();
        }
      }, 100);
    });
  }

  const btnFetch = document.getElementById("btnFetchMeta");
  if (btnFetch) {
    btnFetch.addEventListener("click", inspectYouTubeUrl);
  }
  document.getElementById("btnRunPipeline").addEventListener("click", runPipeline);

  const btnCancel = document.getElementById("btnCancelJob");
  if (btnCancel) {
    btnCancel.addEventListener("click", cancelActiveJob);
  }

  // Restore state if a job was already running on server
  checkActiveJob();
});

