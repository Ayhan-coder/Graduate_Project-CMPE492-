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
    const pillGpu = document.getElementById("pillGpu");
    const dot = pillGpu ? pillGpu.querySelector(".status-dot") : null;

    if (data && data.cuda) {
      if (gpuEl) gpuEl.innerText = "NVIDIA";
      if (pillGpu) pillGpu.setAttribute("title", `${data.gpu} (${data.vram_gb} GB VRAM)`);
      if (dot) dot.classList.add("online");
    } else {
      if (gpuEl) gpuEl.innerText = "Intel";
      if (pillGpu) pillGpu.setAttribute("title", "Intel / CPU Modu (NVIDIA aktif değil)");
      if (dot) dot.classList.remove("online");
    }
  } catch (err) {
    console.error("System info error:", err);
    const gpuEl = document.getElementById("gpuText");
    if (gpuEl) gpuEl.innerText = "Intel";
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

    if (info.is_restricted) {
      showToast(info.restriction_message || "Bu video YouTube tarafından kısıtlanmıştır. Yerel dosya yüklemeniz önerilir.", "warning");
      const tabUpload = document.getElementById("tabUpload");
      if (tabUpload) {
        tabUpload.classList.add("tab-attention-pulse");
        setTimeout(() => tabUpload.classList.remove("tab-attention-pulse"), 6000);
      }
    } else {
      showToast("Video metadata retrieved!", "success");
    }

    // Proactively pre-fetch transcript and terminology in the background so player & helper slide are instant
    fetchTranscriptPreview(url, false);
  } catch (err) {
    titleEl.innerText = "Could not load video info";
    authorEl.innerText = err.message;
    showToast(err.message, "error");
    const tabUpload = document.getElementById("tabUpload");
    if (tabUpload) {
      tabUpload.classList.add("tab-attention-pulse");
      setTimeout(() => tabUpload.classList.remove("tab-attention-pulse"), 6000);
    }
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
    let suggestUpload = false;
    try {
      const payload = JSON.parse(e.data);
      if (payload?.message) message = payload.message;
      if (payload?.error && payload.error !== message) extra = payload.error;
      if (payload?.detail && payload.detail !== message) extra = payload.detail;
      if (payload?.suggest_upload) suggestUpload = true;
    } catch (_) {}
    if (extra) {
      appendLog(`Diagnostic info: ${extra}`);
    }
    if (suggestUpload) {
      appendLog("[Çözüm Önerisi] Bu video YouTube tarafından kısıtlanmıştır. Videoyu indirip 'File Upload' sekmesinden doğrudan yükleyebilirsiniz.");
      const tabUpload = document.getElementById("tabUpload");
      if (tabUpload) {
        tabUpload.classList.add("tab-attention-pulse");
        setTimeout(() => tabUpload.classList.remove("tab-attention-pulse"), 7000);
      }
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
    updateDrawerConceptGraph(result);

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
      events: {
        onStateChange: (event) => {
          if (event && event.data === 1) { // YT.PlayerState.PLAYING
            const chkWatch = document.getElementById("chkWatchPause");
            if (chkWatch && chkWatch.checked && !eyeTrackerActive && typeof window.startEyeTracker === "function") {
              window.startEyeTracker();
            }
          }
        },
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
      focusBadgeHtml = `<span class="term-focus-badge focus-review" title="Review-Priority from student exams (${masteryVal ?? 0}% mastery) — Higher LLM Weight"><svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> Review (${masteryVal ?? 0}%)</span>`;
    } else if (focus === "learning") {
      focusBadgeHtml = `<span class="term-focus-badge focus-learning" title="Still Learning (${masteryVal ?? 50}% mastery)"><svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> Learning</span>`;
    } else if (focus === "known") {
      focusBadgeHtml = `<span class="term-focus-badge focus-known" title="Mastered in student exams (${masteryVal ?? 100}% mastery) — Lower LLM Weight"><svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> Mastered</span>`;
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

    // 3. Cognitive Attention Drop Check & In-Video Pop-Quiz Intervention
    if (typeof checkAttentionPopQuizTrigger === "function") {
      checkAttentionPopQuizTrigger(t);
    }

    // 4. Auto-activate watch tracking if enabled and media is playing
    const chkWatch = document.getElementById("chkWatchPause");
    if (chkWatch && chkWatch.checked && !eyeTrackerActive && typeof window.startEyeTracker === "function" && isAnyMediaPlaying()) {
      window.startEyeTracker();
    }
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
        return `<span class="chip-item weak" title="Mastery: ${mastery}% • Missed: ${t.incorrect || 0} times"><svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> ${escapeHtml(t.term)} (${mastery}%)</span>`;
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
        return `<span class="chip-item strong" title="Mastery: ${mastery}%"><svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> ${escapeHtml(t.term)} (${mastery}%)</span>`;
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
              <svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg> ${escapeHtml(title)}
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
    statusInd.innerText = isAnswered ? "Answer saved" : "Unanswered";
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
    statusInd.innerText = "Answer saved";
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
    statusInd.innerText = "Answer saved";
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
      weakList.innerHTML = terms.map((t) => `<span class="chip-item weak"><svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> ${escapeHtml(t)}</span>`).join("");
    }
  }

  const strongList = document.getElementById("examStrongTermsList");
  if (strongList) {
    const terms = result.strong_terms || [];
    if (terms.length === 0) {
      strongList.innerHTML = `<span class="empty-chip">None</span>`;
    } else {
      strongList.innerHTML = terms.map((t) => `<span class="chip-item strong"><svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> ${escapeHtml(t)}</span>`).join("");
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
let tabSwitchPausedMedia = false;
let tabSwitchCount = 0;
let snoozeAttentionUntil = 0;
let isSimulationMode = false;
let simDistractedState = false;

// Media playback unified helpers
function isAnyMediaPlaying() {
  const ytContainer = document.getElementById("ytPlayerContainer");
  const videoContainer = document.getElementById("htmlVideoContainer");
  const videoPlayer = document.getElementById("videoPlayer");
  const audioPlayer = document.getElementById("audioPlayer");
  const manimatorPlayer = document.getElementById("manimatorVideoPlayer");
  const manimatorDeck = document.getElementById("manimatorResultDeck");

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
  if (manimatorPlayer && manimatorDeck && !manimatorDeck.classList.contains("hidden")) {
    if (!manimatorPlayer.paused && !manimatorPlayer.ended && manimatorPlayer.currentTime > 0) return true;
  }
  return false;
}

function pauseActiveMedia() {
  const ytContainer = document.getElementById("ytPlayerContainer");
  const videoContainer = document.getElementById("htmlVideoContainer");
  const videoPlayer = document.getElementById("videoPlayer");
  const audioPlayer = document.getElementById("audioPlayer");
  const manimatorPlayer = document.getElementById("manimatorVideoPlayer");
  const manimatorDeck = document.getElementById("manimatorResultDeck");
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
  if (manimatorPlayer && manimatorDeck && !manimatorDeck.classList.contains("hidden")) {
    if (!manimatorPlayer.paused && !manimatorPlayer.ended) {
      manimatorPlayer.pause();
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
  const manimatorPlayer = document.getElementById("manimatorVideoPlayer");
  const manimatorDeck = document.getElementById("manimatorResultDeck");

  if (ytPlayer && ytContainer && !ytContainer.classList.contains("hidden") && ytPlayer.playVideo) {
    try {
      ytPlayer.playVideo();
    } catch (_) {}
  } else if (videoPlayer && videoContainer && !videoContainer.classList.contains("hidden")) {
    videoPlayer.play().catch(() => {});
  } else if (audioPlayer) {
    audioPlayer.play().catch(() => {});
  } else if (manimatorPlayer && manimatorDeck && !manimatorDeck.classList.contains("hidden")) {
    manimatorPlayer.play().catch(() => {});
  }
}

// Unified Attention / Tab Pause Overlay Presentation Helper
function showAttentionPauseOverlay(type, customReason) {
  const overlay = document.getElementById("attentionPauseOverlay");
  const alertEmoji = document.getElementById("attentionAlertEmoji");
  const alertTitle = document.getElementById("attentionAlertTitle");
  const reasonEl = document.getElementById("attentionPauseReason");
  const liveMsg = document.getElementById("attentionLiveMsg");

  if (type === "tab") {
    if (alertEmoji) alertEmoji.innerHTML = '<svg class="ui-icon ui-icon-lg" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>';
    if (alertTitle) alertTitle.innerText = "Başka Sekmeye Geçildi — Video Duraklatıldı";
    if (reasonEl) reasonEl.innerText = customReason || "Başka bir sekmeye geçtiğiniz algılandı. Ders konusundan kopmamanız ve terimleri kaçırmamanız için video otomatik duraklatıldı.";
    if (liveMsg) liveMsg.innerText = "Ders sekmesine döndüğünüzde video kaldığı yerden devam edecektir...";
  } else if (type === "blur") {
    if (alertEmoji) alertEmoji.innerHTML = '<svg class="ui-icon ui-icon-lg" viewBox="0 0 24 24"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>';
    if (alertTitle) alertTitle.innerText = "Pencere Odağı Kayboldu — Video Duraklatıldı";
    if (reasonEl) reasonEl.innerText = customReason || "Ders penceresinin dışına çıktığınız algılandı. Dikkatinizi korumak ve konuyu kaçırmamanız için video otomatik duraklatıldı.";
    if (liveMsg) liveMsg.innerText = "Pencereye tekrar odaklandığınızda video devam edecektir...";
  } else {
    if (alertEmoji) alertEmoji.innerHTML = '<svg class="ui-icon ui-icon-lg" viewBox="0 0 24 24"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>';
    if (alertTitle) alertTitle.innerText = "Dikkat Dağınıklığı Algılandı!";
    if (reasonEl) reasonEl.innerText = customReason || "Gözleriniz veya başınız ekrandan uzaklaştığı için video otomatik olarak duraklatıldı.";
    if (liveMsg) liveMsg.innerText = "Ekrana tekrar odaklandığınızda video devam edecektir...";
  }

  if (overlay) overlay.classList.remove("hidden");
  playAttentionChime();
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

// ==========================================================================
// Cognitive Science Attention Pop-Quiz & Interpolated Testing Controller
// Research Citations:
// - Guo, Kim, Rubin (ACM L@S 2014): 6.9M edX video sessions, median attention peak at 6 min (360s).
// - Szpunar, Khan, Schacter (PNAS 2013): mind wandering exceeds 40% after 4-5 min (240-300s);
//   interpolated testing cuts mind wandering by >50% and doubles long-term retention.
// - Bunce, Flaherty, VandenPlas (2010): first attention lapse onset at 270-300s.
// - Risko et al. (2012): progressive mind wandering over lecture duration.
// - Bradbury (2016): cognitive fatigue at 600-900s.
// ==========================================================================

let popQuizActive = false;
let popQuizAnswered = false;
const triggeredPopQuizMilestones = new Set();
let activePopQuizData = null;

const SCIENTIFIC_ATTENTION_MILESTONES = {
  180: {
    badge: "3. Dakika Eşiği (180 sn)",
    source: "Risko et al. (2012) & Bunce et al. (2010)",
    citation: "Erken Zihin Kayması Onset Noktası: Video dersin 3. dakikasından itibaren ilk dikkat dalgalanmaları başlar. 1 hızlı soru ile dikkatinizi toplayın!",
  },
  300: {
    badge: "5. Dakika Eşiği (300 sn)",
    source: "Szpunar & Schacter (PNAS 2013 - Harvard)",
    citation: "Optimal İnterpolated Test Eşiği: 5. dakikada verilen kısa pop-quiz, zihin kaymasını %40'tan %18'e indirir ve dikkati sıfırlar.",
  },
  360: {
    badge: "6. Dakika Eşiği (360 sn)",
    source: "Guo, Kim, Rubin (ACM L@S 2014 - edX 6.9M Video)",
    citation: "Kritik Dikkat Kırılma Sınırı: edX araştırmasına göre online ders videolarında medyan odak 6. dakikada tepe yapar ve ardından hızla çöker.",
  },
  600: {
    badge: "10. Dakika Eşiği (600 sn)",
    source: "Bradbury (2016) / Wilson & Korn (2007)",
    citation: "İkincil Bilişsel Yorgunluk Eşiği: Uzun derslerde 10. dakikadan sonra pasif dinlemede derin dikkat çöküşü yaşanır.",
  },
};

function playSuccessChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(523.25, ctx.currentTime);
    osc.frequency.setValueAtTime(659.25, ctx.currentTime + 0.1);
    osc.frequency.setValueAtTime(783.99, ctx.currentTime + 0.2);
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.46);
  } catch (_) {}
}

function checkAttentionPopQuizTrigger(t) {
  if (popQuizActive) return;
  const selectMode = document.getElementById("selectPopQuizMode");
  const mode = selectMode ? selectMode.value : "scientific";
  if (mode === "disabled") return;

  let thresholds = [180, 300, 360, 600];
  if (mode === "frequent") {
    thresholds = [120, 240, 360, 480, 600];
  } else if (mode === "demo") {
    thresholds = [45, 90, 135, 180, 225, 270];
  }

  for (const s of thresholds) {
    if (t >= s && t <= (s + 2.5) && !triggeredPopQuizMilestones.has(s)) {
      triggeredPopQuizMilestones.add(s);
      triggerAttentionPopQuiz(s);
      break;
    }
  }
}

async function triggerAttentionPopQuiz(targetSeconds) {
  if (popQuizActive) return;
  popQuizActive = true;
  popQuizAnswered = false;

  // 1. Pause video & play gentle attention chime
  pauseActiveMedia();
  playAttentionChime();

  const modal = document.getElementById("popQuizModal");
  if (!modal) return;

  // 2. Resolve or fallback milestone info
  const mInfo = SCIENTIFIC_ATTENTION_MILESTONES[targetSeconds] || {
    badge: `${Math.floor(targetSeconds / 60)}. Dakika Eşiği (${targetSeconds} sn)`,
    source: "Bilişsel Bilim & İnterpolated Testing",
    citation: "Video derslerde düzenli aralıklarla soru çözmek zihin kaymasını önler ve kalıcı hafızayı güçlendirir.",
  };

  const badgeEl = document.getElementById("popquizMilestoneBadge");
  const titleEl = document.getElementById("popquizCitationTitle");
  const descEl = document.getElementById("popquizCitationDesc");
  if (badgeEl) badgeEl.innerHTML = '<svg class="ui-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ' + escapeHtml(mInfo.badge);
  if (titleEl) titleEl.innerText = `Araştırma Dayanağı: ${mInfo.source}`;
  if (descEl) descEl.innerText = mInfo.citation;

  // 3. Prepare or generate question from current lecture terms
  let questionData = null;
  const terms = (currentResult?.llm?.terms && currentResult.llm.terms.length > 0)
    ? currentResult.llm.terms
    : [];

  if (terms.length > 0) {
    // Select terms up to this timestamp
    const earlier = terms.filter((item) => {
      if (item.start_s == null) return true;
      return Number(item.start_s) <= (targetSeconds + 10.0);
    });
    const pool = earlier.length >= 2 ? earlier : terms;
    // Pick first review/unmastered term, or random
    const targetTerm = pool[Math.floor(Math.random() * pool.length)];
    const correctExpl = (targetTerm.explanation || "").trim();

    // Distractors from other terms
    const otherExpls = terms
      .filter((t) => (t.term || "").toLowerCase() !== (targetTerm.term || "").toLowerCase() && (t.explanation || "").trim())
      .map((t) => t.explanation.trim());

    const genericTr = [
      "Süresi ve hedefi belli olmayan genel bir süreç",
      "Yalnızca dokümantasyon amacıyla tutulan geçici bir liste",
      "Sadece ofis ekipmanlarını yönetmek için kullanılan bir araç",
      "Herhangi bir kurala bağlı olmayan serbest çalışma yöntemi",
    ];
    const genericEn = [
      "A general process with no specified time limit",
      "A temporary document listing office equipment",
      "A tool used only for file archiving purposes",
      "An unguided approach without strict project rules",
    ];
    const isTr = (currentResult?.lang || "tr").toLowerCase().startsWith("tr");
    const generics = isTr ? genericTr : genericEn;

    let distractors = otherExpls.slice(0, 3);
    let gi = 0;
    while (distractors.length < 3 && gi < generics.length) {
      if (!distractors.includes(generics[gi]) && generics[gi] !== correctExpl) {
        distractors.push(generics[gi]);
      }
      gi++;
    }

    const options = [correctExpl, ...distractors.slice(0, 3)];
    // Shuffle
    const shuffled = options
      .map((val) => ({ val, sort: Math.random() }))
      .sort((a, b) => a.sort - b.sort)
      .map((obj) => obj.val);

    const answerIdx = shuffled.indexOf(correctExpl);

    questionData = {
      term: targetTerm.term,
      prompt: isTr ? `Bu derste "${targetTerm.term}" kavramı ne anlama gelir?` : `What does "${targetTerm.term}" mean in this lecture?`,
      options: shuffled,
      answer_index: answerIdx,
      explanation: correctExpl,
    };
  } else {
    // Fallback demo question
    questionData = {
      term: "Aktif Öğrenme (Active Retrieval)",
      prompt: "Bilişsel araştırmalara göre video ders izlerken dikkati yeniden zirveye çıkarmanın en etkili yolu nedir?",
      options: [
        "Videonun kritik dakikalarında kısa bir pop-quiz çözerek bilgiyi aktif hatırlamak",
        "Videoyu durdurup 30 dakika beklemek",
        "Videoyu 2 kat hızlandırarak izlemek",
        "Ekrana bakmadan yalnızca sesi dinlemek",
      ],
      answer_index: 0,
      explanation: "Szpunar & Schacter (PNAS 2013) araştırmasına göre interpolated testing (araya serpiştirilen mikro sorular) zihin kaymasını %40'tan %18'e indirir.",
    };
  }

  activePopQuizData = questionData;
  renderPopQuizQuestion(questionData);

  modal.classList.remove("hidden");
  showToast("Dikkat Molası: Video duraklatıldı, 1 hızlı soruyla dikkatinizi toplayın!", "info");
}

function renderPopQuizQuestion(q) {
  const termNameEl = document.getElementById("popquizTermName");
  const promptEl = document.getElementById("popquizPrompt");
  const optionsList = document.getElementById("popquizOptionsList");
  const feedbackBox = document.getElementById("popquizFeedbackBox");

  if (termNameEl) termNameEl.innerText = q.term;
  if (promptEl) promptEl.innerText = q.prompt;
  if (feedbackBox) {
    feedbackBox.className = "popquiz-feedback-box hidden";
  }

  if (optionsList) {
    optionsList.innerHTML = "";
    q.options.forEach((optText, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "popquiz-option-btn";
      btn.dataset.idx = String(idx);
      btn.innerHTML = `
        <span class="opt-key">${idx + 1}</span>
        <span class="opt-text">${escapeHtml(optText)}</span>
      `;
      btn.addEventListener("click", () => handlePopQuizAnswer(idx));
      optionsList.appendChild(btn);
    });
  }
}

async function handlePopQuizAnswer(selectedIdx) {
  if (popQuizAnswered || !activePopQuizData) return;
  popQuizAnswered = true;

  const q = activePopQuizData;
  const isCorrect = selectedIdx === q.answer_index;
  const optionsList = document.getElementById("popquizOptionsList");
  const feedbackBox = document.getElementById("popquizFeedbackBox");
  const fbIcon = document.getElementById("popquizFbIcon");
  const fbTitle = document.getElementById("popquizFbTitle");
  const fbExpl = document.getElementById("popquizFbExpl");
  const masteryTag = document.getElementById("popquizMasteryTag");
  const masteryText = document.getElementById("popquizMasteryText");

  // Highlight buttons
  if (optionsList) {
    const btns = optionsList.querySelectorAll(".popquiz-option-btn");
    btns.forEach((btn, idx) => {
      btn.disabled = true;
      if (idx === q.answer_index) {
        btn.classList.add("correct");
      } else if (idx === selectedIdx && !isCorrect) {
        btn.classList.add("wrong");
      }
    });
  }

  // Show feedback
  if (feedbackBox) {
    feedbackBox.classList.remove("hidden");
    feedbackBox.classList.toggle("correct", isCorrect);
    feedbackBox.classList.toggle("wrong", !isCorrect);
  }

  if (isCorrect) {
    playSuccessChime();
    if (fbIcon) fbIcon.innerHTML = '<svg class="ui-icon ui-icon-lg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>';
    if (fbTitle) fbTitle.innerText = "Harika! Doğru Yanıt — Dikkatiniz Yenilendi!";
    if (masteryTag) masteryTag.classList.remove("hidden");
    if (masteryText) masteryText.innerText = `Öğrenci Profili: "${q.term}" +30% Ustalık Eklendi`;
  } else {
    if (fbIcon) fbIcon.innerHTML = '<svg class="ui-icon ui-icon-lg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
    if (fbTitle) fbTitle.innerText = "Doğru Açıklama:";
    if (masteryTag) masteryTag.classList.remove("hidden");
    if (masteryText) masteryText.innerText = `Öğrenci Profili: "${q.term}" İnceleme Önceliğine Alındı`;
  }
  if (fbExpl) fbExpl.innerText = q.explanation;

  // Auto-focus resume button for quick Space/Enter key resumption
  const btnResume = document.getElementById("btnResumeFromPopQuiz");
  if (btnResume) btnResume.focus();

  // Send answer to backend learner profile
  try {
    const videoId = currentResult?.video_meta?.id || "";
    await fetch("/api/exam/quick-answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        term: q.term,
        correct: isCorrect,
        video_id: videoId,
      }),
    });
    // Refresh student profile badge
    loadLearnerProfile();
  } catch (_) {}
}

function closeAttentionPopQuiz(resumePlayback = true) {
  const modal = document.getElementById("popQuizModal");
  if (modal) modal.classList.add("hidden");
  popQuizActive = false;
  activePopQuizData = null;

  if (resumePlayback) {
    resumeActiveMedia();
    showToast("▶️ Video kaldığı yerden devam ediyor.", "success");
  }
}

function setupAttentionPopQuiz() {
  const btnTest = document.getElementById("btnTestPopQuiz");
  const btnResearch = document.getElementById("btnOpenResearchModal");
  const modalResearch = document.getElementById("attentionResearchModal");
  const btnCloseResearch = document.getElementById("btnCloseResearchModal");
  const btnCloseResearchBottom = document.getElementById("btnCloseResearchModalBottom");
  const btnCloseModal = document.getElementById("btnClosePopQuizModal");
  const btnSkip = document.getElementById("btnSkipPopQuiz");
  const btnResume = document.getElementById("btnResumeFromPopQuiz");

  if (btnTest) {
    btnTest.addEventListener("click", () => {
      let t = 180;
      const ytContainer = document.getElementById("ytPlayerContainer");
      const videoPlayer = document.getElementById("videoPlayer");
      if (ytPlayer && typeof ytPlayer.getCurrentTime === "function" && ytContainer && !ytContainer.classList.contains("hidden")) {
        try { t = ytPlayer.getCurrentTime() || 180; } catch (_) {}
      } else if (videoPlayer && !videoPlayer.paused) {
        t = videoPlayer.currentTime || 180;
      }
      triggerAttentionPopQuiz(Math.round(t) || 300);
    });
  }

  if (btnResearch && modalResearch) {
    btnResearch.addEventListener("click", () => {
      modalResearch.classList.remove("hidden");
    });
  }

  const closeResearch = () => {
    if (modalResearch) modalResearch.classList.add("hidden");
  };
  if (btnCloseResearch) btnCloseResearch.addEventListener("click", closeResearch);
  if (btnCloseResearchBottom) btnCloseResearchBottom.addEventListener("click", closeResearch);

  if (btnCloseModal) btnCloseModal.addEventListener("click", () => closeAttentionPopQuiz(true));
  if (btnSkip) btnSkip.addEventListener("click", () => closeAttentionPopQuiz(true));
  if (btnResume) btnResume.addEventListener("click", () => closeAttentionPopQuiz(true));

  // Keyboard navigation
  window.addEventListener("keydown", (e) => {
    const popModal = document.getElementById("popQuizModal");
    if (!popModal || popModal.classList.contains("hidden")) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;

    if (e.key === "Escape") {
      e.preventDefault();
      closeAttentionPopQuiz(true);
      return;
    }

    if (e.code === "Space" || e.key === "Enter") {
      e.preventDefault();
      closeAttentionPopQuiz(true);
      return;
    }

    if (["1", "2", "3", "4"].includes(e.key)) {
      e.preventDefault();
      const idx = parseInt(e.key, 10) - 1;
      handlePopQuizAnswer(idx);
    }
  });
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

  window.startEyeTracker = startEyeTracker;
  window.stopEyeTracker = stopEyeTracker;
  window.toggleEyeTracker = toggleEyeTracker;

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
      tabSwitchPausedMedia = false;
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
      tabSwitchPausedMedia = false;
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
    const chkWatch = document.getElementById("chkWatchPause");
    const isWatchPauseActive = chkWatch ? chkWatch.checked : true;

    if (isDistracted) {
      if (!isWatchPauseActive) {
        return;
      }
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

              const alertEmoji = document.getElementById("attentionAlertEmoji");
              const alertTitle = document.getElementById("attentionAlertTitle");
              const reasonEl = document.getElementById("attentionPauseReason");
              const liveMsg = document.getElementById("attentionLiveMsg");

              if (alertEmoji) alertEmoji.innerHTML = '<svg class="ui-icon ui-icon-lg" viewBox="0 0 24 24"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>';
              if (alertTitle) alertTitle.innerText = "Dikkat Dağınıklığı Algılandı!";
              if (reasonEl) reasonEl.innerText = `${reason}. Ders videosu odaklanmanız için otomatik duraklatıldı.`;
              if (liveMsg) liveMsg.innerText = "Ekrana tekrar odaklandığınızda video devam edecektir...";

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

function setupManimator() {
  const modal = document.getElementById("manimatorModal");
  const btnNav = document.getElementById("btnNavManimator");
  const btnOpenTrigger = document.getElementById("btnOpenManimator");
  const btnSlideToManimator = document.getElementById("btnSlideToManimator");
  const btnClose = document.getElementById("btnCloseManimatorModal");
  const btnCloseFooter = document.getElementById("btnCloseManimatorFooter");
  
  const tabLecture = document.getElementById("tabManimatorLecture");
  const tabPdf = document.getElementById("tabManimatorPdf");
  const tabCustom = document.getElementById("tabManimatorCustom");
  const panelLecture = document.getElementById("panelManimatorLecture");
  const panelPdf = document.getElementById("panelManimatorPdf");
  const panelCustom = document.getElementById("panelManimatorCustom");

  const pdfInput = document.getElementById("manimatorPdfInput");
  const btnSelectPdf = document.getElementById("btnSelectManimatorPdf");
  const pdfNameBadge = document.getElementById("manimatorSelectedPdfName");

  const btnGen = document.getElementById("btnTriggerManimatorGen");
  const stepper = document.getElementById("manimatorProgressStepper");
  const resultDeck = document.getElementById("manimatorResultDeck");
  const videoPlayer = document.getElementById("manimatorVideoPlayer");
  const subtitlesTimeline = document.getElementById("manimatorSubtitlesTimeline");
  const termsContainer = document.getElementById("manimatorTermsContainer");
  const hookText = document.getElementById("manimatorHookText");
  const codeDisplay = document.getElementById("manimatorCodeDisplay");
  const btnDownload = document.getElementById("btnDownloadManimVideo");
  const btnCopyCode = document.getElementById("btnCopyManimCode");
  const btnSaveLearner = document.getElementById("btnManimatorSaveLearner");

  let currentGeneratedData = null;
  let activeMode = "lecture";

  function switchMode(mode) {
    activeMode = mode;
    [tabLecture, tabPdf, tabCustom].forEach((t) => {
      if (t) t.classList.toggle("active", t.getAttribute("data-mode") === mode);
    });
    if (panelLecture) panelLecture.classList.toggle("hidden", mode !== "lecture");
    if (panelPdf) panelPdf.classList.toggle("hidden", mode !== "pdf");
    if (panelCustom) panelCustom.classList.toggle("hidden", mode !== "custom");
  }

  if (tabLecture) tabLecture.addEventListener("click", () => switchMode("lecture"));
  if (tabPdf) tabPdf.addEventListener("click", () => switchMode("pdf"));
  if (tabCustom) tabCustom.addEventListener("click", () => switchMode("custom"));

  if (btnSelectPdf && pdfInput) {
    btnSelectPdf.addEventListener("click", () => pdfInput.click());
    pdfInput.addEventListener("change", () => {
      if (pdfInput.files && pdfInput.files[0]) {
        pdfNameBadge.textContent = pdfInput.files[0].name;
      } else {
        pdfNameBadge.textContent = "Dosya seçilmedi";
      }
    });
  }

  let activeStuckTime = 0;
  let activeStuckConcept = "";
  let activeStuckQuote = "";

  function openManimatorModal(prefillConcept) {
    if (!modal) return;
    modal.classList.remove("hidden");

    // Pause main lecture video while watching or generating explainer video
    pauseActiveMedia();

    // Get active playback timestamp
    const t = getCurrentVideoTime();
    activeStuckTime = t;
    const tsBadge = document.getElementById("manimatorLectureTimestamp");
    if (tsBadge) tsBadge.innerHTML = `<svg class="ui-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> Anlık Zaman: ${escapeHtml(formatTime(t))}`;

    // Get recent quote & terms
    const asr = currentResult?.asr || {};
    const segments = asr.segments || [];
    const windowStart = Math.max(0, t - 45);
    const windowEnd = t + 5;
    const recentSegs = segments.filter(s => s.start !== undefined && s.end !== undefined && s.end >= windowStart && s.start <= windowEnd);
    const recentText = recentSegs.length ? recentSegs.map(s => s.text).join(" ") : (asr.text || "").slice(0, 300);
    activeStuckQuote = recentText;

    const quoteEl = document.getElementById("manimatorLectureQuoteText");
    if (quoteEl) {
      quoteEl.textContent = recentText ? `"${recentText.slice(0, 240)}..."` : "Dersin bu anındaki içerik analiz edilerek anlatım videosu hazırlanacak.";
    }

    // Concept badge
    let activeTerm = prefillConcept || "";
    if (!activeTerm) {
      const terms = currentResult?.llm?.terms || [];
      const match = terms.find(tObj => tObj.start !== undefined && tObj.start >= windowStart && tObj.start <= windowEnd) || terms[0];
      if (match) activeTerm = match.term;
    }
    activeStuckConcept = activeTerm;
    const conceptBadge = document.getElementById("manimatorActiveConceptBadge");
    if (conceptBadge) {
      conceptBadge.innerHTML = `<svg class="ui-icon" viewBox="0 0 24 24"><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg> Odak Kavram: ${escapeHtml(activeTerm || "Ders Bölümü")}`;
    }
    if (prefillConcept) {
      const customInp = document.getElementById("manimatorCustomTopicInput");
      if (customInp) customInp.value = prefillConcept;
    }
  }

  function closeManimatorModal() {
    if (!modal) return;
    modal.classList.add("hidden");
    if (videoPlayer) videoPlayer.pause();
  }

  if (btnNav) btnNav.addEventListener("click", () => openManimatorModal());
  if (btnOpenTrigger) btnOpenTrigger.addEventListener("click", () => openManimatorModal());
  if (btnSlideToManimator) {
    btnSlideToManimator.addEventListener("click", () => {
      const slideTitle = document.getElementById("slideMainTitle")?.innerText;
      openManimatorModal(slideTitle);
    });
  }
  if (btnClose) btnClose.addEventListener("click", closeManimatorModal);
  if (btnCloseFooter) btnCloseFooter.addEventListener("click", closeManimatorModal);

  // Stepper animation
  function setStep(stepIndex) {
    const steps = [
      document.getElementById("stepQwen"),
      document.getElementById("stepTts"),
      document.getElementById("stepManim"),
      document.getElementById("stepFfmpeg"),
    ];
    steps.forEach((el, idx) => {
      if (!el) return;
      el.classList.remove("active", "completed");
      if (idx < stepIndex) el.classList.add("completed");
      else if (idx === stepIndex) el.classList.add("active");
    });
  }

  // Generation Handler
  if (btnGen) {
    btnGen.addEventListener("click", async () => {
      const origText = btnGen.innerHTML;
      btnGen.disabled = true;
      btnGen.innerHTML = `<svg class="ui-icon spin" viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> <span>Video Üretiliyor...</span>`;
      if (stepper) stepper.classList.remove("hidden");
      if (resultDeck) resultDeck.classList.add("hidden");

      setStep(0); // Qwen script
      const stepTimer1 = setTimeout(() => setStep(1), 2500); // TTS
      const stepTimer2 = setTimeout(() => setStep(2), 5500); // Manim render
      const stepTimer3 = setTimeout(() => setStep(3), 9500); // FFmpeg

      try {
        const style = document.getElementById("manimatorStyleSelect")?.value || "manim_animation";
        const duration = parseInt(document.getElementById("manimatorDurationSelect")?.value || "50", 10);
        const voice = document.getElementById("manimatorVoiceSelect")?.value || "tr-TR-AhmetNeural";
        const modelMode = document.getElementById("manimatorModelSelect")?.value || "fast";
        const useLocalQwen = (modelMode === "deep");

        let responseData = null;

        if (activeMode === "pdf" && pdfInput && pdfInput.files && pdfInput.files[0]) {
          const formData = new FormData();
          formData.append("file", pdfInput.files[0]);
          formData.append("video_style", style);
          formData.append("duration_s", duration.toString());
          formData.append("voice", voice);
          formData.append("lang", currentResult?.lang || "tr");

          const res = await fetch("/api/manimator/pdf-to-video", {
            method: "POST",
            body: formData,
          });
          if (!res.ok) throw new Error(`Sunucu hatası: ${res.statusText}`);
          responseData = await res.json();
        } else {
          // Lecture or Custom mode
          let topicVal = "";
          if (activeMode === "custom") {
            topicVal = document.getElementById("manimatorCustomTopicInput")?.value?.trim() || "";
          } else {
            topicVal = activeStuckConcept || "";
          }

          const targetJobId = currentResult?.job_id || (currentVideoId ? `preview_${currentVideoId}` : null);
          const res = await fetch("/api/manimator/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              job_id: targetJobId || "",
              current_time_s: activeStuckTime || getCurrentVideoTime(),
              topic: topicVal,
              context: activeStuckQuote,
              lang: currentResult?.lang || "tr",
              video_style: style,
              duration_s: duration,
              voice: voice,
              use_local_qwen: useLocalQwen,
            }),
          });
          if (!res.ok) throw new Error(`Sunucu hatası: ${res.statusText}`);
          responseData = await res.json();
        }

        clearTimeout(stepTimer1);
        clearTimeout(stepTimer2);
        clearTimeout(stepTimer3);
        setStep(4); // All done!

        currentGeneratedData = responseData;
        renderManimatorResults(responseData);
        showToast("Özel anlatım videosu başarıyla üretildi!", "success");
      } catch (err) {
        clearTimeout(stepTimer1);
        clearTimeout(stepTimer2);
        clearTimeout(stepTimer3);
        console.error("Manimator generation error:", err);
        showToast(`Video üretimi sırasında hata: ${err.message}`, "error");
      } finally {
        if (stepper) stepper.classList.add("hidden");
        btnGen.disabled = false;
        btnGen.innerHTML = origText;
      }
    });
  }

  function renderManimatorResults(data) {
    if (!resultDeck || !data) return;
    resultDeck.classList.remove("hidden");

    // Video Player
    if (videoPlayer && data.video_url) {
      videoPlayer.src = data.video_url;
      videoPlayer.load();
      videoPlayer.play().catch(() => {});
    }

    // Duration badge
    const durBadge = document.getElementById("manimatorVideoDurationBadge");
    if (durBadge) durBadge.textContent = `${formatTime(data.duration_s || 0)}`;

    // Hook text
    if (hookText) hookText.textContent = data.pedagogical_hook || "Kavram pedagojik benzetmelerle modellenmiştir.";

    // Terms
    if (termsContainer) {
      termsContainer.innerHTML = "";
      (data.terms || []).forEach((t) => {
        const row = document.createElement("div");
        row.className = "manimator-term-pill-card";
        row.innerHTML = `
          <span class="manimator-term-name">${escapeHtml(t.term || "")}</span>
          <span class="manimator-term-desc">${escapeHtml(t.explanation || "")}</span>
        `;
        termsContainer.appendChild(row);
      });
    }

    // Manim Code
    if (codeDisplay) {
      codeDisplay.textContent = data.manim_code || "# Kod bulunamadı";
    }

    // Subtitles Timeline
    if (subtitlesTimeline) {
      subtitlesTimeline.innerHTML = "";
      const subs = data.subtitles || [];
      subs.forEach((cue) => {
        const cueRow = document.createElement("div");
        cueRow.className = "subtitle-cue-row";
        cueRow.dataset.start = cue.start;
        cueRow.dataset.end = cue.end;
        cueRow.innerHTML = `
          <span class="cue-time-badge">${formatTime(cue.start)}</span>
          <span>${escapeHtml(cue.text || cue.title || "")}</span>
        `;
        cueRow.addEventListener("click", () => {
          if (videoPlayer) {
            videoPlayer.currentTime = cue.start;
            videoPlayer.play().catch(() => {});
          }
        });
        subtitlesTimeline.appendChild(cueRow);
      });

      // Synchronize active subtitle cue with video playback
      if (videoPlayer) {
        videoPlayer.ontimeupdate = () => {
          const ct = videoPlayer.currentTime;
          const rows = subtitlesTimeline.querySelectorAll(".subtitle-cue-row");
          rows.forEach((r) => {
            const st = parseFloat(r.dataset.start || "0");
            const et = parseFloat(r.dataset.end || "999");
            if (ct >= st && ct <= et) {
              if (!r.classList.contains("active")) {
                rows.forEach(x => x.classList.remove("active"));
                r.classList.add("active");
                r.scrollIntoView({ behavior: "smooth", block: "nearest" });
              }
            }
          });
        };
      }
    }
  }

  // Download & Copy Code Actions
  if (btnDownload) {
    btnDownload.addEventListener("click", () => {
      if (!currentGeneratedData?.video_url) return;
      const a = document.createElement("a");
      a.href = currentGeneratedData.video_url;
      a.download = `${currentGeneratedData.title || "manimator_video"}.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      showToast("Video indirmesi başlatıldı.", "success");
    });
  }

  if (btnCopyCode) {
    btnCopyCode.addEventListener("click", () => {
      if (!currentGeneratedData?.manim_code) return;
      navigator.clipboard.writeText(currentGeneratedData.manim_code).then(() => {
        showToast("Python Manim kodu panoya kopyalandı!", "success");
      });
    });
  }

  if (btnSaveLearner) {
    btnSaveLearner.addEventListener("click", async () => {
      if (!currentGeneratedData?.terms) return;
      try {
        for (const t of currentGeneratedData.terms) {
          if (t.term) {
            await fetch("/api/popquiz/answer", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                term: t.term,
                correct: true,
                video_id: currentVideoId || "",
              }),
            });
          }
        }
        await loadLearnerProfile();
        showToast("Kavram anlaşıldı olarak profiline işlendi ve pekiştirildi!", "success");
      } catch (e) {
        showToast("Öğrenci profili güncellendi.", "info");
      }
    });
  }
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
      btnFullscreen.innerHTML = windowWrapper.classList.contains("fullscreen")
        ? '<svg class="ui-icon" viewBox="0 0 24 24"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/></svg>'
        : '<svg class="ui-icon" viewBox="0 0 24 24"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>';
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
          const iconSvg = `<svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;
          cardEl.innerHTML = `
            <div class="slide-term-header">
              <span class="slide-term-name">${iconSvg} ${escapeHtml(title)}</span>
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
              <span style="font-weight:bold;color:#c4b5fd;">Kavram Akışı:</span><br/>
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
      mermaid_diagram: "graph LR\n  A[\"Başlangıç & Plan\"] --> B[\"Yürütme & Analiz\"]\n  B --> C[\"Sonuç & Kazanım\"]",
      cards: relevantTerms.map(t => ({
        icon: "",
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
    if (btnFullscreen) btnFullscreen.innerHTML = '<svg class="ui-icon" viewBox="0 0 24 24"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>';
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
      const origText = btnDeepAI.innerHTML;
      btnDeepAI.disabled = true;
      btnDeepAI.innerHTML = `<svg class="ui-icon spin" viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Qwen 3.5 Tasarlıyor...`;
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
        btnDeepAI.innerHTML = origText;
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

// ==========================================================================
// Smart Tab Visibility & Watch Distraction Tracker
// Automatically pauses media when student switches tabs or looks away from video
// ==========================================================================
function setupTabVisibilityTracker() {
  const chkTab = document.getElementById("chkTabPause");
  const chkWatch = document.getElementById("chkWatchPause");

  // Page Visibility API: Detect when user switches tabs or minimizes browser
  document.addEventListener("visibilitychange", () => {
    const isTabPauseActive = chkTab ? chkTab.checked : true;
    const autoResumeChecked = document.getElementById("chkAutoResume")?.checked ?? true;
    const overlay = document.getElementById("attentionPauseOverlay");
    const tabCountEl = document.getElementById("etTabSwitchCount");
    const pauseCountEl = document.getElementById("etPauseCount");

    if (document.hidden || document.visibilityState === "hidden") {
      // User switched away to another tab
      if (isTabPauseActive && isAnyMediaPlaying()) {
        tabSwitchPausedMedia = true;
        pauseActiveMedia();
        tabSwitchCount++;
        attentionPauseCount++;
        if (tabCountEl) tabCountEl.innerText = String(tabSwitchCount);
        if (pauseCountEl) pauseCountEl.innerText = String(attentionPauseCount);
        updateEyeTrackerStatus("DISTRACTED", "Sekme Dışında");
        showAttentionPauseOverlay("tab");
      }
    } else {
      // User returned to this tab
      if (tabSwitchPausedMedia) {
        if (autoResumeChecked) {
          tabSwitchPausedMedia = false;
          if (overlay) overlay.classList.add("hidden");
          resumeActiveMedia();
          updateEyeTrackerStatus("FOCUSED", "Odaklandı");
          showToast("Sekmeye geri dönüldü: Video otomatik devam ediyor.", "success");
        } else {
          updateEyeTrackerStatus("WARNING", "Devam Bekleniyor");
          showToast("Ders sekmesine geri döndünüz. Videoyu devam ettirmek için butona tıklayın.", "info");
        }
      }
    }
  });

  // Window blur / focus: Detect when user clicks another application/window
  let blurDebounce = null;
  window.addEventListener("blur", () => {
    // If user is interacting with an input/file dialog, do not pause
    if (document.activeElement && (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA" || document.activeElement.tagName === "SELECT")) {
      return;
    }
    const isTabPauseActive = chkTab ? chkTab.checked : true;
    if (!isTabPauseActive) return;

    blurDebounce = setTimeout(() => {
      if (isAnyMediaPlaying() && !tabSwitchPausedMedia) {
        tabSwitchPausedMedia = true;
        pauseActiveMedia();
        tabSwitchCount++;
        attentionPauseCount++;
        const tabCountEl = document.getElementById("etTabSwitchCount");
        const pauseCountEl = document.getElementById("etPauseCount");
        if (tabCountEl) tabCountEl.innerText = String(tabSwitchCount);
        if (pauseCountEl) pauseCountEl.innerText = String(attentionPauseCount);
        updateEyeTrackerStatus("DISTRACTED", "Pencere Dışında");
        showAttentionPauseOverlay("blur");
      }
    }, 450);
  });

  window.addEventListener("focus", () => {
    if (blurDebounce) {
      clearTimeout(blurDebounce);
      blurDebounce = null;
    }
    if (tabSwitchPausedMedia) {
      const autoResumeChecked = document.getElementById("chkAutoResume")?.checked ?? true;
      const overlay = document.getElementById("attentionPauseOverlay");
      if (autoResumeChecked) {
        tabSwitchPausedMedia = false;
        if (overlay) overlay.classList.add("hidden");
        resumeActiveMedia();
        updateEyeTrackerStatus("FOCUSED", "Odaklandı");
        showToast("Ekrana geri dönüldü: Video otomatik devam ediyor.", "success");
      }
    }
  });

  // Watch pause checkbox listener
  if (chkWatch) {
    chkWatch.addEventListener("change", (e) => {
      if (e.target.checked) {
        if (!eyeTrackerActive && isAnyMediaPlaying() && typeof window.startEyeTracker === "function") {
          window.startEyeTracker();
        }
        showToast("İzlenmeyince duraklatma (Bakış & Odak Takibi) etkinleştirildi.", "info");
      } else {
        showToast("İzlenmeyince duraklatma kapatıldı.", "warning");
      }
    });
  }

  if (chkTab) {
    chkTab.addEventListener("change", (e) => {
      showToast(e.target.checked ? "Sekmeden ayrılınca duraklatma aktif." : "Sekmeden ayrılınca duraklatma kapatıldı.", "info");
    });
  }

  // Auto-activate eye tracker when HTML5 video or audio plays, if chkWatch is checked
  const vidPlayer = document.getElementById("videoPlayer");
  const audPlayer = document.getElementById("audioPlayer");
  const onMediaPlayTrigger = () => {
    const watchChk = document.getElementById("chkWatchPause");
    if (watchChk && watchChk.checked && !eyeTrackerActive && typeof window.startEyeTracker === "function") {
      window.startEyeTracker();
    }
  };
  if (vidPlayer) vidPlayer.addEventListener("play", onMediaPlayTrigger);
  if (audPlayer) audPlayer.addEventListener("play", onMediaPlayTrigger);
}

// ==========================================================================
// Cornell Note-Taking Study Sheet & Export System (Walter Pauk Pedagogy)
// ==========================================================================
let activeCornellData = null;
let activeCornellMarkdown = "";

function setupCornellNotes() {
  const modal = document.getElementById("cornellModal");
  const btnOpenHeader = document.getElementById("btnNavCornell");
  const btnOpenHub = document.getElementById("btnOpenCornellNotes");
  const btnOpenToolbar = document.getElementById("btnOpenCornellToolbar");
  const btnClose = document.getElementById("btnCloseCornellModal");
  const btnCloseFooter = document.getElementById("btnCloseCornellFooter");
  const btnPrint = document.getElementById("btnPrintCornell");
  const btnDownloadMd = document.getElementById("btnDownloadCornellMd");
  const btnCopyMd = document.getElementById("btnCopyCornellMd");

  const openSheet = async () => {
    await renderCornellNotesModal();
  };

  const closeSheet = () => {
    if (modal) modal.classList.add("hidden");
  };

  if (btnOpenHeader) btnOpenHeader.addEventListener("click", openSheet);
  if (btnOpenHub) btnOpenHub.addEventListener("click", openSheet);
  if (btnOpenToolbar) btnOpenToolbar.addEventListener("click", openSheet);
  if (btnClose) btnClose.addEventListener("click", closeSheet);
  if (btnCloseFooter) btnCloseFooter.addEventListener("click", closeSheet);

  if (btnPrint) {
    btnPrint.addEventListener("click", () => {
      window.print();
    });
  }

  if (btnDownloadMd) {
    btnDownloadMd.addEventListener("click", () => {
      if (!activeCornellMarkdown) return;
      const title = (activeCornellData?.title || "Cornell_Notes").replace(/[^\w\d\-_]/g, "_");
      const blob = new Blob([activeCornellMarkdown], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${title}_Cornell_Notes.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast("Markdown (.md) ders notu indirildi!", "success");
    });
  }

  if (btnCopyMd) {
    btnCopyMd.addEventListener("click", async () => {
      if (!activeCornellMarkdown) return;
      try {
        await navigator.clipboard.writeText(activeCornellMarkdown);
        showToast("Cornell Ders Notu panoya kopyalandı! (Obsidian / Notion uyumlu)", "success");
      } catch (err) {
        showToast("Kopyalama başarısız oldu.", "error");
      }
    });
  }

  // Close on backdrop click
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeSheet();
    });
  }

  // Keyboard shortcut: Press 'C' to open Cornell sheet (when not typing)
  window.addEventListener("keydown", (e) => {
    if (e.key === "c" || e.key === "C") {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable) return;
      if (modal && !modal.classList.contains("hidden")) {
        closeSheet();
      } else {
        openSheet();
      }
    }
  });
}

function generateClientSideCornellData(res) {
  const meta = res?.video_meta || {};
  const title = meta.title || "Ders Notu & Kavram Özeti";
  const durSec = Number(meta.duration || res?.asr?.duration || 0);
  const durStr = durSec > 0 ? `${Math.floor(durSec / 60).toString().padStart(2, '0')}:${Math.floor(durSec % 60).toString().padStart(2, '0')}` : "--:--";
  const dateStr = new Date().toLocaleDateString("tr-TR");
  const rawTerms = res?.llm?.terms || [];

  const cues = [];
  const notes = [];

  rawTerms.forEach((t, idx) => {
    const term = t.term || `Kavram ${idx + 1}`;
    const expl = t.explanation || "Temel ders kavramı.";
    const time = t.timestamp || "--:--";
    const timeS = t.start != null ? Number(t.start) : null;
    const low = (term + " " + expl).toLowerCase ? (term + " " + expl).toLowerCase() : "";

    let question = `Sınav Sorusu: '${term}' kavramının temel işlevi nedir ve neden önemlidir?`;
    if (low.includes("sapma") || low.includes("maliyet") || low.includes("variance")) {
      question = `Hesaplama Sorusu: '${term}' formülü nedir ve negatif/pozitif çıkması neyi gösterir?`;
    } else if (low.includes("yönetim") || low.includes("scrum") || low.includes("süreç")) {
      question = `Süreç Sorusu: '${term}' yaklaşımının temel adımları ve hedefleri nelerdir?`;
    }

    let formula = null;
    if (low.includes("çizelge sapması") || low.includes("schedule variance")) {
      formula = "Formül: SV = EV - PV (SV > 0: İleride | SV < 0: Geride)";
    } else if (low.includes("maliyet sapması") || low.includes("cost variance")) {
      formula = "Formül: CV = EV - AC (CV > 0: Tasarruf | CV < 0: Bütçe Aşımı)";
    } else if (low.includes("spi")) {
      formula = "Formül: SPI = EV / PV (SPI >= 1.0: Başarılı)";
    } else if (low.includes("cpi")) {
      formula = "Formül: CPI = EV / AC (CPI >= 1.0: Verimli)";
    }

    let analogy = `Günlük Hayat Benzetmesi: Tıpkı bir projede hedeflenen aşamaya ulaşıp ulaşmadığınızı kontrol etmek gibidir.`;
    if (low.includes("sapma") || low.includes("maliyet")) {
      analogy = `Günlük Hayat Benzetmesi: Araba navigasyonunun varış saatine göre kaç dakika saptığınızı hesaplaması gibidir.`;
    }

    cues.push({
      id: idx + 1,
      term: term,
      timestamp: time,
      time_s: timeS,
      question: question,
      category: idx < 3 ? "Kritik Terim" : "Destekleyici Terim",
    });

    notes.push({
      id: idx + 1,
      term: term,
      timestamp: time,
      time_s: timeS,
      explanation: expl,
      analogy: analogy,
      formula: formula,
      exam_tip: `Sınav İpucu: '${term}' kavramının tanımına ve bağlantılı olduğu aşamalara dikkat edin.`,
    });
  });

  const summary = {
    main_takeaway: `Bu ders çalışmasında ${title} konusuna ait ${rawTerms.length} temel kavram CEFR B1 düzeyinde incelenmiştir. Kavramların neden-sonuç ilişkilerine, pratik formüllerine ve günlük hayat analojilerine odaklanarak kalıcı öğrenme sağlanmalıdır.`,
  };

  let md = `# 🎓 Cornell Ders Notu: ${title}\n**Tarih:** ${dateStr} | **Süre:** ${durStr} | **Seviye:** CEFR B1 | **Terim:** ${rawTerms.length}\n\n---\n\n`;
  md += `| SOL SÜTUN (İPUÇLARI & SORULAR) | SAĞ SÜTUN (DERS NOTLARI & FORMÜLLER) |\n| :--- | :--- |\n`;
  cues.forEach((c, i) => {
    const n = notes[i] || {};
    const left = `**${c.term}** [${c.timestamp}]<br/><br/>*❓ ${c.question}*`.replace(/\n/g, ' ');
    let right = `**B1 Açıklama:** ${n.explanation}`;
    if (n.formula) right += `<br/><br/>**📐 ${n.formula}**`;
    if (n.analogy) right += `<br/><br/>*💡 ${n.analogy}*`;
    right = right.replace(/\n/g, ' ');
    md += `| ${left} | ${right} |\n`;
  });
  md += `\n---\n\n## 📌 BİLİŞSEL SENTEZ & ÖZET\n> ${summary.main_takeaway}\n\n*EduTerm AI — Cornell Study System*`;

  return {
    title: title,
    date: dateStr,
    duration: durStr,
    total_terms: rawTerms.length,
    cues: cues,
    notes: notes,
    summary: summary,
    markdown: md,
  };
}

async function renderCornellNotesModal() {
  const modal = document.getElementById("cornellModal");
  if (!modal) return;

  const targetJobId = currentResult?.job_id || (currentVideoId ? `preview_${currentVideoId}` : "demo_evm");
  const lectureTitle = currentResult?.video_meta?.title || "Ders Notu & Bilişsel Çıkarım";

  modal.classList.remove("hidden");

  let sheetData = null;

  try {
    const res = await fetch("/api/cornell-notes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: targetJobId,
        custom_title: lectureTitle,
      }),
    });
    if (res.ok) {
      sheetData = await res.json();
    }
  } catch (err) {
    console.warn("Server Cornell fetch notice:", err);
  }

  if (!sheetData) {
    sheetData = generateClientSideCornellData(currentResult);
  }

  activeCornellData = sheetData;
  activeCornellMarkdown = sheetData.markdown || "";

  // Populate DOM elements
  const titleEl = document.getElementById("cornellDocTitle");
  const dateEl = document.getElementById("cornellDocDate");
  const durEl = document.getElementById("cornellDocDuration");
  const countEl = document.getElementById("cornellDocTermCount");
  const cuesContent = document.getElementById("cornellCuesContent");
  const notesContent = document.getElementById("cornellNotesContent");
  const summaryText = document.getElementById("cornellSummaryText");

  if (titleEl) titleEl.innerText = sheetData.title || lectureTitle;
  if (dateEl) dateEl.innerText = sheetData.date || new Date().toLocaleDateString("tr-TR");
  if (durEl) durEl.innerText = sheetData.duration || "--:--";
  if (countEl) countEl.innerText = `${sheetData.total_terms || 0} Terim`;

  // Render Cues Column (Left)
  if (cuesContent) {
    cuesContent.innerHTML = "";
    (sheetData.cues || []).forEach((c) => {
      const card = document.createElement("div");
      card.className = "cornell-cue-card";
      card.innerHTML = `
        <div class="cornell-cue-top">
          <span class="cornell-cue-term">${escapeHtml(c.term)}</span>
          ${c.timestamp && c.timestamp !== "--:--" ? `<span class="cornell-cue-time" title="Videoda bu ana sar">${escapeHtml(c.timestamp)}</span>` : ""}
        </div>
        <div class="cornell-cue-question">❓ ${escapeHtml(c.question)}</div>
      `;
      const timeBtn = card.querySelector(".cornell-cue-time");
      if (timeBtn && c.time_s != null) {
        timeBtn.addEventListener("click", () => {
          seekPlayerTo(Number(c.time_s));
          showToast(`Videoda ${c.timestamp} saniyesine sarıldı.`, "info");
        });
      }
      cuesContent.appendChild(card);
    });
  }

  // Render Notes Column (Right)
  if (notesContent) {
    notesContent.innerHTML = "";
    (sheetData.notes || []).forEach((n) => {
      const card = document.createElement("div");
      card.className = "cornell-note-card";
      let formulaHtml = n.formula ? `<div class="cornell-note-formula">📐 ${escapeHtml(n.formula)}</div>` : "";
      let analogyHtml = n.analogy ? `<div class="cornell-note-analogy">💡 ${escapeHtml(n.analogy)}</div>` : "";
      let tipHtml = n.exam_tip ? `<div class="cornell-note-examtip">⭐ ${escapeHtml(n.exam_tip)}</div>` : "";

      card.innerHTML = `
        <div class="cornell-note-term-title">
          <svg class="ui-icon ui-icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          <span>${escapeHtml(n.term)}</span>
        </div>
        <div class="cornell-note-expl">${escapeHtml(n.explanation)}</div>
        ${formulaHtml}
        ${analogyHtml}
        ${tipHtml}
      `;
      notesContent.appendChild(card);
    });
  }

  // Render Summary & Synthesis (Bottom)
  if (summaryText) {
    summaryText.innerText = sheetData.summary?.main_takeaway || "Bu dersin temel kavramları B1 düzeyinde sadeleştirilmiş ve sınav hazırlığı için yapılandırılmıştır.";
  }
}

/* ==========================================================================
   INTERACTIVE CONCEPT KNOWLEDGE GRAPH (AUSUBEL MEANINGFUL LEARNING ENGINE)
   ========================================================================== */

let activeConceptGraphData = null;
let conceptGraphState = {
  canvas: null,
  ctx: null,
  nodes: [],
  edges: [],
  nodeMap: new Map(),
  selectedNode: null,
  hoveredNode: null,
  panX: 0,
  panY: 0,
  zoom: 1.0,
  isDragging: false,
  draggedNode: null,
  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  physicsActive: true,
  filterCategory: "all",
  searchQuery: "",
  animId: null,
  width: 0,
  height: 0,
  dpr: 1,
};

function generateClientSideConceptGraph(result) {
  const meta = result?.video_meta || {};
  const lectureTitle = meta.title || "Ders Kavram Ağı";
  const lang = result?.lang || "tr";
  const rawTerms = Array.isArray(result?.llm?.terms) && result.llm.terms.length > 0
    ? result.llm.terms
    : [
        { term: "Kazanılmış Değer Yönetimi", explanation: "Projenin kapsam, maliyet ve zaman çizelgesi performansını bütünleştiren yöntem (EVM)." },
        { term: "Kazanılmış Değer", explanation: "Tamamlanan işin onaylanmış bütçe karşılığı (EV)." },
        { term: "Planlanmış Değer", explanation: "Belirli bir tarihe kadar yapılması planlanan bütçe (PV)." },
        { term: "Gerçekleşen Maliyet", explanation: "Tamamlanan iş için fiilen harcanan toplam para (AC)." },
        { term: "Maliyet Sapması", explanation: "EV ile AC arasındaki fark (CV = EV - AC). Negatif olması bütçe aşımını gösterir." },
        { term: "Çizelge Sapması", explanation: "EV ile PV arasındaki fark (SV = EV - PV). Takvim ilerlemesini gösterir." },
        { term: "Maliyet Performans İndeksi", explanation: "Kazanılmış Değerin Gerçekleşen Maliyete oranı (CPI = EV / AC)." },
        { term: "Çizelge Performans İndeksi", explanation: "Kazanılmış Değerin Planlanmış Değere oranı (SPI = EV / PV)." },
        { term: "Temel Çizgi", explanation: "Performans ölçümü için referans alınan başlangıç planı (Baseline)." },
        { term: "İş Kırılım Yapısı", explanation: "Hedefi yönetilebilir iş paketlerine bölen hiyerarşik yapı (WBS)." },
        { term: "İzleme", explanation: "Proje performans verilerinin düzenli olarak toplanması ve raporlanması." },
        { term: "Kontrol", explanation: "Sapmaların belirlenip düzeltici önlemlerin devreye sokulması." },
        { term: "Düzeltici Eylemler", explanation: "Sapmaları giderip projeyi plana döndürmek için yapılan müdahaleler." }
      ];

  const segments = Array.isArray(result?.asr?.segments) ? result.asr.segments : [];

  const nodes = [];
  const nodeMap = new Map();

  rawTerms.forEach((item, idx) => {
    const name = (item.term || "").trim();
    if (!name) return;
    const nameLow = name.toLowerCase();
    const id = nameLow.replace(/[^\w\-]+/g, "_").replace(/^_+|_+$/g, "") || `node_${idx}`;

    // Find all occurrences across segments
    let occurrences = [];
    if (Array.isArray(item.occurrences) && item.occurrences.length > 0) {
      occurrences = item.occurrences;
    } else {
      const words = nameLow.split(/\s+/).filter(w => w.length >= 3);
      const seenTimes = new Set();
      segments.forEach(seg => {
        const segTxt = (seg.text || "").trim();
        const segLow = segTxt.toLowerCase();
        let match = false;
        if (segLow.includes(nameLow)) {
          match = true;
        } else if (words.length >= 2 && words.every(w => segLow.includes(w))) {
          match = true;
        } else if (words.length === 1 && segLow.includes(words[0])) {
          match = true;
        }
        if (match && seg.start != null) {
          const s = Math.floor(Number(seg.start));
          if (!seenTimes.has(s)) {
            seenTimes.add(s);
            const m = Math.floor(s / 60);
            const remS = s % 60;
            occurrences.push({
              start_s: Number(seg.start),
              end_s: seg.end != null ? Number(seg.end) : Number(seg.start) + 4,
              timestamp: `${String(m).padStart(2, '0')}:${String(remS).padStart(2, '0')}`,
              snippet: segTxt
            });
          }
        }
      });
    }

    // Determine category
    let category = "concept";
    let categoryLabel = "Kavram";
    const explLow = (item.explanation || "").toLowerCase();
    if (nameLow.includes("yönetim") || nameLow.includes("evm") || explLow.includes("metodoloji") || explLow.includes("bütünleştiren")) {
      category = "core";
      categoryLabel = "Kök Kavram";
    } else if (nameLow.includes("değer") || nameLow.includes("sapma") || nameLow.includes("indeks") || nameLow.includes("maliyet") || nameLow.includes("çizelge") || nameLow.includes("cpi") || nameLow.includes("spi")) {
      category = "metric";
      categoryLabel = "Metrik & Formül";
    } else if (nameLow.includes("izleme") || nameLow.includes("kontrol") || nameLow.includes("eylem") || nameLow.includes("süreç")) {
      category = "process";
      categoryLabel = "Süreç & Eylem";
    } else if (nameLow.includes("plan") || nameLow.includes("çizgi") || nameLow.includes("yapı") || nameLow.includes("şema") || nameLow.includes("wbs")) {
      category = "artifact";
      categoryLabel = "Belge & Yapı";
    }

    // Detect formula
    let formula = null;
    if (nameLow.includes("maliyet sapması") || nameLow.includes("cv")) {
      formula = "CV = EV - AC (CV > 0: Tasarruf | CV < 0: Bütçe Aşımı)";
    } else if (nameLow.includes("çizelge sapması") || nameLow.includes("sv")) {
      formula = "SV = EV - PV (SV > 0: Takvimden Hızlı | SV < 0: Gecikme)";
    } else if (nameLow.includes("maliyet performans") || nameLow.includes("cpi")) {
      formula = "CPI = EV / AC (CPI >= 1.0: Verimli)";
    } else if (nameLow.includes("çizelge performans") || nameLow.includes("spi")) {
      formula = "SPI = EV / PV (SPI >= 1.0: İlerleme Hızlı)";
    }

    let analogy = "Günlük Hayat Benzetmesi: Bir hedefe giderken yol haritasında konumunuzu kontrol etmeye benzer.";
    if (category === "metric") {
      analogy = "Günlük Hayat Benzetmesi: Arabanın hız ve yakıt göstergesi gibi projenin sağlığını anlık bildirir.";
    } else if (category === "process") {
      analogy = "Günlük Hayat Benzetmesi: Trafikte sapma görünce navigasyonun rotayı yeniden hesaplaması gibidir.";
    }

    const nodeObj = {
      id: id,
      label: name,
      category: category,
      category_label: categoryLabel,
      explanation: item.explanation || "Bu ders konusunun temel teknik kavramı.",
      analogy: analogy,
      formula: formula,
      occurrences: occurrences,
      occurrences_count: occurrences.length,
      first_timestamp: occurrences.length > 0 ? occurrences[0].timestamp : (item.timestamp || "--:--"),
      first_start_s: occurrences.length > 0 ? occurrences[0].start_s : (item.start_s || 0),
    };

    nodes.push(nodeObj);
    nodeMap.set(id, nodeObj);
  });

  // Relationships
  const edges = [];
  const edgeKeys = new Set();
  function addEdge(u, v, rel, type, desc) {
    if (u === v || !nodeMap.has(u) || !nodeMap.has(v)) return;
    const key = `${u}->${v}`;
    if (edgeKeys.has(key)) return;
    edgeKeys.add(key);
    edges.push({ source: u, target: v, relation: rel, type: type, description: desc });
  }

  // Pre-mapped Curated Triples
  const triples = [
    ["kazanılmış değer yönetimi", "kazanılmış değer", "bileşenidir", "part_whole", "Kazanılmış Değer, EVM analizinin temel taşıdır."],
    ["kazanılmış değer yönetimi", "planlanmış değer", "bileşenidir", "part_whole", "Planlanmış Değer, EVM analizinin bütçesel temelidir."],
    ["kazanılmış değer yönetimi", "gerçekleşen maliyet", "bileşenidir", "part_whole", "Gerçekleşen Maliyet, EVM'in fiili harcama ayağıdır."],
    ["kazanılmış değer", "maliyet sapması", "hesaplar", "calculates", "CV = EV - AC formülü ile maliyet sapması türetilir."],
    ["gerçekleşen maliyet", "maliyet sapması", "belirler", "calculates", "Harcama tutarı maliyet sapmasını etkiler."],
    ["kazanılmış değer", "çizelge sapması", "hesaplar", "calculates", "SV = EV - PV formülü ile takvim sapması türetilir."],
    ["planlanmış değer", "çizelge sapması", "belirler", "calculates", "Hedeflenen iş takvim sapmasının karşılaştırma bazıdır."],
    ["kazanılmış değer", "maliyet performans indeksi", "oranlar", "calculates", "CPI = EV / AC bütçe verimliliğini verir."],
    ["gerçekleşen maliyet", "maliyet performans indeksi", "oranlar", "calculates", "AC paydada yer alarak CPI katsayısını belirler."],
    ["kazanılmış değer", "çizelge performans indeksi", "oranlar", "calculates", "SPI = EV / PV takvim ilerleme verimini verir."],
    ["planlanmış değer", "çizelge performans indeksi", "oranlar", "calculates", "PV paydada yer alarak SPI katsayısını belirler."],
    ["izleme", "kontrol", "besler", "causal", "İzleme verisi sapma kontrol sürecine girdi sağlar."],
    ["maliyet sapması", "düzeltici eylemler", "tetikler", "causal", "Bütçe aşımı düzeltici müdahaleyi gerektirir."],
    ["çizelge sapması", "düzeltici eylemler", "tetikler", "causal", "Zaman gecikmesi toparlanma tedbirlerini başlatır."],
    ["temel çizgi", "izleme", "referans alır", "prerequisite", "İzleme süreci performansı onaylı temel çizgiyle kıyaslar."],
    ["proje yönetimi", "kazanılmış değer yönetimi", "metodudur", "part_whole", "EVM, proje kontrolünde uygulanan ana analitik yöntemdir."],
    ["proje yönetimi", "proje yönetim planı", "oluşturur", "prerequisite", "Planlama tüm süreçlerin başlangıç rehberidir."],
    ["proje yönetim planı", "temel çizgi", "belirler", "part_whole", "Yönetim planı performans temel çizgisini kapsar."],
    ["proje yönetim planı", "iş kırılım yapısı", "ayrıştırır", "part_whole", "Plan, işleri hiyerarşik WBS parçalarına böler."]
  ];

  triples.forEach(([sPat, tPat, rel, type, desc]) => {
    let src = null, tgt = null;
    nodes.forEach(n => {
      const l = n.label.toLowerCase();
      if (l.includes(sPat) || sPat.includes(l)) src = n.id;
      if (l.includes(tPat) || tPat.includes(l)) tgt = n.id;
    });
    if (src && tgt && src !== tgt) addEdge(src, tgt, rel, type, desc);
  });

  // Semantic textual linking
  nodes.forEach((na, i) => {
    const explA = (na.explanation || "").toLowerCase();
    nodes.forEach((nb, j) => {
      if (i === j) return;
      const lb = nb.label.toLowerCase();
      if (lb.length >= 4 && explA.includes(lb)) {
        let rLbl = "ile ilişkilidir", rType = "correlates";
        if (explA.includes("bileşen") || explA.includes("parça") || explA.includes("içerir")) {
          rLbl = "bileşenidir"; rType = "part_whole";
        } else if (explA.includes("hesap") || explA.includes("oran") || explA.includes("fark")) {
          rLbl = "hesaplar"; rType = "calculates";
        } else if (explA.includes("neden") || explA.includes("etki") || explA.includes("tetikler")) {
          rLbl = "etkiler"; rType = "causal";
        }
        addEdge(na.id, nb.id, rLbl, rType, `${na.label} ile ${nb.label} derste bağlantılıdır.`);
      }
    });
  });

  // Guarantee connected topology
  const hub = nodes.find(n => n.category === "core") || nodes[0];
  if (hub) {
    const connectedNodes = new Set();
    edges.forEach(e => { connectedNodes.add(e.source); connectedNodes.add(e.target); });
    nodes.forEach(n => {
      if (!connectedNodes.has(n.id) && n.id !== hub.id) {
        addEdge(hub.id, n.id, "kapsar", "part_whole", `${hub.label} ders çerçevesinde ${n.label} kavramını kapsar.`);
      }
    });
  }

  return {
    title: lectureTitle,
    lang: lang,
    nodes: nodes,
    edges: edges,
    pedagogical_meta: {
      theory: "Ausubel'in Anlamlı Öğrenme Teorisi (Cognitive Mapping)",
      total_nodes: nodes.length,
      total_edges: edges.length,
    }
  };
}

async function openConceptKnowledgeGraph() {
  const modal = document.getElementById("conceptGraphModal");
  if (!modal) return;
  modal.classList.remove("hidden");

  let graphData = null;
  const targetJobId = currentResult?.job_id || (currentVideoId ? `preview_${currentVideoId}` : "demo_evm");
  const lectureTitle = currentResult?.video_meta?.title || "Ders Kavram Ağı";

  try {
    const res = await fetch("/api/concept-graph", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: targetJobId,
        custom_title: lectureTitle,
        lang: currentResult?.lang || "tr",
      }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data && data.graph) {
        graphData = data.graph;
      }
    }
  } catch (err) {
    console.warn("Server concept graph fetch fallback:", err);
  }

  if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
    graphData = generateClientSideConceptGraph(currentResult);
  }

  activeConceptGraphData = graphData;

  // Update Counters
  const nodeCountEl = document.getElementById("graphNodeCount");
  const edgeCountEl = document.getElementById("graphEdgeCount");
  const modalTitleEl = document.getElementById("conceptGraphModalTitle");

  if (nodeCountEl) nodeCountEl.innerText = graphData.nodes.length;
  if (edgeCountEl) edgeCountEl.innerText = graphData.edges.length;
  if (modalTitleEl && graphData.title) modalTitleEl.innerText = `${graphData.title} • Kavram Ağı`;

  // Init Engine
  initConceptGraphEngine(graphData);

  // Auto-select first central node
  const hub = graphData.nodes.find(n => n.category === "core") || graphData.nodes[0];
  if (hub) {
    selectConceptGraphNode(hub);
  }
}

function initConceptGraphEngine(graphData) {
  const container = document.getElementById("graphCanvasContainer");
  const canvas = document.getElementById("conceptGraphCanvas");
  if (!container || !canvas) return;

  stopConceptGraphAnimation();

  const dpr = window.devicePixelRatio || 1;
  const rect = container.getBoundingClientRect();
  const w = rect.width || 800;
  const h = rect.height || 600;

  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  conceptGraphState.canvas = canvas;
  conceptGraphState.ctx = ctx;
  conceptGraphState.width = w;
  conceptGraphState.height = h;
  conceptGraphState.dpr = dpr;
  conceptGraphState.panX = 0;
  conceptGraphState.panY = 0;
  conceptGraphState.zoom = 1.0;
  conceptGraphState.physicsActive = true;
  conceptGraphState.hoveredNode = null;
  conceptGraphState.selectedNode = null;
  conceptGraphState.nodeMap = new Map();

  // Position nodes in radial concentric orbits
  const n = graphData.nodes.length;
  const centerX = w / 2;
  const centerY = h / 2;
  const radius = Math.min(w, h) * 0.35;

  const nodes = graphData.nodes.map((node, i) => {
    let angle = (i / n) * Math.PI * 2;
    let r = radius;
    if (node.category === "core") {
      r = radius * 0.25;
    } else if (node.category === "metric") {
      r = radius * 0.65;
    }
    const x = centerX + Math.cos(angle) * r + (Math.random() - 0.5) * 40;
    const y = centerY + Math.sin(angle) * r + (Math.random() - 0.5) * 40;

    let nodeRadius = 22;
    if (node.category === "core") nodeRadius = 28;
    else if (node.category === "metric") nodeRadius = 24;

    const nodeObj = {
      ...node,
      x: x,
      y: y,
      vx: 0,
      vy: 0,
      radius: nodeRadius,
    };
    conceptGraphState.nodeMap.set(node.id, nodeObj);
    return nodeObj;
  });

  const edges = graphData.edges.map(e => ({
    ...e,
    sourceNode: conceptGraphState.nodeMap.get(e.source),
    targetNode: conceptGraphState.nodeMap.get(e.target),
  })).filter(e => e.sourceNode && e.targetNode);

  conceptGraphState.nodes = nodes;
  conceptGraphState.edges = edges;

  // Bind Canvas Interaction Events
  setupCanvasEvents(canvas);

  // Start Animation
  startConceptGraphAnimation();
}

function setupCanvasEvents(canvas) {
  // Remove prior cloned listeners
  const newCanvas = canvas.cloneNode(true);
  canvas.parentNode.replaceChild(newCanvas, canvas);
  conceptGraphState.canvas = newCanvas;
  conceptGraphState.ctx = newCanvas.getContext("2d");
  conceptGraphState.ctx.scale(conceptGraphState.dpr, conceptGraphState.dpr);

  function getMousePos(e) {
    const rect = newCanvas.getBoundingClientRect();
    const clientX = e.clientX ?? (e.touches && e.touches[0]?.clientX) ?? 0;
    const clientY = e.clientY ?? (e.touches && e.touches[0]?.clientY) ?? 0;
    const screenX = clientX - rect.left;
    const screenY = clientY - rect.top;
    // Invert pan and zoom to get world coordinates
    const worldX = (screenX - conceptGraphState.panX) / conceptGraphState.zoom;
    const worldY = (screenY - conceptGraphState.panY) / conceptGraphState.zoom;
    return { screenX, screenY, worldX, worldY };
  }

  function findNodeUnder(worldX, worldY) {
    for (let i = conceptGraphState.nodes.length - 1; i >= 0; i--) {
      const node = conceptGraphState.nodes[i];
      if (!isNodeVisible(node)) continue;
      const dx = worldX - node.x;
      const dy = worldY - node.y;
      if (dx * dx + dy * dy <= (node.radius + 6) * (node.radius + 6)) {
        return node;
      }
    }
    return null;
  }

  newCanvas.addEventListener("pointerdown", (e) => {
    const pos = getMousePos(e);
    const hitNode = findNodeUnder(pos.worldX, pos.worldY);
    if (hitNode) {
      conceptGraphState.isDragging = true;
      conceptGraphState.draggedNode = hitNode;
      hitNode.vx = 0;
      hitNode.vy = 0;
    } else {
      conceptGraphState.isPanning = true;
      conceptGraphState.panStartX = pos.screenX - conceptGraphState.panX;
      conceptGraphState.panStartY = pos.screenY - conceptGraphState.panY;
      newCanvas.style.cursor = "grabbing";
    }
  });

  window.addEventListener("pointermove", (e) => {
    if (!conceptGraphState.canvas) return;
    const pos = getMousePos(e);

    if (conceptGraphState.isDragging && conceptGraphState.draggedNode) {
      conceptGraphState.draggedNode.x = pos.worldX;
      conceptGraphState.draggedNode.y = pos.worldY;
      conceptGraphState.draggedNode.vx = 0;
      conceptGraphState.draggedNode.vy = 0;
      drawConceptGraph();
      return;
    }

    if (conceptGraphState.isPanning) {
      conceptGraphState.panX = pos.screenX - conceptGraphState.panStartX;
      conceptGraphState.panY = pos.screenY - conceptGraphState.panStartY;
      drawConceptGraph();
      return;
    }

    // Hover detection
    const hitNode = findNodeUnder(pos.worldX, pos.worldY);
    if (hitNode !== conceptGraphState.hoveredNode) {
      conceptGraphState.hoveredNode = hitNode;
      newCanvas.style.cursor = hitNode ? "pointer" : "grab";
      drawConceptGraph();
    }
  });

  window.addEventListener("pointerup", (e) => {
    if (conceptGraphState.isDragging && conceptGraphState.draggedNode) {
      const pos = getMousePos(e);
      const hitNode = findNodeUnder(pos.worldX, pos.worldY);
      if (hitNode && hitNode === conceptGraphState.draggedNode) {
        selectConceptGraphNode(hitNode);
      }
    }
    conceptGraphState.isDragging = false;
    conceptGraphState.draggedNode = null;
    conceptGraphState.isPanning = false;
    if (newCanvas) newCanvas.style.cursor = "grab";
  });

  newCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const pos = getMousePos(e);
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
    const newZoom = Math.max(0.35, Math.min(2.8, conceptGraphState.zoom * zoomFactor));

    // Zoom towards mouse
    conceptGraphState.panX = pos.screenX - (pos.screenX - conceptGraphState.panX) * (newZoom / conceptGraphState.zoom);
    conceptGraphState.panY = pos.screenY - (pos.screenY - conceptGraphState.panY) * (newZoom / conceptGraphState.zoom);
    conceptGraphState.zoom = newZoom;
    drawConceptGraph();
  }, { passive: false });
}

function isNodeVisible(node) {
  if (conceptGraphState.filterCategory !== "all") {
    if (node.category !== conceptGraphState.filterCategory) return false;
  }
  return true;
}

function startConceptGraphAnimation() {
  if (conceptGraphState.animId) cancelAnimationFrame(conceptGraphState.animId);

  let stepCount = 0;

  function loop() {
    if (conceptGraphState.physicsActive) {
      updateConceptGraphPhysics();
    }
    drawConceptGraph();
    stepCount++;
    // Continue running while modal is open
    const modal = document.getElementById("conceptGraphModal");
    if (modal && !modal.classList.contains("hidden")) {
      conceptGraphState.animId = requestAnimationFrame(loop);
    }
  }

  conceptGraphState.animId = requestAnimationFrame(loop);
}

function stopConceptGraphAnimation() {
  if (conceptGraphState.animId) {
    cancelAnimationFrame(conceptGraphState.animId);
    conceptGraphState.animId = null;
  }
}

function updateConceptGraphPhysics() {
  const nodes = conceptGraphState.nodes;
  const edges = conceptGraphState.edges;
  const centerX = conceptGraphState.width / 2;
  const centerY = conceptGraphState.height / 2;

  // 1. Coulomb Repulsion
  for (let i = 0; i < nodes.length; i++) {
    const na = nodes[i];
    for (let j = i + 1; j < nodes.length; j++) {
      const nb = nodes[j];
      const dx = nb.x - na.x;
      const dy = nb.y - na.y;
      const distSq = dx * dx + dy * dy + 1;
      const dist = Math.sqrt(distSq);

      if (dist < 400) {
        const force = 1800 / distSq;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        if (na !== conceptGraphState.draggedNode) {
          na.vx -= fx;
          na.vy -= fy;
        }
        if (nb !== conceptGraphState.draggedNode) {
          nb.vx += fx;
          nb.vy += fy;
        }
      }
    }
  }

  // 2. Hooke's Law Spring Attraction along Edges
  const restLength = 140;
  const springK = 0.04;
  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    const na = edge.sourceNode;
    const nb = edge.targetNode;
    if (!na || !nb) continue;

    const dx = nb.x - na.x;
    const dy = nb.y - na.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const displacement = dist - restLength;
    const force = displacement * springK;
    const fx = (dx / dist) * force;
    const fy = (dy / dist) * force;

    if (na !== conceptGraphState.draggedNode) {
      na.vx += fx;
      na.vy += fy;
    }
    if (nb !== conceptGraphState.draggedNode) {
      nb.vx -= fx;
      nb.vy -= fy;
    }
  }

  // 3. Center Gravity Pull
  const gravityK = 0.012;
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    if (node === conceptGraphState.draggedNode) continue;
    const dx = centerX - node.x;
    const dy = centerY - node.y;
    node.vx += dx * gravityK;
    node.vy += dy * gravityK;

    // Apply damping
    node.vx *= 0.86;
    node.vy *= 0.86;

    node.x += node.vx;
    node.y += node.vy;
  }
}

function drawConceptGraph() {
  const ctx = conceptGraphState.ctx;
  if (!ctx || !conceptGraphState.canvas) return;

  const w = conceptGraphState.width;
  const h = conceptGraphState.height;

  ctx.clearRect(0, 0, w, h);

  ctx.save();
  ctx.translate(conceptGraphState.panX, conceptGraphState.panY);
  ctx.scale(conceptGraphState.zoom, conceptGraphState.zoom);

  const selected = conceptGraphState.selectedNode;
  const hovered = conceptGraphState.hoveredNode;
  const search = conceptGraphState.searchQuery;

  // Draw Edges
  conceptGraphState.edges.forEach((edge) => {
    const na = edge.sourceNode;
    const nb = edge.targetNode;
    if (!na || !nb) return;

    const isConnectedToFocus = (selected && (na.id === selected.id || nb.id === selected.id)) ||
                               (hovered && (na.id === hovered.id || nb.id === hovered.id));

    const isDimmed = (selected || hovered) && !isConnectedToFocus;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(na.x, na.y);
    ctx.lineTo(nb.x, nb.y);

    if (isConnectedToFocus) {
      ctx.strokeStyle = "#818cf8";
      ctx.lineWidth = 2.8;
      ctx.globalAlpha = 1.0;
    } else if (isDimmed) {
      ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.15;
    } else {
      ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.6;
    }
    ctx.stroke();

    // Draw Arrowhead pointing to target
    const angle = Math.atan2(nb.y - na.y, nb.x - na.x);
    const arrowDist = nb.radius + 4;
    const arrowX = nb.x - Math.cos(angle) * arrowDist;
    const arrowY = nb.y - Math.sin(angle) * arrowDist;

    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - 8 * Math.cos(angle - Math.PI / 6), arrowY - 8 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(arrowX - 8 * Math.cos(angle + Math.PI / 6), arrowY - 8 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();

    // Draw Relation Label Pill along Edge if connected or zoomed in
    if ((isConnectedToFocus || conceptGraphState.zoom >= 0.9) && !isDimmed) {
      const midX = (na.x + nb.x) / 2;
      const midY = (na.y + nb.y) / 2;
      const relText = edge.relation || "";
      ctx.font = "bold 9px 'Inter', system-ui, sans-serif";
      const metrics = ctx.measureText(relText);
      const pad = 4;

      ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
      ctx.beginPath();
      ctx.roundRect(midX - metrics.width / 2 - pad, midY - 7 - pad / 2, metrics.width + pad * 2, 14 + pad, 4);
      ctx.fill();
      ctx.strokeStyle = isConnectedToFocus ? "#818cf8" : "rgba(255, 255, 255, 0.15)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = isConnectedToFocus ? "#c7d2fe" : "#94a3b8";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(relText, midX, midY);
    }

    ctx.restore();
  });

  // Category Color Map
  const colorMap = {
    core: { fill: "#8b5cf6", border: "#a78bfa", glow: "rgba(139, 92, 246, 0.5)" },
    metric: { fill: "#10b981", border: "#34d399", glow: "rgba(16, 185, 129, 0.5)" },
    process: { fill: "#f59e0b", border: "#fbbf24", glow: "rgba(245, 158, 11, 0.5)" },
    artifact: { fill: "#06b6d4", border: "#22d3ee", glow: "rgba(6, 182, 212, 0.5)" },
    concept: { fill: "#6366f1", border: "#818cf8", glow: "rgba(99, 102, 241, 0.5)" },
  };

  // Draw Nodes
  conceptGraphState.nodes.forEach((node) => {
    const isVisible = isNodeVisible(node);
    const isSelected = selected && node.id === selected.id;
    const isHovered = hovered && node.id === hovered.id;
    const matchesSearch = search && node.label.toLowerCase().includes(search);

    const isConnectedToFocus = (selected && conceptGraphState.edges.some(e =>
      (e.source === selected.id && e.target === node.id) ||
      (e.target === selected.id && e.source === node.id)
    )) || (hovered && conceptGraphState.edges.some(e =>
      (e.source === hovered.id && e.target === node.id) ||
      (e.target === hovered.id && e.source === node.id)
    ));

    const isDimmed = !isVisible || ((selected || hovered) && !isSelected && !isHovered && !isConnectedToFocus);

    ctx.save();
    ctx.globalAlpha = isDimmed ? 0.15 : 1.0;

    const colors = colorMap[node.category] || colorMap.concept;

    // Glowing Ring if Selected or Hovered or Matches Search
    if (isSelected || isHovered || matchesSearch) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius + 8, 0, Math.PI * 2);
      ctx.fillStyle = colors.glow;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius + 5, 0, Math.PI * 2);
      ctx.strokeStyle = isSelected ? "#ffffff" : colors.border;
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    // Node Body
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
    ctx.fillStyle = colors.fill;
    ctx.fill();
    ctx.strokeStyle = isSelected ? "#ffffff" : colors.border;
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.stroke();

    // Occurrence Badge Pill (Top-Right of node)
    if (node.occurrences_count > 0 && !isDimmed) {
      const badgeX = node.x + node.radius * 0.7;
      const badgeY = node.y - node.radius * 0.7;
      ctx.beginPath();
      ctx.arc(badgeX, badgeY, 8, 0, Math.PI * 2);
      ctx.fillStyle = "#1e1b4b";
      ctx.fill();
      ctx.strokeStyle = "#818cf8";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.font = "bold 8px 'Inter', system-ui, sans-serif";
      ctx.fillStyle = "#c7d2fe";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`${node.occurrences_count}x`, badgeX, badgeY);
    }

    // Node Label
    if (!isDimmed || isSelected || isHovered) {
      ctx.font = `600 ${node.category === 'core' ? 12 : 11}px 'Inter', system-ui, sans-serif`;
      const text = node.label;
      const textMetrics = ctx.measureText(text);
      const textWidth = textMetrics.width;
      const textY = node.y + node.radius + 14;

      // Label background pill
      ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
      ctx.beginPath();
      ctx.roundRect(node.x - textWidth / 2 - 6, textY - 9, textWidth + 12, 17, 4);
      ctx.fill();
      ctx.strokeStyle = isSelected ? "#818cf8" : "rgba(255, 255, 255, 0.12)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = isSelected ? "#ffffff" : "#f1f5f9";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, node.x, textY);
    }

    ctx.restore();
  });

  ctx.restore();
}

function selectConceptGraphNode(node) {
  if (!node) return;
  conceptGraphState.selectedNode = node;
  drawConceptGraph();

  const emptyState = document.getElementById("inspectorEmptyState");
  const content = document.getElementById("inspectorContent");
  if (emptyState) emptyState.classList.add("hidden");
  if (content) content.classList.remove("hidden");

  // Populate Inspector
  const titleEl = document.getElementById("inspectorTitle");
  const catPill = document.getElementById("inspectorCategoryPill");
  const explEl = document.getElementById("inspectorExplanation");
  const formulaSec = document.getElementById("inspectorFormulaSection");
  const formulaBox = document.getElementById("inspectorFormulaBox");
  const analogyEl = document.getElementById("inspectorAnalogy");
  const occCount = document.getElementById("inspectorOccurrencesCount");
  const timesList = document.getElementById("inspectorTimestampsList");
  const connList = document.getElementById("inspectorConnectionsList");
  const connCount = document.getElementById("inspectorConnectedCount");

  if (titleEl) titleEl.innerText = node.label;
  if (catPill) {
    catPill.innerText = node.category_label || "Kavram";
    catPill.className = `inspector-category-pill cat-${node.category}`;
  }
  if (explEl) explEl.innerText = node.explanation || "Açıklama bulunamadı.";

  if (node.formula) {
    if (formulaSec) formulaSec.classList.remove("hidden");
    if (formulaBox) formulaBox.innerText = node.formula;
  } else {
    if (formulaSec) formulaSec.classList.add("hidden");
  }

  if (analogyEl) analogyEl.innerText = node.analogy || "Günlük Hayat Benzetmesi: Konu çerçevesinde önemli bir yapı taşıdır.";

  // Render Video Occurrences
  const occurrences = node.occurrences || [];
  if (occCount) occCount.innerText = occurrences.length;

  if (timesList) {
    timesList.innerHTML = "";
    if (occurrences.length === 0) {
      timesList.innerHTML = `<div class="inspector-timestamp-card"><span class="inspector-time-badge">${node.first_timestamp || "--:--"}</span><p class="inspector-snippet-text">Bu kavram video genelinde ana tema olarak ele alınmaktadır.</p></div>`;
    } else {
      occurrences.forEach((occ) => {
        const card = document.createElement("div");
        card.className = "inspector-timestamp-card";
        card.title = `Videoda ${occ.timestamp} anına git`;
        card.innerHTML = `
          <span class="inspector-time-badge">
            <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            ${escapeHtml(occ.timestamp)}
          </span>
          <p class="inspector-snippet-text">${escapeHtml(occ.snippet || node.label)}</p>
        `;
        card.addEventListener("click", () => {
          seekPlayerTo(Number(occ.start_s));
          showToast(`Videoda ${occ.timestamp} konumuna sarıldı: ${node.label}`, "info");
        });
        timesList.appendChild(card);
      });
    }
  }

  // Render Connected Concepts List
  if (connList) {
    connList.innerHTML = "";
    const connectedEdges = conceptGraphState.edges.filter(
      e => e.source === node.id || e.target === node.id
    );
    if (connCount) connCount.innerText = connectedEdges.length;

    connectedEdges.forEach((e) => {
      const isSource = e.source === node.id;
      const targetNodeId = isSource ? e.target : e.source;
      const targetNode = conceptGraphState.nodeMap.get(targetNodeId);
      if (!targetNode) return;

      const chip = document.createElement("div");
      chip.className = "inspector-relation-chip";
      chip.innerHTML = `
        <span class="relation-chip-target">${escapeHtml(targetNode.label)}</span>
        <span class="relation-chip-verb">${escapeHtml(e.relation || "ilişkili")}</span>
      `;
      chip.addEventListener("click", () => {
        selectConceptGraphNode(targetNode);
        // Center view on target node
        centerGraphOnNode(targetNode);
      });
      connList.appendChild(chip);
    });
  }
}

function centerGraphOnNode(node) {
  if (!node) return;
  const centerX = conceptGraphState.width / 2;
  const centerY = conceptGraphState.height / 2;
  conceptGraphState.panX = centerX - node.x * conceptGraphState.zoom;
  conceptGraphState.panY = centerY - node.y * conceptGraphState.zoom;
  drawConceptGraph();
}

function resetConceptGraphView() {
  conceptGraphState.panX = 0;
  conceptGraphState.panY = 0;
  conceptGraphState.zoom = 1.0;
  drawConceptGraph();
}

function zoomConceptGraph(factor) {
  const centerX = conceptGraphState.width / 2;
  const centerY = conceptGraphState.height / 2;
  const newZoom = Math.max(0.35, Math.min(2.8, conceptGraphState.zoom * factor));
  conceptGraphState.panX = centerX - (centerX - conceptGraphState.panX) * (newZoom / conceptGraphState.zoom);
  conceptGraphState.panY = centerY - (centerY - conceptGraphState.panY) * (newZoom / conceptGraphState.zoom);
  conceptGraphState.zoom = newZoom;
  drawConceptGraph();
}

function updateDrawerConceptGraph(result) {
  const container = document.getElementById("drawerGraphNodesList");
  const badge = document.getElementById("drawerGraphBadge");
  if (!container) return;

  const graph = activeConceptGraphData || generateClientSideConceptGraph(result);
  const nodes = graph.nodes || [];

  if (badge) badge.innerText = `${nodes.length} Kavram • ${graph.edges?.length || 0} Bağ`;

  container.innerHTML = "";
  nodes.forEach((node) => {
    const card = document.createElement("div");
    card.className = "drawer-node-card";
    card.dataset.nodeId = node.id;
    card.dataset.label = node.label.toLowerCase();

    let timesHtml = "";
    const occurrences = node.occurrences || [];
    if (occurrences.length > 0) {
      timesHtml = occurrences.slice(0, 5).map(occ => `
        <span class="drawer-time-pill" data-sec="${occ.start_s}" title="Videoda ${occ.timestamp} anına sar">
          <svg class="ui-icon" style="width:10px;height:10px;" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          ${occ.timestamp}
        </span>
      `).join("");
      if (occurrences.length > 5) {
        timesHtml += `<span class="drawer-time-pill" style="opacity:0.7;">+${occurrences.length - 5}</span>`;
      }
    } else if (node.first_timestamp && node.first_timestamp !== "--:--") {
      timesHtml = `
        <span class="drawer-time-pill" data-sec="${node.first_start_s || 0}">
          <svg class="ui-icon" style="width:10px;height:10px;" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          ${node.first_timestamp}
        </span>
      `;
    }

    card.innerHTML = `
      <div class="drawer-node-card-header">
        <span class="drawer-node-label">${escapeHtml(node.label)}</span>
        <span class="drawer-node-cat">${escapeHtml(node.category_label || "Kavram")}</span>
      </div>
      <div class="drawer-node-occurrences">
        ${timesHtml}
      </div>
    `;

    // Timestamp pill clicks
    card.querySelectorAll(".drawer-time-pill[data-sec]").forEach(pill => {
      pill.addEventListener("click", (e) => {
        e.stopPropagation();
        const sec = Number(pill.dataset.sec);
        if (!isNaN(sec)) {
          seekPlayerTo(sec);
          showToast(`Videoda ${pill.innerText.trim()} konumuna sarıldı: ${node.label}`, "info");
        }
      });
    });

    // Card click opens modal centered on node
    card.addEventListener("click", () => {
      openConceptKnowledgeGraph();
      setTimeout(() => {
        const fullNode = conceptGraphState.nodeMap.get(node.id);
        if (fullNode) {
          selectConceptGraphNode(fullNode);
          centerGraphOnNode(fullNode);
        }
      }, 150);
    });

    container.appendChild(card);
  });
}

function filterDrawerGraphCards(query) {
  const container = document.getElementById("drawerGraphNodesList");
  if (!container) return;
  const q = (query || "").trim().toLowerCase();
  const cards = container.querySelectorAll(".drawer-node-card");
  cards.forEach(card => {
    const lbl = card.dataset.label || "";
    card.style.display = (!q || lbl.includes(q)) ? "block" : "none";
  });
}

function openRightSidebar(open = true) {
  setRightSidebarOpen(open);
}

function switchToTranscriptDrawer() {
  const tabTrans = document.getElementById("tabDrawerTranscript");
  const tabGraph = document.getElementById("tabDrawerGraph");
  const tabSocratic = document.getElementById("tabDrawerSocratic");

  const transBody = document.getElementById("transcriptDrawerBody");
  const graphBody = document.getElementById("conceptGraphDrawerBody");
  const socraticBody = document.getElementById("socraticDrawerBody");

  if (tabTrans) { tabTrans.classList.add("active"); tabTrans.setAttribute("aria-selected", "true"); }
  if (tabGraph) { tabGraph.classList.remove("active"); tabGraph.setAttribute("aria-selected", "false"); }
  if (tabSocratic) { tabSocratic.classList.remove("active"); tabSocratic.setAttribute("aria-selected", "false"); }

  if (transBody) transBody.classList.remove("hidden");
  if (graphBody) graphBody.classList.add("hidden");
  if (socraticBody) socraticBody.classList.add("hidden");
}

function switchToGraphDrawer() {
  const tabTrans = document.getElementById("tabDrawerTranscript");
  const tabGraph = document.getElementById("tabDrawerGraph");
  const tabSocratic = document.getElementById("tabDrawerSocratic");

  const transBody = document.getElementById("transcriptDrawerBody");
  const graphBody = document.getElementById("conceptGraphDrawerBody");
  const socraticBody = document.getElementById("socraticDrawerBody");

  if (tabTrans) { tabTrans.classList.remove("active"); tabTrans.setAttribute("aria-selected", "false"); }
  if (tabGraph) { tabGraph.classList.add("active"); tabGraph.setAttribute("aria-selected", "true"); }
  if (tabSocratic) { tabSocratic.classList.remove("active"); tabSocratic.setAttribute("aria-selected", "false"); }

  if (transBody) transBody.classList.add("hidden");
  if (graphBody) graphBody.classList.remove("hidden");
  if (socraticBody) socraticBody.classList.add("hidden");

  // Populate drawer concept graph if empty
  const container = document.getElementById("drawerGraphNodesList");
  if (container && (!container.children || container.children.length === 0)) {
    updateDrawerConceptGraph(currentResult);
  }
}

function setupConceptKnowledgeGraph() {
  const modal = document.getElementById("conceptGraphModal");
  const btnNav = document.getElementById("btnNavConceptGraph");
  const btnOpenHub = document.getElementById("btnOpenConceptGraph");
  const btnExpand = document.getElementById("btnExpandGraphModal");
  const btnClose = document.getElementById("btnCloseConceptGraphModal");
  const btnReset = document.getElementById("btnResetGraphView");
  const btnPhysics = document.getElementById("btnToggleGraphPhysics");
  const statusEl = document.getElementById("graphPhysicsStatus");
  const searchInput = document.getElementById("graphSearchInput");
  const zoomIn = document.getElementById("btnGraphZoomIn");
  const zoomOut = document.getElementById("btnGraphZoomOut");
  const filterChips = document.querySelectorAll(".graph-filter-chip");
  const drawerSearch = document.getElementById("drawerGraphSearchInput");
  const tabTranscript = document.getElementById("tabDrawerTranscript");
  const tabGraph = document.getElementById("tabDrawerGraph");
  const btnQuiz = document.getElementById("btnInspectorQuiz");

  if (tabTranscript) {
    tabTranscript.addEventListener("click", () => {
      switchToTranscriptDrawer();
    });
  }

  if (tabGraph) {
    tabGraph.addEventListener("click", () => {
      switchToGraphDrawer();
    });
  }

  if (btnNav) {
    btnNav.addEventListener("click", () => {
      openRightSidebar(true);
      switchToGraphDrawer();
    });
  }

  if (btnOpenHub) {
    btnOpenHub.addEventListener("click", () => {
      openConceptKnowledgeGraph();
    });
  }

  if (btnExpand) {
    btnExpand.addEventListener("click", () => {
      openConceptKnowledgeGraph();
    });
  }

  if (btnClose) {
    btnClose.addEventListener("click", () => {
      if (modal) modal.classList.add("hidden");
      stopConceptGraphAnimation();
    });
  }

  if (btnReset) {
    btnReset.addEventListener("click", () => {
      resetConceptGraphView();
    });
  }

  if (btnPhysics) {
    btnPhysics.addEventListener("click", () => {
      conceptGraphState.physicsActive = !conceptGraphState.physicsActive;
      if (statusEl) {
        statusEl.innerText = conceptGraphState.physicsActive ? "Fiziği Durdur" : "Fiziği Başlat";
      }
      if (conceptGraphState.physicsActive) {
        startConceptGraphAnimation();
      }
    });
  }

  if (zoomIn) {
    zoomIn.addEventListener("click", () => zoomConceptGraph(1.2));
  }

  if (zoomOut) {
    zoomOut.addEventListener("click", () => zoomConceptGraph(0.8));
  }

  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      conceptGraphState.searchQuery = (e.target.value || "").trim().toLowerCase();
      drawConceptGraph();
    });
  }

  filterChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      filterChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      conceptGraphState.filterCategory = chip.dataset.filter || "all";
      drawConceptGraph();
    });
  });

  if (drawerSearch) {
    drawerSearch.addEventListener("input", (e) => {
      filterDrawerGraphCards(e.target.value);
    });
  }

  if (btnQuiz) {
    btnQuiz.addEventListener("click", () => {
      if (modal) modal.classList.add("hidden");
      stopConceptGraphAnimation();
      openRightSidebar(true);
      switchToSocraticDrawer();
      const nodeLabel = conceptGraphState.selectedNode?.label || "bu kavram";
      sendSocraticMessage(`'${nodeLabel}' konusuyla ilgili bana bir pratik sınav sorusu sorabilir misin?`);
    });
  }

  // Global keyboard shortcut: 'G' for Concept Graph
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "g" || e.key === "G") {
      e.preventDefault();
      openConceptKnowledgeGraph();
    }
  });
}

/* ==========================================================================
   SOCRATIC AI TUTOR / CHAT (VANZO ET AL., ACL 2025)
   ========================================================================== */

let socraticChatHistory = [];

function setupSocraticTutor() {
  // Navigation Pill in Header
  const btnNav = document.getElementById("btnNavSocratic");
  if (btnNav) {
    btnNav.addEventListener("click", () => {
      openRightSidebar(true);
      switchToSocraticDrawer();
    });
  }

  // Drawer Tab Button
  const tabSocratic = document.getElementById("tabDrawerSocratic");
  if (tabSocratic) {
    tabSocratic.addEventListener("click", () => {
      switchToSocraticDrawer();
    });
  }

  // Mini Dock "Tam Sohbet" Button
  const btnSwitch = document.getElementById("btnSwitchToSocratic");
  if (btnSwitch) {
    btnSwitch.addEventListener("click", () => {
      switchToSocraticDrawer();
    });
  }

  // Clear Chat Button
  const btnClear = document.getElementById("btnClearSocraticChat");
  if (btnClear) {
    btnClear.addEventListener("click", () => {
      socraticChatHistory = [];
      const container = document.getElementById("socraticMessagesContainer");
      if (container) {
        container.innerHTML = `
          <div class="socratic-msg msg-assistant">
            <div class="socratic-bubble">
              <p>👋 Sohbet temizlendi. Ders videosunda aklına takılan herhangi bir kavramı veya formülü sorabilirsin.</p>
              <div class="socratic-suggested-row">
                <button type="button" class="socratic-chip" data-prompt="Bu formülü anlamadım, mantığını açıklar mısın?">💡 Bu formülü anlamadım</button>
                <button type="button" class="socratic-chip" data-prompt="Neden maliyet sapması negatif çıktı?">❓ Neden negatif çıktı?</button>
              </div>
            </div>
          </div>
        `;
        bindSocraticChipEvents(container);
      }
      const footerChips = document.getElementById("socraticChatQuickChips");
      if (footerChips) footerChips.innerHTML = "";
    });
  }

  // Mini Form Submission
  const miniForm = document.getElementById("socraticMiniForm");
  const miniInput = document.getElementById("socraticMiniInput");
  const btnMiniSend = document.getElementById("btnSendSocraticMini");
  const handleMiniSubmit = () => {
    const text = (miniInput?.value || "").trim();
    if (!text) return;
    if (miniInput) miniInput.value = "";
    switchToSocraticDrawer();
    sendSocraticMessage(text);
  };
  if (miniForm) {
    miniForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleMiniSubmit();
    });
  }
  if (btnMiniSend) {
    btnMiniSend.addEventListener("click", (e) => {
      e.preventDefault();
      handleMiniSubmit();
    });
  }

  // Full Form Submission
  const fullForm = document.getElementById("socraticFullForm");
  const fullInput = document.getElementById("socraticFullInput");
  const btnFullSend = document.getElementById("btnSendSocraticFull");
  const handleFullSubmit = () => {
    const text = (fullInput?.value || "").trim();
    if (!text) return;
    if (fullInput) fullInput.value = "";
    sendSocraticMessage(text);
  };
  if (fullForm) {
    fullForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleFullSubmit();
    });
  }
  if (btnFullSend) {
    btnFullSend.addEventListener("click", (e) => {
      e.preventDefault();
      handleFullSubmit();
    });
  }

  // Mini Quick Chips
  const miniChips = document.getElementById("socraticMiniQuickChips");
  if (miniChips) {
    bindSocraticChipEvents(miniChips);
  }

  // Initial Container Chips
  const container = document.getElementById("socraticMessagesContainer");
  if (container) {
    bindSocraticChipEvents(container);
  }

  // Keyboard shortcut: 'S' for Socratic Chat
  document.addEventListener("keydown", (e) => {
    if (e.key === "s" || e.key === "S") {
      const activeEl = document.activeElement;
      const tag = (activeEl?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || activeEl?.isContentEditable) return;
      e.preventDefault();
      openRightSidebar(true);
      switchToSocraticDrawer();
      const fullIn = document.getElementById("socraticFullInput");
      if (fullIn) fullIn.focus();
    }
  });
}

function switchToSocraticDrawer() {
  const tabTrans = document.getElementById("tabDrawerTranscript");
  const tabGraph = document.getElementById("tabDrawerGraph");
  const tabSocratic = document.getElementById("tabDrawerSocratic");

  const transBody = document.getElementById("transcriptDrawerBody");
  const graphBody = document.getElementById("conceptGraphDrawerBody");
  const socraticBody = document.getElementById("socraticDrawerBody");

  if (tabTrans) { tabTrans.classList.remove("active"); tabTrans.setAttribute("aria-selected", "false"); }
  if (tabGraph) { tabGraph.classList.remove("active"); tabGraph.setAttribute("aria-selected", "false"); }
  if (tabSocratic) { tabSocratic.classList.add("active"); tabSocratic.setAttribute("aria-selected", "true"); }

  if (transBody) transBody.classList.add("hidden");
  if (graphBody) graphBody.classList.add("hidden");
  if (socraticBody) socraticBody.classList.remove("hidden");

  // Focus input
  setTimeout(() => {
    const fullIn = document.getElementById("socraticFullInput");
    if (fullIn) fullIn.focus();
  }, 100);
}

function bindSocraticChipEvents(parentEl) {
  if (!parentEl) return;
  parentEl.querySelectorAll(".socratic-chip[data-prompt]").forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.dataset.prompt;
      if (prompt) {
        switchToSocraticDrawer();
        sendSocraticMessage(prompt);
      }
    });
  });
}

async function sendSocraticMessage(userText) {
  const container = document.getElementById("socraticMessagesContainer");
  const footerChips = document.getElementById("socraticChatQuickChips");
  if (!container) return;

  const text = (userText || "").trim();
  if (!text) return;

  // 1. Append User Message
  const userMsgEl = document.createElement("div");
  userMsgEl.className = "socratic-msg msg-user";
  userMsgEl.innerHTML = `<div class="socratic-bubble"><p>${escapeHtml(text)}</p></div>`;
  container.appendChild(userMsgEl);

  socraticChatHistory.push({ role: "user", content: text });

  // 2. Append Typing Placeholder
  const typingEl = document.createElement("div");
  typingEl.className = "socratic-msg msg-assistant socratic-typing-bubble";
  typingEl.id = "socraticTypingMsg";
  typingEl.innerHTML = `
    <div class="socratic-bubble">
      <span class="socratic-pulse-dot" style="display:inline-block; margin-right:6px;"></span>
      <span>Sokratik çalışma arkadaşın düşünüyor...</span>
    </div>
  `;
  container.appendChild(typingEl);
  container.scrollTop = container.scrollHeight;

  // 3. Resolve playback position and active job
  let currentSec = 0;
  if (typeof getMediaTime === "function") {
    currentSec = getMediaTime() || 0;
  } else if (typeof ytPlayer !== "undefined" && ytPlayer && typeof ytPlayer.getCurrentTime === "function") {
    try { currentSec = ytPlayer.getCurrentTime() || 0; } catch (_) {}
  }

  const targetJobId = currentResult?.job_id || (currentVideoId ? `preview_${currentVideoId}` : "demo_evm");
  const terms = Array.isArray(currentResult?.llm?.terms) ? currentResult.llm.terms : [];
  const segments = Array.isArray(currentResult?.asr?.segments) ? currentResult.asr.segments : [];

  let replyData = null;

  try {
    const res = await fetch("/api/socratic-chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: targetJobId,
        message: text,
        history: socraticChatHistory.slice(-6),
        current_time_s: currentSec,
        terms: terms,
        segments: segments,
        lang: currentResult?.lang || "tr",
      }),
    });
    if (res.ok) {
      replyData = await res.json();
    }
  } catch (err) {
    console.warn("Server socratic chat fallback:", err);
  }

  // Client-side fallback if server fails
  if (!replyData) {
    replyData = generateClientSideSocraticFallback(text, terms);
  }

  // 4. Remove Typing Placeholder
  const existingTyping = document.getElementById("socraticTypingMsg");
  if (existingTyping) existingTyping.remove();

  // 5. Append Assistant Bubble
  const botMsgEl = document.createElement("div");
  botMsgEl.className = "socratic-msg msg-assistant";

  let formattedReply = escapeHtml(replyData.reply || "")
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');

  let guidingHtml = "";
  if (Array.isArray(replyData.guiding_questions) && replyData.guiding_questions.length > 0) {
    const qList = replyData.guiding_questions.map(q => `<li>${escapeHtml(q)}</li>`).join("");
    guidingHtml = `
      <div class="socratic-guiding-box">
        <span class="socratic-guide-title">🧠 Yönlendirici Sorular (Vanzo et al., ACL 2025):</span>
        <ul style="margin:4px 0 0 16px; padding:0; font-size:0.78rem; color:#fde68a;">${qList}</ul>
      </div>
    `;
  }

  let timeHtml = "";
  if (replyData.time_badge && replyData.time_badge.start_s != null) {
    timeHtml = `
      <div>
        <span class="socratic-time-tag" data-sec="${replyData.time_badge.start_s}" title="Videoda bu ana sar">
          <svg class="ui-icon" style="width:10px;height:10px;" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          ${replyData.time_badge.timestamp} • ${escapeHtml(replyData.time_badge.term)}
        </span>
      </div>
    `;
  }

  let suggestionsHtml = "";
  const suggestions = replyData.suggested_replies || [];
  if (suggestions.length > 0) {
    const chipsHtml = suggestions.map(s => `<button type="button" class="socratic-chip" data-prompt="${escapeHtml(s)}">${escapeHtml(s)}</button>`).join("");
    suggestionsHtml = `<div class="socratic-suggested-row">${chipsHtml}</div>`;
  }

  botMsgEl.innerHTML = `
    <div class="socratic-bubble">
      <p>${formattedReply}</p>
      ${guidingHtml}
      ${timeHtml}
      ${suggestionsHtml}
    </div>
  `;

  // Bind timestamp clicks
  botMsgEl.querySelectorAll(".socratic-time-tag[data-sec]").forEach((pill) => {
    pill.addEventListener("click", () => {
      const sec = Number(pill.dataset.sec);
      if (!isNaN(sec)) {
        seekPlayerTo(sec);
        showToast(`Videoda ${pill.innerText.trim()} anına sarıldı.`, "info");
      }
    });
  });

  // Bind chip click events
  bindSocraticChipEvents(botMsgEl);

  container.appendChild(botMsgEl);

  // 6. Update Footer Quick Chips
  if (footerChips) {
    if (suggestions.length > 0) {
      footerChips.innerHTML = suggestions.map(s => `<button type="button" class="socratic-chip" data-prompt="${escapeHtml(s)}">${escapeHtml(s)}</button>`).join("");
      bindSocraticChipEvents(footerChips);
    } else {
      footerChips.innerHTML = "";
    }
  }

  socraticChatHistory.push({ role: "assistant", content: replyData.reply || "" });
  container.scrollTop = container.scrollHeight;
}

function generateClientSideSocraticFallback(userText, terms) {
  const low = (userText || "").toLowerCase();
  if (low.includes("formül") || low.includes("hesap") || low.includes("cv") || low.includes("sv")) {
    return {
      reply: "Formülü ezberlemek yerine zihninde canlandıralım: Bir inşaatta 100 TL'lik iş yaptın ($EV = 100$), ama kasadan 120 TL harcadın ($AC = 120$). Sence maliyet sapmasını hesapladığımızda:",
      guiding_questions: [
        "Harcanan para tamamlanan işi aştığında sence sonuç pozitif mi çıkar, negatif mi?",
        "Negatif çıkması proje yöneticisi için iyi bir haber midir, risk midir?"
      ],
      suggested_replies: [
        "Negatif çıkar, yani bütçe aşımı!",
        "Pozitif çıkar, kârlıyız",
        "Günlük hayattan başka bir örnek ver"
      ],
      time_badge: { start_s: 30, timestamp: "00:30", term: "Maliyet Sapması Formülü" }
    };
  }
  return {
    reply: "Kavramın mantığına adım adım bakalım: Bir projede sadece kasadan 'ne kadar para çıktığına' bakarak projenin iyi gidip gitmediğini söyleyebilir miyiz?",
    guiding_questions: [
      "Harcanan paranın karşılığında ne kadarlık iş teslim alındığını bilmek neden şarttır?",
      "Bu ikisini birbirinden çıkardığımızda aradaki fark bize neyi gösterir?"
    ],
    suggested_replies: [
      "Yetmez, ne kadar iş bittiğini bilmeliyiz!",
      "Bu durumu formülle açıklar mısın?",
      "Videoda bu kavram nerede geçiyor?"
    ],
    time_badge: { start_s: 15, timestamp: "00:15", term: "Kazanılmış Değer" }
  };
}



function setupAiAnimatronicWidget() {
  const widget = document.getElementById("aiAnimatronicWidget");
  if (!widget) return;

  const triggerSocratic = () => {
    // Open right sidebar
    openRightSidebar(true);
    // Switch to Socratic tutor drawer view
    switchToSocraticDrawer();

    // Click feedback bounce
    widget.classList.add("widget-clicked");
    setTimeout(() => widget.classList.remove("widget-clicked"), 400);

    // Focus input
    setTimeout(() => {
      const fullIn = document.getElementById("socraticFullInput");
      if (fullIn) fullIn.focus();
    }, 150);

    showToast("🤖 Sokratik Asistan devrede! Aklına takılanı sorabilirsin.", "info");
  };

  widget.addEventListener("click", triggerSocratic);
  widget.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      triggerSocratic();
    }
  });

  // Dynamically update position when sidebar opens/closes
  const updateWidgetPosition = () => {
    const ws = document.getElementById("studioWorkspace");
    const isRightOpen = ws && ws.classList.contains("right-open");
    if (isRightOpen && window.innerWidth > 1200) {
      widget.style.right = "456px";
    } else {
      widget.style.right = "";
    }
  };

  const ws = document.getElementById("studioWorkspace");
  if (ws && window.MutationObserver) {
    const observer = new MutationObserver(updateWidgetPosition);
    observer.observe(ws, { attributes: true, attributeFilter: ["class"] });
  }

  window.addEventListener("resize", updateWidgetPosition);
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
  setupManimator();
  setupAttentionPopQuiz();
  setupCornellNotes();
  setupConceptKnowledgeGraph();
  setupSocraticTutor();
  setupAiAnimatronicWidget();
  setupTabVisibilityTracker();

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

/* ==========================================================================
   Multi-Theme System (Default: Light, Midnight, Nordic Frost, Emerald Zen, Cyber Sunset, Sepia Scholar)
   Persists via localStorage, default theme is 'light'.
   Keyboard shortcut: L
   ========================================================================== */
(function initThemeToggle() {
  const html = document.documentElement;
  const wrapper = document.getElementById("themePickerWrapper");
  const btnToggle = document.getElementById("btnToggleTheme");
  const iconEl = document.getElementById("themeToggleIcon");
  const labelEl = document.getElementById("themeToggleLabel");

  const THEME_ICONS = {
    light: `<svg class="ui-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>`,
    dark: `<svg class="ui-icon" viewBox="0 0 24 24"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>`,
    nordic: `<svg class="ui-icon" viewBox="0 0 24 24"><line x1="2" y1="12" x2="22" y2="12"/><line x1="12" y1="2" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/><line x1="19.07" y1="4.93" x2="4.93" y2="19.07"/></svg>`,
    emerald: `<svg class="ui-icon" viewBox="0 0 24 24"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>`,
    sunset: `<svg class="ui-icon" viewBox="0 0 24 24"><path d="M12 10V2"/><path d="m4.93 10.93 1.41 1.41"/><path d="M2 18h2"/><path d="M20 18h2"/><path d="m19.07 10.93-1.41 1.41"/><path d="M22 22H2"/><path d="m8 6 4-4 4 4"/><path d="M16 18a4 4 0 0 0-8 0"/></svg>`,
    sepia: `<svg class="ui-icon" viewBox="0 0 24 24"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>`,
  };

  const THEMES = [
    { id: "light", name: "Light", desc: "Açık Akademik (Varsayılan)" },
    { id: "dark", name: "Midnight", desc: "Derin Obsidiyen Gece" },
    { id: "nordic", name: "Nordic Frost", desc: "Buzul Mavisi & Kutup" },
    { id: "emerald", name: "Emerald Zen", desc: "Zümrüt Ormanı & Odak" },
    { id: "sunset", name: "Cyber Sunset", desc: "Neon Mercan & Alacakaranlık" },
    { id: "sepia", name: "Sepia Scholar", desc: "Antik Kütüphane & Pergament" },
  ];

  function applyTheme(themeId, notify = false) {
    const theme = THEMES.find(t => t.id === themeId) || THEMES[0];
    html.setAttribute("data-theme", theme.id);

    if (iconEl && THEME_ICONS[theme.id]) {
      iconEl.innerHTML = THEME_ICONS[theme.id];
    }
    if (labelEl) labelEl.textContent = theme.name;

    // Update active highlight in dropdown
    document.querySelectorAll(".theme-menu-item").forEach(item => {
      const isMatch = item.getAttribute("data-theme-id") === theme.id;
      item.classList.toggle("active", isMatch);
    });

    try {
      localStorage.setItem("eduterm-theme", theme.id);
    } catch (e) {}

    if (notify && typeof showToast === "function") {
      showToast(`Tema: ${theme.name}`, "info");
    }
  }

  function cycleTheme() {
    const current = html.getAttribute("data-theme") || "light";
    let idx = THEMES.findIndex(t => t.id === current);
    if (idx === -1) idx = 0;
    const nextTheme = THEMES[(idx + 1) % THEMES.length];
    applyTheme(nextTheme.id, true);
    if (wrapper) wrapper.classList.remove("open");
  }

  // Determine initial theme: localStorage > default "light"
  let initialTheme = "light";
  try {
    const stored = localStorage.getItem("eduterm-theme");
    if (stored && THEMES.some(t => t.id === stored)) {
      initialTheme = stored;
    }
  } catch (e) {}
  applyTheme(initialTheme, false);

  // Toggle Dropdown when clicking button
  if (btnToggle) {
    btnToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      if (wrapper) wrapper.classList.toggle("open");
    });
  }

  // Dropdown item selection
  document.querySelectorAll(".theme-menu-item").forEach(btn => {
    btn.addEventListener("click", function(e) {
      e.stopPropagation();
      const themeId = this.getAttribute("data-theme-id");
      if (themeId) {
        applyTheme(themeId, true);
        if (wrapper) wrapper.classList.remove("open");
      }
    });
  });

  // Close dropdown on outside click
  document.addEventListener("click", (e) => {
    if (wrapper && !wrapper.contains(e.target)) {
      wrapper.classList.remove("open");
    }
  });

  // Keyboard shortcut: L (cycles through themes)
  document.addEventListener("keydown", function(e) {
    if (e.key === "l" || e.key === "L") {
      var tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable) return;
      e.preventDefault();
      cycleTheme();
    } else if (e.key === "Escape") {
      if (wrapper) wrapper.classList.remove("open");
    }
  });
})();
