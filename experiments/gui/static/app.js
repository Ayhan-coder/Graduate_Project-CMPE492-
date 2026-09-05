// ==========================================================================
// EduTerm AI — Frontend Application Logic & YouTube Sync
// ==========================================================================

let activeTab = "youtube";
let selectedFile = null;
let currentScope = "full";
let currentResult = null;
let ytPlayer = null;
let ytApiReady = false;

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
window.onYouTubeIframeAPIReady = function () {
  ytApiReady = true;
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

// YouTube Metadata Inspection
async function inspectYouTubeUrl() {
  const url = document.getElementById("ytUrlInput").value.trim();
  if (!url) {
    showToast("Please enter a valid YouTube URL", "error");
    return;
  }
  const previewCard = document.getElementById("videoPreviewCard");
  const titleEl = document.getElementById("previewTitle");
  const authorEl = document.getElementById("previewAuthor");
  const thumbEl = document.getElementById("previewThumb");
  const durEl = document.getElementById("previewDuration");

  titleEl.innerText = "Inspecting video...";
  previewCard.classList.remove("hidden");

  try {
    const res = await fetch("/api/youtube-info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to inspect URL");
    }
    const info = await res.json();
    titleEl.innerText = info.title;
    authorEl.innerText = info.channel;
    durEl.innerText = formatTime(info.duration);
    thumbEl.src = info.thumbnail;
    showToast("Video metadata retrieved!", "success");
  } catch (err) {
    titleEl.innerText = "Could not load video info";
    authorEl.innerText = err.message;
    showToast(err.message, "error");
  }
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

// Pipeline Execution
async function runPipeline() {
  const btnRun = document.getElementById("btnRunPipeline");
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
  } else if (currentScope === "custom") {
    start_s = parseFloat(document.getElementById("inputStartSec").value) || 0.0;
    const durVal = parseFloat(document.getElementById("inputDurationSec").value);
    if (durVal && durVal > 0) duration_s = durVal;
  }

  let jobId = null;
  btnRun.disabled = true;
  progressBox.classList.remove("hidden");
  barFill.style.width = "10%";
  statusMsg.innerText = "Initializing task...";
  logStream.innerHTML = "";

  function appendLog(msg) {
    const item = document.createElement("div");
    item.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logStream.appendChild(item);
    logStream.scrollTop = logStream.scrollHeight;
  }

  try {
    if (activeTab === "youtube") {
      const url = document.getElementById("ytUrlInput").value.trim();
      if (!url) throw new Error("Please enter a YouTube URL.");

      appendLog(`Submitting YouTube URL: ${url}`);
      const res = await fetch("/api/process-youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, start_s, duration_s, asr_model: asrModel, lang, llm_model: llmModel }),
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
      if (!selectedFile) throw new Error("Please select an audio/video file to upload.");

      appendLog(`Uploading local file: ${selectedFile.name}`);
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("start_s", start_s.toString());
      if (duration_s) formData.append("duration_s", duration_s.toString());
      formData.append("asr_model", asrModel);
      formData.append("lang", lang);
      formData.append("llm_model", llmModel);

      const res = await fetch("/api/process-upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        let errMsg = `Upload failed (${res.status})`;
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
    }

    // Connect to SSE Stream
    const eventSource = new EventSource(`/api/stream/${jobId}`);

    eventSource.addEventListener("step", (e) => {
      const payload = JSON.parse(e.data);
      statusMsg.innerText = payload.message;
      appendLog(`Step: ${payload.step}`);
      if (payload.step === "download" || payload.step === "convert") barFill.style.width = "25%";
      if (payload.step === "asr") barFill.style.width = "50%";
      if (payload.step === "llm") barFill.style.width = "75%";
    });

    eventSource.addEventListener("log", (e) => {
      const payload = JSON.parse(e.data);
      appendLog(payload.message);
    });

    eventSource.addEventListener("download_done", (e) => {
      const payload = JSON.parse(e.data);
      appendLog(`Audio extracted: ${payload.duration_s.toFixed(1)}s`);
      barFill.style.width = "40%";
    });

    eventSource.addEventListener("asr_done", (e) => {
      const payload = JSON.parse(e.data);
      appendLog(`ASR complete: ${payload.latency_s}s (RTF: ${payload.rtf})`);
      barFill.style.width = "75%";
    });

    eventSource.addEventListener("llm_done", (e) => {
      const payload = JSON.parse(e.data);
      appendLog(`LLM extracted ${payload.n_terms} terms in ${payload.latency_s}s`);
      barFill.style.width = "95%";
    });

    eventSource.addEventListener("done", (e) => {
      const result = JSON.parse(e.data);
      eventSource.close();
      barFill.style.width = "100%";
      statusMsg.innerText = "Completed successfully!";
      btnRun.disabled = false;
      showToast("Speech-to-Terminology pipeline complete!", "success");
      renderResults(result);
    });

    eventSource.addEventListener("error", (e) => {
      eventSource.close();
      btnRun.disabled = false;
      barFill.style.background = "var(--accent-rose)";
      statusMsg.innerText = "Error occurred";
      showToast("An error occurred during processing.", "error");
      try {
        const payload = JSON.parse(e.data);
        appendLog(`Error: ${payload.message}`);
      } catch (_) {}
    });
  } catch (err) {
    btnRun.disabled = false;
    showToast(err.message, "error");
    appendLog(`Error: ${err.message}`);
  }
}

// Render Results View
function renderResults(result) {
  currentResult = result;
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("resultsView").classList.remove("hidden");

  // Diagnostics
  document.getElementById("diagRtf").innerText = `${result.asr.rtf.toFixed(3)} RTF`;
  document.getElementById("diagAsrLat").innerText = `${result.asr.latency_s}s`;
  document.getElementById("diagLlmLat").innerText = `${result.llm.latency_s}s`;
  document.getElementById("diagTermsCount").innerText = result.llm.n_terms;
  document.getElementById("termCounter").innerText = result.llm.n_terms;

  // Player Setup
  const ytContainer = document.getElementById("ytPlayerContainer");
  const audioContainer = document.getElementById("htmlAudioContainer");
  const meta = result.video_meta || {};

  if (meta.id && meta.url) {
    // YouTube mode
    audioContainer.classList.add("hidden");
    ytContainer.classList.remove("hidden");
    loadYouTubeVideo(meta.id);
  } else {
    // Local Audio mode
    ytContainer.classList.add("hidden");
    audioContainer.classList.remove("hidden");
    const audioPlayer = document.getElementById("audioPlayer");
    audioPlayer.src = `/api/audio/${result.job_id}`;
  }

  renderTermsGrid(result.llm.terms, result.lang);
  renderTranscript(result.asr.segments, result.asr.text);
}

// YouTube Player loader
function loadYouTubeVideo(videoId) {
  if (ytPlayer && ytPlayer.loadVideoById) {
    ytPlayer.loadVideoById(videoId);
  } else if (ytApiReady) {
    ytPlayer = new YT.Player("ytPlayer", {
      videoId: videoId,
      playerVars: {
        playsinline: 1,
        rel: 0,
      },
    });
  }
}

// Seek Player to Timestamp
function seekPlayerTo(seconds) {
  if (seconds === null || isNaN(seconds)) return;
  if (ytPlayer && ytPlayer.seekTo) {
    ytPlayer.seekTo(seconds, true);
    ytPlayer.playVideo();
  } else {
    const audio = document.getElementById("audioPlayer");
    if (audio) {
      audio.currentTime = seconds;
      audio.play();
    }
  }
}

// Render Terminology Cards
function renderTermsGrid(terms, lang) {
  const grid = document.getElementById("termsGrid");
  grid.innerHTML = "";

  if (!terms || terms.length === 0) {
    grid.innerHTML = `<p style="color: var(--text-muted); font-size: 0.9rem; grid-column: 1/-1;">No domain terms identified in this segment.</p>`;
    return;
  }

  terms.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "term-card";
    card.dataset.term = item.term.toLowerCase();
    card.dataset.expl = (item.explanation || "").toLowerCase();

    card.innerHTML = `
      <div>
        <div class="term-header">
          <h4 class="term-name">${escapeHtml(item.term)}</h4>
          <span class="term-badge">${(lang || "tr").toUpperCase()}</span>
        </div>
        <p class="term-explanation">${escapeHtml(item.explanation || "")}</p>
      </div>
      <div class="term-actions">
        <button type="button" class="term-action-btn btn-speak" title="Pronounce term">🔊 Speak</button>
        <button type="button" class="term-action-btn btn-copy" title="Copy term & definition">📋 Copy</button>
      </div>
    `;

    // Speak term
    card.querySelector(".btn-speak").addEventListener("click", () => {
      const utter = new SpeechSynthesisUtterance(item.term);
      utter.lang = (lang || "tr").toLowerCase().startsWith("tr") ? "tr-TR" : "en-US";
      window.speechSynthesis.speak(utter);
    });

    // Copy term
    card.querySelector(".btn-copy").addEventListener("click", () => {
      navigator.clipboard.writeText(`${item.term}: ${item.explanation}`);
      showToast(`Copied "${item.term}" to clipboard!`, "success");
    });

    grid.appendChild(card);
  });
}

// Render Interactive Transcript
function renderTranscript(segments, fullText) {
  const scroll = document.getElementById("transcriptScroll");
  scroll.innerHTML = "";

  if (segments && segments.length > 0) {
    segments.forEach((seg) => {
      const segEl = document.createElement("div");
      segEl.className = "transcript-segment";
      const timeTag = seg.start !== null ? formatTime(seg.start) : "---";

      segEl.innerHTML = `
        <span class="segment-time">${timeTag}</span>
        <span class="segment-text">${escapeHtml(seg.text)}</span>
      `;

      if (seg.start !== null) {
        segEl.addEventListener("click", () => {
          seekPlayerTo(seg.start);
          showToast(`Jumped to ${timeTag}`, "info");
        });
      }
      scroll.appendChild(segEl);
    });
  } else {
    scroll.innerHTML = `<p style="padding: 1rem; color: var(--text-secondary);">${escapeHtml(fullText)}</p>`;
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
    window.location.href = `/api/export-csv/${currentResult.job_id}`;
  });

  document.getElementById("btnExportJson").addEventListener("click", () => {
    if (!currentResult) return;
    const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `eduterm_${currentResult.job_id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Downloaded JSON export", "success");
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

// Subtab switching (Terminology vs Transcript)
function setupSubtabs() {
  const btnTerms = document.getElementById("subtabTermsBtn");
  const btnTrans = document.getElementById("subtabTranscriptBtn");
  const contentTerms = document.getElementById("subtabTermsContent");
  const contentTrans = document.getElementById("subtabTranscriptContent");

  btnTerms.addEventListener("click", () => {
    btnTerms.classList.add("active");
    btnTrans.classList.remove("active");
    contentTerms.classList.remove("hidden");
    contentTrans.classList.add("hidden");
  });

  btnTrans.addEventListener("click", () => {
    btnTrans.classList.add("active");
    btnTerms.classList.remove("active");
    contentTrans.classList.remove("hidden");
    contentTerms.classList.add("hidden");
  });
}

// Quick Sample Chips
function setupDemoChips() {
  document.querySelectorAll(".demo-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const url = chip.dataset.url;
      document.getElementById("ytUrlInput").value = url;
      document.getElementById("tabYoutube").click();
      inspectYouTubeUrl();
    });
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// DOM Ready initialization
document.addEventListener("DOMContentLoaded", () => {
  loadSystemInfo();
  setupTabs();
  setupUpload();
  setupScopePresets();
  setupTermsFilter();
  setupExports();
  setupSubtabs();
  setupDemoChips();

  document.getElementById("btnFetchMeta").addEventListener("click", inspectYouTubeUrl);
  document.getElementById("btnRunPipeline").addEventListener("click", runPipeline);
});
