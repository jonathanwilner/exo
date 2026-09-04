"use strict";

const elements = {
  applicationStatus: document.getElementById("application-status"),
  sidecarStatus: document.getElementById("sidecar-status"),
  xpuStatus: document.getElementById("xpu-status"),
  contract: document.getElementById("contract"),
  implementation: document.getElementById("implementation"),
  model: document.getElementById("model"),
  sidecarEndpoint: document.getElementById("sidecar-endpoint"),
  xpuDevices: document.getElementById("xpu-devices"),
  activeRequests: document.getElementById("active-requests"),
  prompt: document.getElementById("prompt"),
  maxTokens: document.getElementById("max-tokens"),
  temperature: document.getElementById("temperature"),
  send: document.getElementById("send"),
  cancel: document.getElementById("cancel"),
  requestStatus: document.getElementById("request-status"),
  output: document.getElementById("output"),
  reasoning: document.getElementById("reasoning"),
  usage: document.getElementById("usage"),
};

let activeRequestId = null;
let streamController = null;
let finishReason = null;

function setIndicator(element, text, state) {
  element.textContent = text;
  element.dataset.state = state;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    setIndicator(
      elements.applicationStatus,
      status.status === "ready" ? "READY" : "NOT READY",
      status.status === "ready" ? "ready" : "failed",
    );
    setIndicator(
      elements.sidecarStatus,
      status.sidecar.ready ? "READY" : "UNAVAILABLE",
      status.sidecar.ready ? "ready" : "failed",
    );
    const xpuReady = status.hardware_gate.ready;
    setIndicator(
      elements.xpuStatus,
      xpuReady ? "VERIFIED" : "NOT VERIFIED",
      xpuReady ? "ready" : "failed",
    );
    elements.contract.textContent = status.contract;
    elements.implementation.textContent = status.implementation;
    elements.model.textContent = status.model;
    elements.sidecarEndpoint.textContent = status.sidecar_endpoint;
    const names = status.xpu.torch_xpu_device_names || [];
    const sysfsCount = (status.xpu.intel_sysfs_devices || []).length;
    elements.xpuDevices.textContent = names.length
      ? names.join(", ")
      : `${sysfsCount} Intel DRM device(s); PyTorch names unavailable`;
    elements.activeRequests.textContent = String(status.active_requests);
  } catch (error) {
    setIndicator(elements.applicationStatus, "UNAVAILABLE", "failed");
    setIndicator(elements.sidecarStatus, "UNKNOWN", "failed");
    setIndicator(elements.xpuStatus, "UNKNOWN", "failed");
  }
}

function handleEvent(eventName, data) {
  if (eventName === "accepted") {
    activeRequestId = data.request_id;
    elements.requestStatus.textContent = `Accepted request ${activeRequestId}`;
  } else if (eventName === "delta") {
    elements.output.textContent += data.text || "";
  } else if (eventName === "reasoning_delta") {
    elements.reasoning.textContent += data.text || "";
  } else if (eventName === "usage") {
    elements.usage.textContent =
      `Tokens: ${data.prompt_tokens} prompt + ${data.completion_tokens} completion = ${data.total_tokens} total`;
  } else if (eventName === "finish_reason") {
    finishReason = data.reason;
    elements.requestStatus.textContent =
      finishReason === "length"
        ? "Maximum token limit reached; output may be incomplete."
        : `Finishing: ${finishReason}`;
  } else if (eventName === "cancelled") {
    elements.requestStatus.textContent = "Request cancelled.";
  } else if (eventName === "finished") {
    elements.requestStatus.textContent =
      finishReason === "length"
        ? "Maximum token limit reached. Increase Max Tokens and retry."
        : "Request finished.";
  } else if (eventName === "tool_call") {
    elements.requestStatus.textContent = data.message;
  } else if (eventName === "error") {
    elements.requestStatus.textContent = `Error: ${data.message}`;
  }
}

function parseEventBlock(block) {
  let eventName = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return;
  try {
    handleEvent(eventName, JSON.parse(dataLines.join("\n")));
  } catch (error) {
    elements.requestStatus.textContent = "Received an invalid stream event.";
  }
}

async function readEventStream(response) {
  if (!response.body) throw new Error("Response body is unavailable");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) parseEventBlock(block);
    if (done) break;
  }
  if (buffer.trim()) parseEventBlock(buffer);
}

function setGenerating(generating) {
  elements.send.disabled = generating;
  elements.cancel.disabled = !generating;
  elements.prompt.disabled = generating;
  elements.maxTokens.disabled = generating;
  elements.temperature.disabled = generating;
}

async function sendPrompt() {
  const prompt = elements.prompt.value;
  if (!prompt.trim()) {
    elements.requestStatus.textContent = "Enter a non-empty prompt.";
    return;
  }
  setGenerating(true);
  elements.output.textContent = "";
  elements.reasoning.textContent = "";
  elements.usage.textContent = "";
  elements.requestStatus.textContent = "Submitting request.";
  finishReason = null;
  streamController = new AbortController();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_tokens: Number(elements.maxTokens.value),
        temperature: Number(elements.temperature.value),
      }),
      signal: streamController.signal,
    });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`HTTP ${response.status}: ${body.slice(0, 300)}`);
    }
    activeRequestId = response.headers.get("X-Exo-Request-Id");
    await readEventStream(response);
  } catch (error) {
    if (error.name !== "AbortError") {
      elements.requestStatus.textContent = `Request failed: ${error.message}`;
    }
  } finally {
    activeRequestId = null;
    streamController = null;
    setGenerating(false);
    refreshStatus();
  }
}

async function cancelRequest() {
  if (!activeRequestId) return;
  elements.cancel.disabled = true;
  elements.requestStatus.textContent = "Requesting cancellation.";
  try {
    const response = await fetch(
      `/api/requests/${encodeURIComponent(activeRequestId)}/cancel`,
      { method: "POST" },
    );
    if (!response.ok && response.status !== 409) {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    elements.requestStatus.textContent = `Cancellation failed: ${error.message}`;
    streamController?.abort();
  }
}

elements.send.addEventListener("click", sendPrompt);
elements.cancel.addEventListener("click", cancelRequest);
elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) sendPrompt();
});

refreshStatus();
setInterval(refreshStatus, 2000);
