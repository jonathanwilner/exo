<script lang="ts">
  import { onMount } from "svelte";

  interface DemoStatus {
    status?: string;
    model?: string;
    active_requests?: number;
    xpu?: {
      torch_xpu_device_names?: string[];
    };
    simulation?: {
      active?: boolean;
      link_up?: boolean;
    };
  }

  const DEFAULT_PROMPT = "Compute 17 times 23. Give only the integer.";

  let prompt = $state(DEFAULT_PROMPT);
  let maxTokens = $state(1024);
  let temperature = $state(0);
  let output = $state("");
  let reasoning = $state("");
  let usage = $state("");
  let requestStatus = $state("Demo prompt ready. Click Run Demo.");
  let model = $state("deepseek-r1-distill-qwen-1.5b-vllm");
  let xpuName = $state("Checking Intel XPU");
  let ready = $state(false);
  let simulationReady = $state(false);
  let generating = $state(false);
  let activeRequestId = $state<string | null>(null);
  let finishReason = $state<string | null>(null);
  let streamController: AbortController | null = null;
  let waitingForReasoningEnd = false;
  let reasoningBoundaryBuffer = "";

  const REASONING_END = "</think>";

  function recordValue(value: unknown): Record<string, unknown> {
    return typeof value === "object" && value !== null
      ? (value as Record<string, unknown>)
      : {};
  }

  function stringValue(value: unknown): string {
    return typeof value === "string" ? value : "";
  }

  function numberValue(value: unknown): number | null {
    return typeof value === "number" ? value : null;
  }

  function appendContent(text: string): void {
    if (!waitingForReasoningEnd) {
      output += text;
      return;
    }

    reasoningBoundaryBuffer += text;
    const boundaryIndex = reasoningBoundaryBuffer.indexOf(REASONING_END);
    if (boundaryIndex >= 0) {
      const reasoningPart = reasoningBoundaryBuffer.slice(0, boundaryIndex);
      reasoning += reasoning
        ? reasoningPart
        : reasoningPart.replace(/^<think>\s*/, "");
      output += reasoningBoundaryBuffer.slice(
        boundaryIndex + REASONING_END.length,
      );
      reasoningBoundaryBuffer = "";
      waitingForReasoningEnd = false;
      return;
    }

    const safeCharacterCount =
      reasoningBoundaryBuffer.length - (REASONING_END.length - 1);
    if (safeCharacterCount > 0) {
      const reasoningPart = reasoningBoundaryBuffer.slice(
        0,
        safeCharacterCount,
      );
      reasoning += reasoning
        ? reasoningPart
        : reasoningPart.replace(/^<think>\s*/, "");
      reasoningBoundaryBuffer =
        reasoningBoundaryBuffer.slice(safeCharacterCount);
    }
  }

  async function refreshStatus(): Promise<void> {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const status = (await response.json()) as DemoStatus;
      ready = status.status === "ready";
      model = status.model ?? model;
      xpuName =
        status.xpu?.torch_xpu_device_names?.[0] ?? "Intel XPU unavailable";
      simulationReady =
        status.simulation?.active === true &&
        status.simulation?.link_up === true;
    } catch {
      ready = false;
      simulationReady = false;
      xpuName = "Status unavailable";
    }
  }

  function handleEvent(eventName: string, value: unknown): void {
    const data = recordValue(value);
    if (eventName === "accepted") {
      activeRequestId = stringValue(data.request_id) || null;
      requestStatus = activeRequestId
        ? `Accepted request ${activeRequestId}`
        : "Request accepted.";
    } else if (eventName === "delta") {
      appendContent(stringValue(data.text));
    } else if (eventName === "reasoning_delta") {
      waitingForReasoningEnd = false;
      reasoningBoundaryBuffer = "";
      reasoning += stringValue(data.text);
    } else if (eventName === "usage") {
      const promptTokens = numberValue(data.prompt_tokens);
      const completionTokens = numberValue(data.completion_tokens);
      const totalTokens = numberValue(data.total_tokens);
      if (
        promptTokens !== null &&
        completionTokens !== null &&
        totalTokens !== null
      ) {
        usage = `${promptTokens} prompt + ${completionTokens} completion = ${totalTokens} tokens`;
      }
    } else if (eventName === "finish_reason") {
      finishReason = stringValue(data.reason) || null;
      requestStatus =
        finishReason === "length"
          ? "Maximum token limit reached; output may be incomplete."
          : `Finishing: ${finishReason ?? "complete"}`;
    } else if (eventName === "cancelled") {
      requestStatus = "Request cancelled.";
    } else if (eventName === "finished") {
      requestStatus =
        finishReason === "length"
          ? "Maximum token limit reached. Increase Max Tokens and retry."
          : "Request finished.";
    } else if (eventName === "error") {
      requestStatus = `Error: ${stringValue(data.message)}`;
    }
  }

  function parseEventBlock(block: string): void {
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (dataLines.length === 0) return;
    try {
      handleEvent(eventName, JSON.parse(dataLines.join("\n")) as unknown);
    } catch {
      requestStatus = "Received an invalid stream event.";
    }
  }

  async function readEventStream(response: Response): Promise<void> {
    if (!response.body) throw new Error("Response body is unavailable");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() ?? "";
      for (const block of blocks) parseEventBlock(block);
      if (done) break;
    }
    if (buffer.trim()) parseEventBlock(buffer);
  }

  async function runDemo(): Promise<void> {
    if (!prompt.trim() || generating || !ready) return;
    generating = true;
    output = "";
    reasoning = "";
    usage = "";
    finishReason = null;
    waitingForReasoningEnd = model.toLowerCase().includes("deepseek-r1");
    reasoningBoundaryBuffer = "";
    requestStatus = "Submitting request.";
    streamController = new AbortController();
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_tokens: maxTokens,
          temperature,
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
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        requestStatus = `Request failed: ${error instanceof Error ? error.message : "unknown error"}`;
      }
    } finally {
      activeRequestId = null;
      streamController = null;
      generating = false;
      await refreshStatus();
    }
  }

  async function cancelDemo(): Promise<void> {
    if (!activeRequestId) return;
    requestStatus = "Requesting cancellation.";
    try {
      const response = await fetch(
        `/api/requests/${encodeURIComponent(activeRequestId)}/cancel`,
        { method: "POST" },
      );
      if (!response.ok && response.status !== 409) {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      requestStatus = `Cancellation failed: ${error instanceof Error ? error.message : "unknown error"}`;
      streamController?.abort();
    }
  }

  onMount(() => {
    void refreshStatus();
    const statusInterval = window.setInterval(() => void refreshStatus(), 2000);
    return () => {
      window.clearInterval(statusInterval);
      streamController?.abort();
    };
  });
</script>

<section
  class="mx-auto w-full max-w-4xl rounded-lg border border-exo-yellow/25 bg-exo-dark-gray/80 p-4 shadow-[0_0_28px_rgba(255,215,0,0.06)]"
  aria-label="Intel vLLM demonstration"
>
  <div class="mb-3 flex flex-wrap items-center justify-between gap-2">
    <div>
      <h2 class="font-mono text-xs tracking-[0.2em] text-exo-yellow uppercase">
        Intel vLLM Live Demo
      </h2>
      <p class="mt-1 text-xs text-white/50">{model} · {xpuName}</p>
    </div>
    <div class="flex items-center gap-2 text-[10px] font-mono tracking-wider">
      <span
        class="rounded border px-2 py-1 {ready
          ? 'border-green-400/40 bg-green-400/10 text-green-300'
          : 'border-red-400/40 bg-red-400/10 text-red-300'}"
        >XPU {ready ? "READY" : "WAITING"}</span
      >
      <span
        class="rounded border px-2 py-1 {simulationReady
          ? 'border-green-400/40 bg-green-400/10 text-green-300'
          : 'border-red-400/40 bg-red-400/10 text-red-300'}"
        >TB SIM {simulationReady ? "2 NODES" : "DOWN"}</span
      >
    </div>
  </div>

  <label for="dashboard-demo-prompt" class="sr-only">Demo prompt</label>
  <textarea
    id="dashboard-demo-prompt"
    bind:value={prompt}
    rows="2"
    maxlength="32768"
    disabled={generating}
    class="w-full resize-y rounded border border-white/15 bg-exo-black/70 px-3 py-2 font-mono text-sm text-white outline-none transition-colors focus:border-exo-yellow/60 disabled:opacity-60"
  ></textarea>

  <div class="mt-3 flex flex-wrap items-end gap-3">
    <label class="text-[10px] font-mono tracking-wider text-white/50 uppercase">
      Max tokens
      <input
        bind:value={maxTokens}
        type="number"
        min="1"
        max="1024"
        disabled={generating}
        class="mt-1 block w-28 rounded border border-white/15 bg-exo-black px-2 py-1.5 text-sm text-white outline-none focus:border-exo-yellow/60"
      />
    </label>
    <label class="text-[10px] font-mono tracking-wider text-white/50 uppercase">
      Temperature
      <input
        bind:value={temperature}
        type="number"
        min="0"
        max="2"
        step="0.1"
        disabled={generating}
        class="mt-1 block w-28 rounded border border-white/15 bg-exo-black px-2 py-1.5 text-sm text-white outline-none focus:border-exo-yellow/60"
      />
    </label>
    <button
      type="button"
      onclick={runDemo}
      disabled={!ready || generating || !prompt.trim()}
      class="rounded bg-exo-yellow px-5 py-2 font-mono text-xs font-semibold tracking-wider text-exo-black uppercase transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-35"
      >Run Demo</button
    >
    <button
      type="button"
      onclick={cancelDemo}
      disabled={!generating || !activeRequestId}
      class="rounded border border-white/20 px-4 py-2 font-mono text-xs tracking-wider text-white/70 uppercase transition-colors hover:border-red-400/50 hover:text-red-300 disabled:cursor-not-allowed disabled:opacity-30"
      >Cancel</button
    >
    <a
      href="/contract"
      class="ml-auto py-2 font-mono text-[10px] tracking-wider text-white/45 uppercase hover:text-exo-yellow"
      >Open diagnostic view</a
    >
  </div>

  <p class="mt-3 font-mono text-[11px] text-white/55" role="status">
    {requestStatus}
  </p>

  {#if output || reasoning || usage}
    <div class="mt-3 grid gap-3 lg:grid-cols-2">
      <div class="min-w-0">
        <h3
          class="mb-1 font-mono text-[10px] tracking-wider text-white/40 uppercase"
        >
          Streamed output
        </h3>
        <pre
          class="max-h-48 overflow-auto whitespace-pre-wrap rounded border border-white/10 bg-exo-black/70 p-3 text-xs leading-relaxed text-white/80"
          aria-live="polite">{output}</pre>
      </div>
      <div class="min-w-0">
        <h3
          class="mb-1 font-mono text-[10px] tracking-wider text-white/40 uppercase"
        >
          Reasoning channel
        </h3>
        <pre
          class="max-h-48 overflow-auto whitespace-pre-wrap rounded border border-white/10 bg-exo-black/70 p-3 text-xs leading-relaxed text-white/65">{reasoning ||
            "No separate reasoning stream reported."}</pre>
      </div>
    </div>
    {#if usage}
      <p class="mt-2 font-mono text-[10px] text-white/40">{usage}</p>
    {/if}
  {/if}
</section>
