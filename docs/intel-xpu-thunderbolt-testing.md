# Intel XPU and Thunderbolt test harness

This test path exercises two-node Exo behavior when only one Intel XPU machine is
available. It separates network simulation from XPU simulation so local device
traffic is never reported as Thunderbolt performance.

## Safety boundaries

- The network harness owns only `exo-tb-node-a`, `exo-tb-node-b`, and its state
  file. Interface aliases contain a random ownership token. Cleanup stops if the
  live token does not match the state file.
- The Intel XPU probe performs read-only inventory by default.
- Experimental Level Zero multi-root enumeration requires
  `--enable-experimental-multi-root`.
- The XCCL smoke test additionally requires `--run-xccl-smoke`. It allocates four
  `float32` values per rank and has bounded timeouts.
- Neither harness treats simulated results as two-machine performance evidence.

## Simulated Thunderbolt network

Inspect the exact commands without changing the host:

```bash
uv run python scripts/exo_thunderbolt_netns.py start --dry-run \
  --latency 50us --loss 0.01% --rate 20gbit
```

Create the two namespaces on a Linux host with network-administration privileges:

```bash
sudo uv run python scripts/exo_thunderbolt_netns.py start \
  --latency 50us --loss 0.01% --rate 20gbit
sudo uv run python scripts/exo_thunderbolt_netns.py status
```

Exercise link loss and recovery:

```bash
sudo uv run python scripts/exo_thunderbolt_netns.py fail
sudo uv run python scripts/exo_thunderbolt_netns.py restore
sudo uv run python scripts/exo_thunderbolt_netns.py stop
```

The production Linux classifier requires `thunderbolt_net` driver, module, or
Thunderbolt bus evidence in sysfs. A simulated interface named `thunderbolt0`
therefore remains `unknown` unless a test supplies an injected sysfs tree.

## Intel XPU multi-root simulation

Run baseline inventory:

```bash
uv run python scripts/intel_xpu_multi_root_probe.py
```

Request experimental two-device enumeration without starting a collective:

```bash
uv run python scripts/intel_xpu_multi_root_probe.py \
  --enable-experimental-multi-root \
  --force-preemption-mode-3
```

Run the bounded two-rank XCCL collective only after enumeration reports exactly
two XPU devices:

```bash
uv run python scripts/intel_xpu_multi_root_probe.py \
  --enable-experimental-multi-root \
  --run-xccl-smoke \
  --force-preemption-mode-3
```

## Distributed vLLM configuration

`VllmXpuDistributedConfig` validates the selected addresses, interface,
parallelism, and memory limits. Its pure builders generate:

- Ray head or worker commands bound to the selected address.
- A vLLM XPU serving command using Ray.
- `VLLM_HOST_IP`, Intel XPU process settings, `GLOO_SOCKET_IFNAME`,
  `CCL_ATL_TRANSPORT`, `FI_PROVIDER`, and `FI_TCP_IFACE` for the selected network
  interface.

Use the builders for orchestration tests. A real two-node acceptance test must
still prove that XCCL byte counters increase on the physical `thunderbolt0`
interfaces and that cable removal reconstructs the distributed instance.

## Verified Panther Lake smoke result

On 4 September 2026, an Intel Arc B390 Panther Lake system completed these
bounded checks with `NEOReadDebugKeys=1`, `CreateMultipleRootDevices=2`, and
`ForcePreemptionMode=3`:

- Level Zero and PyTorch reported two simulated XPU devices.
- Two XCCL ranks completed an all-reduce with the expected values.
- vLLM `0.21.1.dev0` loaded a 1.5B BF16 model with tensor parallel size 2.
- The OpenAI-compatible completion endpoint returned HTTP 200.
- The checked kernel-log window contained no matching `xe` GPU fault, reset, or
  hang.

This result proves a useful single-machine functional test. It does not prove
two-machine correctness, Thunderbolt bandwidth, or production stability.
