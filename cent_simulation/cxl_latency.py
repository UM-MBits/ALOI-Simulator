"""CXL communication model: one-shot all-to-all collectives over PCIe.

Every inter-device transfer in the simulator uses this model:

- Bandwidth is per lane, per direction, in GB/s (10**9 B/s), set by the PCIe
  generation (``PCIE_GB_PER_LANE``).  Links are unicast; there is no
  multicast bandwidth discount.
- Values travel at the simulation's element precision (FP8 = 1 B, BF16 = 2 B,
  see ``wire_bytes_per_value``), packed into 256 B flits that carry 192 B of
  payload (3 x 64 B messages per PBR flit).
- A TP all-reduce is a one-shot all-to-all: every rank sends its full partial
  vector to all peers at once (full duplex) and reduces locally.  Time is
  bounded by the (TP - 1) vectors arriving at each rank's ingress, plus a
  fixed ``TP_FIXED_NS``.
- A PP handoff is one vector transfer plus ``PP_FIXED_NS``.
- Communication is not overlapped with compute.

Latencies are returned in microseconds unless a function name ends in ``_ms``.
The precision (or bytes per value) is always passed explicitly, so a run can
never silently use another precision on the wire.
"""

import math

from tp_mapping import pim_precision_geometry

PCIE_GB_PER_LANE = {5: 4, 6: 8, 7: 16}
DEFAULT_PCIE_GEN = 6
BYTES_PER_GB = 10**9

FLIT_BYTES = 256
FLIT_PAYLOAD_BYTES = 192
# Fixed per-transfer overhead: 4 CXL ports (25 ns) + 2 PCIe PHY traversals
# (5 + 20 + 5 ns) + 1 CXL switch (20 ns) = 180 ns, for every TP and PP transfer.
CXL_PORT_NS = 25
PCIE_PHY_NS = 5 + 20 + 5
CXL_SWITCH_NS = 20
TRANSFER_FIXED_NS = 4 * CXL_PORT_NS + 2 * PCIE_PHY_NS + CXL_SWITCH_NS
TP_FIXED_NS = TRANSFER_FIXED_NS
PP_FIXED_NS = TRANSFER_FIXED_NS


def wire_bytes_per_value(precision):
    """Bytes per element on the wire for a simulation precision ("fp8", "bf16")."""
    bits = pim_precision_geometry(str(precision))["element_bits"]
    if bits % 8:
        raise ValueError(f"Precision {precision!r} is not byte aligned")
    return bits // 8


def pcie_lane_bandwidth(pcie_gen=DEFAULT_PCIE_GEN):
    """Per-lane, per-direction bandwidth (GB/s) for a PCIe generation."""
    try:
        return PCIE_GB_PER_LANE[int(pcie_gen)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Unsupported PCIe generation {pcie_gen!r}; choose from {sorted(PCIE_GB_PER_LANE)}") from None


def device_bandwidth(lanes_per_device, pcie_gen=DEFAULT_PCIE_GEN):
    """Per-device, per-direction bandwidth (GB/s)."""
    if lanes_per_device <= 0:
        raise ValueError("PCIe lanes per device must be positive")
    return lanes_per_device * pcie_lane_bandwidth(pcie_gen)


def wire_bytes(values, *, bytes_per_value, flit_packing=True):
    """Bytes on the link for one vector; flit packing adds the 256/192 overhead."""
    if values <= 0 or bytes_per_value <= 0:
        raise ValueError("values and bytes_per_value must be positive")
    payload = values * bytes_per_value
    if not flit_packing:
        return payload
    return math.ceil(payload / FLIT_PAYLOAD_BYTES) * FLIT_BYTES


def _wire_us(nbytes, bandwidth, bytes_per_bandwidth_unit):
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    return nbytes / (bandwidth * bytes_per_bandwidth_unit) * 1e6


def transfer_us(values, bandwidth, *, bytes_per_value, fixed_ns=PP_FIXED_NS, flit_packing=True,
                bytes_per_bandwidth_unit=BYTES_PER_GB):
    """One unicast vector transfer (``bandwidth`` in units of ``bytes_per_bandwidth_unit``/s)."""
    return fixed_ns / 1000 + _wire_us(wire_bytes(values, bytes_per_value=bytes_per_value,
                                                 flit_packing=flit_packing),
                                      bandwidth, bytes_per_bandwidth_unit)


def allreduce_us(values, bandwidth, tp=2, *, bytes_per_value, flit_packing=True,
                 fixed_ns=TP_FIXED_NS, bytes_per_bandwidth_unit=BYTES_PER_GB):
    """One-shot all-to-all all-reduce of a ``values``-long partial vector."""
    if tp < 1:
        raise ValueError("tp must be positive")
    if tp == 1:
        return 0.0
    ingress = (tp - 1) * wire_bytes(values, bytes_per_value=bytes_per_value, flit_packing=flit_packing)
    return fixed_ns / 1000 + _wire_us(ingress, bandwidth, bytes_per_bandwidth_unit)


def allgather_us(values, bandwidth, devices, *, bytes_per_value, flit_packing=True,
                 fixed_ns=TP_FIXED_NS, bytes_per_bandwidth_unit=BYTES_PER_GB):
    """One-shot all-to-all all-gather: each rank sends its 1/devices shard to all peers."""
    if devices < 1:
        raise ValueError("devices must be positive")
    if devices == 1:
        return 0.0
    shard = math.ceil(values / devices)
    ingress = (devices - 1) * wire_bytes(shard, bytes_per_value=bytes_per_value, flit_packing=flit_packing)
    return fixed_ns / 1000 + _wire_us(ingress, bandwidth, bytes_per_bandwidth_unit)


def pipeline_comm_us(layers, pp_stages, values, bandwidth, tp=2, *, bytes_per_value,
                     flit_packing=True, allreduces_per_layer=2, tp_fixed_ns=TP_FIXED_NS,
                     pp_fixed_ns=PP_FIXED_NS, bytes_per_bandwidth_unit=BYTES_PER_GB):
    """Communication for ``layers`` decoder layers split over ``pp_stages`` stages.

    Each layer does ``allreduces_per_layer`` TP all-reduces (after O and Down);
    each stage boundary does one PP handoff.  Returns (tp_us, pp_us).
    """
    if layers < 1 or not 1 <= pp_stages <= layers:
        raise ValueError("need 1 <= pp_stages <= layers")
    common = dict(bytes_per_value=bytes_per_value, flit_packing=flit_packing,
                  bytes_per_bandwidth_unit=bytes_per_bandwidth_unit)
    ar = allreduce_us(values, bandwidth, tp, fixed_ns=tp_fixed_ns, **common)
    handoff = transfer_us(values, bandwidth, fixed_ns=pp_fixed_ns, **common)
    return layers * allreduces_per_layer * ar, (pp_stages - 1) * handoff


def block_cxl_costs_us(values, lanes_per_device, tp, pp, pcie_gen=DEFAULT_PCIE_GEN, *, precision):
    """Per-block CXL of the KV-head TP mapping, keyed by the kernel it follows."""
    bandwidth = device_bandwidth(lanes_per_device, pcie_gen)
    nbytes = wire_bytes_per_value(precision)
    reduction = allreduce_us(values, bandwidth, tp, bytes_per_value=nbytes)
    return {
        "attention.o_projection": reduction,
        "ffn.down_projection": reduction,
        "aux.pp_handoff": transfer_us(values, bandwidth, bytes_per_value=nbytes) if pp > 1 else 0.0,
    }


# --- run_sim interface (milliseconds) -------------------------------------

def kv_head_tp_latency_ms(hidden_dim, lanes_per_device, devices, total_devices, pcie_gen=DEFAULT_PCIE_GEN, *,
                          precision):
    """The two row-parallel TP all-reduces (after O and Down) of one block."""
    if devices <= 0 or total_devices <= 0:
        raise ValueError("device counts must be positive")
    if devices > total_devices:
        raise ValueError("TP devices cannot exceed total source devices")
    if hidden_dim <= 0 or lanes_per_device <= 0:
        raise ValueError("hidden dimension and PCIe lanes must be positive")
    if devices == 1:
        return 0.0
    return 2 * allreduce_us(hidden_dim, device_bandwidth(lanes_per_device, pcie_gen), devices,
                            bytes_per_value=wire_bytes_per_value(precision)) / 1000


def pp_handoff_latency_ms(size, lanes_per_device, pcie_gen=DEFAULT_PCIE_GEN, *, precision):
    """One activation-vector handoff between pipeline stages."""
    return transfer_us(size, device_bandwidth(lanes_per_device, pcie_gen),
                       bytes_per_value=wire_bytes_per_value(precision)) / 1000


def cent_tp_latency_ms(size, lanes_per_device, devices, total_devices, pcie_gen=DEFAULT_PCIE_GEN, *, precision):
    """CENT's original (non-KV-head) TP mapping per block.

    That mapping exchanges column-parallel shards with five gather-then-
    broadcast steps on the embedding vector and one on the FFN vector; each
    pair is now one one-shot all-to-all all-gather.
    """
    if devices > total_devices:
        raise ValueError("TP devices cannot exceed total source devices")
    embedding, ffn = size
    bandwidth = device_bandwidth(lanes_per_device, pcie_gen)
    nbytes = wire_bytes_per_value(precision)
    return (5 * allgather_us(embedding, bandwidth, devices, bytes_per_value=nbytes)
            + allgather_us(ffn, bandwidth, devices, bytes_per_value=nbytes)) / 1000


def kv_head_tp_pcie_bits(hidden_dim, devices, *, precision):
    """Return (main, per-helper, system) transmitted payload bits per block.

    Each of the two all-reduces has every rank send its partial vector to each
    of the (TP - 1) peers; ranks are symmetric.
    """
    if hidden_dim <= 0 or devices <= 0:
        raise ValueError("hidden dimension and TP devices must be positive")
    if devices == 1:
        return 0, 0, 0
    per_rank = 2 * (devices - 1) * hidden_dim * wire_bytes_per_value(precision) * 8
    return per_rank, per_rank, devices * per_rank


def pp_handoff_pcie_bits(size, *, precision):
    """Payload bits of one PP handoff."""
    return size * wire_bytes_per_value(precision) * 8
