"""Network-intrusion module (flow-based IDS, docs/NETWORK_IDS_PLAN.md).

schema: frozen shared NetFlow-v2 features and the label taxonomy.
evidence: network evidence, actions and the evidence package builder.
splits: sampling and host-group splits for the protocols.
live: nfstream flow to feature mapping (lazy import; no capture at import time).
pi_monitor: Raspberry Pi runner (lazy imports of nfstream and onnxruntime).
"""
