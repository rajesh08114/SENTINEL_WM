"""SENTINEL-WM live-capture agent.

A host-side process that enumerates local network interfaces, sniffs a chosen
one (optionally filtered to a host/CIDR), reassembles packets into
bidirectional CICFlowMeter-schema flow rows, and streams them to the backend
over ``WS /agent`` for real-time forecasting.
"""
__version__ = "0.1.0"
