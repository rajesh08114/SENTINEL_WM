"""Enumerate local network interfaces (Wireshark-style picker source).

psutil gives the addresses + up/down; on Windows we also pull friendly
descriptions from scapy so the console shows "Intel(R) Wi-Fi 6 AX201" rather
than a GUID.
"""
from __future__ import annotations

import socket
from dataclasses import asdict, dataclass
from typing import Optional

import psutil


@dataclass
class Interface:
    name: str
    description: str
    ipv4: Optional[str]
    netmask: Optional[str]
    mac: Optional[str]
    is_up: bool
    is_loopback: bool


def _windows_descriptions() -> dict[str, str]:
    try:
        from scapy.arch.windows import get_windows_if_list  # type: ignore
    except Exception:
        return {}
    out: dict[str, str] = {}
    for d in get_windows_if_list():
        # scapy exposes both the friendly name and the description
        for key in ("name", "netid", "guid"):
            v = d.get(key)
            if v:
                out[str(v)] = d.get("description") or d.get("name") or str(v)
        if d.get("name"):
            out[str(d["name"])] = d.get("description") or str(d["name"])
    return out


def list_interfaces() -> list[Interface]:
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()
    descs = _windows_descriptions()
    out: list[Interface] = []
    for name, addr_list in addrs.items():
        ipv4 = netmask = mac = None
        for a in addr_list:
            if a.family == socket.AF_INET:
                ipv4, netmask = a.address, a.netmask
            elif getattr(a, "family", None) == getattr(psutil, "AF_LINK", -1):
                mac = a.address
        st = stats.get(name)
        lname = name.lower()
        is_lo = ("loopback" in lname or lname == "lo"
                 or lname.startswith(("lo:", "lo "))
                 or ipv4 == "127.0.0.1" or ipv4 == "::1")
        out.append(Interface(
            name=name,
            description=descs.get(name, name),
            ipv4=ipv4, netmask=netmask, mac=mac,
            is_up=bool(st.isup) if st else False,
            is_loopback=is_lo,
        ))
    out.sort(key=lambda i: (not i.is_up, i.is_loopback, i.name.lower()))
    return out


def to_dicts(ifaces: list[Interface]) -> list[dict]:
    return [asdict(i) for i in ifaces]
