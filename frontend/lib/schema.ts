// Mirrors backend app/sentinel_infer/forecast.py — the column contract. Used
// client-side only for a fast "does this CSV match?" report; the backend
// re-validates on upload.

export const REQUIRED = [
  "flow_start_epoch",
  "Source IP", "Destination IP", "Destination Port", "Protocol",
  "Flow Duration", "Flow IAT Mean",
  "Total Fwd Packets", "Total Backward Packets",
  "Total Length of Fwd Packets", "Total Length of Bwd Packets",
] as const;

export const ALIASES: Record<string, string> = {
  "flow id": "Flow ID", "src ip": "Source IP", "dst ip": "Destination IP",
  "src port": "Source Port", "dst port": "Destination Port",
  "protocol": "Protocol", "timestamp": "Timestamp",
  "tot fwd pkts": "Total Fwd Packets", "tot bwd pkts": "Total Backward Packets",
  "total fwd packet": "Total Fwd Packets", "total bwd packet": "Total Backward Packets",
  "totlen fwd pkts": "Total Length of Fwd Packets", "totlen bwd pkts": "Total Length of Bwd Packets",
  "total length of fwd packet": "Total Length of Fwd Packets",
  "total length of bwd packet": "Total Length of Bwd Packets",
  "flow duration": "Flow Duration", "flow iat mean": "Flow IAT Mean",
  "fwd iat tot": "Fwd IAT Total", "bwd iat tot": "Bwd IAT Total",
};

export const TIER2_OPTIONAL = [
  "pkt_ttl_mean", "pkt_win_mean", "pkt_payload_var", "pkt_payload_nonzero_ratio",
  "pkt_payload_skew", "pkt_payload_kurtosis", "retransmission_count",
  "frag_more_flag_present", "frag_dont_flag_present",
  "port_scan_entropy", "port_scan_sequential_ratio",
  "port_scan_max_sequential_run", "unique_dst_ports_per_src",
];

export const FLAG_FROM: Record<string, string> = {
  flag_true_fin: "FIN Flag Count", flag_true_syn: "SYN Flag Count",
  flag_true_rst: "RST Flag Count", flag_true_psh: "PSH Flag Count",
  flag_true_ack: "ACK Flag Count", flag_true_urg: "URG Flag Count",
};

export const PROGRESSION_STATES = [
  "NORMAL", "PRE_ATTACK", "ONSET", "ACTIVE", "CONTINUATION",
] as const;

export const FEATURE_GROUPS: Record<string, string[]> = {
  "volume / rate": ["flow_count", "packet_count", "byte_count", "packet_rate", "byte_rate", "fwd_bwd_ratio"],
  "connection dynamics": ["unique_src", "unique_dst", "unique_pairs", "unique_dst_ports", "fan_out", "fan_in", "failed_conn_ratio"],
  "TCP flag rates": ["syn_rate", "ack_rate", "rst_rate", "fin_rate", "psh_rate", "urg_rate"],
  "timing / burstiness": ["flow_duration_mean", "iat_mean", "iat_std", "burstiness"],
  "packet layer (Tier-2)": ["ttl_mean", "ttl_var", "tcp_window_mean", "tcp_window_std", "payload_var_mean", "payload_nonzero_ratio", "payload_skew_mean", "payload_kurt_mean", "retransmission_rate", "frag_present_ratio"],
  "scan signatures": ["port_scan_entropy", "port_scan_seq_ratio", "port_scan_max_run", "unique_dst_ports_per_src"],
  "protocol / time": ["tcp_ratio", "udp_ratio", "time_since_prev_window", "window_index_in_day"],
  "distribution entropy": ["dstport_entropy", "dstip_entropy", "srcip_entropy", "flowsize_entropy"],
  "first differences": ["d_packet_rate", "d_byte_rate", "d_syn_rate", "d_rst_rate", "d_unique_dst", "d_unique_dst_ports", "d_fan_out", "d_failed_conn_ratio"],
};

export interface ColumnMatch {
  ok: boolean;
  requiredHit: string[];
  requiredMiss: string[];
  aliased: { from: string; to: string }[];
  tier2Hit: string[];
  flagHit: string[];
  extra: string[];
  labelled: boolean;
  synthTimeAxis: boolean;
}

export function matchColumns(headers: string[]): ColumnMatch {
  const have = new Set(headers.map((h) => h.trim()));
  const lower = new Map(headers.map((h) => [h.trim().toLowerCase(), h.trim()]));
  const canon = new Set(have);
  const aliased: { from: string; to: string }[] = [];
  for (const [lc, c] of Object.entries(ALIASES)) {
    if (lower.has(lc) && !have.has(c)) {
      canon.add(c);
      aliased.push({ from: lower.get(lc)!, to: c });
    }
  }
  const hasTimestamp = canon.has("Timestamp") || canon.has("flow_start_epoch");
  const requiredHit = REQUIRED.filter(
    (c) => canon.has(c) || (c === "flow_start_epoch" && hasTimestamp)
  );
  const requiredMiss = REQUIRED.filter(
    (c) => !(canon.has(c) || (c === "flow_start_epoch" && hasTimestamp))
  );
  const tier2Hit = TIER2_OPTIONAL.filter((c) => have.has(c));
  const flagHit = Object.entries(FLAG_FROM)
    .filter(([t, s]) => have.has(t) || have.has(s))
    .map(([t, s]) => (have.has(t) ? t : `${t}  (from ${s})`));
  const known = new Set<string>([
    ...REQUIRED, ...Object.values(ALIASES), ...TIER2_OPTIONAL,
    ...Object.keys(FLAG_FROM), ...Object.values(FLAG_FROM),
    "Timestamp", "Source Port", "Flow ID", "Label", "attack_family",
    "source_day", "source_file", "day",
  ]);
  const extra = headers
    .map((h) => h.trim())
    .filter((h) => !known.has(h) && !Object.keys(ALIASES).includes(h.toLowerCase()));
  return {
    ok: requiredMiss.length === 0,
    requiredHit: [...requiredHit],
    requiredMiss: [...requiredMiss],
    aliased,
    tier2Hit,
    flagHit,
    extra,
    labelled: have.has("Label"),
    synthTimeAxis: !canon.has("flow_start_epoch") && canon.has("Timestamp"),
  };
}
