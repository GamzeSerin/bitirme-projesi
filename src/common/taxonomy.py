"""Ekibin ortak saldırı taksonomisi.

Her veri seti kendi etiketlerini bu ortak kategorilere eşler. DSS (main.py)
yalnızca bu kategorileri ve ciddiyet seviyelerini bilir. Yeni bir kategori
gerekiyorsa buraya ekleyin ve PR açın; herkes aynı listeyi kullanmalı.
"""

# Ortak kategori -> ciddiyet (0 = zararsız, 3 = kritik)
SEVERITY = {
    "BENIGN": 0,
    "RECON": 1,
    "SPOOFING": 2,
    "BRUTEFORCE": 2,
    "WEB": 2,
    "MALWARE": 3,
    "DOS": 3,
    "DDOS": 3,
    "MIRAI": 3,
}

COMMON_CLASSES = list(SEVERITY)


def ciciot2023_category(label: str) -> str:
    """CICIoT2023 34 sınıflık etiketi 8 ortak kategoriye eşler (resmi gruplama)."""
    label = label.strip().upper()
    if label == "BENIGN":
        return "BENIGN"
    if label.startswith("DDOS-"):
        return "DDOS"
    if label.startswith("DOS-"):
        return "DOS"
    if label.startswith("MIRAI-"):
        return "MIRAI"
    if label.startswith("RECON-") or label == "VULNERABILITYSCAN":
        return "RECON"
    if label in {"DNS_SPOOFING", "MITM-ARPSPOOFING"}:
        return "SPOOFING"
    if label == "DICTIONARYBRUTEFORCE":
        return "BRUTEFORCE"
    if label in {"BROWSERHIJACKING", "XSS", "UPLOADING_ATTACK", "SQLINJECTION",
                 "COMMANDINJECTION", "BACKDOOR_MALWARE"}:
        return "WEB"
    raise ValueError(f"Eşlenmemiş CICIoT2023 etiketi: {label}")
