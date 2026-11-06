"""Çok şirketli (çok kiracılı) model: demo sigorta şirketleri ve poliçe → şirket eşlemesi.

Saf Python; Streamlit ve pandas içe aktarmaz.

Bir müşteri tek bir şirketle çalışmaz: poliçeleri farklı şirketlerde olabilir. Bu yüzden şirket MÜŞTERİYE değil
POLİÇEYE bağlıdır (``sirket`` sütunu). Bir sigortacı yalnızca kendi şirketinin poliçelerini görür; bir müşteri
tüm poliçelerini, her birinin şirket adıyla birlikte görür.

Dürüst sınır: buradaki şirketler **kurgusaldır** (gerçek bir kurumla ilgisi yoktur) ve poliçelerin şirketlere dağılımı
sentetiktir; poliçe numarasından kararlı bir özetle türetilir (rastgele sayı üretecini etkilemez). Gerçek kullanımda
şirket bilgisi müşterinin kendi portföy dosyasından gelir.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

__all__ = ["Company", "COMPANIES", "DEFAULT_COMPANY", "company_ids", "valid_company", "company_name", "company_short",
           "assign_company", "assign_companies"]


@dataclass(frozen=True)
class Company:
    """Kurgusal demo sigorta şirketi.

    Attributes:
        id: Kararlı kimlik (hesap yapılandırmasında ve veri sütununda kullanılır).
        name: Ekranda görünen ad (kurgusal olduğu adda belirtilir).
        short: Kısa kod (rozet ve tablo için).
    """

    id: str
    name: str
    short: str


#: Kurgusal demo şirketleri. Adların hiçbir gerçek kurumla ilişkisi yoktur; "(demo)" eki bunu görünür kılar.
COMPANIES: Tuple[Company, ...] = (
    Company("boztepe", "Boztepe Sigorta (demo)", "BOZ"),
    Company("yaylacik", "Yaylacık Sigorta (demo)", "YAY"),
    Company("camliova", "Çamlıova Sigorta (demo)", "ÇAM"),
)
#: ``company`` anahtarı verilmeyen sigortacı hesabının bağlandığı şirket (kısıtlı tarafa düşer, tümünü görmez).
DEFAULT_COMPANY: str = COMPANIES[0].id
_BY_ID: Dict[str, Company] = {c.id: c for c in COMPANIES}
#: Eşleme özetinin tuzu: değiştirilirse poliçelerin şirketleri değişir. Demo için, varsayılan tohum ve kurallarla
#: ilk üç demo müşterinin poliçelerinin üç şirkete de dağılması için seçilmiştir (``test_casework.py`` bunu denetler).
_SALT = "orbital-tenant-v16"


def company_ids() -> Tuple[str, ...]:
    """Tüm şirket kimlikleri (sırayla)."""
    return tuple(c.id for c in COMPANIES)


def valid_company(value: object) -> bool:
    """``value`` bilinen bir şirket kimliği mi? (Metin olmayan değerler geçersizdir.)"""
    return isinstance(value, str) and value in _BY_ID


def company_name(company_id: object) -> str:
    """Şirket kimliğinin ekran adı; bilinmiyorsa ``—``."""
    return _BY_ID[company_id].name if valid_company(company_id) else "—"          # type: ignore[index]


def company_short(company_id: object) -> str:
    """Şirket kimliğinin kısa kodu; bilinmiyorsa ``—``."""
    return _BY_ID[company_id].short if valid_company(company_id) else "—"         # type: ignore[index]


def assign_company(police_no: str) -> str:
    """Poliçe numarasından kararlı biçimde bir şirket seçer (rastgele sayı üreteci kullanılmaz).

    Args:
        police_no: Poliçe numarası.

    Returns:
        Şirket kimliği.
    """
    digest = hashlib.sha256(f"{_SALT}|{police_no}".encode("utf-8")).digest()
    return COMPANIES[int.from_bytes(digest[:4], "big") % len(COMPANIES)].id


def assign_companies(police_nos: Sequence[str]) -> Tuple[str, ...]:
    """:func:`assign_company` işlemini bir poliçe listesine uygular."""
    return tuple(assign_company(p) for p in police_nos)
